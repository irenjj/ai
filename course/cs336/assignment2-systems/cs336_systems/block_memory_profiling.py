"""Measure live allocations and saved storages for one block in a real training step."""
import argparse
from collections import defaultdict
import json
import linecache
from pathlib import Path

import torch
from cs336_basics import model, optimizer, nn_utils


def profile_block(output_dir, context_length=128, block_index=12):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "result.json").exists() or (output_dir / "snapshot.pickle").exists():
        raise FileExistsError("Use a fresh output directory to preserve previous captures")
    torch.manual_seed(42)
    m = model.BasicsTransformerLM(10000, context_length, 1024, 24, 16, 4096).cuda()
    opt = optimizer.AdamW(m.parameters())
    tokens = torch.randint(10000, (4, context_length + 1), device="cuda")
    x, target = tokens[:, :-1], tokens[:, 1:]

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.cuda.nvtx.range("forward"):
            logits = m(x)
        with torch.cuda.nvtx.range("loss"):
            loss = nn_utils.cross_entropy(logits.reshape(-1, 10000), target.reshape(-1))
        with torch.cuda.nvtx.range("backward"):
            loss.backward()
        with torch.cuda.nvtx.range("optimizer"):
            opt.step()
        torch.cuda.synchronize()

    for _ in range(5):
        step()
    block = m.layers[block_index]
    parameter_ptrs = {p.untyped_storage().data_ptr() for p in m.parameters()}
    saved = {}
    boundaries = {}

    def capture(name):
        torch.cuda.synchronize()
        snapshot = torch.cuda.memory._snapshot()
        live = {}
        for segment in snapshot["segments"]:
            for b in segment["blocks"]:
                if b["state"] == "active_allocated":
                    live[b["address"]] = dict(size=b["size"], requested_size=b["requested_size"],
                                              frames=b.get("frames", []))
        allocated = torch.cuda.memory_allocated()
        boundaries[name] = dict(allocated_bytes=allocated, live=live)
        torch.cuda.nvtx.mark(f"block{block_index}/{name}/allocated_bytes={allocated}")

    def pack(t):
        storage = t.untyped_storage()
        addr = storage.data_ptr()
        entry = saved.setdefault(addr, dict(bytes=storage.nbytes(), parameter=addr in parameter_ptrs,
                                           shapes=[], saves=0))
        entry["saves"] += 1
        entry["shapes"].append(list(t.shape))
        # The reporting structures retain metadata only, never tensor references.
        return t.detach()

    original = block.forward

    def measured_forward(*args, **kwargs):
        with torch.cuda.nvtx.range(f"block{block_index}/forward"):
            capture("forward_before")
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                result = original(*args, **kwargs)
            capture("forward_after")
            return result

    block.forward = measured_forward
    backward_range = None

    def backward_pre(module, grad_output):
        nonlocal backward_range
        # Hooks straddle autograd's nested NVTX ranges; use an independent range ID.
        backward_range = torch.cuda.nvtx.range_start(f"block{block_index}/backward")
        capture("backward_before")

    def backward_post(module, grad_input, grad_output):
        capture("backward_after")
        torch.cuda.nvtx.range_end(backward_range)

    handles = [block.register_full_backward_pre_hook(backward_pre),
               block.register_full_backward_hook(backward_post)]
    torch.cuda.memory._record_memory_history(max_entries=300000)
    torch.cuda.profiler.start()
    try:
        with torch.autograd.profiler.emit_nvtx(record_shapes=True):
            step()
        torch.cuda.memory._dump_snapshot(str(output_dir / "snapshot.pickle"))
    finally:
        torch.cuda.profiler.stop()
        torch.cuda.memory._record_memory_history(enabled=None)
        for h in handles:
            h.remove()
        block.forward = original

    before = boundaries["forward_before"]["live"]
    after = boundaries["forward_after"]["live"]
    groups = defaultdict(lambda: dict(bytes=0, storages=0))
    new_saved = 0
    for addr, entry in saved.items():
        entry["preexisting"] = addr in before
        if entry["parameter"] or entry["preexisting"]:
            continue
        assert addr in after, "Saved tensor missing at forward boundary"
        frames = after[addr]["frames"]
        frame = next((f for f in frames if "/cs336_basics/" in f["filename"]), None)
        assert frame is not None, "Missing allocation source"
        source = f"{Path(frame['filename']).name}:{frame['line']} {linecache.getline(frame['filename'], frame['line']).strip()}"
        entry["source"] = source
        new_saved += entry["bytes"]
        groups[source]["bytes"] += entry["bytes"]
        groups[source]["storages"] += 1

    pre = boundaries["backward_before"]["live"]
    post = boundaries["backward_after"]["live"]
    freed = sum(v["size"] for a,v in pre.items() if a not in post)
    added = sum(v["size"] for a,v in post.items() if a not in pre)
    resized = sum(post[a]["size"]-pre[a]["size"] for a in pre.keys() & post.keys())
    parameter_bytes = sum(p.numel()*p.element_size() for p in block.parameters())
    gradient_bytes = sum(p.grad.numel()*p.grad.element_size() for p in block.parameters() if p.grad is not None)
    assert parameter_bytes == gradient_bytes
    inferred_gradient_bytes = new_saved + boundaries["backward_after"]["allocated_bytes"] - boundaries["backward_before"]["allocated_bytes"]
    assert inferred_gradient_bytes == gradient_bytes, "Boundary balance differs from parameter gradients; inspect shared tensors"
    assert added-freed+resized == boundaries["backward_after"]["allocated_bytes"]-boundaries["backward_before"]["allocated_bytes"]
    result = dict(context_length=context_length, batch_size=4, block_index=block_index,
                  precision="fp32", model="medium", warmup_steps=5, measured_steps=1,
                  torch_version=torch.__version__, device=torch.cuda.get_device_name(),
                  new_saved_storage_bytes=new_saved,
                  preexisting_nonparameter_saved_bytes=sum(e["bytes"] for e in saved.values() if e["preexisting"] and not e["parameter"]),
                  referenced_parameter_storage_bytes=sum(e["bytes"] for e in saved.values() if e["parameter"]),
                  groups=[dict(source=s,**v,share_percent=100*v["bytes"]/new_saved) for s,v in sorted(groups.items(),key=lambda kv:-kv[1]["bytes"])],
                  parameter_bytes=parameter_bytes,parameter_gradient_bytes=gradient_bytes,
                  inferred_parameter_gradient_bytes=inferred_gradient_bytes,
                  backward_boundary_freed_bytes=freed,backward_boundary_added_bytes=added,
                  backward_boundary_same_address_size_delta=resized,
                  boundaries={k:v["allocated_bytes"] for k,v in boundaries.items()},
                  saved_storages=saved)
    (output_dir / "result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='saved_storages'},indent=2))
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--context-length",type=int,default=128)
    args=parser.parse_args()
    profile_block(args.output_dir,args.context_length)
