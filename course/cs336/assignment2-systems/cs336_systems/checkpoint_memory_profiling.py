"""Compare non-nested checkpoint groups in independent CUDA processes."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from cs336_basics import model, optimizer
from cs336_systems.benchmark import run


class CheckpointGroup(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def compute(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        return checkpoint(self.compute, x, use_reentrant=False)


def group_layers(m, size):
    if size:
        layers = list(m.layers)
        m.layers = nn.ModuleList(CheckpointGroup(layers[i:i+size]) for i in range(0,len(layers),size))


def check_correctness():
    import copy
    torch.manual_seed(42)
    baseline=model.BasicsTransformerLM(64,16,32,4,4,64).cuda()
    x=torch.randint(64,(2,16),device='cuda')
    expected=baseline(x)
    expected.square().mean().backward()
    for size in (1,2,3,4):
        candidate=copy.deepcopy(baseline)
        candidate.zero_grad(set_to_none=True)
        group_layers(candidate,size)
        actual=candidate(x)
        torch.testing.assert_close(actual,expected)
        actual.square().mean().backward()
        for a,b in zip(candidate.parameters(),baseline.parameters()):
            torch.testing.assert_close(a.grad,b.grad)
    print('Outputs and all parameter gradients match for k=1,2,3,4',flush=True)


def worker(size):
    torch.manual_seed(42)
    result=dict(group_size=size,device=torch.cuda.get_device_name(),torch_version=torch.__version__,
                model='medium',batch_size=4,context_length=2048,precision='fp32',
                warmup_steps=5,measurement_steps=10,use_reentrant=False)
    phase='initialization'
    try:
        m=model.BasicsTransformerLM(10000,2048,1024,24,16,4096).cuda()
        layers=list(m.layers)
        calls=[0]*24
        def counter(i):
            def hook(module,args):calls[i]+=1
            return hook
        handles=[layer.register_forward_pre_hook(counter(i)) for i,layer in enumerate(layers)]
        group_layers(m,size)
        opt=optimizer.AdamW(m.parameters())
        origin=torch.randint(10000,(4*2048+1,),device='cuda')
        x=origin[:-1].view(4,2048);target=origin[1:].view(4,2048)
        phase='warmup'
        for _ in range(5):run(m,opt,3,x,target,False)
        calls[:]=[0]*24
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline=torch.cuda.memory_allocated()
        phase='measurement'
        samples=[run(m,opt,3,x,target,True) for _ in range(10)]
        assert calls==[20 if size else 10]*24, calls
        result.update(status='ok',start_allocated_bytes=baseline,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      mean_step_ms=statistics.mean(s['total']*1000 for s in samples),
                      std_step_ms=statistics.pstdev(s['total']*1000 for s in samples),
                      forward_calls_per_layer=calls)
        for h in handles:h.remove()
    except torch.cuda.OutOfMemoryError as e:
        result.update(status='OOM',failed_phase=phase,error=str(e))
    print('RESULT_JSON:'+json.dumps(result),flush=True)


def sweep(output_dir):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    # Run correctness checks separately so the coordinator owns no CUDA context.
    subprocess.run([sys.executable,'-m',__spec__.name,'--check'],check=True)
    results=[]
    for size in (0,1,2,3,4,6,8,12,24):
        print(f'Running k={size}',flush=True)
        p=subprocess.run([sys.executable,'-m',__spec__.name,'--worker',str(size)],capture_output=True,text=True)
        (output_dir/f'k{size}.log').write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError(f'Worker k={size} failed; see log')
        r=json.loads(next(s.removeprefix('RESULT_JSON:') for s in p.stdout.splitlines() if s.startswith('RESULT_JSON:')))
        results.append(r)
        (output_dir/'results.json').write_text(json.dumps(results,indent=2))
        print(r,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--worker',type=int)
    p.add_argument('--check',action='store_true')
    p.add_argument('--output-dir')
    args=p.parse_args()
    if args.check:check_correctness()
    elif args.worker is not None:worker(args.worker)
    elif args.output_dir:sweep(args.output_dir)
    else:p.error('--output-dir is required')
