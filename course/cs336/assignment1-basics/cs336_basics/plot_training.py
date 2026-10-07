"""Generate an offline HTML dashboard from training JSONL logs, or serve live updates."""

import argparse
import errno
import glob
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def find_logs(patterns: list[str]) -> list[Path]:
    files = set()
    for pattern in patterns:
        for name in glob.glob(pattern, recursive=True):
            path = Path(name)
            candidates = path.rglob("*.jsonl") if path.is_dir() else [path]
            files.update(candidate.resolve() for candidate in candidates
                         if candidate.is_file() and candidate.suffix == ".jsonl")
    return sorted(files)


def number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def load_logs(patterns: list[str]) -> dict:
    files = find_logs(patterns)
    if not files:
        raise ValueError("No JSONL logs found: " + ", ".join(patterns))
    sessions = []
    warnings = []
    for path in files:
        groups = {}
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    if not line.endswith("\n"):
                        warnings.append(f"{path.name}: ignored unfinished final line")
                        break
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                key = (str(record.get("run_id", path.parent.name)),
                       str(record.get("session_id", path.stem)))
                session = groups.setdefault(key, {
                    "id": f"{path}:{key}", "run_id": key[0], "session_id": key[1],
                    "file": str(path), "config": {}, "status": "running",
                    "train": [], "validation": [], "diagnostics": [],
                })
                if record.get("event") == "start":
                    session["config"] = record.get("config", {})
                if record.get("event") == "finished":
                    session["status"] = "finished"
                event = record.get("event")
                if event == "diagnostics":
                    point = {field: number(record.get(field)) for field in ("step", "elapsed_seconds")}
                    point["activations"] = record.get("activations", {})
                    for field in ("weights", "gradients_before_clip", "gradients_after_clip"):
                        point[field] = record.get(field, {})
                    session["diagnostics"].append(point)
                if event in ("train", "validation"):
                    point = {field: number(record.get(field)) for field in
                             ("step", "elapsed_seconds", "train_loss", "val_loss", "lr", "averaged_steps")}
                    if point["step"] is not None or point["elapsed_seconds"] is not None:
                        session[event].append(point)
        sessions.extend(groups.values())
    return {"sessions": sessions, "warnings": warnings}


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CS336 训练曲线</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#17243a;font:15px system-ui,sans-serif}
main{max-width:1250px;margin:32px auto;padding:0 22px}h1{font-size:27px;margin-bottom:7px}
p{color:#596779;line-height:1.6}.card{background:white;border:1px solid #e0e6ef;border-radius:12px;padding:20px;margin:18px 0}
.controls,.legend{display:flex;gap:18px;align-items:center;flex-wrap:wrap}select,button{font:inherit;padding:6px 10px;border:1px solid #ccd5e2;border-radius:6px;background:white}
label{cursor:pointer}h2{font-size:18px;margin:0 0 12px}.chart{position:relative}svg{width:100%;height:320px;display:block}
.tip{display:none;position:absolute;pointer-events:none;background:#17243a;color:white;padding:9px 12px;border-radius:7px;font-size:13px;white-space:pre;z-index:2}
.swatch{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px}
.small{font-size:13px;color:#596779}details{margin-top:12px}pre{overflow:auto;background:#f4f6fa;padding:14px;font-size:12px}
.error{color:#aa2525}.stats{display:flex;gap:24px;flex-wrap:wrap;margin-top:15px}.stats strong{display:block;font-size:21px;color:#17243a}
</style></head><body><main>
<h1>CS336 训练曲线</h1>
<p>鼠标悬停查看数值；可切换横轴、对数刻度和实验。每个启动 session 单独绘制，耗时从该次启动计起。</p>
<div class="card"><div class="controls">
<label>横轴 <select id="axis"><option value="step">训练步数 step</option><option value="elapsed_seconds">本次启动耗时（秒）</option></select></label>
<label><input id="log" type="checkbox"> loss 对数刻度</label>
<label><input id="train" type="checkbox" checked>训练 loss（实线）</label>
<label><input id="val" type="checkbox" checked>验证 loss（虚线）</label>
<button id="refresh">刷新</button><span id="status" class="small"></span>
</div><div id="legend" class="legend" style="margin-top:18px"></div><div id="stats" class="stats"></div>
<p id="message" class="small"></p></div>
<section class="card"><h2>Loss</h2><div class="chart"><svg id="loss" viewBox="0 0 1000 320" role="img" aria-label="Loss 曲线"></svg><div class="tip"></div></div></section>
<section class="card"><h2>学习率</h2><div class="chart"><svg id="lr" viewBox="0 0 1000 320" role="img" aria-label="学习率曲线"></svg><div class="tip"></div></div></section>
<section class="card"><h2>范数诊断</h2>
<div class="controls">
<label>指标 <select id="norm-metric"><option value="rms">RMS</option><option value="l2">L2 范数</option></select></label>
<label><input id="norm-log" type="checkbox">范数对数刻度</label>
<label>激活模块 <select id="activation-module"></select></label>
<label>权重 / 梯度 <select id="parameter-name"></select></label>
</div>
<p class="small">当前采样步快照；权重为更新前数值。梯度实线为裁剪前，虚线为裁剪后。RMS 便于比较不同大小的张量。对数刻度不显示零值。</p>
<p id="norm-message" class="small"></p>
<h2>激活</h2><div class="chart"><svg id="activation-norm" viewBox="0 0 1000 320" role="img" aria-label="激活范数"></svg><div class="tip"></div></div>
<h2>权重</h2><div class="chart"><svg id="weight-norm" viewBox="0 0 1000 320" role="img" aria-label="权重范数"></svg><div class="tip"></div></div>
<h2>梯度</h2><div class="chart"><svg id="gradient-norm" viewBox="0 0 1000 320" role="img" aria-label="梯度范数"></svg><div class="tip"></div></div>
</section>
<section class="card"><h2>实验配置与日志来源</h2><div id="configs"></div></section>
</main><script id="initial-data" type="application/json">__DATA__</script><script>
let data=JSON.parse(document.getElementById('initial-data').textContent);
const live=__LIVE__, hidden=new Set(), colors=['#2563eb','#dc6b16','#12825a','#9333ea','#db2777','#0891b2'];
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=v=>!Number.isFinite(v)?'—':Math.abs(v)<0.001&&v!==0?v.toExponential(2):Number(v.toPrecision(5)).toString();
function chart(id, series, logarithmic){
 const svg=$(id), tip=svg.parentElement.querySelector('.tip'), xfield=$('axis').value;
 const left=85,right=975,top=15,bottom=270;
 const valid=series.map(s=>({...s,points:s.points.filter(p=>Number.isFinite(p[xfield])&&Number.isFinite(p.value)&&(!logarithmic||p.value>0))}));
 const all=valid.flatMap(s=>s.points.map(p=>({...p,label:s.label,color:s.color})));
 if(!all.length){svg.innerHTML='<text x="90" y="145" fill="#596779">尚无可显示的数据</text>';svg.onmousemove=null;tip.style.display='none';return;}
 let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
 for(const p of all){const y=logarithmic?Math.log10(p.value):p.value;xmin=Math.min(xmin,p[xfield]);xmax=Math.max(xmax,p[xfield]);ymin=Math.min(ymin,y);ymax=Math.max(ymax,y);}
 if(xmax===xmin){xmin-=0.5;xmax+=0.5;}if(ymax===ymin){ymin-=0.5;ymax+=0.5;}
 if(!logarithmic){ymin=Math.min(0,ymin);ymax+=Math.max((ymax-ymin)*0.05,1e-12);}
 const X=v=>left+(v-xmin)/(xmax-xmin)*(right-left),Y=v=>bottom-((logarithmic?Math.log10(v):v)-ymin)/(ymax-ymin)*(bottom-top);
 let parts=[];
 for(let i=0;i<=5;i++){
  let x=left+(right-left)*i/5,y=bottom-(bottom-top)*i/5;
  parts.push(`<path d="M ${left} ${y} H ${right}" stroke="#e6ebf2"/><text x="${left-10}" y="${y+4}" text-anchor="end" font-size="12" fill="#596779">${fmt(logarithmic?10**(ymin+(ymax-ymin)*i/5):ymin+(ymax-ymin)*i/5)}</text>`);
  parts.push(`<text x="${x}" y="${bottom+23}" text-anchor="middle" font-size="12" fill="#596779">${fmt(xmin+(xmax-xmin)*i/5)}</text>`);
 }
 parts.push(`<text x="530" y="315" text-anchor="middle" fill="#596779" font-size="13">${xfield==='step'?'训练步数 step':'本次启动耗时（秒）'}</text>`);
 for(const s of valid){
  const points=[...s.points].sort((a,b)=>a[xfield]-b[xfield]);
  const d=points.map((p,i)=>`${i?'L':'M'} ${X(p[xfield])} ${Y(p.value)}`).join(' ');
  parts.push(`<path class="series" d="${d}" fill="none" stroke="${s.color}" stroke-width="2" ${s.dash?'stroke-dasharray="7 5"':''}/>`);
  for(const p of points)parts.push(`<circle cx="${X(p[xfield])}" cy="${Y(p.value)}" r="2.5" fill="${s.color}"/>`);
 }
 parts.push('<circle id="'+id+'-hover" r="5" fill="white" stroke="#17243a" stroke-width="2" visibility="hidden"/>');
 svg.innerHTML=parts.join('');
 svg.onmousemove=e=>{
  const cursor=svg.createSVGPoint();cursor.x=e.clientX;cursor.y=e.clientY;
  const p=cursor.matrixTransform(svg.getScreenCTM().inverse());
  let best=all[0],distance=Infinity;
  for(const candidate of all){const d=(X(candidate[xfield])-p.x)**2+(Y(candidate.value)-p.y)**2;if(d<distance){best=candidate;distance=d;}}
  const marker=$(id+'-hover');marker.setAttribute('cx',X(best[xfield]));marker.setAttribute('cy',Y(best.value));marker.setAttribute('visibility','visible');
  tip.textContent=`${best.label}\nstep: ${best.step}\n耗时: ${fmt(best.elapsed_seconds)} s\n数值: ${fmt(best.value)}`+(best.averaged_steps?`\n平均步数: ${best.averaged_steps}`:'');
  tip.style.display='block';const rect=svg.parentElement.getBoundingClientRect();
  tip.style.left=Math.max(0,Math.min(e.clientX-rect.left+12,rect.width-tip.offsetWidth))+'px';tip.style.top=Math.max(0,e.clientY-rect.top-tip.offsetHeight-8)+'px';
 };
 svg.onmouseleave=()=>{tip.style.display='none';$(id+'-hover').setAttribute('visibility','hidden');};
}
function normCharts(){
 const visible=data.sessions.filter(s=>!hidden.has(s.id)),modules=new Set(),parameters=new Set();
 for(const s of visible)for(const p of s.diagnostics||[]){
  Object.keys(p.activations||{}).forEach(k=>modules.add(k));
  for(const field of ['weights','gradients_before_clip','gradients_after_clip'])Object.keys(p[field]?.per_parameter||{}).forEach(k=>parameters.add(k));
 }
 function options(id,items,global){
  const previous=$(id).value;
  $(id).innerHTML=(global?'<option value="">整个模型</option>':'')+[...items].sort().map(k=>`<option value="${esc(k)}">${esc(k)}</option>`).join('');
  if([...$(id).options].some(o=>o.value===previous))$(id).value=previous;
 }
 options('activation-module',modules,false);options('parameter-name',parameters,true);
 const activation=[],weights=[],gradients=[],metric=$('norm-metric').value,module=$('activation-module').value,parameter=$('parameter-name').value;
 let count=0,invalid=0,missing=0;
 function value(stats){if(stats?.finite===false)invalid++;if(!stats)missing++;return stats?.[metric]??null;}
 data.sessions.forEach((s,i)=>{
  if(hidden.has(s.id))return;
  const points=s.diagnostics||[],color=colors[i%colors.length],label=`${s.run_id} / ${s.session_id}`;count+=points.length;
  const calls=Math.max(0,...points.map(p=>(p.activations?.[module]||[]).length));
  for(let call=0;call<calls;call++)activation.push({label:`${label} / ${module} / 调用 ${call+1} / ${metric}`,color,dash:call%2===1,points:points.map(p=>({...p,value:value(p.activations?.[module]?.[call])}))});
  for(const [field,target,dash] of [['weights',weights,false],['gradients_before_clip',gradients,false],['gradients_after_clip',gradients,true]]){
   const kind={weights:'权重',gradients_before_clip:'裁剪前梯度',gradients_after_clip:'裁剪后梯度'}[field];
   target.push({label:`${label} / ${parameter||'整个模型'} / ${kind} / ${metric}`,color,dash,points:points.map(p=>({...p,value:value(parameter?p[field]?.per_parameter?.[parameter]:p[field])}))});
  }
 });
 const old=visible.filter(s=>!(s.diagnostics||[]).length).length;
 $('norm-message').textContent=(count?`已读取 ${count} 个诊断采样步。`:'尚无范数日志；旧训练进程不会自动加载新代码，需使用更新后的代码启动训练。')+(old?` ${old} 个 session 没有诊断记录。`:'')+(invalid?` 当前选择含 ${invalid} 个非有限统计，未绘制，请检查原始日志。`:'')+(missing?` 当前选择含 ${missing} 个缺失统计，未绘制。`:'');
 for(const [id,series] of [['activation-norm',activation],['weight-norm',weights],['gradient-norm',gradients]])chart(id,series,$('norm-log').checked);
}
function render(){
 const loss=[],lr=[];let configs='',legend='',stats='';
 data.sessions.forEach((s,i)=>{
  const color=colors[i%colors.length],label=`${s.run_id} / ${s.session_id}`;
  legend+=`<label><input type="checkbox" data-index="${i}" ${hidden.has(s.id)?'':'checked'}><span class="swatch" style="background:${color}"></span>${esc(label)}</label>`;
  configs+=`<details><summary>${esc(label)} · ${s.status==='finished'?'已结束':'尚无结束记录'}</summary><p class="small">${esc(s.file)}</p><pre>${esc(JSON.stringify(s.config,null,2))}</pre></details>`;
  if(hidden.has(s.id))return;
  const trains=s.train.filter(p=>Number.isFinite(p.train_loss)),vals=s.validation.filter(p=>Number.isFinite(p.val_loss));
  if($('train').checked)loss.push({label:label+' / train loss',color,points:trains.map(p=>({...p,value:p.train_loss}))});
  if($('val').checked)loss.push({label:label+' / validation loss',color,dash:true,points:vals.map(p=>({...p,value:p.val_loss}))});
  lr.push({label:label+' / lr',color,points:s.train.map(p=>({...p,value:p.lr}))});
  const last=trains.at(-1),val=vals.at(-1);
  stats+=`<div class="small">${esc(s.run_id)}<strong>${last?fmt(last.train_loss):'—'}</strong>最新训练 loss${last?' · step '+last.step:''}${val?'<br>最新验证 loss '+fmt(val.val_loss):''}</div>`;
 });
 $('legend').innerHTML=legend;$('configs').innerHTML=configs;$('stats').innerHTML=stats;
 for(const box of $('legend').querySelectorAll('input'))box.onchange=()=>{const id=data.sessions[box.dataset.index].id;box.checked?hidden.delete(id):hidden.add(id);render();};
 $('message').textContent='训练 loss 为日志记录区间的平均值。固定 batch 实验主要观察训练 loss，验证 loss 不要求趋近 0。'+(data.warnings.length?' '+data.warnings.join('；'):'');
 chart('loss',loss,$('log').checked);chart('lr',lr,false);
 normCharts();
}
async function refresh(){
 if(!live){$('status').textContent='离线快照：重新运行脚本可更新 HTML。';return;}
 try{const response=await fetch('/data.json',{cache:'no-store'});if(!response.ok)throw Error(await response.text());data=await response.json();render();$('status').textContent='实时更新 · '+new Date().toLocaleTimeString();$('status').className='small';}
 catch(error){$('status').textContent='刷新失败：'+error.message;$('status').className='small error';}
}
for(const id of ['axis','log','train','val','norm-metric','norm-log','activation-module','parameter-name'])$(id).onchange=render;
$('refresh').onclick=refresh;render();refresh();if(live)setInterval(refresh,3000);
</script></body></html>'''


def make_html(data: dict, live: bool = False) -> str:
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return HTML.replace("__LIVE__", "true" if live else "false").replace("__DATA__", encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+", default=["runs"], help="JSONL files, directories, or quoted glob patterns")
    parser.add_argument("--output", type=Path, default=Path("runs/training_curves.html"))
    parser.add_argument("--serve", action="store_true", help="Serve live charts with automatic refresh every 3 seconds")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        data = load_logs(args.logs)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(make_html(data), encoding="utf-8")
    print(f"HTML: {args.output.resolve()}", flush=True)
    if not args.serve:
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split("?", 1)[0]
            if route not in ("/", "/data.json"):
                self.send_error(404)
                return
            try:
                latest = load_logs(args.logs)
                content = (make_html(latest, live=True) if route == "/" else
                           json.dumps(latest, ensure_ascii=False, allow_nan=False))
            except (ValueError, OSError) as error:
                self.send_error(503, str(error))
                return
            body = content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8" if route == "/" else
                             "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            parser.error(f"Port {args.port} is already in use. Open http://{args.host}:{args.port} "
                         "if a dashboard is already running, or choose another port with --port.")
        parser.error(f"Cannot start dashboard on {args.host}:{args.port}: {error}")
    with server:
        print(f"Live dashboard: http://{args.host}:{server.server_port} (Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
