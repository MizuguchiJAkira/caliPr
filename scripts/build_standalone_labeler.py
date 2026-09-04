"""Build a single-file landmark labeler that needs no install.

The server-backed labeler is the right tool for someone who already has the repo
and a terminal. For classmates who do not, it is a harder sell than ImageJ, which
defeats the point. This emits one .html file: open it, pick your photographs from
a file dialog, click landmarks, export. No Python, no server, no repo.

The schema is baked in at build time from ``landmark_config`` (minus any dataset
profile), so the standalone cannot drift from what the measurement engine and the
pose model expect. Rebuild it whenever the schema changes.

Landmarks only -- no polygons, no ruler, no calibration. That is everything
geomorph needs, and it is what makes the task explainable in two sentences.

Usage::

    python scripts/build_standalone_labeler.py --profile data/alewife \\
        --title "Alewife landmarks" --out dist/calipr-alewife.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fish_morpho.landmark_config import KEYPOINTS, View  # noqa: E402


def landmarks_for(profile_dir: Path | None) -> list[dict]:
    drop: set[str] = set()
    if profile_dir is not None:
        f = profile_dir / "schema.json"
        if f.is_file():
            drop = set(json.loads(f.read_text()).get("exclude_keypoints") or [])
    return [
        {"name": k.name, "hint": k.labeling_hint, "desc": k.description}
        for k in KEYPOINTS
        if k.view == View.LATERAL and k.name not in drop
    ]


THEMES = {
    # The canvas ground stays dark in BOTH themes. The surround must not out-shine
    # the photograph: these specimens are shot against near-black tanks, and a
    # bright panel beside a dark image forces constant eye adaptation, which costs
    # precision on exactly the faint margins that are hardest to judge.
    "dark": ("--bg:#12151a;--panel:#1b2029;--line:#2b3340;--fg:#e6ebf2;--mut:#8a97a8;"
             "--accent:#4aa3ff;--good:#37c871;--warn:#ffb454;--kp:#ff5d6c;"
             "--btn:#28303c;--btnhover:#313b49;--hover:#232a35;--sel:#26303d;"
             "--dot:#3a4553;--hintbg:#161b22;--kbd:#0e1218;--canvas:#0b0e12;"
             "--dropfg:#e6ebf2;--dropmut:#8a97a8;"),
    "light": ("--bg:#ffffff;--panel:#f7f7f7;--line:#d9d9d9;--fg:#1a1a1a;--mut:#666;"
              "--accent:#1a6bb5;--good:#2e7d32;--warn:#b26a00;--kp:#c62828;"
              "--btn:#ffffff;--btnhover:#eee;--hover:#f0f0f0;--sel:#e4eef7;"
              "--dot:#c4c4c4;--hintbg:#f0f0f0;--kbd:#eee;--canvas:#3a3a3a;"
              "--dropfg:#f0f0f0;--dropmut:#c8c8c8;"),
}
FONTS = {
    "dark": "-apple-system,Segoe UI,Roboto,sans-serif",
    "light": "'Lucida Grande',Helvetica,Arial,sans-serif",
}

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
 :root{__THEME__}
 *{box-sizing:border-box}
 html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
   font:13px/1.45 __FONT__;overflow:hidden}
 #app{display:grid;grid-template-columns:300px 1fr;height:100vh}
 #side{background:var(--panel);border-right:1px solid var(--line);display:flex;
   flex-direction:column;min-height:0}
 h1{font-size:14px;margin:0;padding:12px 14px;border-bottom:1px solid var(--line)}
 h1 small{color:var(--mut);font-weight:400}
 .sec{padding:10px 14px;border-bottom:1px solid var(--line)}
 button{background:var(--btn);color:var(--fg);border:1px solid var(--line);
   border-radius:6px;padding:6px 9px;cursor:pointer;font-size:12px}
 button:hover{background:var(--btnhover)}
 button.primary{background:var(--good);border-color:var(--good);color:#052;font-weight:600}
 .row{display:flex;gap:6px}.row button{flex:1}
 #hint{padding:9px 14px;background:var(--hintbg);color:var(--mut);font-size:12px;
   border-bottom:1px solid var(--line);min-height:60px;max-height:150px;overflow:auto;flex:none}
 #hint b{color:var(--fg)}
 #tasks{overflow:auto;flex:1;min-height:120px}
 .task{display:flex;align-items:center;gap:8px;padding:5px 14px;cursor:pointer;
   border-left:3px solid transparent}
 .task:hover{background:var(--hover)}
 .task.active{background:var(--sel);border-left-color:var(--accent)}
 .task .dot{width:9px;height:9px;border-radius:50%;background:var(--dot);flex:none}
 .task.set .dot{background:var(--good)}
 .task .nm{flex:1}
 .task .n{color:var(--mut);font-size:10px;font-variant-numeric:tabular-nums}
 #specWrap{overflow:auto;max-height:26vh;border-top:1px solid var(--line)}
 .spec{padding:5px 14px;cursor:pointer;display:flex;justify-content:space-between;gap:6px}
 .spec:hover{background:var(--hover)}.spec.active{background:var(--sel)}
 .spec .c{color:var(--mut);font-size:11px;font-variant-numeric:tabular-nums}
 .spec.done .c{color:var(--good)}
 #main{position:relative;min-width:0}
 canvas{display:block;width:100%;height:100%;cursor:crosshair;background:var(--canvas)}
 #hud{position:absolute;top:10px;left:12px;background:#0d1117cc;color:#c8d2df;padding:6px 10px;
   border-radius:6px;color:var(--mut);font-size:11px;pointer-events:none}
 #toast{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);
   background:#0d1117ee;color:#e6ebf2;border:1px solid #2b3340;padding:8px 14px;border-radius:8px;
   opacity:0;transition:opacity .2s;pointer-events:none}
 #toast.show{opacity:1}
 kbd{background:var(--kbd);border:1px solid var(--line);border-radius:3px;padding:0 4px}
 #drop{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
   flex-direction:column;gap:14px;background:var(--canvas);color:var(--dropfg);
   text-align:center;padding:40px}
 #drop.hide{display:none}
 #drop p{color:var(--dropmut);max-width:460px;line-height:1.6}
 label.file{background:var(--accent);color:#04121f;font-weight:600;padding:10px 18px;
   border-radius:8px;cursor:pointer}
 #imgctrl{display:flex;gap:8px;align-items:center;font-size:11px;color:var(--mut);margin-top:8px}
 #imgctrl input{flex:1;min-width:0;accent-color:var(--accent)}
</style></head><body>
<div id="app">
 <div id="side">
  <h1>caliPr <small>— __TITLE__</small></h1>
  <div style="padding:6px 14px;font-size:10.5px;color:var(--mut);
       border-bottom:1px solid var(--line);line-height:1.5">
    __PROVENANCE__<br>Runs offline · your photographs never leave this computer
  </div>
  <div class="sec">
   <div class="row"><label class="file" style="flex:1;text-align:center">
     Choose photos<input id="files" type="file" accept="image/*" multiple hidden></label></div>
   <div id="imgctrl"><span>contrast</span><input id="ctrst" type="range" min="1" max="3.5" step=".05" value="1">
     <span>bright</span><input id="brt" type="range" min=".6" max="1.7" step=".05" value="1"></div>
  </div>
  <div id="hint">Choose your photographs to begin.</div>
  <div id="tasks"></div>
  <div class="sec">
   <div class="row" style="margin-bottom:6px">
     <button id="undo" title="Undo the selected landmark (Z)">↶ Undo</button>
     <button id="fit" title="Fit image (F)">Fit</button></div>
   <div class="row">
     <button id="prev">← Prev</button><button id="next">Next →</button></div>
  </div>
  <div class="sec">
   <div class="row" style="margin-bottom:6px">
     <button id="expJson" class="primary">Export labels</button></div>
   <div class="row"><button id="expTps">Export .tps for R</button></div>
   <div class="row" style="margin-top:6px"><button id="reset"
     title="Delete all saved landmarks in this browser">Reset all labels</button></div>
   <div id="prog" style="margin-top:8px;color:var(--mut);font-size:11px"></div>
  </div>
  <div id="specWrap"></div>
 </div>
 <div id="main">
  <canvas id="cv"></canvas>
  <div id="hud"></div>
  <div id="toast"></div>
  <div id="drop">
   <h2 style="margin:0;font-size:18px">Landmark labeling</h2>
   <p>Choose your specimen photographs. Nothing is uploaded anywhere — the images
      stay on your computer and this page runs entirely in your browser.</p>
   <p>Pick a landmark on the left, then click it on the fish. It advances to the
      next one automatically. <kbd>wheel</kbd> zoom · <kbd>drag</kbd> pan ·
      <kbd>Z</kbd> undo · <kbd>F</kbd> fit</p>
   <label class="file">Choose photos<input id="files2" type="file" accept="image/*" multiple hidden></label>
   <p style="font-size:11px">Your work saves in this browser automatically. If you
      close the page, reopen it and choose the same photos to carry on.</p>
  </div>
 </div>
</div>
<script>
const LANDMARKS = __LANDMARKS__;
const STORE = "calipr_standalone___KEY__";
// Above this many image pixels per screen pixel, a click cannot be placed
// carefully; 3 keeps a landmark inside a few pixels of where it was aimed.
const COARSE_PX = 3;
const $ = s => document.querySelector(s);
const cv = $("#cv"), ctx = cv.getContext("2d");

let files = [];           // {name, file, url}
let idx = -1;             // current specimen
let data = load();        // {filename: {kp:{name:[x,y]}, w, h}}
let img = new Image(), imgW = 0, imgH = 0;
let vs = {scale:1, ox:0, oy:0};
let active = null;        // landmark name
let drag = null, panning = null, moved = false;
let filter = "none";

function load(){ try { return JSON.parse(localStorage.getItem(STORE)) || {}; }
                 catch(e){ return {}; } }
function save(){ try { localStorage.setItem(STORE, JSON.stringify(data)); } catch(e){} }
function rec(){ const n = files[idx] && files[idx].name;
  if(!n) return null;
  if(!data[n]) data[n] = {kp:{}, w:imgW, h:imgH};
  return data[n]; }
function toast(m){ const t=$("#toast"); t.textContent=m; t.classList.add("show");
  clearTimeout(t._h); t._h=setTimeout(()=>t.classList.remove("show"),1500); }

function resize(){ const r=cv.getBoundingClientRect();
  cv.width=Math.max(1,r.width|0); cv.height=Math.max(1,r.height|0); draw(); }
function fit(){ if(!imgW) return; const r=cv.getBoundingClientRect();
  if(!r.width||!r.height){ requestAnimationFrame(fit); return; }
  const s=Math.min(r.width/imgW, r.height/imgH)*0.96;
  vs.scale=s; vs.ox=(r.width-imgW*s)/2; vs.oy=(r.height-imgH*s)/2; draw(); }
const toImg=(x,y)=>[(x-vs.ox)/vs.scale,(y-vs.oy)/vs.scale];
const toScr=(x,y)=>[x*vs.scale+vs.ox, y*vs.scale+vs.oy];

function draw(){
  ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,cv.width,cv.height);
  if(imgW){ ctx.filter=filter;
    ctx.drawImage(img, vs.ox, vs.oy, imgW*vs.scale, imgH*vs.scale); ctx.filter="none"; }
  const r=rec(); if(r){
    for(const [nm,p] of Object.entries(r.kp)){
      const [x,y]=toScr(p[0],p[1]);
      ctx.beginPath(); ctx.arc(x,y,5,0,7); ctx.fillStyle=(nm===active)?"#7fd0ff":"#ff5d6c";
      ctx.fill(); ctx.lineWidth=1.5; ctx.strokeStyle="#0b0e12"; ctx.stroke();
      if(nm===active){ ctx.font="12px sans-serif";
        const w=ctx.measureText(nm).width;
        ctx.fillStyle="#000a"; ctx.fillRect(x+7,y-8,w+6,14);
        ctx.fillStyle="#fff"; ctx.fillText(nm,x+10,y+3); }
    }
  }
  const done = r ? Object.keys(r.kp).length : 0;
  // One screen pixel covers 1/scale image pixels, and that is the floor on
  // placement precision no matter how steady the hand. At fit on a 6000 px photo
  // it is 5-14 image pixels, so a landmark placed without zooming is imprecise
  // by construction rather than by carelessness. Say so, rather than let someone
  // label a whole series from the fitted view and find out afterwards.
  const perPx = vs.scale>0 ? 1/vs.scale : 0;
  const coarse = perPx > COARSE_PX;
  const hud=$("#hud");
  hud.textContent = (files[idx] ? files[idx].name : "—") +
    "  ·  " + done + "/" + LANDMARKS.length + "  ·  " + ((vs.scale*100)|0) + "%" +
    "  ·  ±" + perPx.toFixed(1) + " px" + (coarse ? "  — zoom in to place accurately" : "");
  hud.style.color = coarse ? "#ffb454" : "";
}

function buildTasks(){
  const box=$("#tasks"); box.innerHTML="";
  const r=rec();
  LANDMARKS.forEach((L,i)=>{
    const set = r && r.kp[L.name];
    const el=document.createElement("div");
    el.className="task"+(set?" set":"")+(active===L.name?" active":"");
    el.innerHTML=`<span class="dot"></span><span class="nm">${L.name}</span><span class="n">${i+1}</span>`;
    el.onclick=()=>{ active=L.name; showHint(); buildTasks(); draw(); };
    box.appendChild(el);
  });
  renderSpecs();
}
function showHint(){
  const L=LANDMARKS.find(x=>x.name===active);
  $("#hint").innerHTML = L
    ? `<b>${L.name}</b> — ${L.hint}`
    : 'Pick a landmark from the list, then click it on the fish. '+
      '<kbd>wheel</kbd> zoom · <kbd>drag</kbd> pan · <kbd>Z</kbd> undo · <kbd>F</kbd> fit';
}
function renderSpecs(){
  const w=$("#specWrap"); w.innerHTML="";
  files.forEach((f,i)=>{
    const d=data[f.name], n=d?Object.keys(d.kp).length:0;
    const el=document.createElement("div");
    el.className="spec"+(i===idx?" active":"")+(n>=LANDMARKS.length?" done":"");
    el.innerHTML=`<span>${f.name.length>26?f.name.slice(0,25)+"…":f.name}</span>`+
                 `<span class="c">${n}/${LANDMARKS.length}</span>`;
    el.onclick=()=>select(i);
    w.appendChild(el);
  });
  const total=files.length, complete=files.filter(f=>{
    const d=data[f.name]; return d && Object.keys(d.kp).length>=LANDMARKS.length; }).length;
  $("#prog").textContent = `${complete}/${total} specimens complete`;
}

function select(i){
  if(i<0||i>=files.length) return;
  idx=i;
  img=new Image();
  img.onload=()=>{ imgW=img.naturalWidth; imgH=img.naturalHeight;
    const r=rec(); r.w=imgW; r.h=imgH; save();
    // resume at the first unplaced landmark
    const nxt=LANDMARKS.find(L=>!r.kp[L.name]);
    active = nxt ? nxt.name : LANDMARKS[0].name;
    resize(); fit(); buildTasks(); showHint(); draw(); };
  img.src=files[i].url;
}
function advance(){
  const r=rec(); if(!r) return;
  const i=LANDMARKS.findIndex(L=>L.name===active);
  const rest=[...LANDMARKS.slice(i+1),...LANDMARKS.slice(0,i+1)];
  const nxt=rest.find(L=>!r.kp[L.name]);
  active = nxt ? nxt.name : null;
  if(!active) toast("All landmarks placed — Next → for the following specimen");
  showHint();
}
let warnedCoarse=false;
function place(x,y){
  const r=rec(); if(!r||!active){ toast("Pick a landmark first"); return; }
  if(!warnedCoarse && vs.scale>0 && 1/vs.scale > COARSE_PX){
    warnedCoarse=true;
    toast("Zoomed out — each click lands within ~"+(1/vs.scale).toFixed(0)+
          " image px. Scroll to zoom in for accurate placement.");
  }
  r.kp[active]=[Math.round(x),Math.round(y)];
  save(); advance(); buildTasks(); draw();
}
function nearPoint(mx,my,thresh=12){
  const r=rec(); if(!r) return null;
  for(const [nm,p] of Object.entries(r.kp)){
    const [sx,sy]=toScr(p[0],p[1]);
    if(Math.hypot(sx-mx,sy-my)<thresh) return nm; }
  return null;
}

cv.addEventListener("mousedown",e=>{
  const r=cv.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const hit=nearPoint(mx,my);
  moved=false;
  if(hit){ drag=hit; active=hit; showHint(); buildTasks(); }
  else panning={mx,my,ox:vs.ox,oy:vs.oy};
});
window.addEventListener("mousemove",e=>{
  const r=cv.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  if(drag){ const [ix,iy]=toImg(mx,my); rec().kp[drag]=[Math.round(ix),Math.round(iy)];
    moved=true; draw(); }
  else if(panning){ const dx=mx-panning.mx, dy=my-panning.my;
    if(Math.abs(dx)+Math.abs(dy)>3) moved=true;
    vs.ox=panning.ox+dx; vs.oy=panning.oy+dy; draw(); }
});
window.addEventListener("mouseup",e=>{
  if(drag){ save(); toast("Moved "+drag); drag=null; buildTasks(); draw(); return; }
  if(panning){ const wasPan=moved; panning=null;
    if(!wasPan){ const r=cv.getBoundingClientRect();
      const [ix,iy]=toImg(e.clientX-r.left, e.clientY-r.top);
      if(ix>=0&&iy>=0&&ix<=imgW&&iy<=imgH) place(ix,iy); } }
});
cv.addEventListener("wheel",e=>{
  e.preventDefault();
  const r=cv.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const [ix,iy]=toImg(mx,my);
  let d=e.deltaY; if(e.deltaMode===1) d*=16;
  d=Math.max(-60,Math.min(60,d));
  const ns=Math.max(0.02,Math.min(40, vs.scale*Math.exp(-d*0.0010)));
  vs.scale=ns; vs.ox=mx-ix*ns; vs.oy=my-iy*ns; draw();
},{passive:false});

window.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT") return;
  if(e.key==="z"||e.key==="Z"){ e.preventDefault();
    const r=rec(); if(r&&active&&r.kp[active]){ delete r.kp[active]; save();
      buildTasks(); draw(); toast("Cleared "+active); }
    else toast("Nothing to undo for "+(active||"—")); }
  if(e.key==="f"||e.key==="F"){ e.preventDefault(); fit(); }
  if(e.key==="ArrowRight"){ e.preventDefault(); select(idx+1); }
  if(e.key==="ArrowLeft"){ e.preventDefault(); select(idx-1); }
});
$("#undo").onclick=()=>window.dispatchEvent(new KeyboardEvent("keydown",{key:"z"}));
$("#fit").onclick=()=>fit();
$("#next").onclick=()=>select(idx+1);
$("#prev").onclick=()=>select(idx-1);

function loadFiles(list){
  const imgs=[...list].filter(f=>/^image\//.test(f.type))
    .sort((a,b)=>a.name.localeCompare(b.name));
  if(!imgs.length){ toast("No images in that selection"); return; }
  // Landmarks are stored under the FILENAME, so two files with the same name --
  // easy if photos are gathered from several folders -- would share one record
  // and silently overwrite each other. Refuse rather than lose work.
  const seen={}, dups=[];
  for(const f of imgs){ if(seen[f.name]) dups.push(f.name); seen[f.name]=1; }
  if(dups.length){
    const uniq=[...new Set(dups)];
    alert("Two or more of your files have the same name:\n\n  "+
          uniq.slice(0,8).join("\n  ")+
          (uniq.length>8?"\n  …and "+(uniq.length-8)+" more":"")+
          "\n\nLandmarks are saved per filename, so these would overwrite each "+
          "other. Rename them so every file is unique, then choose them again.");
    return;
  }
  files=imgs.map(f=>({name:f.name, file:f, url:URL.createObjectURL(f)}));
  $("#drop").classList.add("hide");
  select(0);
}
$("#files").onchange=e=>loadFiles(e.target.files);
$("#files2").onchange=e=>loadFiles(e.target.files);
$("#ctrst").oninput=$("#brt").oninput=()=>{
  filter=`contrast(${$("#ctrst").value}) brightness(${$("#brt").value})`; draw(); };

function download(name, text){
  const b=new Blob([text],{type:"application/octet-stream"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(b); a.download=name; a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),2000);
}
$("#expJson").onclick=()=>{
  const out={format:"calipr-landmarks/1", landmark_order:LANDMARKS.map(L=>L.name),
             exported:new Date().toISOString(), specimens:{}};
  let n=0;
  for(const [fn,d] of Object.entries(data)){
    if(!d.kp||!Object.keys(d.kp).length) continue;
    out.specimens[fn]={width:d.w, height:d.h, keypoints:d.kp}; n++; }
  if(!n){ toast("Nothing labeled yet"); return; }
  const part=Object.values(out.specimens)
    .filter(s2=>Object.keys(s2.keypoints).length<LANDMARKS.length).length;
  if(part && !confirm(`${n} specimen(s) to export, of which ${part} are only `+
      `partly labelled.\n\nExport anyway? Partly labelled specimens are still `+
      `useful — missing landmarks are recorded as missing, not guessed.`)) return;
  download("calipr_labels___KEY__.json", JSON.stringify(out,null,1));
  toast(`Exported ${n} specimens${part?` (${part} partial)`:""} — send this file back`);
};
$("#expTps").onclick=()=>{
  // Only landmarks that at least one specimen has: an all-NA column makes
  // geomorph's estimate.missing() fail with an unhelpful subscript error.
  const present=LANDMARKS.map(L=>L.name).filter(n=>
    Object.values(data).some(d=>d.kp&&d.kp[n]));
  if(!present.length){ toast("Nothing labeled yet"); return; }
  const lines=[];
  let n=0;
  for(const [fn,d] of Object.entries(data)){
    if(!d.kp||!Object.keys(d.kp).length||!d.h) continue;
    lines.push("LM="+present.length);
    for(const nm of present){
      const p=d.kp[nm];
      // TPS y is Cartesian from the bottom-left; image y is from the top-left.
      lines.push(p ? `${p[0]} ${d.h-p[1]}` : "-1 -1");
    }
    lines.push("IMAGE="+fn);
    lines.push("ID="+fn.replace(/\.[^.]+$/,""));
    lines.push(""); n++;
  }
  download("landmarks.tps", lines.join("\n"));
  download("landmark_names.csv",
    "index,name\n"+present.map((n2,i)=>`${i+1},${n2}`).join("\n")+"\n");
  const miss=(lines.join("\n").match(/-1 -1/g)||[]).length;
  toast(`Exported ${n} specimens to TPS`+
        (miss?` — ${miss} missing landmark(s) written as -1; read with negNA=TRUE`:""));
};

// Records are kept for every filename ever labelled in this browser, so a second
// batch does not lose the first. The cost is that an export can include
// specimens not in the current selection, which is why both exports state how
// many they cover — and why there has to be a way to start clean.
$("#reset").onclick=()=>{
  const n=Object.keys(data).length;
  if(!n){ toast("Nothing saved"); return; }
  if(!confirm(`Delete saved landmarks for ${n} specimen(s) in this browser?\n\n`+
              `This cannot be undone. Export first if you have not already.`)) return;
  data={}; save();
  buildTasks(); draw(); toast("Cleared");
};

window.addEventListener("resize",resize);
showHint(); buildTasks(); resize();
</script></body></html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="build_standalone_labeler")
    ap.add_argument("--profile", type=Path, default=None,
                    help="Dataset directory holding schema.json, to narrow the "
                         "landmark set (e.g. data/alewife).")
    ap.add_argument("--title", default="landmarks")
    ap.add_argument("--key", default="default",
                    help="Namespaces browser storage and the export filename, so "
                         "two studies on one machine cannot overwrite each other.")
    ap.add_argument("--theme", choices=sorted(THEMES), default="dark",
                    help="Chrome colour. The image canvas stays dark either way.")
    ap.add_argument("--provenance", default="Cornell University Museum of Vertebrates",
                    help="Shown under the title, so someone opening an emailed "
                         "HTML file can see where it came from.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    lms = landmarks_for(args.profile)
    html = (TEMPLATE
            .replace("__LANDMARKS__", json.dumps(lms, indent=1))
            .replace("__TITLE__", args.title)
            .replace("__THEME__", THEMES[args.theme])
            .replace("__FONT__", FONTS[args.theme])
            .replace("__PROVENANCE__", args.provenance)
            .replace("__KEY__", args.key))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"wrote {args.out}  ({len(lms)} landmarks, {len(html)//1024} KB)")
    for L in lms:
        print(f"  {L['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
