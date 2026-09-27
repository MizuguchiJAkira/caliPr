"""Build a single-file landmark labeler that needs no install.

The server-backed labeler is the right tool for someone who already has the repo
and a terminal. For collaborators who do not, it is a harder sell than ImageJ,
which defeats the point. This emits one .html file: open it, pick your photographs from
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
    # The canvas ground stays dark in EVERY theme. The surround must not out-shine
    # the photograph: these specimens are shot against near-black tanks, and a
    # bright panel beside a dark image forces constant eye adaptation, which costs
    # precision on exactly the faint margins that are hardest to judge.
    #
    # "rstudio" is the default, and the look of the server-backed labeller: a grey
    # frame, white panes with a tab strip, flat toolbar buttons. The people this
    # file is sent to do their analysis in RStudio.
    "rstudio": ("--bg:#E3E6EA;--panel:#ffffff;--line:#D6DADF;--edge:#C3C9D0;--fg:#22262A;"
                "--mut:#6B7279;--accent:#3A78B5;--good:#22864D;--goodfill:#35A263;"
                "--warn:#AE6600;--kp:#C9323F;--btn1:#ffffff;--btn2:#EDEFF2;"
                "--btnhover:#E3E7EB;--hover:#EEF4FB;--sel:#D8E6F5;--dot:#B5BCC4;"
                "--hintbg:#F8FAFC;--kbd:#F7F8FA;--tab1:#F4F5F7;--tab2:#E6E9ED;"
                "--bar1:#FDFDFE;--bar2:#F1F3F5;--canvas:#2E3236;"
                "--dropfg:#EEF0F2;--dropmut:#AEB5BD;"),
    "dark": ("--bg:#0d1014;--panel:#1b2029;--line:#2b3340;--edge:#333c4a;--fg:#e6ebf2;"
             "--mut:#8a97a8;--accent:#4aa3ff;--good:#37c871;--goodfill:#37c871;"
             "--warn:#ffb454;--kp:#ff5d6c;--btn1:#2c3440;--btn2:#262d38;"
             "--btnhover:#313b49;--hover:#232a35;--sel:#26303d;--dot:#3a4553;"
             "--hintbg:#161b22;--kbd:#0e1218;--tab1:#20262f;--tab2:#1b2029;"
             "--bar1:#1f252e;--bar2:#1b2029;--canvas:#0b0e12;"
             "--dropfg:#e6ebf2;--dropmut:#8a97a8;"),
    "light": ("--bg:#ffffff;--panel:#f7f7f7;--line:#d9d9d9;--edge:#cfcfcf;--fg:#1a1a1a;"
              "--mut:#666;--accent:#1a6bb5;--good:#2e7d32;--goodfill:#2e7d32;"
              "--warn:#b26a00;--kp:#c62828;--btn1:#ffffff;--btn2:#ffffff;"
              "--btnhover:#eee;--hover:#f0f0f0;--sel:#e4eef7;--dot:#c4c4c4;"
              "--hintbg:#f0f0f0;--kbd:#eee;--tab1:#f7f7f7;--tab2:#f0f0f0;"
              "--bar1:#fafafa;--bar2:#f3f3f3;--canvas:#3a3a3a;"
              "--dropfg:#f0f0f0;--dropmut:#c8c8c8;"),
}
FONTS = {
    "rstudio": "'Lucida Grande','Lucida Sans Unicode','Segoe UI',Helvetica,Arial,sans-serif",
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
   font:12px/1.45 __FONT__;overflow:hidden}
 /* Panes on a frame, each with a tab strip, as in RStudio. */
 #app{display:grid;grid-template-columns:300px 1fr;gap:6px;padding:6px;height:100vh}
 #side{display:flex;flex-direction:column;gap:6px;min-height:0}
 .pane{display:flex;flex-direction:column;min-height:0;overflow:hidden;flex:none;
   background:var(--panel);border:1px solid var(--edge);border-radius:4px}
 .pane.grow{flex:1}
 .tabs{flex:none;display:flex;align-items:flex-end;height:28px;padding:0 6px 0 4px;
   background:linear-gradient(var(--tab1),var(--tab2));border-bottom:1px solid var(--line)}
 .tab{display:inline-flex;align-items:center;gap:6px;height:24px;margin-bottom:-1px;padding:0 11px;
   font-weight:bold;background:var(--panel);border:1px solid var(--line);border-bottom:0;
   border-radius:4px 4px 0 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%}
 .tab small{font-weight:normal;color:var(--mut)}
 .toolbar{flex:none;display:flex;align-items:center;gap:1px;min-height:30px;padding:2px 5px;
   background:linear-gradient(var(--bar1),var(--bar2));border-bottom:1px solid var(--line)}
 .toolbar.foot{border-bottom:0;border-top:1px solid var(--line)}
 .spacer{flex:1}
 .tsep{width:1px;height:16px;margin:0 4px;background:var(--edge);flex:none}
 .body{padding:8px 10px}
 #prov{padding:6px 10px;font-size:10.5px;color:var(--mut);line-height:1.5;
   border-bottom:1px solid var(--line)}
 button,label.file{font:12px/16px __FONT__;color:var(--fg);cursor:pointer;
   background:linear-gradient(var(--btn1),var(--btn2));border:1px solid var(--edge);
   border-radius:3px;padding:3px 9px}
 button:hover,label.file:hover{background:var(--btnhover)}
 button:disabled{opacity:.5;cursor:default}
 button.primary,label.file.primary{background:linear-gradient(#63A0DA,#3A78B5);
   border-color:#2E649B;color:#fff;font-weight:bold}
 .tb{display:inline-flex;align-items:center;gap:5px;height:24px;padding:2px 6px;
   background:transparent;border-color:transparent;white-space:nowrap}
 .tb:hover{background:linear-gradient(var(--btn1),var(--btnhover));border-color:var(--edge)}
 [data-ico]::before{content:'';width:14px;height:14px;flex:none;background:center/contain no-repeat}
 [data-ico=undo]::before{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 14 14'%3E%3Cpath d='M4.5 2.5 1.5 5.5l3 3' fill='none' stroke='%234A6F96' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M2 5.5h6.3a3.6 3.6 0 0 1 0 7.2H5.5' fill='none' stroke='%234A6F96' stroke-width='1.6' stroke-linecap='round'/%3E%3C/svg%3E")}
 [data-ico=fit]::before{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 14 14'%3E%3Cpath d='M1.5 5V1.5H5M9 1.5h3.5V5M12.5 9v3.5H9M5 12.5H1.5V9' fill='none' stroke='%234A6F96' stroke-width='1.5'/%3E%3C/svg%3E")}
 [data-ico=folder]::before{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 14 14'%3E%3Cpath d='M1 3h4l1.3 1.4H13v7.6H1z' fill='%23F3C969' stroke='%23B8912F'/%3E%3C/svg%3E")}
 [data-ico=sheet]::before{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 14 14'%3E%3Cpath d='M2.5 1.5h6l3 3v8h-9z' fill='%23fff' stroke='%23707780'/%3E%3Cpath d='M4.5 6.5h5M4.5 8.5h5M4.5 10.5h3' stroke='%233A78B5'/%3E%3C/svg%3E")}
 [data-ico=rdoc]::before{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 14 14'%3E%3Ccircle cx='7' cy='7' r='6' fill='%23fff' stroke='%233A78B5'/%3E%3Ctext x='7' y='10.2' font-family='Georgia' font-size='9' font-weight='bold' text-anchor='middle' fill='%233A78B5'%3ER%3C/text%3E%3C/svg%3E")}
 [data-ico=zip]::before{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 14 14'%3E%3Cpath d='M2.5 1.5h6l3 3v8h-9z' fill='%23fff' stroke='%23707780'/%3E%3Cpath d='M6 2v1M7 3v1M6 4v1M7 5v1M6 6v1' stroke='%23707780'/%3E%3Crect x='5.6' y='7.4' width='1.8' height='2.2' fill='%23707780'/%3E%3C/svg%3E")}
 [data-ico=clear]::before{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 14 14'%3E%3Ccircle cx='7' cy='7' r='5.6' fill='%23D2504B'/%3E%3Cpath d='M4.8 4.8l4.4 4.4M9.2 4.8l-4.4 4.4' stroke='%23fff' stroke-width='1.6' stroke-linecap='round'/%3E%3C/svg%3E")}
 #imgctrl{display:grid;grid-template-columns:auto 1fr;gap:4px 8px;align-items:center;
   font-size:11px;color:var(--mut);margin-top:8px}
 #imgctrl input{min-width:0;accent-color:var(--accent)}
 #hint{flex:none;padding:7px 10px;background:var(--hintbg);color:var(--mut);
   border-bottom:1px solid var(--line);min-height:60px;max-height:150px;overflow:auto}
 #hint b{color:var(--fg)}
 #tasks{overflow:auto;flex:1;min-height:120px}
 .task{display:flex;align-items:center;gap:8px;padding:2px 10px;cursor:pointer;
   font:12px/19px Monaco,Menlo,Consolas,monospace}
 .task:hover{background:var(--hover)}
 .task.active{background:var(--sel);box-shadow:inset 3px 0 0 var(--accent)}
 .task .dot{width:9px;height:9px;border-radius:50%;background:var(--panel);
   box-shadow:inset 0 0 0 1.5px var(--dot);flex:none}
 .task.set .dot{background:var(--goodfill);box-shadow:none}
 .task .nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
 .task .n{color:var(--mut);font-size:10px;font-variant-numeric:tabular-nums}
 #specWrap{overflow:auto;max-height:26vh}
 .spec{padding:2px 10px;cursor:pointer;display:flex;justify-content:space-between;gap:6px;
   border-bottom:1px solid var(--line);font:12px/19px Monaco,Menlo,Consolas,monospace}
 .spec:hover{background:var(--hover)}
 .spec.active{background:var(--sel);box-shadow:inset 3px 0 0 var(--accent)}
 .spec .c{color:var(--mut);font-size:11px;font-variant-numeric:tabular-nums}
 .spec.done .c{color:var(--good)}
 .exp{display:flex;flex-direction:column;gap:5px;padding:8px 10px}
 .exp button{display:flex;align-items:center;gap:6px;width:100%;text-align:left}
 #prog{padding:0 10px 8px;color:var(--mut);font-size:11px}
 #main{min-width:0}
 #stage{position:relative;flex:1;min-height:0;overflow:hidden;background:var(--canvas)}
 canvas{position:absolute;inset:0;display:block;width:100%;height:100%;cursor:crosshair;
   background:var(--canvas)}
 .status{flex:none;display:flex;align-items:center;height:23px;padding:0 9px;
   background:linear-gradient(var(--bar1),var(--bar2));border-top:1px solid var(--line);
   font:11px Monaco,Menlo,Consolas,monospace;color:var(--mut)}
 #hud{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 #toast{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);
   background:var(--panel);color:var(--fg);border:1px solid var(--edge);padding:6px 12px;
   border-radius:4px;box-shadow:0 4px 14px #0000002b;opacity:0;transition:opacity .2s;
   pointer-events:none}
 #toast.show{opacity:1}
 kbd{font:11px Monaco,Menlo,monospace;background:var(--kbd);border:1px solid var(--edge);
   border-bottom-width:2px;border-radius:3px;padding:0 4px;color:var(--fg)}
 #drop{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
   flex-direction:column;gap:14px;background:var(--canvas);color:var(--dropfg);
   text-align:center;padding:40px}
 #drop.hide{display:none}
 #drop p{color:var(--dropmut);max-width:460px;line-height:1.6}
 #drop label.file{padding:7px 16px}
</style></head><body>
<div id="app">
 <div id="side">
  <section class="pane">
   <div class="tabs"><span class="tab">caliPr <small>__TITLE__</small></span></div>
   <div id="prov">__PROVENANCE__<br>Runs offline · your photographs never leave this computer</div>
   <div class="body">
    <label class="file primary" style="display:inline-flex;align-items:center;gap:6px">
      Choose photos<input id="files" type="file" accept="image/*" multiple hidden></label>
    <div id="imgctrl"><span>contrast</span><input id="ctrst" type="range" min="1" max="3.5" step=".05" value="1">
      <span>brightness</span><input id="brt" type="range" min=".6" max="1.7" step=".05" value="1"></div>
   </div>
  </section>
  <section class="pane grow">
   <div class="tabs"><span class="tab">Landmarks</span></div>
   <div id="hint">Choose your photographs to begin.</div>
   <div id="tasks"></div>
   <div class="toolbar foot">
     <button id="undo" class="tb" data-ico="undo" title="Undo the selected landmark (Z)">Undo</button>
     <button id="fit" class="tb" data-ico="fit" title="Fit image (F)">Fit</button>
     <span class="spacer"></span>
     <button id="prev" class="tb">← Prev</button><button id="next" class="tb">Next →</button></div>
  </section>
  <section class="pane">
   <div class="tabs"><span class="tab">Specimens</span></div>
   <div id="specWrap"></div>
  </section>
  <section class="pane">
   <div class="tabs"><span class="tab">Export</span></div>
   <div class="exp">
     <button id="expBundle" class="primary" data-ico="zip"
       title="Labels plus the original photo files, so the other end can rebuild
the training set exactly. Large.">Export bundle (+ photos)</button>
     <button id="expJson" data-ico="sheet" title="Coordinates only. Small, but only usable by
someone who already has these exact photographs.">Export labels only</button>
     <button id="expTps" data-ico="rdoc">Export .tps for R</button>
     <button id="reset" data-ico="clear"
       title="Delete all saved landmarks in this browser">Reset all labels</button>
   </div>
   <div id="prog"></div>
  </section>
 </div>
 <section id="main" class="pane">
  <div class="tabs"><span class="tab" id="phototab">Photograph</span></div>
  <div id="stage">
   <canvas id="cv"></canvas>
   <div id="toast"></div>
   <div id="drop">
    <h2 style="margin:0;font-size:18px">Landmark labeling</h2>
    <p>Choose your specimen photographs. Nothing is uploaded anywhere — the images
       stay on your computer and this page runs entirely in your browser.</p>
    <p>Pick a landmark on the left, then click it on the fish. It advances to the
       next one automatically. <kbd>wheel</kbd> zoom · <kbd>drag</kbd> pan ·
       <kbd>Z</kbd> undo · <kbd>F</kbd> fit</p>
    <label class="file primary">Choose photos<input id="files2" type="file" accept="image/*" multiple hidden></label>
    <p style="font-size:11px">Your work saves in this browser automatically. If you
       close the page, reopen it and choose the same photos to carry on.</p>
   </div>
  </div>
  <div class="status"><span id="hud"></span></div>
 </section>
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
  hud.style.color = coarse ? "var(--warn)" : "";
  $("#phototab").textContent = files[idx] ? files[idx].name : "Photograph";
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
  // Hash in the background: it is only needed at export, and on a large batch
  // it takes long enough that blocking the first click on it would be rude.
  fingerprintAll();
}
$("#files").onchange=e=>loadFiles(e.target.files);
$("#files2").onchange=e=>loadFiles(e.target.files);
$("#ctrst").oninput=$("#brt").oninput=()=>{
  filter=`contrast(${$("#ctrst").value}) brightness(${$("#brt").value})`; draw(); };

function download(name, text){
  downloadBlob(name, new Blob([text],{type:"application/octet-stream"}));
}
function downloadBlob(name, blob){
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob); a.download=name; a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),2000);
}

// ---- integrity fingerprints ------------------------------------------------
// A landmark set is only valid for the exact pixels it was drawn on. The likely
// accident is a resize — a phone download, a Preview re-export, an email client
// shrinking an attachment — which scales every coordinate by a constant factor
// while leaving the labels looking perfectly sane. Recording a hash of the
// original bytes lets the receiving end prove the photograph is the same one.
const CRCT=(()=>{const t=new Uint32Array(256);
  for(let i=0;i<256;i++){let c=i;
    for(let k=0;k<8;k++) c = (c&1) ? (0xEDB88320^(c>>>1)) : (c>>>1);
    t[i]=c>>>0;}
  return t;})();
function crc32(u8){let c=0xFFFFFFFF;
  for(let i=0;i<u8.length;i++) c = CRCT[(c^u8[i])&0xFF] ^ (c>>>8);
  return (c^0xFFFFFFFF)>>>0;}
const hex=buf=>[...new Uint8Array(buf)].map(b=>b.toString(16).padStart(2,"0")).join("");
async function sha256(u8){
  if(self.crypto && crypto.subtle && crypto.subtle.digest){
    try { return hex(await crypto.subtle.digest("SHA-256", u8)); } catch(e){}
  }
  // A browser that withholds SubtleCrypto from file:// still needs to say
  // something about these bytes. FNV-1a is weaker but catches a resize or a
  // re-encode, which is all this has to do.
  let h1=0x811c9dc5, h2=0x01000193;
  for(let i=0;i<u8.length;i++){
    h1=Math.imul(h1^u8[i],16777619)>>>0;
    if((i&2047)===0) h2=Math.imul(h2^h1,16777619)>>>0; }
  return "fnv1a:"+h1.toString(16).padStart(8,"0")+h2.toString(16).padStart(8,"0");
}
async function fingerprintAll(){
  let n=0;
  for(const f of files){
    const r=data[f.name] || (data[f.name]={kp:{},w:0,h:0});
    if(r.fp && r.fp.bytes===f.file.size) { n++; continue; }
    $("#prog").textContent=`checking photo ${++n}/${files.length}…`;
    try {
      const u8=new Uint8Array(await f.file.arrayBuffer());
      r.fp={bytes:f.file.size, modified:f.file.lastModified||0,
            sha256:await sha256(u8), crc32:crc32(u8)};
    } catch(e){ r.fp={bytes:f.file.size, error:String(e)}; }
  }
  save(); renderSpecs();
}

// ---- zip (store mode; JPEGs are already compressed) ------------------------
function zipStore(entries){
  const enc=new TextEncoder(), parts=[], central=[];
  let off=0;
  const u16=v=>{const b=new Uint8Array(2); new DataView(b.buffer).setUint16(0,v,true); return b;};
  const u32=v=>{const b=new Uint8Array(4); new DataView(b.buffer).setUint32(0,v>>>0,true); return b;};
  for(const e of entries){
    const nm=enc.encode(e.name);
    parts.push(u32(0x04034b50),u16(20),u16(0x0800),u16(0),u16(0),u16(33),
               u32(e.crc),u32(e.size),u32(e.size),u16(nm.length),u16(0),nm,e.body);
    central.push([u32(0x02014b50),u16(20),u16(20),u16(0x0800),u16(0),u16(0),u16(33),
                  u32(e.crc),u32(e.size),u32(e.size),u16(nm.length),u16(0),u16(0),
                  u16(0),u16(0),u32(0),u32(off),nm]);
    off += 30+nm.length+e.size;
  }
  const cdStart=off; let cdLen=0;
  for(const row of central) for(const p of row){ parts.push(p); cdLen+=p.length; }
  parts.push(u32(0x06054b50),u16(0),u16(0),u16(central.length),u16(central.length),
             u32(cdLen),u32(cdStart),u16(0));
  return new Blob(parts,{type:"application/zip"});
}
$("#expJson").onclick=()=>{
  const out={format:"calipr-landmarks/1", landmark_order:LANDMARKS.map(L=>L.name),
             exported:new Date().toISOString(), specimens:{}};
  let n=0;
  for(const [fn,d] of Object.entries(data)){
    if(!d.kp||!Object.keys(d.kp).length) continue;
    out.specimens[fn]={width:d.w, height:d.h, keypoints:d.kp, file:d.fp||null};
    n++; }
  if(!n){ toast("Nothing labeled yet"); return; }
  const part=Object.values(out.specimens)
    .filter(s2=>Object.keys(s2.keypoints).length<LANDMARKS.length).length;
  if(part && !confirm(`${n} specimen(s) to export, of which ${part} are only `+
      `partly labelled.\n\nExport anyway? Partly labelled specimens are still `+
      `useful — missing landmarks are recorded as missing, not guessed.`)) return;
  download("calipr_labels___KEY__.json", JSON.stringify(out,null,1));
  toast(`Exported ${n} specimens${part?` (${part} partial)`:""} — send this file back`);
};
function labelPayload(){
  const out={format:"calipr-landmarks/1", landmark_order:LANDMARKS.map(L=>L.name),
             exported:new Date().toISOString(), specimens:{}};
  let n=0;
  for(const [fn,d] of Object.entries(data)){
    if(!d.kp||!Object.keys(d.kp).length) continue;
    out.specimens[fn]={width:d.w, height:d.h, keypoints:d.kp, file:d.fp||null};
    n++; }
  return [out,n];
}
$("#expBundle").onclick=async()=>{
  const [out,n]=labelPayload();
  if(!n){ toast("Nothing labeled yet"); return; }
  const want=new Set(Object.keys(out.specimens));
  const have=files.filter(f=>want.has(f.name));
  if(have.length<want.size){
    alert(`${want.size-have.length} labelled specimen(s) are not among the photos `+
          `currently loaded, so their images cannot go in the bundle.\n\n`+
          `Choose all the photos again, then export.`);
    return; }
  await fingerprintAll();
  const total=have.reduce((a,f)=>a+f.file.size,0);
  const mb=(total/1048576).toFixed(0);
  if(total > 3.5*1024*1024*1024){
    alert(`That bundle would be ${mb} MB, past the 4 GB limit of this zip `+
          `format. Export in smaller batches.`); return; }
  if(!confirm(`Bundle ${have.length} photo(s) + labels — about ${mb} MB.\n\n`+
              `The photographs go in unmodified, byte for byte, so the other end `+
              `can rebuild the training set exactly.\n\nContinue?`)) return;
  $("#prog").textContent="building bundle…";
  const entries=[{name:"labels.json",
                  body:new Blob([JSON.stringify(out,null,1)]),
                  size:new Blob([JSON.stringify(out,null,1)]).size,
                  crc:crc32(new TextEncoder().encode(JSON.stringify(out,null,1)))}];
  for(const f of have){
    const fp=(data[f.name]||{}).fp||{};
    entries.push({name:"images/"+f.name, body:f.file, size:f.file.size,
                  crc:fp.crc32>>>0});
  }
  downloadBlob("calipr_bundle___KEY__.zip", zipStore(entries));
  $("#prog").textContent="";
  toast(`Bundle exported — ${have.length} photos, ${mb} MB`);
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
    ap.add_argument("--theme", choices=sorted(THEMES), default="rstudio",
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
