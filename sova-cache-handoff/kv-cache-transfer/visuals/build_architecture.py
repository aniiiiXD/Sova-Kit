"""Create an offline animated SVG, a static SVG poster and an HTML player."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
ROOT.mkdir(parents=True, exist_ok=True)
BLUE, COPPER, INK, MUTED = "#35769a", "#a96547", "#24384c", "#65798c"
parts = []


def add(value):
    parts.append(value)


def text(x, y, value, size=15, color=INK, weight=400, anchor="start"):
    from html import escape
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>'


def rect(x, y, w, h, fill="#fff", stroke="#d6e0e8", radius=14, extra=""):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" {extra}/>'


def wire(path, color="#9aaebb", dashed=False, marker="arrow"):
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.8" {"stroke-dasharray=\"5 5\"" if dashed else ""} marker-end="url(#{marker})"/>'


def packet(path, color, start, end, radius=4):
    return f'''<circle class="motion" r="{radius}" fill="{color}" opacity="0">
      <animateMotion dur="12s" repeatCount="indefinite" path="{path}" keyPoints="0;0;1;1" keyTimes="0;{start};{end};1" calcMode="linear"/>
      <animate attributeName="opacity" dur="12s" repeatCount="indefinite" values="0;0;1;1;0;0" keyTimes="0;{start-.001};{start};{end};{end+.001};1"/>
    </circle>'''


add('''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1600 1050" width="1600" height="1050" role="img" aria-labelledby="title desc">
<title id="title">SOVA cross-model KV cache transfer architecture</title>
<desc id="desc">The same prompt is evaluated by a native large-model path and a transfer path. A smaller Qwen3 model reads the prefix; a fitted mapper translates its keys and values; the larger Qwen3 model imports the cache, processes one tail token and generates an answer. A proposed SageMaker GPU worker reads inputs from S3 and returns traces, quality scores and timing to S3. Motion illustrates flow, not measured latency.</desc>
<defs>
 <pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M24 0H0V24" fill="none" stroke="#dce5ed" stroke-width=".65"/></pattern>
 <marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M1 1L7 4L1 7" fill="none" stroke="#91a7b7" stroke-width="1.3"/></marker>
 <marker id="blue-arrow" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M1 1L7 4L1 7" fill="none" stroke="#35769a" stroke-width="1.4"/></marker>
 <marker id="copper-arrow" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M1 1L7 4L1 7" fill="none" stroke="#a96547" stroke-width="1.4"/></marker>
 <style>text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}path,rect,circle{vector-effect:non-scaling-stroke}@media(prefers-reduced-motion:reduce){.motion{display:none}}</style>
</defs>
<rect width="1600" height="1050" fill="#f6f8fb"/>
<rect width="1600" height="1050" fill="url(#grid)" opacity=".62"/>
''')
add(text(60, 49, "sova.lab", 20, INK, 700))
add(text(220, 49, "[ CACHE STATE HANDOFF ]", 12, MUTED))
add(rect(1260, 26, 280, 34, "#e8eef4", "none", 7))
add(text(1400, 48, "Experiment design / unmeasured", 12, MUTED, anchor="middle"))
add(text(60, 107, "Smaller read. Larger decode.", 38, INK, 650))
add(text(60, 141, "Translate the prefix state instead of asking the target to read it all again.", 17, MUTED))

# Native comparator: intentionally no comparative speed animation.
add('<g id="native-control">')
add(rect(300, 180, 1240, 190, "#edf1f5", "#dbe3ea", 18))
add(text(324, 208, "Control / native large model", 13, MUTED))
add(wire("M260 418H280Q300 418 300 398V285H320"))
add(rect(320, 230, 630, 114, "#f9fafc", "#d0dbe4", 12))
add(text(344, 264, "Large-model prefill", 21, "#5a6d80", 600))
add(text(344, 293, "Target reads the full N-token prompt.", 14, MUTED))
add(text(344, 322, "Builds its own cache before answering", 12, "#8192a2"))
for r in range(3):
    for c in range(10):
        add(rect(738+c*18, 253+r*23, 12, 16, "#d3dde6" if c%3 else "#bdccd9", "none", 2))
add(wire("M950 287H1060"))
add(text(1005, 267, "KV", 12, MUTED, anchor="middle"))
add(rect(1065, 230, 310, 114, "#f9fafc", "#d0dbe4", 12))
add(text(1089, 265, "Large-model decode", 21, "#5a6d80", 600))
add(text(1089, 294, "Generates tokens one by one", 13, MUTED))
add(text(1089, 322, "Native quality baseline", 12, "#8192a2"))
add(wire("M1375 287H1440"))
add(text(1463, 292, "Answer", 13, MUTED, anchor="middle"))
add('</g>')

# Shared input, shown only once.
add(rect(60, 380, 200, 220, "#fff", "#bdcddb", 14))
add(text(82, 410, "Same input", 17, INK, 600))
add('<path d="M91 432H184L207 455V521H91Z" fill="#f7f9fc" stroke="#c8d5e1"/><path d="M184 432V455H207" fill="none" stroke="#c8d5e1"/>')
for y, width in [(451,64),(468,92),(485,78),(502,96)]:
    add(f'<path d="M104 {y}h{width}" stroke="#b2c4d3" stroke-width="3" stroke-linecap="round"/>')
add(text(82, 552, "Document + query", 13, MUTED))
add(text(82, 577, "identical token IDs", 11, "#7d91a3"))

add(rect(300, 430, 1240, 246, "#eef4f8", "#cfdee9", 18))
add(text(324, 458, "Transfer / source prefix to target state", 13, BLUE))
add(wire("M260 512H285Q300 512 300 540V548H320", BLUE, marker="blue-arrow"))
add(rect(320, 480, 250, 172, "#fff", "#9fbccc", 12))
add(text(344, 512, "Small-model prefill", 18, INK, 600))
add(text(344, 537, "Reads the N−1 prefix", 13, MUTED))
for r in range(3):
    for c in range(7):
        x, y = 344+c*28, 554+r*20
        add(rect(x,y,20,13,"#b7d2e1" if (r+c)%3 else "#79a9c3","none",3))
        begin = .11+(r*7+c)*.005
        add(f'<rect class="motion" x="{x}" y="{y}" width="20" height="13" rx="3" fill="{BLUE}" opacity="0"><animate attributeName="opacity" dur="12s" repeatCount="indefinite" values="0;0;.9;.9;0;0" keyTimes="0;{begin-.001:.3f};{begin:.3f};.27;.30;1"/></rect>')
add(text(344, 631, "Source K / V cache", 12, BLUE))

add(wire("M570 564H650", BLUE, marker="blue-arrow"))
add(text(611, 541, "K_s / V_s", 11, BLUE, anchor="middle"))
add(rect(660, 480, 280, 172, "#fff", "#94afc3", 12))
add(text(684, 512, "KV translator", 21, INK, 600))
add(text(684, 537, "Fitted once per model pair", 12, MUTED))
for c in range(4):
    for r in range(3):
        add(rect(687+c*13, 551+r*13, 9, 9, "#83afc6", "none", 1.5))
        add(rect(861+c*13, 551+r*13, 9, 9, "#c3957e", "none", 1.5))
add('<path d="M745 570H770M830 570H851" fill="none" stroke="#9bafbd" stroke-width="1.5"/><path d="M800 547L824 570L800 593L776 570Z" fill="#edf2f6" stroke="#8ba5ba"/>')
add(text(800, 576, "W", 19, BLUE, 600, "middle"))
add(text(684, 617, "undo RoPE / map / reapply", 12, MUTED))
add(text(684, 639, "Affine ridge, per KV head", 11, "#7e91a3"))
add(rect(660, 391, 280, 29, "#edf1f6", "#d4dfe8", 6))
add(text(800, 410, "Offline fit / frozen mapper", 11, MUTED, anchor="middle"))
add(wire("M800 420V471", "#9eafbd", dashed=True))

add(wire("M940 564H1055", COPPER, marker="copper-arrow"))
add(text(998, 541, "K_t / V_t", 11, COPPER, anchor="middle"))
add(rect(1065, 480, 310, 172, "#fff", "#c4ac9d", 12))
add(text(1089, 512, "Large-model decode", 21, INK, 600))
add(text(1089, 537, "Imports N−1 cached positions", 13, MUTED))
add(rect(1089, 552, 260, 31, "#f7eee8", "none", 5))
add(text(1219, 572, "Processes 1 prompt tail token", 11, COPPER, anchor="middle"))
for i in range(5):
    add(rect(1089+i*49, 600, 38, 25, "#f4f6f9", "#dbe3e9", 4))
    add(text(1108+i*49, 617, f"t{i+1}", 12, "#7d8e9f", anchor="middle"))
    if i<4:
        add('<path d="M'+str(1129+i*49)+' 613h7" stroke="#b5c3ce"/>')
    begin=.73+i*.025
    add(f'<rect class="motion" x="{1089+i*49}" y="600" width="38" height="25" rx="4" fill="{COPPER}" opacity="0"><animate attributeName="opacity" dur="12s" repeatCount="indefinite" values="0;0;.2;.2;0;0" keyTimes="0;{begin-.001:.3f};{begin:.3f};.92;.96;1"/></rect>')
add(wire("M1375 564H1440", COPPER, marker="copper-arrow"))
add(text(1464, 516, "Answer", 13, INK, 600, "middle"))
add(rect(1440, 539, 50, 58, "#fff", "#adc2d1", 7))
for y,w in [(554,27),(566,20),(578,27)]:
    add(f'<path d="M1451 {y}h{w}" stroke="#829eaf" stroke-width="2"/>')

# All animated objects live in a removable motion layer.
add(packet("M262 512H285Q300 512 300 540V548H318", BLUE, .035, .10))
add(packet("M574 564H650", BLUE, .27, .34))
add(packet("M744 570H858", BLUE, .37, .48))
add(packet("M944 564H1055", COPPER, .56, .65))
add(packet("M1378 564H1437", COPPER, .85, .91))
for x,y,w,h,color,start,end in [(320,480,250,172,BLUE,.10,.27),(660,480,280,172,BLUE,.34,.56),(1065,480,310,172,COPPER,.65,.91)]:
    add(f'<rect class="motion" x="{x-2}" y="{y-2}" width="{w+4}" height="{h+4}" rx="14" fill="none" stroke="{color}" stroke-width="2" opacity="0"><animate attributeName="opacity" dur="12s" repeatCount="indefinite" values="0;0;.8;.8;0;0" keyTimes="0;{start-.001};{start};{end};{end+.001};1"/></rect>')

# Complete measurement boundaries, with no fake time values.
add(text(320, 718, "Time the complete cold request", 16, INK, 600))
add(rect(320, 735, 1055, 56, "#fff", "#d6e1ea", 9))
for x,label,color in [(344,"source prefill",BLUE),(584,"map + move/import",BLUE),(919,"target tail + first token",COPPER)]:
    add(text(x,769,label,14,color))
add(text(548,769,"+",16,MUTED));add(text(883,769,"+",16,MUTED))
add(text(320, 817, "Native and transferred answers are scored on the same held-out questions.", 13, MUTED))

# Proposed execution infrastructure, not an active deployment.
add('<g id="execution-plan">')
add('<path d="M60 849H1540" stroke="#cfdae4"/>')
add(text(60, 880, "Where it runs", 16, INK, 600))
add(text(1540, 880, "Proposed AWS worker / no deployment launched", 11, MUTED, anchor="end"))
for x,w,title,sub in [(60,260,"Laptop / submit","CLI + job monitoring"),(390,240,"S3 / inputs","checkpoints + data"),(710,440,"SageMaker / GPU job","calibrate · compare 5 arms · measure"),(1230,310,"S3 / results","traces + quality + memory")]:
    add(rect(x,901,w,80,"#fff","#d1dde7",10))
    add(text(x+20,932,title,15,INK,600))
    add(text(x+20,958,sub,11,MUTED))
for start,end in [(320,380),(630,700),(1150,1220)]:
    add(wire(f"M{start} 941H{end}"))
add(text(930,1005,"Job exits after artifacts are saved",11,MUTED,anchor="middle"))
add('</g>')
add(text(60,1032,"Initial adapter: dense Qwen3. Hybrid Qwen3.5/3.8 needs recurrent-state support.",11,MUTED))
add(text(1540,1032,"Motion illustrates sequence, not measured speed.",11,MUTED,anchor="end"))
add('</svg>')
svg = "\n".join(parts)
# Confirm well-formed XML before writing the assets.
ET.fromstring(svg)
(ROOT/"cache-transfer-animated.svg").write_text(svg)
static_root=ET.fromstring(svg)
ns="{http://www.w3.org/2000/svg}"
for parent in list(static_root.iter()):
    for child in list(parent):
        if child.get("class")=="motion" or child.tag in (ns+"animate",ns+"animateMotion"):
            parent.remove(child)
ET.register_namespace("", "http://www.w3.org/2000/svg")
static=ET.tostring(static_root,encoding="unicode")
(ROOT/"cache-transfer-static.svg").write_text(static)

html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SOVA / Cache transfer architecture</title>
<style>
:root{color-scheme:light;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#24384c;background:#eaf0f5}*{box-sizing:border-box}body{margin:0}header,main,footer{max-width:1512px;margin:auto;padding-left:32px;padding-right:32px}header{display:flex;align-items:center;justify-content:space-between;gap:16px;min-height:78px}.brand{font-weight:750;font-size:23px;letter-spacing:-1px}.brand span{font-weight:400;font-size:13px;letter-spacing:0;margin-left:14px;color:#65798c}a{color:#35769a;text-underline-offset:4px}.downloads{display:flex;gap:18px;font-size:12px}.player{border:1px solid #cddbe5;border-radius:17px;background:#f6f8fb;overflow:hidden;box-shadow:0 16px 54px #203d5710}.toolbar{display:flex;align-items:center;gap:14px;padding:17px 24px;border-bottom:1px solid #d8e2ea;background:#f9fbfd}button{font:inherit;cursor:pointer;border:1px solid #bdcdda;border-radius:7px;padding:9px 14px;color:#24384c;background:#fff;white-space:nowrap;font-size:13px}button:hover{background:#edf4f9}button:active{transform:translateY(1px)}button:focus-visible,a:focus-visible,input:focus-visible{outline:3px solid #35769a;outline-offset:4px}button.primary{background:#294d69;color:#fff;border-color:#294d69;min-width:90px}label{font-size:12px;color:#65798c}.track{flex:1;display:flex;align-items:center;gap:12px}.track input{width:100%;accent-color:#35769a;cursor:pointer}.time{font-variant-numeric:tabular-nums;font-size:12px;color:#65798c;min-width:71px}.check{display:flex;align-items:center;gap:7px;white-space:nowrap}.check input{accent-color:#35769a}.canvas{overflow-x:auto}.canvas svg{display:block;width:100%;height:auto;min-width:900px}.steps{display:grid;grid-template-columns:repeat(5,1fr);border-top:1px solid #d8e2ea}.steps button{border:0;border-radius:0;background:#f9fbfd;text-align:left;padding:17px 21px;border-right:1px solid #d8e2ea;font-size:12px;white-space:normal;color:#65798c}.steps button:last-child{border-right:0}.steps button[aria-pressed=true]{background:#e3eef5;color:#244f6e;box-shadow:inset 0 3px #35769a}.steps span{display:block;font-size:10px;margin-bottom:7px;opacity:.7}.explainer{display:grid;grid-template-columns:300px 1fr;gap:30px;padding:28px 8px 32px}h1{font-size:23px;letter-spacing:-.7px;margin:0 0 10px;font-weight:620}.explainer p{font-size:14px;color:#536b7f;line-height:1.7;margin:0;max-width:870px}.explain-label{font-size:12px;color:#65798c;margin-bottom:9px}.legend{display:flex;gap:20px;margin-top:15px;font-size:12px;color:#65798c}.legend i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px}.scope{border-top:1px solid #cddbe5;padding-top:20px;padding-bottom:26px;display:flex;justify-content:space-between;gap:25px;font-size:12px;color:#65798c;line-height:1.6}.scope p{margin:0;max-width:760px}footer{padding-bottom:28px;font-size:12px;color:#65798c}#native-control.hidden-control{display:none}@media(max-width:800px){header,main,footer{padding-left:15px;padding-right:15px}.brand span{display:none}.toolbar{flex-wrap:wrap;padding:14px;gap:10px}.track{order:3;flex-basis:100%}.check{margin-left:auto}.steps{grid-template-columns:repeat(5,minmax(0,1fr))}.steps button{font-size:10px;padding:14px 8px}.explainer{grid-template-columns:1fr;gap:12px;padding:24px 3px}.scope{flex-direction:column;gap:12px}.downloads{gap:10px;font-size:11px}h1{font-size:21px}}@media print{.toolbar,.steps,header,.downloads{display:none}.player{border:0;box-shadow:none}.canvas svg{min-width:0}.explainer,.scope{break-inside:avoid}body{background:#fff}}
</style></head><body><header><div class="brand">sova<span>Cache transfer / architecture lab</span></div><nav class="downloads" aria-label="Download diagrams"><a href="cache-transfer-animated.svg" download>Animated SVG</a><a href="cache-transfer-static.svg" download>Static SVG</a><a href="cache-transfer-poster.png" download>PNG poster</a></nav></header><main>
<section class="player" aria-label="Architecture animation"><div class="toolbar"><button type="button" class="primary" id="play">Pause</button><button type="button" id="replay">Replay</button><div class="track"><label for="scrub">Sequence</label><input id="scrub" type="range" min="0" max="11.99" step="0.01" value="0" aria-label="Illustrative animation timeline"><output id="time" class="time">0.0 / 12s</output></div><label class="check"><input type="checkbox" id="control" checked>Show native control</label></div><div class="canvas">__SVG__</div><nav class="steps" aria-label="Explore architecture stages"><button data-stage="0" aria-pressed="true"><span>01 / Input</span>Same prompt</button><button data-stage="1" aria-pressed="false"><span>02 / Read</span>Source prefill</button><button data-stage="2" aria-pressed="false"><span>03 / Translate</span>Map cache state</button><button data-stage="3" aria-pressed="false"><span>04 / Reuse</span>Target decode</button><button data-stage="4" aria-pressed="false"><span>05 / Verify</span>Compare answers</button></nav></section>
<section class="explainer" aria-live="polite"><div><div class="explain-label" id="stage-label">Stage 1 of 5</div><h1 id="stage-title">One input. Two execution paths.</h1><div class="legend"><span><i style="background:#35769a"></i>Source state</span><span><i style="background:#a96547"></i>Target state</span></div></div><p id="stage-body"></p></section><div class="scope"><p><strong>Built code:</strong> dense, full-attention Qwen3 with a per-head affine ridge mapper. This first mechanism experiment does not implement recurrent/convolution-state transfer for the advertised Qwen3.5 → Qwen3.8 pair.</p><p>12-second illustrative sequence.<br>Performance and quality gains remain unmeasured.</p></div></main><footer>Offline artifact. No API calls, GPU jobs or cloud resources are launched by this player.</footer>
<script>
const svg=document.querySelector('.canvas svg'),play=document.getElementById('play'),scrub=document.getElementById('scrub'),clock=document.getElementById('time');
const stages=[
 {at:.4,title:'One input. Two execution paths.',body:'Both paths receive exactly the same document and question. The native control sends every prompt token through the larger model. The transfer path asks the smaller model to read the prefix, then passes translated internal state to the larger model.'},
 {at:2.1,title:'The small model reads the prefix.',body:'Prefill computes the source model’s keys and values for the first N−1 prompt tokens. These are model-specific tensors, not a text summary. The larger model has not processed that prefix. Cold-request timing starts before this source prefill.'},
 {at:5.1,title:'Translate the model’s internal state.',body:'A mapper fitted offline on separate calibration documents removes the source keys’ rotary position encoding, applies a learned per-head affine map to keys and values, then applies the target’s position encoding. The saved artifact is tied to the checkpoint pair and tokenizer; it is not fitted on the held-out questions.'},
 {at:8.8,title:'Import the cache. Decode on the target.',body:'The target imports N−1 mapped cache positions and processes exactly one remaining prompt token to obtain its own next-token logits. It then generates answer tokens autoregressively. Attention is still present; its prompt-prefix cache has been supplied by the mapper instead of being recomputed by the target.'},
 {at:10.8,title:'Verify the tradeoff with real traces.',body:'The SageMaker GPU experiment will compare native and transferred answers, time the complete path, and record cache-import counts, errors and peak memory. Faster first-token latency only helps if quality remains acceptable. The drawing shows the proposed architecture; the smoke suite and GPU experiment have not been run yet.'}
];
const buttons=[...document.querySelectorAll('[data-stage]')];let playing=!matchMedia('(prefers-reduced-motion: reduce)').matches,last=-1,raf;
function show(t){const phase=t<1.2?0:t<3.6?1:t<7.8?2:t<10.3?3:4;if(phase!==last){last=phase;document.getElementById('stage-label').textContent='Stage '+(phase+1)+' of 5';document.getElementById('stage-title').textContent=stages[phase].title;document.getElementById('stage-body').textContent=stages[phase].body;buttons.forEach((b,i)=>b.setAttribute('aria-pressed',String(i===phase)));}scrub.value=t;clock.textContent=t.toFixed(1)+' / 12s';}
function state(){if(playing){svg.unpauseAnimations();play.textContent='Pause';}else{svg.pauseAnimations();play.textContent='Play';}play.setAttribute('aria-label',playing?'Pause architecture animation':'Play architecture animation');}
function frame(){show(svg.getCurrentTime()%12);if(playing)raf=requestAnimationFrame(frame);}
function pause(){playing=false;cancelAnimationFrame(raf);state();}
play.addEventListener('click',()=>{playing=!playing;cancelAnimationFrame(raf);state();if(playing)frame();});
document.getElementById('replay').addEventListener('click',()=>{svg.setCurrentTime(0);playing=true;cancelAnimationFrame(raf);state();frame();});
scrub.addEventListener('input',()=>{pause();svg.setCurrentTime(Number(scrub.value));show(Number(scrub.value));});
buttons.forEach((b,i)=>b.addEventListener('click',()=>{pause();svg.setCurrentTime(stages[i].at);show(stages[i].at);}));
document.getElementById('control').addEventListener('change',event=>document.getElementById('native-control').classList.toggle('hidden-control',!event.target.checked));
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
svg.setCurrentTime(0);state();show(0);if(playing)frame();
</script></body></html>'''
(ROOT/"architecture.html").write_text(html.replace("__SVG__",svg))
print("Created animated SVG, static SVG and offline HTML player in",ROOT)
