from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import urllib.parse
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "kaggle_output" / "source16-cut-pilot" / "source16_cut_pilot"
INDEX_CSV = PILOT_ROOT / "outputs" / "LISTENING_INDEX.csv"
CLIPS_ROOT = PILOT_ROOT / "clips"
OUT = ROOT / "outputs" / "source16_cut_pilot_review"
STATE_JSON = OUT / "REVIEW_STATE.json"
RESULTS_CSV = OUT / "REVIEW_RESULTS.csv"
EXPECTED = 160
VERDICTS = {"GOOD", "BAD_BOUNDARY", "BAD_TEXT", "BAD_BOTH", "UNSURE"}
FLAGS = {"clipped_start", "clipped_end", "mid_phrase_cut", "too_much_silence", "noise_or_music", "speaker_issue"}


def priority(row: dict[str, str]) -> tuple[int, int]:
    d = row.get("decision", "")
    r = row.get("pilot_reason", "")
    if d == "REVIEW":
        p = 0
    elif r == "keep_weakest_boundary":
        p = 1
    elif r == "keep_shortest":
        p = 2
    elif r == "keep_longest":
        p = 3
    elif d == "KEEP_WHISPER_PRIMARY":
        p = 4
    else:
        p = 5
    return p, int(float(row.get("pilot_order", "0") or 0))


def load_items():
    if not INDEX_CSV.exists():
        raise SystemExit("Missing LISTENING_INDEX.csv. Download source16-cut-pilot output first.")
    with INDEX_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXPECTED:
        raise SystemExit(f"Expected {EXPECTED} rows, got {len(rows)}")

    items = []
    audio_map = {}
    for review_order, row in enumerate(sorted(rows, key=priority), 1):
        clip_id = row["clip_id"]
        rel = str(row.get("clip_file", "")).replace("/", os.sep)
        audio = PILOT_ROOT / rel
        if not audio.exists():
            audio = CLIPS_ROOT / row["video_id"] / f"{clip_id}.wav"
        if not audio.exists():
            raise SystemExit(f"Missing WAV: {audio}")
        audio_map[clip_id] = audio
        items.append({
            "review_order": review_order,
            "pilot_order": int(float(row.get("pilot_order", "0") or 0)),
            "pilot_reason": row.get("pilot_reason", ""),
            "video_id": row.get("video_id", ""),
            "clip_id": clip_id,
            "decision": row.get("decision", ""),
            "quality_score": row.get("quality_score", ""),
            "actual_duration_sec": row.get("actual_duration_sec", ""),
            "text": row.get("text", ""),
            "audio_url": "/audio/" + urllib.parse.quote(clip_id),
        })
    return items, audio_map


def load_state():
    if not STATE_JSON.exists():
        return {}
    data = json.loads(STATE_JSON.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


ITEMS, AUDIO = load_items()
ITEM_BY_ID = {x["clip_id"]: x for x in ITEMS}
STATE = load_state()
LOCK = threading.Lock()


def summary():
    counts = {x: 0 for x in VERDICTS}
    for r in STATE.values():
        v = r.get("verdict", "")
        if v in counts:
            counts[v] += 1
    reviewed = sum(counts.values())
    return {"total": len(ITEMS), "reviewed": reviewed, "remaining": len(ITEMS) - reviewed, "counts": counts}


def flush():
    with LOCK:
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = STATE_JSON.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(STATE, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(STATE_JSON)

        fields = ["review_order","pilot_order","pilot_reason","video_id","clip_id","decision","quality_score","actual_duration_sec","text","verdict","flags","note","reviewed_at"]
        tmp_csv = RESULTS_CSV.with_suffix(".csv.tmp")
        with tmp_csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for item in ITEMS:
                r = STATE.get(item["clip_id"], {})
                w.writerow({
                    **{k: item.get(k, "") for k in fields[:9]},
                    "verdict": r.get("verdict", ""),
                    "flags": ";".join(r.get("flags", [])),
                    "note": r.get("note", ""),
                    "reviewed_at": r.get("reviewed_at", ""),
                })
        tmp_csv.replace(RESULTS_CSV)


flush()

HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Source16 Pilot Review</title>
<style>
body{font-family:system-ui;background:#111;color:#eee;margin:0}main{max-width:1050px;margin:auto;padding:18px}.card{background:#1b1b1b;border:1px solid #333;border-radius:12px;padding:16px;margin-bottom:14px}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.badge{background:#292929;padding:6px 9px;border-radius:999px;font-size:13px}#text{font-size:24px;line-height:1.45;margin:16px 0}audio{width:100%}button{background:#2a2a2a;color:#fff;border:1px solid #555;border-radius:8px;padding:10px 12px;cursor:pointer}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}.flags{display:flex;gap:10px;flex-wrap:wrap}.flags label{background:#252525;padding:7px;border-radius:7px}textarea{width:100%;min-height:65px;background:#151515;color:#fff;border:1px solid #444;border-radius:8px;padding:8px;box-sizing:border-box}.active{outline:3px solid #aaa}kbd{background:#333;border:1px solid #555;border-radius:4px;padding:1px 5px}.muted{color:#aaa}#bar{height:9px;background:#333;border-radius:9px;margin-top:10px;overflow:hidden}#fill{height:100%;background:#999;width:0}
</style>
<main>
<div class="card"><div class="row"><b>Source16 pilot review</b><span id="progress" class="badge"></span><span id="counts" class="badge"></span></div><div id="bar"><div id="fill"></div></div></div>
<div class="card">
<div class="row"><span id="order" class="badge"></span><span id="clip" class="badge"></span><span id="decision" class="badge"></span><span id="reason" class="badge"></span><span id="quality" class="badge"></span><span id="duration" class="badge"></span></div>
<div id="text"></div>
<audio id="audio" controls preload="auto"></audio>
<div class="row"><button onclick="replay()">Replay R</button><button onclick="prev()">Prev ←</button><button onclick="next()">Next →</button><button onclick="nextUnreviewed()">Next unreviewed N</button><label><input id="autoplay" type="checkbox" checked> autoplay next</label></div>
</div>
<div class="card grid">
<button id="vGOOD" onclick="saveVerdict('GOOD')">1 · GOOD</button>
<button id="vBAD_BOUNDARY" onclick="saveVerdict('BAD_BOUNDARY')">2 · BAD BOUNDARY</button>
<button id="vBAD_TEXT" onclick="saveVerdict('BAD_TEXT')">3 · BAD TEXT</button>
<button id="vBAD_BOTH" onclick="saveVerdict('BAD_BOTH')">4 · BAD BOTH</button>
<button id="vUNSURE" onclick="saveVerdict('UNSURE')">5 · UNSURE</button>
</div>
<div class="card">
<div class="flags">
<label><input class="flag" type="checkbox" value="clipped_start"> mất âm đầu</label>
<label><input class="flag" type="checkbox" value="clipped_end"> mất âm cuối</label>
<label><input class="flag" type="checkbox" value="mid_phrase_cut"> cắt giữa câu</label>
<label><input class="flag" type="checkbox" value="too_much_silence"> dư silence</label>
<label><input class="flag" type="checkbox" value="noise_or_music"> noise/nhạc</label>
<label><input class="flag" type="checkbox" value="speaker_issue"> speaker issue</label>
</div>
<p><textarea id="note" placeholder="Ghi chú nếu cần..."></textarea></p>
<div class="row"><button onclick="clearRating()">Clear rating</button><span class="muted">Mỗi verdict được checkpoint ngay xuống CSV/JSON local.</span></div>
</div>
<div class="card muted">Space play/pause · 1 good · 2 bad boundary · 3 bad text · 4 bad both · 5 unsure · ←/→ chuyển clip · R replay · N clip chưa review tiếp.</div>
</main>
<script>
let items=[],state={},idx=0;
const g=(id)=>document.getElementById(id),audio=g("audio");
async function api(path,opt={}){const r=await fetch(path,opt),d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d}
function cur(){return items[idx]}
function flags(){return [...document.querySelectorAll(".flag:checked")].map(x=>x.value)}
function setFlags(a){let s=new Set(a||[]);document.querySelectorAll(".flag").forEach(x=>x.checked=s.has(x.value))}
function update(s){let c=s.counts||{};g("progress").textContent=s.reviewed+"/"+s.total+" reviewed · "+s.remaining+" remaining";g("counts").textContent="GOOD "+(c.GOOD||0)+" · BND "+(c.BAD_BOUNDARY||0)+" · TXT "+(c.BAD_TEXT||0)+" · BOTH "+(c.BAD_BOTH||0)+" · ? "+(c.UNSURE||0);g("fill").style.width=(100*s.reviewed/Math.max(1,s.total))+"%"}
function render(play=false){let x=cur();g("order").textContent="review "+(idx+1)+"/"+items.length+" · pilot #"+x.pilot_order;g("clip").textContent=x.clip_id;g("decision").textContent=x.decision;g("reason").textContent=x.pilot_reason;g("quality").textContent="Q "+x.quality_score;g("duration").textContent=x.actual_duration_sec+"s";g("text").textContent=x.text||"(empty)";audio.src=x.audio_url;audio.load();let r=state[x.clip_id]||{};setFlags(r.flags);g("note").value=r.note||"";document.querySelectorAll("[id^='v']").forEach(b=>b.classList.remove("active"));if(r.verdict&&g("v"+r.verdict))g("v"+r.verdict).classList.add("active");if(play&&g("autoplay").checked)audio.play().catch(()=>{})}
async function saveVerdict(v){let x=cur(),f=flags(),n=g("note").value,s=await api("/api/review",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({clip_id:x.clip_id,verdict:v,flags:f,note:n})});state[x.clip_id]={verdict:v,flags:f,note:n};update(s);nextUnreviewed(true)}
async function clearRating(){let x=cur(),s=await api("/api/clear",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({clip_id:x.clip_id})});delete state[x.clip_id];update(s);render(false)}
function prev(){idx=Math.max(0,idx-1);render(false)}function next(play=false){idx=Math.min(items.length-1,idx+1);render(play)}
function nextUnreviewed(play=false){for(let k=1;k<=items.length;k++){let j=(idx+k)%items.length;if(!state[items[j].clip_id]||!state[items[j].clip_id].verdict){idx=j;render(play);return}}next(play)}
function replay(){audio.currentTime=0;audio.play().catch(()=>{})}
document.addEventListener("keydown",(e)=>{if(e.target.tagName==="TEXTAREA"||e.target.tagName==="INPUT")return;if(e.code==="Space"){e.preventDefault();audio.paused?audio.play():audio.pause()}else if(e.key==="1")saveVerdict("GOOD");else if(e.key==="2")saveVerdict("BAD_BOUNDARY");else if(e.key==="3")saveVerdict("BAD_TEXT");else if(e.key==="4")saveVerdict("BAD_BOTH");else if(e.key==="5")saveVerdict("UNSURE");else if(e.key==="ArrowLeft")prev();else if(e.key==="ArrowRight")next();else if(e.key.toLowerCase()==="r")replay();else if(e.key.toLowerCase()==="n")nextUnreviewed(true)});
(async()=>{let d=await api("/api/bootstrap");items=d.items;state=d.state;update(d.summary);let first=items.findIndex(x=>!state[x.clip_id]||!state[x.clip_id].verdict);idx=first>=0?first:0;render(false)})();
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[review] " + (fmt % args), flush=True)

    def send_json(self, obj, status=200):
        data=json.dumps(obj,ensure_ascii=False).encode("utf-8")
        self.send_response(status);self.send_header("Content-Type","application/json; charset=utf-8");self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)

    def do_GET(self):
        path=urllib.parse.urlparse(self.path).path
        if path=="/":
            data=HTML.encode("utf-8");self.send_response(200);self.send_header("Content-Type","text/html; charset=utf-8");self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data);return
        if path=="/api/bootstrap":
            self.send_json({"items":ITEMS,"state":STATE,"summary":summary()});return
        if path.startswith("/audio/"):
            clip=urllib.parse.unquote(path[7:]);p=AUDIO.get(clip)
            if not p or not p.exists():self.send_json({"error":"audio not found"},404);return
            data=p.read_bytes();self.send_response(200);self.send_header("Content-Type","audio/wav");self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data);return
        self.send_json({"error":"not found"},404)

    def do_POST(self):
        try:
            n=int(self.headers.get("Content-Length","0"));payload=json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            clip=str(payload.get("clip_id",""))
            if clip not in ITEM_BY_ID:raise ValueError("unknown clip_id")
            if self.path=="/api/review":
                verdict=str(payload.get("verdict",""))
                if verdict not in VERDICTS:raise ValueError("invalid verdict")
                STATE[clip]={"verdict":verdict,"flags":sorted(set(payload.get("flags",[])) & FLAGS),"note":str(payload.get("note",""))[:1000],"reviewed_at":datetime.now(timezone.utc).isoformat(timespec="seconds")}
            elif self.path=="/api/clear":
                STATE.pop(clip,None)
            else:
                self.send_json({"error":"not found"},404);return
            flush();self.send_json(summary())
        except Exception as e:
            self.send_json({"error":f"{type(e).__name__}: {e}"},400)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--host",default="127.0.0.1")
    p.add_argument("--port",type=int,default=8765)
    p.add_argument("--no-browser",action="store_true")
    a=p.parse_args()
    server=ThreadingHTTPServer((a.host,a.port),Handler)
    url=f"http://{a.host}:{a.port}/"
    print("="*90)
    print("SOURCE16 PILOT REVIEW")
    print(f"clips={len(ITEMS)} reviewed={summary()['reviewed']} remaining={summary()['remaining']}")
    print(f"results={RESULTS_CSV}")
    print(f"url={url}")
    print("Verdicts checkpoint immediately. Ctrl+C stops server; rerun resumes.")
    print("="*90)
    if not a.no_browser:threading.Timer(0.5,lambda:webbrowser.open(url)).start()
    try:server.serve_forever()
    except KeyboardInterrupt:print("\nSTOPPED_LOCAL_REVIEW_SERVER")
    finally:server.server_close();flush()


if __name__=="__main__":
    main()
