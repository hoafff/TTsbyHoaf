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
PILOT_ROOT = (
    ROOT
    / "kaggle_output"
    / "source16-group-sample-cut"
    / "source16_group_sample_cut"
)
INDEX_CSV = PILOT_ROOT / "outputs" / "LISTENING_INDEX.csv"
CLIPS_ROOT = PILOT_ROOT / "clips"
OUT = ROOT / "outputs" / "source16_group_sample_review"
STATE_JSON = OUT / "REVIEW_STATE.json"
RESULTS_CSV = OUT / "REVIEW_RESULTS.csv"

EXPECTED = 15
GROUPS = [
    "ACCEPT_CAPTION_CANDIDATE",
    "REVIEW_ALIGNMENT",
    "REVIEW_WHISPER_PRIMARY",
]
VERDICTS = {"GOOD", "BAD_TEXT", "BAD_BOUNDARY", "BAD_BOTH", "UNSURE"}


def load_items():
    if not INDEX_CSV.exists():
        raise SystemExit(
            "Missing LISTENING_INDEX.csv. Download source16-group-sample-cut output first."
        )

    with INDEX_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != EXPECTED:
        raise SystemExit(f"Expected {EXPECTED} rows, got {len(rows)}")

    counts = {}
    for row in rows:
        group = str(row.get("audit_group", ""))
        counts[group] = counts.get(group, 0) + 1
    if any(counts.get(g, 0) != 5 for g in GROUPS):
        raise SystemExit(f"Expected 5 clips per group, got {counts}")

    items = []
    audio_map = {}
    for row in sorted(rows, key=lambda x: int(float(x.get("sample_order", "0") or 0))):
        clip_id = row["clip_id"]
        rel = str(row.get("clip_file", "")).replace("/", os.sep)
        audio = PILOT_ROOT / rel
        if not audio.exists():
            audio = CLIPS_ROOT / row["audit_group"] / f"{clip_id}.wav"
        if not audio.exists():
            raise SystemExit(f"Missing WAV: {audio}")

        audio_map[clip_id] = audio
        items.append(
            {
                "sample_order": int(float(row.get("sample_order", "0") or 0)),
                "audit_group": row.get("audit_group", ""),
                "video_id": row.get("video_id", ""),
                "clip_id": clip_id,
                "legacy_decision": row.get("legacy_decision", ""),
                "quality_score": row.get("quality_score", ""),
                "proposal_similarity": row.get("proposal_similarity", ""),
                "actual_duration_sec": row.get("actual_duration_sec", ""),
                "whisper_text": row.get("whisper_text", ""),
                "caption_candidate_text": row.get("caption_candidate_text", ""),
                "display_text": row.get("display_text", ""),
                "audio_url": "/audio/" + urllib.parse.quote(clip_id),
            }
        )
    return items, audio_map


def load_state():
    if not STATE_JSON.exists():
        return {}
    obj = json.loads(STATE_JSON.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


ITEMS, AUDIO = load_items()
ITEM_BY_ID = {x["clip_id"]: x for x in ITEMS}
STATE = load_state()
LOCK = threading.Lock()


def summary():
    counts = {x: 0 for x in VERDICTS}
    by_group = {g: {"reviewed": 0, "GOOD": 0, "BAD_TEXT": 0, "BAD_BOUNDARY": 0, "BAD_BOTH": 0, "UNSURE": 0} for g in GROUPS}
    for clip_id, r in STATE.items():
        v = r.get("verdict", "")
        if v not in counts:
            continue
        counts[v] += 1
        group = ITEM_BY_ID.get(clip_id, {}).get("audit_group", "")
        if group in by_group:
            by_group[group]["reviewed"] += 1
            by_group[group][v] += 1
    reviewed = sum(counts.values())
    return {
        "total": len(ITEMS),
        "reviewed": reviewed,
        "remaining": len(ITEMS) - reviewed,
        "counts": counts,
        "by_group": by_group,
    }


def flush():
    with LOCK:
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = STATE_JSON.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(STATE, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(STATE_JSON)

        fields = [
            "sample_order",
            "audit_group",
            "video_id",
            "clip_id",
            "legacy_decision",
            "quality_score",
            "proposal_similarity",
            "actual_duration_sec",
            "whisper_text",
            "caption_candidate_text",
            "verdict",
            "note",
            "reviewed_at",
        ]
        tmp_csv = RESULTS_CSV.with_suffix(".csv.tmp")
        with tmp_csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for item in ITEMS:
                r = STATE.get(item["clip_id"], {})
                w.writerow(
                    {
                        **{k: item.get(k, "") for k in fields[:10]},
                        "verdict": r.get("verdict", ""),
                        "note": r.get("note", ""),
                        "reviewed_at": r.get("reviewed_at", ""),
                    }
                )
        tmp_csv.replace(RESULTS_CSV)


flush()

HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Source16 Group Sample Review</title>
<style>
body{font-family:system-ui;background:#111;color:#eee;margin:0}main{max-width:1100px;margin:auto;padding:18px}.card{background:#1b1b1b;border:1px solid #333;border-radius:12px;padding:16px;margin-bottom:14px}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.badge{background:#292929;padding:6px 9px;border-radius:999px;font-size:13px}.text{font-size:21px;line-height:1.45;margin:10px 0;padding:10px;background:#151515;border-radius:8px}.label{font-size:12px;color:#aaa;margin-top:12px}.help{color:#bbb;line-height:1.4}.good{border-left:4px solid #888}audio{width:100%}button{background:#2a2a2a;color:#fff;border:1px solid #555;border-radius:8px;padding:10px 12px;cursor:pointer}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}textarea{width:100%;min-height:65px;background:#151515;color:#fff;border:1px solid #444;border-radius:8px;padding:8px;box-sizing:border-box}.active{outline:3px solid #aaa}.muted{color:#aaa}#bar{height:9px;background:#333;border-radius:9px;margin-top:10px;overflow:hidden}#fill{height:100%;background:#999;width:0}
</style>
<main>
<div class="card">
<div class="row"><b>Source16 · 5 mẫu/nhóm · 22050 Hz</b><span id="progress" class="badge"></span><span id="counts" class="badge"></span></div>
<div id="bar"><div id="fill"></div></div>
</div>
<div class="card">
<div class="row"><span id="order" class="badge"></span><span id="group" class="badge"></span><span id="source" class="badge"></span><span id="clip" class="badge"></span><span id="sim" class="badge"></span><span id="duration" class="badge"></span></div>
<div id="help" class="help"></div>
<div class="label">WHISPER TEXT</div><div id="whisper" class="text"></div>
<div class="label">CAPTION CANDIDATE (nếu có)</div><div id="caption" class="text"></div>
<audio id="audio" controls preload="auto"></audio>
<div class="row"><button onclick="replay()">Replay R</button><button onclick="prev()">Prev ←</button><button onclick="next()">Next →</button><label><input id="autoplay" type="checkbox" checked> autoplay next</label></div>
</div>
<div class="card grid">
<button id="vGOOD" onclick="saveVerdict('GOOD')">1 · GOOD</button>
<button id="vBAD_TEXT" onclick="saveVerdict('BAD_TEXT')">2 · BAD TEXT</button>
<button id="vBAD_BOUNDARY" onclick="saveVerdict('BAD_BOUNDARY')">3 · BAD BOUNDARY</button>
<button id="vBAD_BOTH" onclick="saveVerdict('BAD_BOTH')">4 · BAD BOTH</button>
<button id="vUNSURE" onclick="saveVerdict('UNSURE')">5 · UNSURE</button>
</div>
<div class="card">
<textarea id="note" placeholder="Ghi chú: caption đúng hơn / Whisper đúng hơn / lỗi từ nào..."></textarea>
<div class="row"><button onclick="clearRating()">Clear rating</button><span class="muted">Lưu ngay xuống CSV/JSON local.</span></div>
</div>
</main>
<script>
let items=[],state={},idx=0;
const g=(id)=>document.getElementById(id),audio=g("audio");
async function api(path,opt={}){const r=await fetch(path,opt),d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d}
function cur(){return items[idx]}
function groupHelp(x){
 if(x.audit_group==="ACCEPT_CAPTION_CANDIDATE")return "Nhóm A: caption alignment đã pass. GOOD khi audio khớp caption candidate và boundary sạch.";
 if(x.audit_group==="REVIEW_ALIGNMENT")return "Nhóm B: caption alignment chưa chắc. So tai với cả Whisper và caption; ghi chú cái nào đúng hơn.";
 return "Nhóm C: caption nguồn low-trust; Whisper đang là text chính. GOOD khi audio khớp Whisper và boundary sạch.";
}
function update(s){let c=s.counts||{};g("progress").textContent=s.reviewed+"/"+s.total+" reviewed";g("counts").textContent="GOOD "+(c.GOOD||0)+" · TXT "+(c.BAD_TEXT||0)+" · BND "+(c.BAD_BOUNDARY||0)+" · BOTH "+(c.BAD_BOTH||0)+" · ? "+(c.UNSURE||0);g("fill").style.width=(100*s.reviewed/Math.max(1,s.total))+"%"}
function render(play=false){let x=cur();g("order").textContent=(idx+1)+"/"+items.length;g("group").textContent=x.audit_group;g("source").textContent=x.video_id;g("clip").textContent=x.clip_id;g("sim").textContent=x.proposal_similarity?"sim "+x.proposal_similarity:"sim n/a";g("duration").textContent=x.actual_duration_sec+"s";g("help").textContent=groupHelp(x);g("whisper").textContent=x.whisper_text||"(empty)";g("caption").textContent=x.caption_candidate_text||"(không dùng caption cho nhóm này)";audio.src=x.audio_url;audio.load();let r=state[x.clip_id]||{};g("note").value=r.note||"";document.querySelectorAll("[id^='v']").forEach(b=>b.classList.remove("active"));if(r.verdict&&g("v"+r.verdict))g("v"+r.verdict).classList.add("active");if(play&&g("autoplay").checked)audio.play().catch(()=>{})}
async function saveVerdict(v){let x=cur(),n=g("note").value,s=await api("/api/review",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({clip_id:x.clip_id,verdict:v,note:n})});state[x.clip_id]={verdict:v,note:n};update(s);next(true)}
async function clearRating(){let x=cur(),s=await api("/api/clear",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({clip_id:x.clip_id})});delete state[x.clip_id];update(s);render(false)}
function prev(){idx=Math.max(0,idx-1);render(false)}function next(play=false){idx=Math.min(items.length-1,idx+1);render(play)}function replay(){audio.currentTime=0;audio.play().catch(()=>{})}
document.addEventListener("keydown",(e)=>{if(e.target.tagName==="TEXTAREA"||e.target.tagName==="INPUT")return;if(e.code==="Space"){e.preventDefault();audio.paused?audio.play():audio.pause()}else if(e.key==="1")saveVerdict("GOOD");else if(e.key==="2")saveVerdict("BAD_TEXT");else if(e.key==="3")saveVerdict("BAD_BOUNDARY");else if(e.key==="4")saveVerdict("BAD_BOTH");else if(e.key==="5")saveVerdict("UNSURE");else if(e.key==="ArrowLeft")prev();else if(e.key==="ArrowRight")next();else if(e.key.toLowerCase()==="r")replay()});
(async()=>{let d=await api("/api/bootstrap");items=d.items;state=d.state;update(d.summary);let first=items.findIndex(x=>!state[x.clip_id]||!state[x.clip_id].verdict);idx=first>=0?first:0;render(false)})();
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[review] " + (fmt % args), flush=True)

    def send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/bootstrap":
            self.send_json({"items": ITEMS, "state": STATE, "summary": summary()})
            return
        if path.startswith("/audio/"):
            clip = urllib.parse.unquote(path[7:])
            p = AUDIO.get(clip)
            if not p or not p.exists():
                self.send_json({"error": "audio not found"}, 404)
                return
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            clip = str(payload.get("clip_id", ""))
            if clip not in ITEM_BY_ID:
                raise ValueError("unknown clip_id")

            if self.path == "/api/review":
                verdict = str(payload.get("verdict", ""))
                if verdict not in VERDICTS:
                    raise ValueError("invalid verdict")
                STATE[clip] = {
                    "verdict": verdict,
                    "note": str(payload.get("note", ""))[:1000],
                    "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            elif self.path == "/api/clear":
                STATE.pop(clip, None)
            else:
                self.send_json({"error": "not found"}, 404)
                return

            flush()
            self.send_json(summary())
        except Exception as exc:
            self.send_json({"error": f"{type(exc).__name__}: {exc}"}, 400)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print("=" * 90)
    print("SOURCE16 GROUP SAMPLE REVIEW · 5 CLIPS PER GROUP")
    print(f"clips={len(ITEMS)} reviewed={summary()['reviewed']}")
    print(f"results={RESULTS_CSV}")
    print(f"url={url}")
    print("Ctrl+C stops local reviewer; results are checkpointed.")
    print("=" * 90)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSTOPPED_LOCAL_REVIEW_SERVER")
    finally:
        server.server_close()
        flush()


if __name__ == "__main__":
    main()
