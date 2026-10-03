from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

KEEP_16 = [
    "RiAsGTSORZ4",
    "HdbsrkbYzHM",
    "-XC3WibmUKQ",
    "YaxBsFrzwZE",
    "I29SlhGqXgg",
    "0_gFaWnMiCY",
    "Pf9QTip2hqI",
    "S9YN9llAFv4",
    "arAbGLeCUsU",
    "E_AgFYj41Nk",
    "vimmfFJdrYM",
    "hDNesxEPAws",
    "HAe5kdfMP9U",
    "Itad_gcdHHM",
    "0Wjq6gqd1Sc",
    "9XJ78IeFBxY",
]

EXCLUDE_ONLY = "0nJXkX0L3kQ"

OUT = Path("/kaggle/working/source16_audit")
OUT.mkdir(parents=True, exist_ok=True)
LOGS = OUT / "logs"
LOGS.mkdir(exist_ok=True)

assert len(KEEP_16) == 16
assert len(set(KEEP_16)) == 16
assert EXCLUDE_ONLY not in KEEP_16


def install_ytdlp() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-U",
            "--pre",
            "yt-dlp[default]",
        ]
    )


def run_json(video_id: str) -> tuple[dict | None, str]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--skip-download",
        "--dump-single-json",
        "--no-warnings",
        url,
    ]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    combined_log = (p.stderr or "")[-8000:]
    if p.returncode != 0:
        return None, combined_log
    try:
        return json.loads(p.stdout), combined_log
    except Exception as exc:
        return None, f"JSON parse error: {exc!r}\n{combined_log}\n{p.stdout[-2000:]}"


def vi_keys(obj: object) -> list[str]:
    if not isinstance(obj, dict):
        return []
    out = []
    for key in obj:
        k = str(key)
        low = k.lower()
        if low == "vi" or low.startswith("vi-") or low.startswith("vi."):
            out.append(k)
    return sorted(out)


install_ytdlp()

try:
    import yt_dlp
    print("yt-dlp:", yt_dlp.version.__version__)
except Exception:
    pass

rows = []

print("=" * 100)
print("MAYMAY SOURCE16 TRANSCRIPT/CAPTION AUDIT")
print("=" * 100)

for idx, video_id in enumerate(KEEP_16, start=1):
    print(f"[{idx:02d}/16] {video_id}", flush=True)
    info, log = run_json(video_id)
    (LOGS / f"{idx:02d}_{video_id}.log").write_text(
        log,
        encoding="utf-8",
        errors="ignore",
    )

    if info is None:
        rows.append(
            {
                "index": idx,
                "video_id": video_id,
                "status": "EXTRACT_FAIL",
                "canonical_text_candidate": "UNKNOWN",
            }
        )
        print("  EXTRACT_FAIL", flush=True)
        continue

    manual = info.get("subtitles") or {}
    automatic = info.get("automatic_captions") or {}
    manual_vi = vi_keys(manual)
    auto_vi = vi_keys(automatic)

    if manual_vi:
        candidate = "MANUAL_VI"
    elif auto_vi:
        candidate = "AUTO_VI"
    else:
        candidate = "NEED_ASR"

    duration = info.get("duration")
    try:
        duration = float(duration)
    except Exception:
        duration = None

    rows.append(
        {
            "index": idx,
            "video_id": video_id,
            "status": "OK",
            "title": info.get("title", ""),
            "channel": info.get("channel") or info.get("uploader") or "",
            "duration_sec": duration,
            "duration_hours": duration / 3600 if duration else None,
            "manual_vi": bool(manual_vi),
            "manual_vi_keys": ",".join(manual_vi),
            "auto_vi": bool(auto_vi),
            "auto_vi_keys": ",".join(auto_vi),
            "manual_language_count": len(manual),
            "auto_language_count": len(automatic),
            "canonical_text_candidate": candidate,
            "webpage_url": info.get("webpage_url", ""),
        }
    )
    print(
        f"  {candidate} | "
        f"{duration / 3600:.3f} h" if duration else f"  {candidate} | duration=?",
        flush=True,
    )

df = pd.DataFrame(rows)
csv_path = OUT / "SOURCE16_TRANSCRIPT_AUDIT.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

ok = df[df["status"] == "OK"].copy()
candidate_col = ok.get("canonical_text_candidate", pd.Series(dtype=str))

summary = {
    "stage": "SOURCE16_TRANSCRIPT_AUDIT_V1",
    "sources_total": 16,
    "sources_ok": int(len(ok)),
    "sources_failed": int((df["status"] != "OK").sum()),
    "manual_vi_sources": int((candidate_col == "MANUAL_VI").sum()),
    "auto_vi_sources": int((candidate_col == "AUTO_VI").sum()),
    "need_asr_sources": int((candidate_col == "NEED_ASR").sum()),
    "total_source_hours": float(
        pd.to_numeric(ok.get("duration_hours"), errors="coerce").sum()
    ),
    "excluded_video_id": EXCLUDE_ONLY,
    "critical_rule": "Exclude only 0nJXkX0L3kQ; all 16 kept sources use the same V2 pipeline.",
}

(OUT / "SUMMARY.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("=" * 100)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("OUTPUT:", OUT)
