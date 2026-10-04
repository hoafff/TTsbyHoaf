from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path("/kaggle/working/source16_keep_auto_cut_pilot")
CLIPS = ROOT / "clips"
OUT = ROOT / "outputs"

MEDIA_SLUG = "maymay-source16-media"
MANIFEST_SLUG = "maymay-source16-keep-auto-pilot-manifest-v1"
EXPECTED_ROWS = 160
TARGET_SAMPLE_RATE = 22050


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.check_call(cmd)


def find_manifest() -> Path:
    matches = [
        p for p in Path("/kaggle/input").rglob("PILOT_MANIFEST.csv")
        if p.is_file() and MANIFEST_SLUG in str(p).lower()
    ]
    if not matches:
        raise FileNotFoundError("Missing KEEP_AUTO pilot manifest dataset")
    return sorted(matches)[0]


def find_audio(video_id: str, expected_name: str) -> Path:
    exact = [
        p for p in Path("/kaggle/input").rglob(expected_name)
        if p.is_file() and MEDIA_SLUG in str(p).lower()
    ]
    if exact:
        return sorted(exact)[0]

    matches = []
    for p in Path("/kaggle/input").rglob("%s.*" % video_id):
        if not p.is_file() or MEDIA_SLUG not in str(p).lower():
            continue
        if p.suffix.lower() in {".json3", ".json", ".csv", ".txt", ".part", ".ytdl"}:
            continue
        matches.append(p)
    if not matches:
        raise FileNotFoundError("Missing source media for %s" % video_id)
    return sorted(matches)[0]


def validate_manifest(df: pd.DataFrame) -> None:
    required = {
        "pilot_order", "pilot_reason", "video_id", "clip_id", "decision",
        "cut_start_sec", "cut_end_sec", "text", "audio_file", "quality_score",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError("Manifest missing columns: %r" % missing)
    if len(df) != EXPECTED_ROWS:
        raise RuntimeError("Expected %d rows, got %d" % (EXPECTED_ROWS, len(df)))
    if set(df["decision"].astype(str)) != {"KEEP_AUTO"}:
        raise RuntimeError("Pilot must be KEEP_AUTO-only")
    if df["clip_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate clip_id")

    starts = pd.to_numeric(df["cut_start_sec"], errors="coerce")
    ends = pd.to_numeric(df["cut_end_sec"], errors="coerce")
    invalid = starts.isna() | ends.isna() | (starts < 0) | (ends <= starts)
    if invalid.any():
        raise RuntimeError("Invalid intervals: %d" % int(invalid.sum()))


def probe(path: Path) -> tuple[float, int, int]:
    p = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels:format=duration",
            "-of", "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    data = json.loads(p.stdout)
    stream = data["streams"][0]
    duration = float(data["format"]["duration"])
    return duration, int(stream["sample_rate"]), int(stream["channels"])


def cut_one(src: Path, dst: Path, start: float, duration: float) -> tuple[float, int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", "%.6f" % start,
        "-i", str(src),
        "-t", "%.6f" % duration,
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(dst),
    ])
    actual, sample_rate, channels = probe(dst)
    if abs(actual - duration) > 0.35:
        raise RuntimeError("Duration mismatch %s requested=%.3f actual=%.3f" % (dst.name, duration, actual))
    if sample_rate != TARGET_SAMPLE_RATE:
        raise RuntimeError("Wrong sample rate %s: %d" % (dst.name, sample_rate))
    if channels != 1:
        raise RuntimeError("Wrong channel count %s: %d" % (dst.name, channels))
    return actual, sample_rate, channels


def main() -> int:
    for p in (ROOT, CLIPS, OUT):
        p.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe unavailable")

    manifest = find_manifest()
    df = pd.read_csv(manifest, low_memory=False)
    validate_manifest(df)
    expected_sources = int(df["video_id"].nunique())

    print("=" * 100)
    print("SOURCE16 KEEP_AUTO CUT PILOT V1")
    print("rows=%d sources=%d" % (len(df), expected_sources))
    print("audio_output=WAV mono PCM16 @ %d Hz" % TARGET_SAMPLE_RATE)
    print("=" * 100)

    result_rows = []
    failures = []

    for pos, row in enumerate(df.sort_values("pilot_order").itertuples(index=False), start=1):
        video_id = str(row.video_id)
        clip_id = str(row.clip_id)
        start = float(row.cut_start_sec)
        end = float(row.cut_end_sec)
        duration = end - start
        src = find_audio(video_id, str(row.audio_file))
        dst = CLIPS / video_id / ("%s.wav" % clip_id)

        print("CUT [%03d/%03d] %s %.3f->%.3f (%.3fs)" %
              (pos, EXPECTED_ROWS, clip_id, start, end, duration), flush=True)
        try:
            actual, sample_rate, channels = cut_one(src, dst, start, duration)
            result_rows.append({
                "pilot_order": int(row.pilot_order),
                "pilot_reason": str(row.pilot_reason),
                "video_id": video_id,
                "clip_id": clip_id,
                "decision": str(row.decision),
                "quality_score": float(row.quality_score),
                "requested_start_sec": start,
                "requested_end_sec": end,
                "requested_duration_sec": duration,
                "actual_duration_sec": actual,
                "output_sample_rate_hz": sample_rate,
                "output_channels": channels,
                "text": str(row.text),
                "source_audio_file": src.name,
                "clip_file": str(dst.relative_to(ROOT)).replace("\\", "/"),
                "clip_bytes": int(dst.stat().st_size),
                "status": "OK",
            })
        except Exception as exc:
            failures.append({
                "pilot_order": int(row.pilot_order),
                "video_id": video_id,
                "clip_id": clip_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print("CUT_FAILED %s: %s: %s" % (clip_id, type(exc).__name__, exc), flush=True)

    result = pd.DataFrame(result_rows)
    pd.DataFrame(failures).to_csv(OUT / "FAILURES.csv", index=False, encoding="utf-8-sig")
    result.to_csv(OUT / "LISTENING_INDEX.csv", index=False, encoding="utf-8-sig")

    clips_ok = int(len(result))
    sources_ok = int(result["video_id"].nunique()) if not result.empty else 0
    sr_ok = bool(not result.empty and (result["output_sample_rate_hz"] == TARGET_SAMPLE_RATE).all())
    mono_ok = bool(not result.empty and (result["output_channels"] == 1).all())

    summary = {
        "stage": "SOURCE16_KEEP_AUTO_CUT_PILOT_V1",
        "expected_clips": EXPECTED_ROWS,
        "clips_ok": clips_ok,
        "clips_failed": int(len(failures)),
        "sources_expected": expected_sources,
        "sources_ok": sources_ok,
        "sample_rate_check": sr_ok,
        "mono_check": mono_ok,
        "total_audio_seconds": float(result["actual_duration_sec"].sum()) if not result.empty else 0.0,
        "encoding": {
            "container": "wav",
            "codec": "pcm_s16le",
            "channels": 1,
            "sample_rate_hz": TARGET_SAMPLE_RATE,
        },
        "purpose": "Validate KEEP_AUTO segmentation at final 22050 Hz training format before bulk cutting.",
    }
    (OUT / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures or clips_ok != EXPECTED_ROWS or sources_ok != expected_sources or not sr_ok or not mono_ok:
        raise RuntimeError("KEEP_AUTO pilot failed validation")

    print("KEEP_AUTO_CUT_PILOT_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
