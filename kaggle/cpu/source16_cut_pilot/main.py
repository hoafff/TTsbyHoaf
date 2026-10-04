from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path("/kaggle/working/source16_cut_pilot")
CLIPS = ROOT / "clips"
OUT = ROOT / "outputs"

MEDIA_SLUG = "maymay-source16-media"
MANIFEST_SLUG = "maymay-source16-cut-pilot-manifest-v1"
EXPECTED_ROWS = 160
EXPECTED_SOURCES = 16
EXPECTED_PER_SOURCE = 10
ALLOWED_DECISIONS = {"KEEP_AUTO", "KEEP_WHISPER_PRIMARY", "REVIEW"}


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.check_call(cmd)


def find_manifest() -> Path:
    matches = [
        p for p in Path("/kaggle/input").rglob("PILOT_MANIFEST.csv")
        if p.is_file() and MANIFEST_SLUG in str(p).lower()
    ]
    if not matches:
        raise FileNotFoundError(f"Mounted pilot dataset {MANIFEST_SLUG} is missing PILOT_MANIFEST.csv")
    return sorted(matches)[0]


def find_audio(video_id: str, expected_name: str) -> Path:
    exact = [
        p for p in Path("/kaggle/input").rglob(expected_name)
        if p.is_file() and MEDIA_SLUG in str(p).lower()
    ]
    if exact:
        return sorted(exact)[0]

    matches = []
    for p in Path("/kaggle/input").rglob(f"{video_id}.*"):
        if not p.is_file() or MEDIA_SLUG not in str(p).lower():
            continue
        if p.suffix.lower() in {".json3", ".json", ".csv", ".txt", ".part", ".ytdl"}:
            continue
        matches.append(p)

    if not matches:
        raise FileNotFoundError(f"Missing source media for {video_id} (expected {expected_name})")
    return sorted(matches)[0]


def validate_manifest(df: pd.DataFrame) -> None:
    required = {
        "pilot_order", "pilot_reason", "video_id", "clip_id", "decision",
        "cut_start_sec", "cut_end_sec", "cut_duration_sec", "text", "audio_file",
        "quality_score",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Pilot manifest missing columns: {missing}")

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, got {len(df)}")
    if df["clip_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate clip_id in pilot manifest")

    decisions = set(df["decision"].astype(str))
    bad = sorted(decisions - ALLOWED_DECISIONS)
    if bad:
        raise RuntimeError(f"Forbidden pilot decisions: {bad}")

    counts = df.groupby("video_id").size()
    if len(counts) != EXPECTED_SOURCES or not (counts == EXPECTED_PER_SOURCE).all():
        raise RuntimeError(f"Expected 10 pilot rows/source across 16 sources: {counts.to_dict()}")

    starts = pd.to_numeric(df["cut_start_sec"], errors="coerce")
    ends = pd.to_numeric(df["cut_end_sec"], errors="coerce")
    durations = pd.to_numeric(df["cut_duration_sec"], errors="coerce")
    invalid = starts.isna() | ends.isna() | durations.isna() | (starts < 0) | (ends <= starts) | (durations <= 0)
    if invalid.any():
        raise RuntimeError(f"Pilot manifest has {int(invalid.sum())} invalid intervals")


def ffprobe_duration(path: Path) -> float:
    p = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return float(p.stdout.strip())


def cut_one(src: Path, dst: Path, start: float, duration: float) -> float:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-y",
        "-ss", f"{start:.6f}",
        "-i", str(src),
        "-t", f"{duration:.6f}",
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(dst),
    ])
    actual = ffprobe_duration(dst)
    if actual <= 0:
        raise RuntimeError(f"Zero-duration output: {dst}")
    if abs(actual - duration) > 0.35:
        raise RuntimeError(
            f"Output duration mismatch for {dst.name}: requested={duration:.3f}s actual={actual:.3f}s"
        )
    return actual


def main() -> int:
    for p in (ROOT, CLIPS, OUT):
        p.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe are required but not available")

    manifest_path = find_manifest()
    df = pd.read_csv(manifest_path, low_memory=False)
    validate_manifest(df)

    print("=" * 100)
    print("SOURCE16 CUT PILOT V1")
    print(f"manifest={manifest_path}")
    print("rows=160 sources=16 clips_per_source=10")
    print("audio_output=WAV mono PCM16; source sample rate preserved")
    print("=" * 100)

    result_rows = []
    failures = []

    for pos, row in enumerate(df.sort_values("pilot_order").itertuples(index=False), start=1):
        video_id = str(row.video_id)
        clip_id = str(row.clip_id)
        start = float(row.cut_start_sec)
        end = float(row.cut_end_sec)
        duration = end - start
        expected_audio = str(row.audio_file)
        src = find_audio(video_id, expected_audio)
        dst = CLIPS / video_id / f"{clip_id}.wav"

        print(
            f"CUT [{pos:03d}/{EXPECTED_ROWS}] {clip_id} "
            f"{start:.3f}->{end:.3f} ({duration:.3f}s) "
            f"decision={row.decision} reason={row.pilot_reason}",
            flush=True,
        )

        try:
            actual = cut_one(src, dst, start, duration)
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
            print(f"CUT_FAILED {clip_id}: {type(exc).__name__}: {exc}", flush=True)

    result = pd.DataFrame(result_rows)
    failure_df = pd.DataFrame(failures)
    result.to_csv(OUT / "LISTENING_INDEX.csv", index=False, encoding="utf-8-sig")
    failure_df.to_csv(OUT / "FAILURES.csv", index=False, encoding="utf-8-sig")

    clips_ok = int(len(result))
    sources_ok = int(result["video_id"].nunique()) if not result.empty else 0
    total_sec = float(result["actual_duration_sec"].sum()) if not result.empty else 0.0

    summary = {
        "stage": "SOURCE16_CUT_PILOT_V1",
        "expected_clips": EXPECTED_ROWS,
        "clips_ok": clips_ok,
        "clips_failed": int(len(failures)),
        "sources_ok": sources_ok,
        "total_audio_seconds": total_sec,
        "total_audio_minutes": total_sec / 60.0,
        "encoding": {
            "container": "wav",
            "codec": "pcm_s16le",
            "channels": 1,
            "sample_rate": "preserved_from_source",
        },
        "purpose": "Listening-only boundary/text pilot before bulk Source16 cutting.",
    }
    (OUT / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "README.txt").write_text(
        "Listen to clips in clips/<video_id>/ and compare them with outputs/LISTENING_INDEX.csv.\n"
        "Check especially: clipped initial/final phonemes, mid-sentence cuts, excessive silence, wrong text, and REVIEW rows.\n"
        "Do not use this pilot output as the final training dataset until listening QC passes.\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("OUTPUT", ROOT)

    if failures or clips_ok != EXPECTED_ROWS or sources_ok != EXPECTED_SOURCES:
        raise RuntimeError(
            f"Pilot incomplete: clips_ok={clips_ok}/{EXPECTED_ROWS} "
            f"sources_ok={sources_ok}/{EXPECTED_SOURCES} failures={len(failures)}"
        )

    print("CUT_PILOT_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
