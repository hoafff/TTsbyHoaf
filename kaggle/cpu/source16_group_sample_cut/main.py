from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path("/kaggle/working/source16_group_sample_cut")
CLIPS = ROOT / "clips"
OUT = ROOT / "outputs"

MEDIA_SLUG = "maymay-source16-media"
MANIFEST_SLUG = "maymay-source16-group-sample-manifest-v1"
EXPECTED_ROWS = 15
PER_GROUP = 5
TARGET_SAMPLE_RATE = 22050
EXPECTED_GROUPS = {
    "ACCEPT_CAPTION_CANDIDATE",
    "REVIEW_ALIGNMENT",
    "REVIEW_WHISPER_PRIMARY",
}


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.check_call(cmd)


def find_manifest() -> Path:
    matches = [
        p
        for p in Path("/kaggle/input").rglob("SAMPLE_MANIFEST.csv")
        if p.is_file() and MANIFEST_SLUG in str(p).lower()
    ]
    if not matches:
        raise FileNotFoundError("Missing Source16 group-sample manifest dataset")
    return sorted(matches)[0]


def find_audio(video_id: str, expected_name: str) -> Path:
    exact = [
        p
        for p in Path("/kaggle/input").rglob(expected_name)
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
        raise FileNotFoundError(f"Missing source media for {video_id}")
    return sorted(matches)[0]


def validate_manifest(df: pd.DataFrame) -> None:
    required = {
        "sample_order",
        "audit_group",
        "video_id",
        "clip_id",
        "cut_start_sec",
        "cut_end_sec",
        "audio_file",
        "whisper_text",
        "caption_candidate_text",
        "display_text",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Manifest missing columns: {missing}")
    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, got {len(df)}")
    if set(df["audit_group"].astype(str)) != EXPECTED_GROUPS:
        raise RuntimeError("Unexpected/missing audit groups")
    counts = df["audit_group"].value_counts().to_dict()
    if any(int(counts.get(g, 0)) != PER_GROUP for g in EXPECTED_GROUPS):
        raise RuntimeError(f"Expected {PER_GROUP} per group, got {counts}")
    if df["clip_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate clip_id")

    starts = pd.to_numeric(df["cut_start_sec"], errors="coerce")
    ends = pd.to_numeric(df["cut_end_sec"], errors="coerce")
    invalid = starts.isna() | ends.isna() | (starts < 0) | (ends <= starts)
    if invalid.any():
        raise RuntimeError(f"Invalid intervals: {int(invalid.sum())}")


def probe(path: Path) -> tuple[float, int, int]:
    p = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels:format=duration",
            "-of",
            "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    data = json.loads(p.stdout)
    stream = data["streams"][0]
    return (
        float(data["format"]["duration"]),
        int(stream["sample_rate"]),
        int(stream["channels"]),
    )


def cut_one(src: Path, dst: Path, start: float, duration: float) -> tuple[float, int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.6f}",
            "-i",
            str(src),
            "-t",
            f"{duration:.6f}",
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    )
    actual, sample_rate, channels = probe(dst)
    if abs(actual - duration) > 0.35:
        raise RuntimeError(
            f"Duration mismatch {dst.name} requested={duration:.3f} actual={actual:.3f}"
        )
    if sample_rate != TARGET_SAMPLE_RATE or channels != 1:
        raise RuntimeError(
            f"Wrong output format {dst.name}: sr={sample_rate} channels={channels}"
        )
    return actual, sample_rate, channels


def main() -> int:
    for p in (ROOT, CLIPS, OUT):
        p.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe unavailable")

    df = pd.read_csv(find_manifest(), low_memory=False).fillna("")
    validate_manifest(df)

    print("=" * 100)
    print("SOURCE16 15-CLIP GROUP SAMPLE CUT")
    print("5 x ACCEPT_CAPTION_CANDIDATE")
    print("5 x REVIEW_ALIGNMENT")
    print("5 x REVIEW_WHISPER_PRIMARY")
    print(f"audio_output=WAV mono PCM16 @ {TARGET_SAMPLE_RATE} Hz")
    print("=" * 100)

    rows = []
    failures = []
    for pos, row in enumerate(df.sort_values("sample_order").itertuples(index=False), 1):
        start = float(row.cut_start_sec)
        end = float(row.cut_end_sec)
        duration = end - start
        video_id = str(row.video_id)
        clip_id = str(row.clip_id)
        src = find_audio(video_id, str(row.audio_file))
        dst = CLIPS / str(row.audit_group) / f"{clip_id}.wav"

        print(
            f"CUT [{pos:02d}/{EXPECTED_ROWS:02d}] {row.audit_group} "
            f"{clip_id} {start:.3f}->{end:.3f}",
            flush=True,
        )
        try:
            actual, sr, ch = cut_one(src, dst, start, duration)
            rows.append(
                {
                    "sample_order": int(row.sample_order),
                    "audit_group": str(row.audit_group),
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "legacy_decision": str(row.decision),
                    "quality_score": str(row.quality_score),
                    "proposal_similarity": str(row.proposal_similarity),
                    "requested_start_sec": start,
                    "requested_end_sec": end,
                    "requested_duration_sec": duration,
                    "actual_duration_sec": actual,
                    "output_sample_rate_hz": sr,
                    "output_channels": ch,
                    "whisper_text": str(row.whisper_text),
                    "caption_candidate_text": str(row.caption_candidate_text),
                    "display_text": str(row.display_text),
                    "source_audio_file": src.name,
                    "clip_file": str(dst.relative_to(ROOT)).replace("\\", "/"),
                    "status": "OK",
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "sample_order": int(row.sample_order),
                    "audit_group": str(row.audit_group),
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    result = pd.DataFrame(rows)
    pd.DataFrame(failures).to_csv(
        OUT / "FAILURES.csv", index=False, encoding="utf-8-sig"
    )
    result.to_csv(
        OUT / "LISTENING_INDEX.csv", index=False, encoding="utf-8-sig"
    )

    group_counts = (
        {str(k): int(v) for k, v in result["audit_group"].value_counts().to_dict().items()}
        if not result.empty
        else {}
    )
    summary = {
        "stage": "SOURCE16_GROUP_SAMPLE_CUT_V1",
        "expected_clips": EXPECTED_ROWS,
        "clips_ok": int(len(result)),
        "clips_failed": int(len(failures)),
        "group_counts": group_counts,
        "sample_rate_check": bool(
            not result.empty
            and (result["output_sample_rate_hz"] == TARGET_SAMPLE_RATE).all()
        ),
        "mono_check": bool(
            not result.empty and (result["output_channels"] == 1).all()
        ),
        "encoding": {
            "container": "wav",
            "codec": "pcm_s16le",
            "channels": 1,
            "sample_rate_hz": TARGET_SAMPLE_RATE,
        },
    }
    (OUT / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if (
        failures
        or len(result) != EXPECTED_ROWS
        or any(group_counts.get(g, 0) != PER_GROUP for g in EXPECTED_GROUPS)
        or not summary["sample_rate_check"]
        or not summary["mono_check"]
    ):
        raise RuntimeError("Source16 group sample cut failed validation")

    print("SOURCE16_GROUP_SAMPLE_CUT_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
