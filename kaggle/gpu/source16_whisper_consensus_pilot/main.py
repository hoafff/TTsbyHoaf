from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

PILOT_VIDEO_ID = "Pf9QTip2hqI"
MODEL_SIZE = "large-v3"

ROOT = Path("/kaggle/working/source16_whisper_consensus_pilot")
AUDIO_DIR = ROOT / "audio"
CAPTION_DIR = ROOT / "youtube_json3"
OUT_DIR = ROOT / "outputs"

for p in (ROOT, AUDIO_DIR, CAPTION_DIR, OUT_DIR):
    p.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def clean_ws(text: str) -> str:
    text = html.unescape(str(text))
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()


def lexical_norm(text: str) -> str:
    text = clean_ws(text).lower()
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹĐđ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    aa = lexical_norm(a)
    bb = lexical_norm(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb).ratio()


def install_deps() -> None:
    run([
        sys.executable, "-m", "pip", "install", "-q", "-U",
        "yt-dlp[default]", "faster-whisper==1.2.1", "av>=11,<19"
    ])


def download_source() -> tuple[Path, Path]:
    url = f"https://www.youtube.com/watch?v={PILOT_VIDEO_ID}"

    audio_tmpl = str(AUDIO_DIR / "%(id)s.%(ext)s")
    run([
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "-f", "bestaudio/best",
        "--no-warnings",
        "-o", audio_tmpl,
        url,
    ])

    run([
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--skip-download",
        "--write-auto-subs",
        "--sub-langs", "vi",
        "--sub-format", "json3",
        "--no-warnings",
        "-o", str(CAPTION_DIR / "%(id)s.%(ext)s"),
        url,
    ])

    audio_matches = sorted(
        p for p in AUDIO_DIR.glob(f"{PILOT_VIDEO_ID}.*")
        if p.is_file()
    )
    caption_matches = sorted(CAPTION_DIR.glob(f"{PILOT_VIDEO_ID}.vi.json3"))

    if not audio_matches:
        raise RuntimeError("Audio download did not produce a file.")
    if not caption_matches:
        raise RuntimeError("Vietnamese automatic caption JSON3 was not found.")

    return audio_matches[0], caption_matches[0]


def parse_youtube_json3(path: Path) -> pd.DataFrame:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for event_index, event in enumerate(obj.get("events", [])):
        segs = event.get("segs") or []
        raw_text = "".join(str(seg.get("utf8", "")) for seg in segs)
        text = clean_ws(raw_text)
        if not text:
            continue

        start_ms = float(event.get("tStartMs") or 0)
        dur_ms = float(event.get("dDurationMs") or 0)
        rows.append({
            "event_index": event_index,
            "start_sec": start_ms / 1000.0,
            "end_sec": (start_ms + dur_ms) / 1000.0,
            "duration_sec": dur_ms / 1000.0,
            "text": text,
        })
    return pd.DataFrame(rows)


def transcribe(audio_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        MODEL_SIZE,
        device="cuda",
        compute_type="float16",
    )

    segments_gen, info = model.transcribe(
        str(audio_path),
        language="vi",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )

    segment_rows = []
    word_rows = []

    for seg in segments_gen:
        segment_rows.append({
            "segment_id": int(seg.id),
            "start_sec": float(seg.start),
            "end_sec": float(seg.end),
            "duration_sec": float(seg.end - seg.start),
            "text": clean_ws(seg.text),
            "avg_logprob": float(seg.avg_logprob),
            "no_speech_prob": float(seg.no_speech_prob),
            "compression_ratio": float(seg.compression_ratio),
        })

        prev_end = None
        for word_index, word in enumerate(seg.words or []):
            start = float(word.start)
            end = float(word.end)
            word_rows.append({
                "segment_id": int(seg.id),
                "word_index": int(word_index),
                "word": clean_ws(word.word),
                "start_sec": start,
                "end_sec": end,
                "duration_sec": max(0.0, end - start),
                "gap_before_sec": None if prev_end is None else max(0.0, start - prev_end),
                "probability": float(word.probability),
            })
            prev_end = end

    meta = {
        "language": str(info.language),
        "language_probability": float(info.language_probability),
        "duration_sec": float(info.duration),
        "duration_after_vad_sec": float(info.duration_after_vad),
        "model": MODEL_SIZE,
        "word_timestamps": True,
        "vad_filter": True,
        "beam_size": 5,
    }
    return pd.DataFrame(segment_rows), pd.DataFrame(word_rows), meta


def overlap_text(captions: pd.DataFrame, start: float, end: float) -> str:
    if captions.empty:
        return ""
    g = captions[(captions["end_sec"] > start) & (captions["start_sec"] < end)]
    return clean_ws(" ".join(g["text"].astype(str).tolist()))


def build_consensus(
    whisper_segments: pd.DataFrame,
    captions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for row in whisper_segments.itertuples(index=False):
        yt_text = overlap_text(captions, float(row.start_sec), float(row.end_sec))
        score = similarity(str(row.text), yt_text)

        if score >= 0.85:
            band = "HIGH"
        elif score >= 0.65:
            band = "MID"
        else:
            band = "LOW"

        rows.append({
            "segment_id": int(row.segment_id),
            "start_sec": float(row.start_sec),
            "end_sec": float(row.end_sec),
            "whisper_text": str(row.text),
            "youtube_text_overlap": yt_text,
            "lexical_similarity": score,
            "match_band": band,
        })
    return pd.DataFrame(rows)


def build_pause_features(words: pd.DataFrame) -> pd.DataFrame:
    if words.empty:
        return words.copy()

    out = words.copy()
    out["gap_before_sec"] = pd.to_numeric(out["gap_before_sec"], errors="coerce")

    valid = out["gap_before_sec"].dropna()
    if len(valid) >= 5:
        ranks = valid.rank(method="average", pct=True)
        out.loc[valid.index, "gap_before_percentile"] = ranks
    else:
        out["gap_before_percentile"] = None

    return out


def quantiles(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


def main() -> int:
    print("=" * 100)
    print("SOURCE16 FASTER-WHISPER CONSENSUS PILOT")
    print("video:", PILOT_VIDEO_ID)
    print("model:", MODEL_SIZE)
    print("=" * 100)

    install_deps()
    audio_path, caption_path = download_source()

    captions = parse_youtube_json3(caption_path)
    whisper_segments, words, whisper_meta = transcribe(audio_path)
    consensus = build_consensus(whisper_segments, captions)
    pause_features = build_pause_features(words)

    captions.to_csv(OUT_DIR / "YOUTUBE_CAPTION_EVENTS.csv", index=False, encoding="utf-8-sig")
    whisper_segments.to_csv(OUT_DIR / "WHISPER_SEGMENTS.csv", index=False, encoding="utf-8-sig")
    words.to_csv(OUT_DIR / "WHISPER_WORDS.csv", index=False, encoding="utf-8-sig")
    pause_features.to_csv(OUT_DIR / "PAUSE_FEATURES.csv", index=False, encoding="utf-8-sig")
    consensus.to_csv(OUT_DIR / "CONSENSUS_WINDOWS.csv", index=False, encoding="utf-8-sig")

    # Keep exact YouTube raw provenance in job output as well.
    (OUT_DIR / caption_path.name).write_bytes(caption_path.read_bytes())

    match_counts = (
        consensus["match_band"].value_counts().to_dict()
        if not consensus.empty
        else {}
    )

    summary = {
        "stage": "SOURCE16_WHISPER_CONSENSUS_PILOT_V1",
        "video_id": PILOT_VIDEO_ID,
        "audio_file": audio_path.name,
        "youtube_caption_events": int(len(captions)),
        "whisper_segments": int(len(whisper_segments)),
        "whisper_words": int(len(words)),
        "match_band_counts": {str(k): int(v) for k, v in match_counts.items()},
        "global_whisper_vs_youtube_similarity": similarity(
            " ".join(whisper_segments["text"].astype(str).tolist()),
            " ".join(captions["text"].astype(str).tolist()),
        ),
        "gap_before_sec_quantiles": quantiles(pause_features.get("gap_before_sec", pd.Series(dtype=float))),
        "word_duration_sec_quantiles": quantiles(pause_features.get("duration_sec", pd.Series(dtype=float))),
        "whisper": whisper_meta,
        "notes": [
            "This pilot does not modify semantic_v2 review results.",
            "Similarity is diagnostic only; no automatic KEEP/DROP decision is made here.",
            "YouTube punctuation is preserved as evidence but is not treated as ground truth.",
            "Faster-Whisper word timestamps are retained for later pause/prosody QC.",
        ],
    }

    (OUT_DIR / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Do not keep the ~200 MB downloaded source audio in Kaggle output.
    # Raw caption provenance is already copied into OUT_DIR above.
    try:
        audio_path.unlink(missing_ok=True)
    except Exception as exc:
        print("WARN: audio cleanup failed:", exc)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("OUTPUT:", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
