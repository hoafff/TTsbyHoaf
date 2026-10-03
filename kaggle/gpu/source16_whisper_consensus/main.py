from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from difflib import SequenceMatcher
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

MODEL_SIZE = "large-v3"
ROOT = Path("/kaggle/working/source16_whisper_consensus")
AUDIO_DIR = ROOT / "audio"
CAPTION_DIR = ROOT / "youtube_json3"
OUT_DIR = ROOT / "outputs"
SECRETS_DIR = Path("/kaggle/input/maymay-runtime-secrets")
COOKIE_PATH = SECRETS_DIR / "cookies.txt"
HF_TOKEN_PATH = SECRETS_DIR / "hf_token.txt"

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


def configure_hf_token() -> None:
    if HF_TOKEN_PATH.exists():
        token = HF_TOKEN_PATH.read_text(encoding="utf-8-sig").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            print("HF token loaded from private Kaggle input.", flush=True)
            return
    print("HF token not present; continuing with unauthenticated HF access.", flush=True)


def require_cookie() -> None:
    if not COOKIE_PATH.exists():
        raise FileNotFoundError(f"Missing private cookie input: {COOKIE_PATH}")
    print("Using private YouTube cookie input:", COOKIE_PATH, flush=True)


def download_source(video_id: str) -> tuple[Path, Path]:
    url = f"https://www.youtube.com/watch?v={video_id}"

    for old in AUDIO_DIR.glob(f"{video_id}.*"):
        old.unlink(missing_ok=True)
    for old in CAPTION_DIR.glob(f"{video_id}.*"):
        old.unlink(missing_ok=True)

    audio_tmpl = str(AUDIO_DIR / "%(id)s.%(ext)s")
    common = [
        sys.executable, "-m", "yt_dlp",
        "--cookies", str(COOKIE_PATH),
        "--no-playlist",
        "--no-warnings",
    ]

    run(common + [
        "-f", "bestaudio/best",
        "-o", audio_tmpl,
        url,
    ])

    run(common + [
        "--skip-download",
        "--write-auto-subs",
        "--sub-langs", "vi",
        "--sub-format", "json3",
        "-o", str(CAPTION_DIR / "%(id)s.%(ext)s"),
        url,
    ])

    audio_matches = sorted(
        p for p in AUDIO_DIR.glob(f"{video_id}.*")
        if p.is_file()
    )
    caption_matches = sorted(CAPTION_DIR.glob(f"{video_id}.vi.json3"))

    if not audio_matches:
        raise RuntimeError(f"Audio download failed for {video_id}")
    if not caption_matches:
        raise RuntimeError(f"Vietnamese auto-caption JSON3 missing for {video_id}")

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


def load_model():
    from faster_whisper import WhisperModel
    print("Loading Faster-Whisper model:", MODEL_SIZE, flush=True)
    return WhisperModel(
        MODEL_SIZE,
        device="cuda",
        compute_type="float16",
    )


def transcribe(model, audio_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
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


def build_consensus(whisper_segments: pd.DataFrame, captions: pd.DataFrame) -> pd.DataFrame:
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
        out.loc[valid.index, "gap_before_percentile"] = valid.rank(method="average", pct=True)
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


def process_one(model, video_id: str) -> dict:
    print("=" * 100, flush=True)
    print("SOURCE:", video_id, flush=True)

    source_dir = OUT_DIR / video_id
    source_dir.mkdir(parents=True, exist_ok=True)

    audio_path, caption_path = download_source(video_id)
    captions = parse_youtube_json3(caption_path)
    whisper_segments, words, whisper_meta = transcribe(model, audio_path)
    consensus = build_consensus(whisper_segments, captions)
    pause_features = build_pause_features(words)

    captions.to_csv(source_dir / "YOUTUBE_CAPTION_EVENTS.csv", index=False, encoding="utf-8-sig")
    whisper_segments.to_csv(source_dir / "WHISPER_SEGMENTS.csv", index=False, encoding="utf-8-sig")
    words.to_csv(source_dir / "WHISPER_WORDS.csv", index=False, encoding="utf-8-sig")
    pause_features.to_csv(source_dir / "PAUSE_FEATURES.csv", index=False, encoding="utf-8-sig")
    consensus.to_csv(source_dir / "CONSENSUS_WINDOWS.csv", index=False, encoding="utf-8-sig")
    (source_dir / caption_path.name).write_bytes(caption_path.read_bytes())

    match_counts = (
        consensus["match_band"].value_counts().to_dict()
        if not consensus.empty
        else {}
    )

    summary = {
        "video_id": video_id,
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
        "status": "OK",
    }

    (source_dir / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    audio_path.unlink(missing_ok=True)
    for tmp in CAPTION_DIR.glob(f"{video_id}.*"):
        tmp.unlink(missing_ok=True)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> int:
    print("=" * 100)
    print("SOURCE16 FASTER-WHISPER CONSENSUS")
    print("sources:", len(KEEP_16))
    print("model:", MODEL_SIZE)
    print("=" * 100)

    install_deps()
    require_cookie()
    configure_hf_token()
    model = load_model()

    all_summaries = []
    failures = []

    for idx, video_id in enumerate(KEEP_16, start=1):
        print(f"[{idx:02d}/{len(KEEP_16):02d}] {video_id}", flush=True)
        try:
            all_summaries.append(process_one(model, video_id))
        except Exception as exc:
            failures.append({
                "video_id": video_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print("FAILED:", video_id, repr(exc), flush=True)

    overall = {
        "stage": "SOURCE16_WHISPER_CONSENSUS_V1",
        "sources_total": len(KEEP_16),
        "sources_ok": len(all_summaries),
        "sources_failed": len(failures),
        "model": MODEL_SIZE,
        "word_timestamps": True,
        "cookie_input": "private Kaggle dataset",
        "hf_token_present": HF_TOKEN_PATH.exists(),
        "sources": all_summaries,
        "failures": failures,
    }

    (OUT_DIR / "SUMMARY_ALL.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("OUTPUT:", OUT_DIR)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
