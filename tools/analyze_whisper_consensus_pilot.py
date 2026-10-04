from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "kaggle_output" / "source16-whisper-consensus-pilot" / "source16_whisper_consensus_pilot" / "outputs"


def norm(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹĐđ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def toks(text: str) -> list[str]:
    n = norm(text)
    return n.split() if n else []


def token_ratio(a: str, b: str) -> float:
    aa = toks(a)
    bb = toks(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb, autojunk=False).ratio()


def resolve_out(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if not p.is_absolute():
            p = ROOT / p
        return p.resolve()

    if DEFAULT_OUT.exists():
        return DEFAULT_OUT.resolve()

    candidates = list((ROOT / "kaggle_output" / "source16-whisper-consensus-pilot").rglob("WHISPER_SEGMENTS.csv"))
    if not candidates:
        raise SystemExit("Pilot output not found. Download it first with kaggle_bridge.py output.")
    return candidates[0].parent.resolve()


def bucket_text(df: pd.DataFrame, text_col: str, bucket_sec: int) -> dict[int, str]:
    if df.empty:
        return {}
    mid = (pd.to_numeric(df["start_sec"]) + pd.to_numeric(df["end_sec"])) / 2.0
    bucket = (mid // bucket_sec).astype(int)
    tmp = df.copy()
    tmp["_bucket"] = bucket
    return {
        int(k): " ".join(g[text_col].astype(str).tolist())
        for k, g in tmp.groupby("_bucket", sort=True)
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    p.add_argument("--bucket-sec", type=int, default=30)
    args = p.parse_args()

    out = resolve_out(args.out)
    yt = pd.read_csv(out / "YOUTUBE_CAPTION_EVENTS.csv", low_memory=False)
    ws = pd.read_csv(out / "WHISPER_SEGMENTS.csv", low_memory=False)
    ww = pd.read_csv(out / "WHISPER_WORDS.csv", low_memory=False)

    # Diagnose rolling/duplicate YouTube caption events.
    yt_norm = yt["text"].astype(str).map(norm)
    consecutive_exact_dup = int((yt_norm == yt_norm.shift(1)).sum())
    unique_ratio = float(yt_norm.nunique() / max(1, len(yt_norm)))

    # Robust text agreement at coarse time buckets, avoiding segment-boundary mismatch
    # and avoiding SequenceMatcher on one enormous transcript.
    yt_b = bucket_text(yt, "text", args.bucket_sec)
    ws_b = bucket_text(ws, "text", args.bucket_sec)
    common = sorted(set(yt_b) & set(ws_b))

    bucket_rows = []
    for b in common:
        score = token_ratio(yt_b[b], ws_b[b])
        bucket_rows.append({
            "bucket": b,
            "start_sec": b * args.bucket_sec,
            "end_sec": (b + 1) * args.bucket_sec,
            "token_similarity": score,
            "youtube_text": yt_b[b],
            "whisper_text": ws_b[b],
        })
    buckets = pd.DataFrame(bucket_rows)

    # Global chronological word gaps, including transitions between Whisper segments.
    words = ww.sort_values(["start_sec", "end_sec"], kind="stable").reset_index(drop=True).copy()
    words["prev_end_sec_global"] = pd.to_numeric(words["end_sec"], errors="coerce").shift(1)
    words["gap_before_global_sec"] = (
        pd.to_numeric(words["start_sec"], errors="coerce") - words["prev_end_sec_global"]
    ).clip(lower=0)
    words["cross_segment_boundary"] = (
        pd.to_numeric(words["segment_id"], errors="coerce")
        != pd.to_numeric(words["segment_id"], errors="coerce").shift(1)
    )

    gaps = words["gap_before_global_sec"].dropna()
    durations = pd.to_numeric(words["duration_sec"], errors="coerce").dropna()

    def q(s: pd.Series) -> dict:
        if s.empty:
            return {}
        return {
            "p50": float(s.quantile(.50)),
            "p90": float(s.quantile(.90)),
            "p95": float(s.quantile(.95)),
            "p99": float(s.quantile(.99)),
            "p999": float(s.quantile(.999)),
            "max": float(s.max()),
        }

    bucket_scores = buckets["token_similarity"] if not buckets.empty else pd.Series(dtype=float)
    summary = {
        "output_dir": str(out),
        "youtube_events": int(len(yt)),
        "whisper_segments": int(len(ws)),
        "whisper_words": int(len(ww)),
        "youtube_consecutive_exact_duplicate_events": consecutive_exact_dup,
        "youtube_unique_normalized_event_ratio": unique_ratio,
        "bucket_seconds": args.bucket_sec,
        "bucket_count": int(len(buckets)),
        "bucket_similarity": q(bucket_scores),
        "bucket_similarity_mean": float(bucket_scores.mean()) if not bucket_scores.empty else None,
        "bucket_similarity_ge_085": int((bucket_scores >= .85).sum()) if not bucket_scores.empty else 0,
        "bucket_similarity_065_085": int(((bucket_scores >= .65) & (bucket_scores < .85)).sum()) if not bucket_scores.empty else 0,
        "bucket_similarity_lt_065": int((bucket_scores < .65).sum()) if not bucket_scores.empty else 0,
        "global_word_gap_sec": q(gaps),
        "word_duration_sec": q(durations),
        "cross_segment_boundaries": int(words["cross_segment_boundary"].sum()),
        "notes": [
            "30-second bucket similarity is diagnostic and is more robust than the pilot's per-segment overlap score.",
            "Global word gaps include pauses across Whisper segment boundaries; the original pilot gap_before_sec did not.",
            "No text is rewritten and no KEEP/DROP decision is made.",
        ],
    }

    buckets.to_csv(out / "CONSENSUS_BUCKETS_30S.csv", index=False, encoding="utf-8-sig")
    words.to_csv(out / "WHISPER_WORDS_GLOBAL_GAPS.csv", index=False, encoding="utf-8-sig")

    if not buckets.empty:
        buckets.nsmallest(20, "token_similarity").to_csv(
            out / "CONSENSUS_BUCKETS_LOWEST_20.csv", index=False, encoding="utf-8-sig"
        )
        buckets.nlargest(20, "token_similarity").to_csv(
            out / "CONSENSUS_BUCKETS_HIGHEST_20.csv", index=False, encoding="utf-8-sig"
        )

    (out / "POSTPROCESS_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
