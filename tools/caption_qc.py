from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PCTS = [0.50, 0.75, 0.90, 0.95, 0.99]


def qdict(series: pd.Series, prefix: str) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    out: dict[str, float] = {
        f"{prefix}_count": int(len(s)),
        f"{prefix}_mean": float(s.mean()),
        f"{prefix}_max": float(s.max()),
    }
    for q in PCTS:
        out[f"{prefix}_p{int(q*100):02d}"] = float(s.quantile(q))
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="QC the 16-source YouTube caption structure before semantic sentence planning."
    )
    p.add_argument(
        "input_dir",
        nargs="?",
        default="kaggle_output/source16-captions/source16_captions",
        help="Downloaded source16-captions output folder.",
    )
    p.add_argument(
        "--out",
        default="outputs/caption_qc_v1",
        help="QC output folder.",
    )
    args = p.parse_args()

    src = Path(args.input_dir)
    if not src.is_absolute():
        src = ROOT / src
    src = src.resolve()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    events_path = src / "ALL_CAPTION_EVENTS.csv"
    sentences_path = src / "ALL_SENTENCE_UNITS.csv"

    if not events_path.exists() or not sentences_path.exists():
        raise SystemExit(
            "Missing ALL_CAPTION_EVENTS.csv / ALL_SENTENCE_UNITS.csv under: "
            f"{src}"
        )

    events = pd.read_csv(events_path, low_memory=False)
    sentences = pd.read_csv(sentences_path, low_memory=False)

    for c in ("start_sec", "end_sec", "duration_sec"):
        events[c] = pd.to_numeric(events[c], errors="coerce")

    for c in ("start_sec", "end_sec", "char_count", "word_count"):
        sentences[c] = pd.to_numeric(sentences[c], errors="coerce")

    sentences["unit_duration_sec"] = sentences["end_sec"] - sentences["start_sec"]
    sentences["events_in_unit"] = (
        pd.to_numeric(sentences["last_event_index"], errors="coerce")
        - pd.to_numeric(sentences["first_event_index"], errors="coerce")
        + 1
    )

    # Caption event gaps are useful evidence for future sentence planning.
    gap_frames = []
    for video_id, g in events.groupby("video_id", sort=False):
        g = g.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)
        gap = g["start_sec"].shift(-1) - g["end_sec"]
        frame = pd.DataFrame(
            {
                "video_id": video_id,
                "event_index": g["event_index"],
                "start_sec": g["start_sec"],
                "end_sec": g["end_sec"],
                "next_start_sec": g["start_sec"].shift(-1),
                "gap_after_sec": gap,
                "text_ws_clean": g["text_ws_clean"],
                "ends_terminal_punctuation": g["ends_terminal_punctuation"],
            }
        )
        gap_frames.append(frame)

    gaps = pd.concat(gap_frames, ignore_index=True)
    valid_gaps = pd.to_numeric(gaps["gap_after_sec"], errors="coerce").dropna()

    thresholds = [0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 1.50]
    gap_counts = {
        f"gap_ge_{str(t).replace('.', '_')}s": int((valid_gaps >= t).sum())
        for t in thresholds
    }

    summary = {
        "caption_events": int(len(events)),
        "punctuation_sentence_units": int(len(sentences)),
        "sources": int(events["video_id"].nunique()),
        **qdict(sentences["char_count"], "unit_chars"),
        **qdict(sentences["word_count"], "unit_words"),
        **qdict(sentences["unit_duration_sec"], "unit_duration_sec"),
        **qdict(sentences["events_in_unit"], "events_per_unit"),
        **qdict(valid_gaps, "event_gap_sec"),
        **gap_counts,
        "units_over_300_chars": int((sentences["char_count"] > 300).sum()),
        "units_over_500_chars": int((sentences["char_count"] > 500).sum()),
        "units_over_800_chars": int((sentences["char_count"] > 800).sum()),
        "units_over_1200_chars": int((sentences["char_count"] > 1200).sum()),
        "units_over_15_sec": int((sentences["unit_duration_sec"] > 15).sum()),
        "units_over_25_sec": int((sentences["unit_duration_sec"] > 25).sum()),
        "units_over_40_sec": int((sentences["unit_duration_sec"] > 40).sum()),
        "interpretation": (
            "These punctuation-derived units are provenance/review blocks only. "
            "They are not approved TTS sentences. The next planner must combine "
            "text semantics, punctuation, event timing/gaps, and duration limits."
        ),
    }

    # Per-source table.
    per_source_rows = []
    for video_id, s in sentences.groupby("video_id", sort=False):
        e = events[events["video_id"] == video_id]
        g = gaps[gaps["video_id"] == video_id]
        vg = pd.to_numeric(g["gap_after_sec"], errors="coerce").dropna()

        row = {
            "video_id": video_id,
            "events": int(len(e)),
            "punctuation_units": int(len(s)),
            "text_chars": int(pd.to_numeric(s["char_count"], errors="coerce").sum()),
            "unit_chars_p50": float(s["char_count"].quantile(0.50)),
            "unit_chars_p90": float(s["char_count"].quantile(0.90)),
            "unit_chars_p99": float(s["char_count"].quantile(0.99)),
            "unit_chars_max": float(s["char_count"].max()),
            "unit_duration_p50": float(s["unit_duration_sec"].quantile(0.50)),
            "unit_duration_p90": float(s["unit_duration_sec"].quantile(0.90)),
            "unit_duration_p99": float(s["unit_duration_sec"].quantile(0.99)),
            "unit_duration_max": float(s["unit_duration_sec"].max()),
            "gap_ge_0_5s": int((vg >= 0.5).sum()),
            "gap_ge_0_7s": int((vg >= 0.7).sum()),
            "gap_ge_1_0s": int((vg >= 1.0).sum()),
        }
        per_source_rows.append(row)

    per_source = pd.DataFrame(per_source_rows)

    # Largest units are exactly what we need to inspect before defining the new planner.
    largest = sentences.sort_values(
        ["char_count", "unit_duration_sec"],
        ascending=False,
    ).head(200).copy()

    # Candidate pause boundaries for threshold calibration.
    pause_candidates = gaps[
        pd.to_numeric(gaps["gap_after_sec"], errors="coerce") >= 0.30
    ].sort_values("gap_after_sec", ascending=False).copy()

    sentences.to_csv(out / "PUNCTUATION_UNITS_WITH_STATS.csv", index=False, encoding="utf-8-sig")
    per_source.to_csv(out / "PER_SOURCE_QC.csv", index=False, encoding="utf-8-sig")
    largest.to_csv(out / "TOP_200_LONG_UNITS.csv", index=False, encoding="utf-8-sig")
    pause_candidates.to_csv(out / "EVENT_GAP_CANDIDATES.csv", index=False, encoding="utf-8-sig")
    (out / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("CAPTION STRUCTURE QC V1")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("Per-source:")
    print(per_source.to_string(index=False))
    print()
    print("Output:", out)
    print()
    print("IMPORTANT: punctuation-derived units are NOT TTS sentence cuts.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
