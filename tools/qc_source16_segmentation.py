from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = (
    ROOT
    / "kaggle_output"
    / "source16-segmentation"
    / "source16_segmentation"
    / "outputs"
    / "CUT_MANIFEST.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "source16_segmentation_qc"

KEEP_DECISIONS = {"KEEP_AUTO", "KEEP_WHISPER_PRIMARY"}
EXPECTED_DECISIONS = KEEP_DECISIONS | {"REVIEW", "DROP"}

REQUIRED_COLUMNS = {
    "manifest_schema",
    "video_id",
    "clip_id",
    "raw_start_sec",
    "raw_end_sec",
    "raw_duration_sec",
    "cut_start_sec",
    "cut_end_sec",
    "cut_duration_sec",
    "word_count",
    "text",
    "boundary_basis",
    "boundary_score",
    "gap_before_clip_sec",
    "gap_after_clip_sec",
    "mean_word_probability",
    "word_probability_p10",
    "median_segment_avg_logprob",
    "max_segment_no_speech_prob",
    "source_consensus_mean",
    "clip_consensus_mean",
    "caption_reference_trusted",
    "decision",
    "decision_reason",
    "quality_score",
    "audio_file",
}

EXPECTED_SOURCES = [
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


def as_float(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def quantiles(series: pd.Series) -> dict[str, float | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"min": None, "p10": None, "p50": None, "p90": None, "p99": None, "max": None}
    return {
        "min": float(s.min()),
        "p10": float(s.quantile(0.10)),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


def choose_evenly(group: pd.DataFrame, count: int, reason: str) -> list[dict]:
    if group.empty or count <= 0:
        return []
    g = group.sort_values(["quality_score", "raw_start_sec"], kind="stable").reset_index()
    if len(g) <= count:
        idxs = list(range(len(g)))
    elif count == 1:
        idxs = [len(g) // 2]
    else:
        idxs = sorted(
            set(round(i * (len(g) - 1) / (count - 1)) for i in range(count))
        )

    rows = []
    for i in idxs:
        row = g.iloc[i].to_dict()
        row["_pilot_reason"] = reason
        rows.append(row)
    return rows


def build_pilot(df: pd.DataFrame, per_source: int) -> pd.DataFrame:
    chosen: list[dict] = []

    for video_id in EXPECTED_SOURCES:
        src = df[df["video_id"] == video_id].copy()
        if src.empty:
            continue

        keep = src[src["decision"].isin(KEEP_DECISIONS)].copy()
        review = src[src["decision"] == "REVIEW"].copy()

        candidates: list[dict] = []
        candidates.extend(choose_evenly(keep, 5, "keep_quality_spread"))
        candidates.extend(choose_evenly(review, 2, "review_quality_spread"))

        if not keep.empty:
            shortest = keep.sort_values(
                ["cut_duration_sec", "quality_score"], ascending=[True, True], kind="stable"
            ).iloc[0].to_dict()
            shortest["_pilot_reason"] = "keep_shortest"
            candidates.append(shortest)

            longest = keep.sort_values(
                ["cut_duration_sec", "quality_score"], ascending=[False, True], kind="stable"
            ).iloc[0].to_dict()
            longest["_pilot_reason"] = "keep_longest"
            candidates.append(longest)

            lowest_boundary = keep.sort_values(
                ["boundary_score", "quality_score", "raw_start_sec"], kind="stable"
            ).iloc[0].to_dict()
            lowest_boundary["_pilot_reason"] = "keep_weakest_boundary"
            candidates.append(lowest_boundary)

        seen: set[str] = set()
        deduped: list[dict] = []
        for row in candidates:
            clip_id = str(row["clip_id"])
            if clip_id in seen:
                continue
            seen.add(clip_id)
            deduped.append(row)
            if len(deduped) >= per_source:
                break

        if len(deduped) < per_source:
            fill = src[~src["clip_id"].astype(str).isin(seen)].sort_values(
                ["quality_score", "raw_start_sec"], kind="stable"
            )
            for _, r in fill.iterrows():
                row = r.to_dict()
                row["_pilot_reason"] = "fill_source_quota"
                deduped.append(row)
                seen.add(str(row["clip_id"]))
                if len(deduped) >= per_source:
                    break

        chosen.extend(deduped)

    if not chosen:
        return pd.DataFrame()

    pilot = pd.DataFrame(chosen)
    if "index" in pilot.columns:
        pilot = pilot.drop(columns=["index"])
    pilot = pilot.rename(columns={"_pilot_reason": "pilot_reason"})
    pilot.insert(0, "pilot_order", range(1, len(pilot) + 1))
    return pilot


def source_interval_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for video_id, group in df.groupby("video_id", sort=False):
        g = group.sort_values(["raw_start_sec", "raw_end_sec"], kind="stable").reset_index(drop=True)
        prev_raw_end = g["raw_end_sec"].shift(1)
        prev_cut_end = g["cut_end_sec"].shift(1)
        raw_delta = g["raw_start_sec"] - prev_raw_end
        cut_delta = g["cut_start_sec"] - prev_cut_end

        for i in range(1, len(g)):
            raw_overlap = max(0.0, -float(raw_delta.iloc[i]))
            cut_overlap = max(0.0, -float(cut_delta.iloc[i]))
            if raw_overlap > 0.05 or cut_overlap > 0.05:
                rows.append(
                    {
                        "video_id": video_id,
                        "prev_clip_id": str(g.iloc[i - 1]["clip_id"]),
                        "clip_id": str(g.iloc[i]["clip_id"]),
                        "raw_overlap_sec": raw_overlap,
                        "cut_overlap_sec": cut_overlap,
                        "raw_gap_sec": float(raw_delta.iloc[i]),
                        "cut_gap_sec": float(cut_delta.iloc[i]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(
        description="QC Source16 segmentation manifest and select a balanced listening pilot."
    )
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--pilot-per-source", type=int, default=10)
    args = p.parse_args()

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = (ROOT / manifest).resolve()
    out = Path(args.out)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        raise SystemExit(f"Missing manifest: {manifest}")

    df = pd.read_csv(manifest, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise SystemExit(f"Manifest missing required columns: {missing}")

    numeric = [
        "raw_start_sec",
        "raw_end_sec",
        "raw_duration_sec",
        "cut_start_sec",
        "cut_end_sec",
        "cut_duration_sec",
        "word_count",
        "boundary_score",
        "gap_before_clip_sec",
        "gap_after_clip_sec",
        "mean_word_probability",
        "word_probability_p10",
        "median_segment_avg_logprob",
        "max_segment_no_speech_prob",
        "source_consensus_mean",
        "clip_consensus_mean",
        "quality_score",
    ]
    as_float(df, numeric)

    df["video_id"] = df["video_id"].astype(str)
    df["clip_id"] = df["clip_id"].astype(str)
    df["decision"] = df["decision"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    structural = []
    duplicate_ids = df[df["clip_id"].duplicated(keep=False)].copy()
    if not duplicate_ids.empty:
        structural.append(f"duplicate_clip_ids={duplicate_ids['clip_id'].nunique()}")

    unexpected_decisions = sorted(set(df["decision"]) - EXPECTED_DECISIONS)
    if unexpected_decisions:
        structural.append(f"unexpected_decisions={unexpected_decisions}")

    source_set = list(dict.fromkeys(df["video_id"].tolist()))
    missing_sources = [x for x in EXPECTED_SOURCES if x not in set(source_set)]
    extra_sources = [x for x in source_set if x not in set(EXPECTED_SOURCES)]
    if missing_sources:
        structural.append(f"missing_sources={missing_sources}")
    if extra_sources:
        structural.append(f"extra_sources={extra_sources}")

    invalid_interval = df[
        df["raw_start_sec"].isna()
        | df["raw_end_sec"].isna()
        | df["cut_start_sec"].isna()
        | df["cut_end_sec"].isna()
        | (df["raw_end_sec"] <= df["raw_start_sec"])
        | (df["cut_end_sec"] <= df["cut_start_sec"])
        | (df["cut_start_sec"] < -1e-6)
    ].copy()
    if not invalid_interval.empty:
        structural.append(f"invalid_intervals={len(invalid_interval)}")

    empty_text = df[df["text"].str.strip().eq("")].copy()
    if not empty_text.empty:
        structural.append(f"empty_text={len(empty_text)}")

    duration_mismatch = df[
        (df["raw_duration_sec"] - (df["raw_end_sec"] - df["raw_start_sec"])).abs() > 0.02
    ].copy()
    if not duration_mismatch.empty:
        structural.append(f"raw_duration_mismatch={len(duration_mismatch)}")

    cut_duration_mismatch = df[
        (df["cut_duration_sec"] - (df["cut_end_sec"] - df["cut_start_sec"])).abs() > 0.02
    ].copy()
    if not cut_duration_mismatch.empty:
        structural.append(f"cut_duration_mismatch={len(cut_duration_mismatch)}")

    overlaps = source_interval_audit(df)
    severe_overlap = (
        overlaps[(overlaps["raw_overlap_sec"] > 0.25) | (overlaps["cut_overlap_sec"] > 0.25)]
        if not overlaps.empty
        else overlaps
    )
    if not severe_overlap.empty:
        structural.append(f"severe_overlap_pairs={len(severe_overlap)}")

    keep = df[df["decision"].isin(KEEP_DECISIONS)].copy()
    review = df[df["decision"] == "REVIEW"].copy()
    drop = df[df["decision"] == "DROP"].copy()

    suspicious_keep = keep[
        (keep["raw_duration_sec"] < 3.0)
        | (keep["raw_duration_sec"] > 14.0)
        | (keep["word_count"] < 3)
        | (keep["mean_word_probability"] < 0.78)
        | (keep["word_probability_p10"] < 0.45)
        | (keep["max_segment_no_speech_prob"] > 0.35)
        | (keep["median_segment_avg_logprob"] < -0.90)
    ].copy()

    pilot = build_pilot(df, max(1, args.pilot_per_source))

    source_rows = []
    for video_id in EXPECTED_SOURCES:
        src = df[df["video_id"] == video_id]
        if src.empty:
            continue
        skeep = src[src["decision"].isin(KEEP_DECISIONS)]
        source_rows.append(
            {
                "video_id": video_id,
                "candidates": int(len(src)),
                "KEEP_AUTO": int((src["decision"] == "KEEP_AUTO").sum()),
                "KEEP_WHISPER_PRIMARY": int((src["decision"] == "KEEP_WHISPER_PRIMARY").sum()),
                "REVIEW": int((src["decision"] == "REVIEW").sum()),
                "DROP": int((src["decision"] == "DROP").sum()),
                "keep_hours": float(skeep["cut_duration_sec"].sum()) / 3600.0,
                "keep_quality_p10": quantiles(skeep["quality_score"])["p10"],
                "keep_quality_p50": quantiles(skeep["quality_score"])["p50"],
                "source_consensus_mean": (
                    float(pd.to_numeric(src["source_consensus_mean"], errors="coerce").dropna().iloc[0])
                    if pd.to_numeric(src["source_consensus_mean"], errors="coerce").notna().any()
                    else None
                ),
            }
        )

    summary = {
        "stage": "SOURCE16_SEGMENTATION_QC_V1",
        "manifest": str(manifest),
        "rows": int(len(df)),
        "sources": int(df["video_id"].nunique()),
        "decision_counts": {k: int(v) for k, v in df["decision"].value_counts().to_dict().items()},
        "hours": {
            "KEEP": float(keep["cut_duration_sec"].sum()) / 3600.0,
            "REVIEW": float(review["cut_duration_sec"].sum()) / 3600.0,
            "DROP": float(drop["cut_duration_sec"].sum()) / 3600.0,
        },
        "keep_duration_sec": quantiles(keep["cut_duration_sec"]),
        "keep_quality_score": quantiles(keep["quality_score"]),
        "keep_word_probability": quantiles(keep["mean_word_probability"]),
        "suspicious_keep_rows": int(len(suspicious_keep)),
        "overlap_pairs_gt_50ms": int(len(overlaps)),
        "severe_overlap_pairs_gt_250ms": int(len(severe_overlap)),
        "pilot_rows": int(len(pilot)),
        "structural_errors": structural,
        "qc_pass": not structural,
    }

    (out / "QC_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(source_rows).to_csv(
        out / "SOURCE_QC.csv", index=False, encoding="utf-8-sig"
    )
    suspicious_keep.sort_values(
        ["quality_score", "video_id", "raw_start_sec"], kind="stable"
    ).to_csv(out / "SUSPICIOUS_KEEP.csv", index=False, encoding="utf-8-sig")
    overlaps.to_csv(out / "OVERLAP_ISSUES.csv", index=False, encoding="utf-8-sig")
    invalid_interval.to_csv(out / "INVALID_INTERVALS.csv", index=False, encoding="utf-8-sig")
    duplicate_ids.to_csv(out / "DUPLICATE_CLIP_IDS.csv", index=False, encoding="utf-8-sig")
    pilot.to_csv(out / "PILOT_MANIFEST.csv", index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("SOURCE16 SEGMENTATION QC V1")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("PILOT:", out / "PILOT_MANIFEST.csv")
    print("SUSPICIOUS:", out / "SUSPICIOUS_KEEP.csv")
    print("OVERLAPS:", out / "OVERLAP_ISSUES.csv")
    print("QC_PASS" if summary["qc_pass"] else "QC_FAILED")
    return 0 if summary["qc_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
