from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "kaggle_output"
    / "source16-segmentation"
    / "source16_segmentation"
    / "outputs"
    / "CUT_MANIFEST.csv"
)
ALIGNMENT = (
    ROOT
    / "outputs"
    / "source16_keep_auto_alignment_scan"
    / "KEEP_AUTO_ALIGNMENT_SCAN.csv"
)
OUT = ROOT / "outputs" / "source16_full_keep_audit"

EXPECTED_KEEP_AUTO = 17495
EXPECTED_KEEP_WHISPER_PRIMARY = 4655
EXPECTED_KEEP_TOTAL = 22150


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def main() -> int:
    require(MANIFEST)
    require(ALIGNMENT)

    manifest = pd.read_csv(MANIFEST, low_memory=False)
    align = pd.read_csv(ALIGNMENT, low_memory=False)

    original_keep = manifest[
        manifest["decision"].isin(["KEEP_AUTO", "KEEP_WHISPER_PRIMARY"])
    ].copy()

    keep_auto = original_keep[original_keep["decision"] == "KEEP_AUTO"].copy()
    keep_wp = original_keep[
        original_keep["decision"] == "KEEP_WHISPER_PRIMARY"
    ].copy()

    if len(keep_auto) != EXPECT_KEEP_AUTO:
        raise RuntimeError(
            f"KEEP_AUTO count mismatch: got={len(keep_auto)} expected={EXPECTED_KEEP_AUTO}"
        )
    if len(keep_wp) != EXPECT_KEEP_WHISPER_PRIMARY:
        raise RuntimeError(
            "KEEP_WHISPER_PRIMARY count mismatch: "
            f"got={len(keep_wp)} expected={EXPECTED_KEEP_WHISPER_PRIMARY}"
        )
    if len(original_keep) != EXPECT_KEEP_TOTAL:
        raise RuntimeError(
            f"Original KEEP total mismatch: got={len(original_keep)} expected={EXPECTED_KEEP_TOTAL}"
        )

    needed = {
        "clip_id",
        "alignment_decision",
        "alignment_reason",
        "proposal_text",
        "proposal_similarity",
        "proposal_token_ratio",
    }
    missing = needed - set(align.columns)
    if missing:
        raise RuntimeError(f"Alignment scan missing columns: {sorted(missing)}")

    if len(align) != EXPECT_KEEP_AUTO:
        raise RuntimeError(
            f"Alignment scan row count mismatch: got={len(align)} expected={EXPECTED_KEEP_AUTO}"
        )
    if align["clip_id"].duplicated().any():
        raise RuntimeError("Alignment scan contains duplicate clip_id rows.")

    keep_auto = keep_auto.merge(
        align[
            [
                "clip_id",
                "alignment_decision",
                "alignment_reason",
                "proposal_text",
                "proposal_similarity",
                "proposal_token_ratio",
            ]
        ],
        on="clip_id",
        how="left",
        validate="one_to_one",
    )

    if keep_auto["alignment_decision"].isna().any():
        missing_ids = keep_auto.loc[
            keep_auto["alignment_decision"].isna(), "clip_id"
        ].head(20).tolist()
        raise RuntimeError(
            "KEEP_AUTO alignment missing for some rows; examples="
            + ",".join(missing_ids)
        )

    keep_auto["audit_route"] = "TRUSTED_CAPTION_ALIGNMENT"
    keep_auto["audit_decision"] = keep_auto["alignment_decision"].map(
        {
            "ACCEPT_CAPTION": "ACCEPT_CAPTION_CANDIDATE",
            "REVIEW_ALIGNMENT": "REVIEW_ALIGNMENT",
        }
    )
    if keep_auto["audit_decision"].isna().any():
        bad = sorted(keep_auto["alignment_decision"].dropna().astype(str).unique())
        raise RuntimeError(f"Unexpected alignment decisions: {bad}")

    keep_wp["audit_route"] = "WHISPER_PRIMARY_LOW_CAPTION_TRUST"
    keep_wp["audit_decision"] = "REVIEW_WHISPER_PRIMARY"
    keep_wp["alignment_decision"] = ""
    keep_wp["alignment_reason"] = "caption_reference_low_trust"
    keep_wp["proposal_text"] = ""
    keep_wp["proposal_similarity"] = None
    keep_wp["proposal_token_ratio"] = None

    common_cols = sorted(set(keep_auto.columns) | set(keep_wp.columns))
    for df in (keep_auto, keep_wp):
        for col in common_cols:
            if col not in df.columns:
                df[col] = None

    full = pd.concat(
        [keep_auto[common_cols], keep_wp[common_cols]],
        ignore_index=True,
    )

    if len(full) != EXPECT_KEEP_TOTAL:
        raise RuntimeError(
            f"Full audit row count mismatch: got={len(full)} expected={EXPECTED_KEEP_TOTAL}"
        )
    if full["clip_id"].duplicated().any():
        raise RuntimeError("Full audit contains duplicate clip_id rows.")

    OUT.mkdir(parents=True, exist_ok=True)
    full.to_csv(
        OUT / "MASTER_KEEP_AUDIT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    accepted = full[
        full["audit_decision"] == "ACCEPT_CAPTION_CANDIDATE"
    ].copy()
    unresolved = full[
        full["audit_decision"] != "ACCEPT_CAPTION_CANDIDATE"
    ].copy()

    accepted.to_csv(
        OUT / "ACCEPT_CAPTION_CANDIDATES.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unresolved.to_csv(
        OUT / "UNRESOLVED_KEEP.csv",
        index=False,
        encoding="utf-8-sig",
    )

    source_summary = (
        full.groupby(["video_id", "decision", "audit_route", "audit_decision"])
        .size()
        .reset_index(name="rows")
        .sort_values(["video_id", "audit_decision"])
    )
    source_summary.to_csv(
        OUT / "SOURCE_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "stage": "SOURCE16_FULL_KEEP_AUDIT_V1",
        "original_keep_rows": int(len(full)),
        "original_decision_counts": {
            str(k): int(v)
            for k, v in full["decision"].value_counts().to_dict().items()
        },
        "audit_decision_counts": {
            str(k): int(v)
            for k, v in full["audit_decision"].value_counts().to_dict().items()
        },
        "routes": {
            "TRUSTED_CAPTION_ALIGNMENT": int(
                (full["audit_route"] == "TRUSTED_CAPTION_ALIGNMENT").sum()
            ),
            "WHISPER_PRIMARY_LOW_CAPTION_TRUST": int(
                (full["audit_route"] == "WHISPER_PRIMARY_LOW_CAPTION_TRUST").sum()
            ),
        },
        "currently_auto_accepted_candidates": int(len(accepted)),
        "currently_unresolved": int(len(unresolved)),
        "bulk_training_approved": False,
        "note": (
            "This is the 22,150-row master audit over the original KEEP pool. "
            "Only caption-aligned rows are current auto-accept candidates. "
            "KEEP_WHISPER_PRIMARY remains unresolved and requires a separate "
            "low-caption-trust transcript strategy before training approval."
        ),
    }

    (OUT / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("SOURCE16 FULL ORIGINAL KEEP AUDIT")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("OUTPUT", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
