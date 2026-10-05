from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "kaggle_output" / "source16-segmentation" / "source16_segmentation" / "outputs" / "CUT_MANIFEST.csv"
CAPTION_ROOT = ROOT / "kaggle_output" / "source16-whisper-consensus"
OUT = ROOT / "outputs" / "source16_keep_auto_alignment_scan"

WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+")
SENT_RE = re.compile(r'.*?[.!?…]+(?:["”’\'»)\]]+)?(?=\s|$)', re.S)

MIN_ACCEPT_SIM = 0.90
MAX_TOKEN_RATIO_DEV = 0.35
NEAR_PAD_SEC = 6.0
TARGET_DECISION = "KEEP_AUTO"


def toks(s: str) -> list[str]:
    return WORD_RE.findall(str(s).lower())


def sim(a: str, b: str) -> float:
    aa, bb = toks(a), toks(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb, autojunk=False).ratio()


def clean_join(parts: list[str]) -> str:
    text = " ".join(str(x).strip() for x in parts if str(x).strip())
    return re.sub(r"\s+", " ", text).strip()


def complete_sentences(text: str) -> list[str]:
    found = []
    for m in SENT_RE.finditer(text):
        s = re.sub(r"\s+", " ", m.group(0)).strip()
        if s:
            found.append(s)
    return found


def find_events(video_id: str) -> Path:
    matches = [
        p for p in CAPTION_ROOT.rglob("YOUTUBE_CAPTION_EVENTS.csv")
        if video_id in str(p.parent)
    ]
    if not matches:
        raise FileNotFoundError(
            f"Caption events not found for {video_id} under {CAPTION_ROOT}"
        )
    return sorted(matches)[0]


def best_sentence_span(whisper_text: str, caption_window: str) -> dict:
    sentences = complete_sentences(caption_window)
    wt = toks(whisper_text)
    if not wt or not sentences:
        return {
            "proposal_text": "",
            "proposal_similarity": 0.0,
            "proposal_token_ratio": None,
            "proposal_sentence_count": 0,
            "alignment_decision": "REVIEW_ALIGNMENT",
            "alignment_reason": "no_complete_caption_sentence",
        }

    best = None
    for i in range(len(sentences)):
        for j in range(i, min(len(sentences), i + 6)):
            proposal = clean_join(sentences[i:j + 1])
            pt = toks(proposal)
            if not pt:
                continue
            token_ratio = len(pt) / len(wt)
            if not (1.0 - MAX_TOKEN_RATIO_DEV <= token_ratio <= 1.0 + MAX_TOKEN_RATIO_DEV):
                continue

            ratio = sim(whisper_text, proposal)
            length_penalty = 0.04 * abs(len(pt) - len(wt)) / max(1, len(wt))
            score = ratio - length_penalty

            candidate = {
                "proposal_text": proposal,
                "proposal_similarity": ratio,
                "proposal_token_ratio": token_ratio,
                "proposal_sentence_count": j - i + 1,
                "_score": score,
            }
            if best is None or candidate["_score"] > best["_score"]:
                best = candidate

    if best is None:
        return {
            "proposal_text": "",
            "proposal_similarity": 0.0,
            "proposal_token_ratio": None,
            "proposal_sentence_count": 0,
            "alignment_decision": "REVIEW_ALIGNMENT",
            "alignment_reason": "no_length_compatible_complete_sentence_span",
        }

    accepted = best["proposal_similarity"] >= MIN_ACCEPT_SIM
    best.pop("_score", None)
    best["alignment_decision"] = "ACCEPT_CAPTION" if accepted else "REVIEW_ALIGNMENT"
    best["alignment_reason"] = (
        "caption_sentence_alignment_pass"
        if accepted
        else "caption_sentence_similarity_low"
    )
    return best


def main() -> int:
    if not MANIFEST.exists():
        raise SystemExit(f"Missing segmentation manifest: {MANIFEST}")

    manifest = pd.read_csv(MANIFEST, low_memory=False)
    keep = manifest[manifest["decision"].astype(str) == TARGET_DECISION].copy()
    if keep.empty:
        raise SystemExit("No KEEP_AUTO rows found.")

    print("=" * 100)
    print("SOURCE16 KEEP_AUTO FULL ALIGNMENT SCAN")
    print(f"input_keep_auto={len(keep)}")
    print("NO AUDIO IS CUT. NO TRAINING TEXT IS REPLACED IN PLACE.")
    print("=" * 100)

    rows = []
    source_summaries = []

    for video_id, group in keep.groupby("video_id", sort=False):
        events = pd.read_csv(find_events(str(video_id)), low_memory=False)
        es = pd.to_numeric(events["start_sec"], errors="coerce")
        ee = pd.to_numeric(events["end_sec"], errors="coerce")

        accepted = 0
        reviewed = 0
        for r in group.itertuples(index=False):
            start = float(r.cut_start_sec)
            end = float(r.cut_end_sec)
            near = events[
                (ee >= start - NEAR_PAD_SEC) & (es <= end + NEAR_PAD_SEC)
            ].sort_values(["start_sec", "end_sec"], kind="stable")

            window = clean_join(near["text"].astype(str).tolist())
            result = best_sentence_span(str(r.text), window)

            if result["alignment_decision"] == "ACCEPT_CAPTION":
                accepted += 1
            else:
                reviewed += 1

            rows.append({
                "video_id": str(video_id),
                "clip_id": str(r.clip_id),
                "original_decision": str(r.decision),
                "raw_start_sec": float(r.raw_start_sec),
                "raw_end_sec": float(r.raw_end_sec),
                "cut_start_sec": start,
                "cut_end_sec": end,
                "cut_duration_sec": float(r.cut_duration_sec),
                "whisper_text": str(r.text),
                **result,
            })

        source_summaries.append({
            "video_id": str(video_id),
            "keep_auto_rows": int(len(group)),
            "accepted_caption": int(accepted),
            "review_alignment": int(reviewed),
            "accepted_rate": float(accepted / len(group)) if len(group) else 0.0,
        })
        print(
            f"SOURCE_DONE {video_id} keep={len(group)} "
            f"accept={accepted} review={reviewed} "
            f"accept_rate={accepted/len(group):.3f}",
            flush=True,
        )

    out = pd.DataFrame(rows)
    src = pd.DataFrame(source_summaries)

    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "KEEP_AUTO_ALIGNMENT_SCAN.csv", index=False, encoding="utf-8-sig")
    out[out["alignment_decision"] == "ACCEPT_CAPTION"].to_csv(
        OUT / "KEEP_AUTO_ACCEPT_CAPTION.csv", index=False, encoding="utf-8-sig"
    )
    out[out["alignment_decision"] == "REVIEW_ALIGNMENT"].to_csv(
        OUT / "KEEP_AUTO_REVIEW_ALIGNMENT.csv", index=False, encoding="utf-8-sig"
    )
    src.to_csv(OUT / "SOURCE_SUMMARY.csv", index=False, encoding="utf-8-sig")

    sims = pd.to_numeric(out["proposal_similarity"], errors="coerce").dropna()
    summary = {
        "stage": "SOURCE16_KEEP_AUTO_FULL_ALIGNMENT_SCAN_V1",
        "input_keep_auto_rows": int(len(out)),
        "params": {
            "min_accept_similarity": MIN_ACCEPT_SIM,
            "max_token_ratio_deviation": MAX_TOKEN_RATIO_DEV,
            "near_pad_sec": NEAR_PAD_SEC,
            "require_complete_caption_sentence": True,
        },
        "decision_counts": {
            str(k): int(v)
            for k, v in out["alignment_decision"].value_counts().to_dict().items()
        },
        "accepted_rate": float(
            (out["alignment_decision"] == "ACCEPT_CAPTION").mean()
        ),
        "proposal_similarity": {
            "p10": float(sims.quantile(0.10)) if len(sims) else None,
            "p50": float(sims.quantile(0.50)) if len(sims) else None,
            "p90": float(sims.quantile(0.90)) if len(sims) else None,
        },
        "source_count": int(out["video_id"].nunique()),
        "note": (
            "Dry-run only. ACCEPT_CAPTION rows are candidates for a new caption-canonical "
            "KEEP class; REVIEW_ALIGNMENT rows remain excluded until further work."
        ),
    }
    (OUT / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("OUTPUT", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
