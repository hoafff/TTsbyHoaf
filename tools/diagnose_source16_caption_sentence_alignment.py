from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "outputs" / "source16_keep_auto_pilot_review" / "REVIEW_RESULTS.csv"
PILOT = ROOT / "kaggle_output" / "source16-keep-auto-cut-pilot" / "source16_keep_auto_cut_pilot" / "outputs" / "LISTENING_INDEX.csv"
WHISPER_ROOT = ROOT / "kaggle_output" / "source16-whisper-consensus"
OUT = ROOT / "outputs" / "source16_keep_auto_caption_sentence_alignment"

WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+")
SENT_RE = re.compile(r'.*?[.!?…]+(?:["”’\'»)\]]+)?(?=\s|$)', re.S)

MIN_ACCEPT_SIM = 0.90
MAX_TOKEN_RATIO_DEV = 0.35
NEAR_PAD_SEC = 6.0


def toks(s: str) -> list[str]:
    return WORD_RE.findall(str(s).lower())


def sim(a: str, b: str) -> float:
    aa, bb = toks(a), toks(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb, autojunk=False).ratio()


def find_events(video_id: str) -> Path:
    matches = []
    for p in WHISPER_ROOT.rglob("YOUTUBE_CAPTION_EVENTS.csv"):
        if video_id in str(p.parent):
            matches.append(p)
    if not matches:
        raise FileNotFoundError(f"Caption events not found for {video_id}")
    return sorted(matches)[0]


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


def best_sentence_span(whisper_text: str, caption_window: str) -> dict:
    sentences = complete_sentences(caption_window)
    wt = toks(whisper_text)
    if not wt or not sentences:
        return {
            "proposal": "",
            "similarity": 0.0,
            "token_ratio": None,
            "sentence_count": 0,
            "accepted": False,
            "reason": "no_complete_caption_sentence",
        }

    best = None
    for i in range(len(sentences)):
        for j in range(i, min(len(sentences), i + 6)):
            proposal = clean_join(sentences[i:j + 1])
            pt = toks(proposal)
            if not pt:
                continue
            token_ratio = len(pt) / len(wt)
            if token_ratio < 1.0 - MAX_TOKEN_RATIO_DEV or token_ratio > 1.0 + MAX_TOKEN_RATIO_DEV:
                continue

            ratio = sim(whisper_text, proposal)
            length_penalty = 0.04 * abs(len(pt) - len(wt)) / max(1, len(wt))
            score = ratio - length_penalty

            candidate = {
                "proposal": proposal,
                "similarity": ratio,
                "token_ratio": token_ratio,
                "sentence_count": j - i + 1,
                "score": score,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    if best is None:
        return {
            "proposal": "",
            "similarity": 0.0,
            "token_ratio": None,
            "sentence_count": 0,
            "accepted": False,
            "reason": "no_length_compatible_complete_sentence_span",
        }

    best["accepted"] = bool(best["similarity"] >= MIN_ACCEPT_SIM)
    best["reason"] = "caption_sentence_alignment_pass" if best["accepted"] else "caption_sentence_similarity_low"
    return best


def main() -> int:
    review = pd.read_csv(REVIEW, low_memory=False).fillna("")
    review = review[review["verdict"].astype(str).str.len() > 0].copy()
    pilot = pd.read_csv(PILOT, low_memory=False)

    merged = review.merge(
        pilot[["clip_id", "requested_start_sec", "requested_end_sec", "text"]],
        on="clip_id",
        how="left",
        suffixes=("_review", "_pilot"),
    )

    cache: dict[str, pd.DataFrame] = {}
    rows = []

    for r in merged.itertuples(index=False):
        video_id = str(r.video_id)
        if video_id not in cache:
            cache[video_id] = pd.read_csv(find_events(video_id), low_memory=False)

        events = cache[video_id]
        start = float(r.requested_start_sec)
        end = float(r.requested_end_sec)
        es = pd.to_numeric(events["start_sec"], errors="coerce")
        ee = pd.to_numeric(events["end_sec"], errors="coerce")
        near = events[
            (ee >= start - NEAR_PAD_SEC) & (es <= end + NEAR_PAD_SEC)
        ].sort_values(["start_sec", "end_sec"], kind="stable")

        window = clean_join(near["text"].astype(str).tolist())
        best = best_sentence_span(str(r.text_pilot), window)

        rows.append({
            "review_order": int(r.review_order),
            "verdict": str(r.verdict),
            "video_id": video_id,
            "clip_id": str(r.clip_id),
            "whisper_text": str(r.text_pilot),
            "proposal_text": best["proposal"],
            "proposal_similarity": best["similarity"],
            "proposal_token_ratio": best["token_ratio"],
            "proposal_sentence_count": best["sentence_count"],
            "alignment_decision": "ACCEPT_CAPTION" if best["accepted"] else "REVIEW_ALIGNMENT",
            "alignment_reason": best["reason"],
        })

    out = pd.DataFrame(rows).sort_values("review_order")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "SENTENCE_ALIGNMENT.csv", index=False, encoding="utf-8-sig")

    summary = {
        "stage": "SOURCE16_CAPTION_SENTENCE_ALIGNMENT_DIAGNOSTIC_V1",
        "rows": int(len(out)),
        "params": {
            "min_accept_similarity": MIN_ACCEPT_SIM,
            "max_token_ratio_deviation": MAX_TOKEN_RATIO_DEV,
            "near_pad_sec": NEAR_PAD_SEC,
            "require_complete_caption_sentence": True,
        },
        "decision_counts": {
            str(k): int(v) for k, v in out["alignment_decision"].value_counts().to_dict().items()
        },
        "by_human_verdict": {},
    }

    for verdict, g in out.groupby("verdict"):
        summary["by_human_verdict"][str(verdict)] = {
            "count": int(len(g)),
            "accepted_caption": int((g["alignment_decision"] == "ACCEPT_CAPTION").sum()),
            "review_alignment": int((g["alignment_decision"] == "REVIEW_ALIGNMENT").sum()),
        }

    (OUT / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n===== BAD_TEXT SENTENCE PROPOSALS =====")
    bad = out[out["verdict"].isin(["BAD_TEXT", "BAD_BOTH"])]
    print(
        bad[
            [
                "review_order",
                "video_id",
                "clip_id",
                "alignment_decision",
                "proposal_similarity",
                "proposal_token_ratio",
                "whisper_text",
                "proposal_text",
            ]
        ].to_string(index=False)
    )
    print("\nOUTPUT", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
