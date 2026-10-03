from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def repeated_token_hint(text: str) -> bool:
    toks = [re.sub(r"[^0-9A-Za-zÀ-ỹĐđ]+", "", t).lower() for t in norm(text).split()]
    toks = [t for t in toks if t]
    return any(a == b for a, b in zip(toks, toks[1:]))


def initial_crowding_hint(text: str) -> bool:
    toks = [re.sub(r"[^A-Za-zÀ-ỹĐđ]+", "", t).lower() for t in norm(text).split()]
    toks = [t for t in toks if len(t) >= 2]
    if len(toks) < 5:
        return False

    initials = [t[0] for t in toks]
    for i in range(len(initials) - 4):
        window = initials[i:i+5]
        if max(window.count(ch) for ch in set(window)) >= 4:
            return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build a deterministic 16-source semantic-QC pilot before full LLM execution."
    )
    p.add_argument(
        "pieces_csv",
        nargs="?",
        default="outputs/semantic_tasks_v2/PRESEGMENTED_PIECES.csv",
    )
    p.add_argument("--out", default="outputs/semantic_pilot_v1")
    p.add_argument("--seed", type=int, default=260103)
    p.add_argument("--long-per-source", type=int, default=4)
    p.add_argument("--tiny-per-source", type=int, default=4)
    p.add_argument("--risk-per-source", type=int, default=2)
    p.add_argument("--control-per-source", type=int, default=4)
    args = p.parse_args()

    src = Path(args.pieces_csv)
    if not src.is_absolute():
        src = ROOT / src
    src = src.resolve()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src, low_memory=False)
    df["text_exact"] = df["text_exact"].astype(str)
    df["repeat_hint"] = df["text_exact"].map(repeated_token_hint)
    df["initial_crowding_hint"] = df["text_exact"].map(initial_crowding_hint)

    rng = random.Random(args.seed)
    rows = []

    def choose(group: pd.DataFrame, mask: pd.Series, n: int, kind: str) -> None:
        candidates = group[mask].copy()
        if candidates.empty or n <= 0:
            return
        indices = list(candidates.index)
        rng.shuffle(indices)
        for idx in indices[:n]:
            r = df.loc[idx].to_dict()
            r["pilot_bucket"] = kind
            rows.append(r)

    for video_id, g in df.groupby("video_id", sort=False):
        long_mask = g["needs_ai_split_hint"].fillna(False).astype(bool)
        tiny_mask = g["tiny_fragment_hint"].fillna(False).astype(bool)
        risk_mask = (
            g["repeat_hint"].fillna(False).astype(bool)
            | g["initial_crowding_hint"].fillna(False).astype(bool)
        )

        control_mask = (
            ~long_mask
            & ~tiny_mask
            & ~risk_mask
            & g["word_count"].between(6, 28)
            & g["char_count"].between(25, 160)
        )

        choose(g, long_mask, args.long_per_source, "LONG_SPLIT")
        choose(g, tiny_mask, args.tiny_per_source, "TINY_FRAGMENT")
        choose(g, risk_mask, args.risk_per_source, "REPEAT_OR_CROWDING")
        choose(g, control_mask, args.control_per_source, "NORMAL_CONTROL")

    pilot = pd.DataFrame(rows).drop_duplicates(subset=["piece_id"]).reset_index(drop=True)

    # Add previous/next context from the same source video.
    pos = {pid: i for i, pid in enumerate(df["piece_id"].astype(str).tolist())}
    records = []
    for r in pilot.itertuples(index=False):
        i = pos[str(r.piece_id)]
        video = str(r.video_id)

        prev_text = ""
        next_text = ""

        if i > 0 and str(df.iloc[i - 1]["video_id"]) == video:
            prev_text = str(df.iloc[i - 1]["text_exact"])
        if i + 1 < len(df) and str(df.iloc[i + 1]["video_id"]) == video:
            next_text = str(df.iloc[i + 1]["text_exact"])

        d = r._asdict()
        d["prev_context"] = prev_text
        d["next_context"] = next_text
        records.append(d)

    pilot = pd.DataFrame(records)
    pilot["pilot_id"] = [f"P{i:04d}" for i in range(1, len(pilot) + 1)]

    pilot.to_csv(out / "SEMANTIC_PILOT.csv", index=False, encoding="utf-8-sig")

    tasks = []
    for r in pilot.itertuples(index=False):
        tasks.append(
            {
                "pilot_id": r.pilot_id,
                "video_id": r.video_id,
                "piece_id": r.piece_id,
                "pilot_bucket": r.pilot_bucket,
                "prev_context": r.prev_context,
                "source_text": r.text_exact,
                "next_context": r.next_context,
                "hints": {
                    "needs_ai_split_hint": bool(r.needs_ai_split_hint),
                    "tiny_fragment_hint": bool(r.tiny_fragment_hint),
                    "repeat_hint": bool(r.repeat_hint),
                    "initial_crowding_hint": bool(r.initial_crowding_hint),
                },
            }
        )

    payload = {
        "stage": "SEMANTIC_PILOT_V1",
        "rules": [
            "Review source_text in prev/next context.",
            "Do not rewrite source_text.",
            "If splitting source_text, concatenated segments must equal source_text after whitespace normalization.",
            "Tiny fragments are not automatic drops.",
            "Phonetic crowding is a risk label, not automatic drop.",
            "Return one primary label per resulting segment.",
        ],
        "allowed_labels": [
            "KEEP_CANDIDATE",
            "DROP_FRAGMENT",
            "BROKEN_TEXT",
            "WEIRD_PHRASE",
            "PARALINGUISTIC",
            "STUTTER_OR_REPEAT",
            "PHONETIC_CROWDING_RISK",
            "NEED_CONTEXT",
        ],
        "items": tasks,
    }

    (out / "SEMANTIC_PILOT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "stage": "SEMANTIC_PILOT_V1",
        "items": int(len(pilot)),
        "sources": int(pilot["video_id"].nunique()),
        "bucket_counts": {
            str(k): int(v)
            for k, v in pilot["pilot_bucket"].value_counts().to_dict().items()
        },
        "seed": args.seed,
        "purpose": (
            "Validate semantic split/labels across all 16 sources before full LLM batch execution."
        ),
    }

    (out / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Output:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
