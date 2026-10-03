from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Export compact GitHub review packs for ChatGPT review (no external LLM API)."
    )
    p.add_argument(
        "pieces_csv",
        nargs="?",
        default="outputs/semantic_tasks_v2/PRESEGMENTED_PIECES.csv",
    )
    p.add_argument(
        "--out",
        default="review_queue/semantic_v2",
    )
    p.add_argument(
        "--target-chars",
        type=int,
        default=45000,
        help="Target source-text characters per pack. Never splits a piece.",
    )
    p.add_argument(
        "--soft-max-chars",
        type=int,
        default=55000,
    )
    p.add_argument(
        "--context-items",
        type=int,
        default=2,
    )
    args = p.parse_args()

    src = Path(args.pieces_csv)
    if not src.is_absolute():
        src = ROOT / src
    src = src.resolve()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out = out.resolve()

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src, low_memory=False)
    required = {
        "video_id",
        "parent_block_id",
        "piece_id",
        "text_exact",
        "char_count",
        "word_count",
        "needs_ai_split_hint",
        "tiny_fragment_hint",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    # Stable ordering is essential.
    df = df.reset_index(drop=True)

    packs = []
    pack_no = 1

    for video_id, g in df.groupby("video_id", sort=False):
        g = g.reset_index(drop=True)

        start = 0
        while start < len(g):
            items = []
            chars = 0
            i = start

            while i < len(g):
                r = g.iloc[i]
                text = str(r["text_exact"])
                add = len(text)

                if items and chars >= args.target_chars and chars + add > args.soft_max_chars:
                    break

                prev_context = []
                for k in range(max(0, i - args.context_items), i):
                    prev_context.append(
                        {
                            "piece_id": str(g.iloc[k]["piece_id"]),
                            "text": str(g.iloc[k]["text_exact"]),
                        }
                    )

                next_context = []
                for k in range(i + 1, min(len(g), i + 1 + args.context_items)):
                    next_context.append(
                        {
                            "piece_id": str(g.iloc[k]["piece_id"]),
                            "text": str(g.iloc[k]["text_exact"]),
                        }
                    )

                items.append(
                    {
                        "piece_id": str(r["piece_id"]),
                        "parent_block_id": str(r["parent_block_id"]),
                        "text_exact": text,
                        "char_count": int(r["char_count"]),
                        "word_count": int(r["word_count"]),
                        "hints": {
                            "needs_ai_split_hint": bool(r["needs_ai_split_hint"]),
                            "tiny_fragment_hint": bool(r["tiny_fragment_hint"]),
                        },
                        "prev_context": prev_context,
                        "next_context": next_context,
                    }
                )

                chars += add
                i += 1

                if chars >= args.soft_max_chars:
                    break

            if not items:
                r = g.iloc[start]
                items = [
                    {
                        "piece_id": str(r["piece_id"]),
                        "parent_block_id": str(r["parent_block_id"]),
                        "text_exact": str(r["text_exact"]),
                        "char_count": int(r["char_count"]),
                        "word_count": int(r["word_count"]),
                        "hints": {
                            "needs_ai_split_hint": bool(r["needs_ai_split_hint"]),
                            "tiny_fragment_hint": bool(r["tiny_fragment_hint"]),
                        },
                        "prev_context": [],
                        "next_context": [],
                    }
                ]
                i = start + 1
                chars = len(str(r["text_exact"]))

            pack_id = f"review_pack_{pack_no:04d}"
            payload = {
                "pack_id": pack_id,
                "video_id": str(video_id),
                "rules": [
                    "ChatGPT reviews directly from GitHub. No external LLM API is required.",
                    "Do not rewrite source text in this stage.",
                    "Every output segment must preserve source words/order exactly.",
                    "Use context for tiny fragments; tiny does not automatically mean DROP.",
                    "PHONETIC_CROWDING_RISK is not an automatic DROP.",
                    "If splitting a piece further, concatenated text must equal text_exact after whitespace normalization.",
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
                "items": items,
            }

            path = out / f"{pack_id}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            packs.append(
                {
                    "pack_id": pack_id,
                    "video_id": str(video_id),
                    "item_count": len(items),
                    "source_text_chars": chars,
                    "first_piece_id": items[0]["piece_id"],
                    "last_piece_id": items[-1]["piece_id"],
                    "file": path.name,
                }
            )

            pack_no += 1
            start = i

    index = pd.DataFrame(packs)
    index.to_csv(out / "INDEX.csv", index=False, encoding="utf-8-sig")

    summary = {
        "stage": "GIT_REVIEW_QUEUE_SEMANTIC_V2",
        "packs": int(len(index)),
        "sources": int(index["video_id"].nunique()),
        "items": int(index["item_count"].sum()),
        "source_text_chars": int(index["source_text_chars"].sum()),
        "target_chars": args.target_chars,
        "soft_max_chars": args.soft_max_chars,
        "review_mode": "ChatGPT reads packs directly from GitHub and commits review results back to GitHub.",
        "external_llm_api_required": False,
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
