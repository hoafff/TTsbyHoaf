from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

LABELS = [
    "KEEP_CANDIDATE",
    "DROP_FRAGMENT",
    "BROKEN_TEXT",
    "WEIRD_PHRASE",
    "PARALINGUISTIC",
    "STUTTER_OR_REPEAT",
    "PHONETIC_CROWDING_RISK",
    "NEED_CONTEXT",
]

# Terminal punctuation followed by optional closing quote/bracket.
# Boundary only when punctuation is followed by whitespace or end-of-text,
# so decimal dots etc. are not blindly split.
BOUNDARY_RE = re.compile(r'(?:[.!?]+|…+)(?:["”’»)\]]+)?(?=\s+|$)')

PROMPT = """You are preparing Vietnamese narration data for TTS training.

Each SOURCE_BLOCK below is already split losslessly at obvious internal terminal punctuation.
For every SOURCE_BLOCK:
- keep it whole if it is already a good spoken training unit;
- otherwise split it further at natural semantic/clause boundaries;
- assign exactly one PRIMARY label to each output segment.

STRICT RULES
1. DO NOT rewrite, correct, normalize, add, or delete words in this stage.
2. The concatenation of segment text, after whitespace normalization, MUST equal the SOURCE_BLOCK text after whitespace normalization.
3. Keep original word order exactly.
4. Punctuation is evidence, not ground truth.
5. Prefer coherent spoken units roughly suitable for 4-12 seconds, but semantic completeness outranks a hard duration target.
6. Tiny standalone fragments may be labeled DROP_FRAGMENT rather than forced into unrelated context.
7. PHONETIC_CROWDING_RISK is meaningful text that may be articulation-risky; it is not an automatic drop.
8. STUTTER_OR_REPEAT is for suspicious accidental repetition/stutter.
9. PARALINGUISTIC covers cough/laugh/non-lexical acoustic actions unsuitable as ordinary phonemic narration.
10. Return JSON only.

Allowed labels:
KEEP_CANDIDATE
DROP_FRAGMENT
BROKEN_TEXT
WEIRD_PHRASE
PARALINGUISTIC
STUTTER_OR_REPEAT
PHONETIC_CROWDING_RISK
NEED_CONTEXT

OUTPUT SCHEMA
{
  "task_id": "...",
  "blocks": [
    {
      "block_id": "...",
      "segments": [
        {
          "text_exact": "...",
          "label": "KEEP_CANDIDATE",
          "reason_short": "..."
        }
      ]
    }
  ]
}
"""


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def split_internal_punctuation(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0

    for match in BOUNDARY_RE.finditer(text):
        end = match.end()
        raw = text[start:end]
        left_trim = len(raw) - len(raw.lstrip())
        right_trimmed = raw.rstrip()
        effective_end = start + len(right_trimmed)

        piece = text[start + left_trim:effective_end]
        if piece:
            spans.append((start + left_trim, effective_end, piece))

        # consume whitespace after boundary, but preserve offsets
        start = end
        while start < len(text) and text[start].isspace():
            start += 1

    if start < len(text):
        raw = text[start:]
        left_trim = len(raw) - len(raw.lstrip())
        piece_start = start + left_trim
        piece = text[piece_start:].rstrip()
        if piece:
            spans.append((piece_start, piece_start + len(piece), piece))

    if not spans and text.strip():
        s = len(text) - len(text.lstrip())
        piece = text.strip()
        spans = [(s, s + len(piece), piece)]

    # Safety: punctuation split must be lossless modulo whitespace.
    if norm(" ".join(p[2] for p in spans)) != norm(text):
        raise ValueError("Lossless punctuation split failed")

    return spans


def build_pieces(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for row in df.itertuples(index=False):
        text = str(row.text)
        spans = split_internal_punctuation(text)
        block_duration = float(row.end_sec) - float(row.start_sec)
        total_chars = max(1, len(text))

        for idx, (char_start, char_end, piece) in enumerate(spans, start=1):
            frac_start = char_start / total_chars
            frac_end = char_end / total_chars

            approx_start = float(row.start_sec) + block_duration * frac_start
            approx_end = float(row.start_sec) + block_duration * frac_end

            words = len(norm(piece).split())

            rows.append(
                {
                    "video_id": str(row.video_id),
                    "parent_block_id": str(row.sentence_id),
                    "piece_id": f"{row.sentence_id}_P{idx:03d}",
                    "parent_start_sec": float(row.start_sec),
                    "parent_end_sec": float(row.end_sec),
                    "char_start": int(char_start),
                    "char_end": int(char_end),
                    "text_exact": piece,
                    "char_count": len(piece),
                    "word_count": words,
                    "approx_start_sec_proportional": approx_start,
                    "approx_end_sec_proportional": approx_end,
                    "needs_ai_split_hint": bool(words > 42 or len(piece) > 240),
                    "tiny_fragment_hint": bool(words <= 2 or len(piece) <= 12),
                    "split_basis": "internal_terminal_punctuation"
                    if len(spans) > 1
                    else "parent_block_unsplit",
                }
            )

    return pd.DataFrame(rows)


def build_tasks(pieces: pd.DataFrame, target_chars: int, soft_max: int) -> list[dict]:
    tasks = []
    task_no = 1

    for video_id, group in pieces.groupby("video_id", sort=False):
        current = []
        current_chars = 0

        def flush():
            nonlocal current, current_chars, task_no
            if not current:
                return
            task_id = f"semantic_v2_task_{task_no:04d}"
            tasks.append(
                {
                    "task_id": task_id,
                    "video_id": video_id,
                    "instructions": PROMPT,
                    "blocks": current,
                }
            )
            task_no += 1
            current = []
            current_chars = 0

        for row in group.itertuples(index=False):
            block = {
                "block_id": str(row.piece_id),
                "parent_block_id": str(row.parent_block_id),
                "source_text": str(row.text_exact),
                "source_char_count": int(row.char_count),
                "source_word_count": int(row.word_count),
                "needs_ai_split_hint": bool(row.needs_ai_split_hint),
                "tiny_fragment_hint": bool(row.tiny_fragment_hint),
                "parent_time_sec": [
                    float(row.parent_start_sec),
                    float(row.parent_end_sec),
                ],
                "char_span_in_parent": [
                    int(row.char_start),
                    int(row.char_end),
                ],
            }

            add_chars = len(block["source_text"]) + 300

            if current and current_chars >= target_chars and current_chars + add_chars > soft_max:
                flush()

            current.append(block)
            current_chars += add_chars

            if current_chars >= soft_max:
                flush()

        flush()

    return tasks


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "input_csv",
        nargs="?",
        default="kaggle_output/source16-captions/source16_captions/ALL_SENTENCE_UNITS.csv",
    )
    p.add_argument("--out", default="outputs/semantic_tasks_v2")
    p.add_argument("--target-chars", type=int, default=30000)
    p.add_argument("--soft-max-chars", type=int, default=42000)
    args = p.parse_args()

    src = Path(args.input_csv)
    if not src.is_absolute():
        src = ROOT / src
    src = src.resolve()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    task_dir = out / "tasks"
    task_dir.mkdir(exist_ok=True)
    for old in task_dir.glob("semantic_v2_task_*.json"):
        old.unlink()

    df = pd.read_csv(src, low_memory=False)
    pieces = build_pieces(df)

    # Parent-level exact coverage check before any LLM use.
    coverage_errors = []
    original = {str(r.sentence_id): str(r.text) for r in df.itertuples(index=False)}
    for parent_id, g in pieces.groupby("parent_block_id", sort=False):
        reconstructed = " ".join(g.sort_values("char_start")["text_exact"].astype(str))
        if norm(reconstructed) != norm(original[parent_id]):
            coverage_errors.append(parent_id)

    if coverage_errors:
        raise SystemExit(f"Pre-segmentation coverage failed for {len(coverage_errors)} parent blocks")

    tasks = build_tasks(pieces, args.target_chars, args.soft_max_chars)

    index_rows = []
    for task in tasks:
        path = task_dir / f"{task['task_id']}.json"
        payload = json.dumps(task, ensure_ascii=False, indent=2)
        path.write_text(payload, encoding="utf-8")

        blocks = task["blocks"]
        index_rows.append(
            {
                "task_id": task["task_id"],
                "video_id": task["video_id"],
                "block_count": len(blocks),
                "source_text_chars": sum(len(b["source_text"]) for b in blocks),
                "file_chars": len(payload),
                "first_block_id": blocks[0]["block_id"],
                "last_block_id": blocks[-1]["block_id"],
                "file": path.name,
            }
        )

    pieces.to_csv(out / "PRESEGMENTED_PIECES.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(index_rows).to_csv(out / "TASK_INDEX.csv", index=False, encoding="utf-8-sig")

    summary = {
        "stage": "SEMANTIC_TASKS_V2_INTERNAL_PUNCTUATION_PRESEG",
        "parent_blocks": int(len(df)),
        "presegmented_pieces": int(len(pieces)),
        "tasks": int(len(tasks)),
        "sources": int(pieces["video_id"].nunique()),
        "pieces_needing_ai_split_hint": int(pieces["needs_ai_split_hint"].sum()),
        "tiny_fragment_hints": int(pieces["tiny_fragment_hint"].sum()),
        "piece_chars_p50": float(pieces["char_count"].quantile(0.50)),
        "piece_chars_p90": float(pieces["char_count"].quantile(0.90)),
        "piece_chars_p99": float(pieces["char_count"].quantile(0.99)),
        "piece_words_p50": float(pieces["word_count"].quantile(0.50)),
        "piece_words_p90": float(pieces["word_count"].quantile(0.90)),
        "piece_words_p99": float(pieces["word_count"].quantile(0.99)),
        "lossless_parent_coverage_errors": 0,
        "hard_rules": [
            "Internal terminal punctuation is split deterministically before LLM use.",
            "Every task contains only one source video.",
            "LLM may segment/label only; it may not rewrite source text.",
            "Character spans preserve deterministic provenance back to the parent caption block.",
            "Proportional timestamps are hints only, never final audio boundaries.",
        ],
    }

    (out / "MANIFEST.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "PROMPT.txt").write_text(PROMPT, encoding="utf-8")

    print("=" * 100)
    print("SEMANTIC TASK BUILDER V2")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Output:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
