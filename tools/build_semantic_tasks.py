from __future__ import annotations

import argparse
import json
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

PROMPT = """You are preparing Vietnamese narration data for TTS training.

For every SOURCE_BLOCK, split its text into coherent spoken sentence/clause segments.

STRICT RULES
1. DO NOT rewrite, correct, normalize, add, or delete words in this stage.
2. The concatenation of segment text, after whitespace normalization, MUST equal the SOURCE_BLOCK text after whitespace normalization.
3. Keep original word order exactly.
4. Split on semantic boundaries. Punctuation is evidence, not ground truth.
5. Avoid tiny standalone fragments when they only make sense as context.
6. Prefer trainable spoken units that would plausibly be about 4-12 seconds, but semantic completeness is more important than a hard duration target.
7. If a long sentence needs splitting, split at a natural clause boundary.
8. Label every segment with exactly one primary label:
KEEP_CANDIDATE, DROP_FRAGMENT, BROKEN_TEXT, WEIRD_PHRASE, PARALINGUISTIC,
STUTTER_OR_REPEAT, PHONETIC_CROWDING_RISK, NEED_CONTEXT.
9. PHONETIC_CROWDING_RISK is not an automatic drop.
10. STUTTER_OR_REPEAT is for suspicious accidental repetition, not intentional expressive repetition.
11. Return JSON only.

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


def hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    sec -= h * 3600
    m = int(sec // 60)
    sec -= m * 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def build_tasks(df: pd.DataFrame, target_chars: int, soft_max: int) -> list[dict]:
    tasks = []
    current_blocks = []
    current_chars = 0
    task_no = 1

    def flush():
        nonlocal current_blocks, current_chars, task_no
        if not current_blocks:
            return
        task_id = f"semantic_task_{task_no:04d}"
        tasks.append({"task_id": task_id, "instructions": PROMPT, "blocks": current_blocks})
        task_no += 1
        current_blocks = []
        current_chars = 0

    current_video = None

    for row in df.itertuples(index=False):
        row_video = str(row.video_id)

        # Never mix two source videos inside one AI task.
        if current_blocks and current_video is not None and row_video != current_video:
            flush()

        current_video = row_video

        text = str(row.text).strip()
        duration = float(row.end_sec) - float(row.start_sec)
        block = {
            "block_id": str(row.sentence_id),
            "video_id": row_video,
            "source_start_sec": float(row.start_sec),
            "source_end_sec": float(row.end_sec),
            "source_duration_sec": duration,
            "source_time": f"{hms(row.start_sec)} --> {hms(row.end_sec)}",
            "source_text": text,
            "source_char_count": int(row.char_count),
            "source_word_count": int(row.word_count),
            "estimated_target_segments_at_8s": max(1, int(round(duration / 8.0))),
        }

        add_chars = len(text) + 450
        if current_blocks and current_chars >= target_chars and current_chars + add_chars > soft_max:
            flush()

        current_blocks.append(block)
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
    p.add_argument("--out", default="outputs/semantic_tasks_v1")
    p.add_argument("--target-chars", type=int, default=40000)
    p.add_argument("--soft-max-chars", type=int, default=55000)
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

    df = pd.read_csv(src, low_memory=False)
    tasks = build_tasks(df, args.target_chars, args.soft_max_chars)

    index_rows = []
    for task in tasks:
        path = task_dir / f"{task['task_id']}.json"
        payload = json.dumps(task, ensure_ascii=False, indent=2)
        path.write_text(payload, encoding="utf-8")
        blocks = task["blocks"]
        index_rows.append({
            "task_id": task["task_id"],
            "block_count": len(blocks),
            "source_text_chars": sum(len(b["source_text"]) for b in blocks),
            "file_chars": len(payload),
            "first_block_id": blocks[0]["block_id"],
            "last_block_id": blocks[-1]["block_id"],
            "videos": ",".join(sorted({b["video_id"] for b in blocks})),
            "file": path.name,
        })

    pd.DataFrame(index_rows).to_csv(out / "TASK_INDEX.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "stage": "SEMANTIC_TASKS_V1",
        "source_blocks": int(len(df)),
        "tasks": int(len(tasks)),
        "source_text_chars": int(df["text"].astype(str).str.len().sum()),
        "target_chars": args.target_chars,
        "soft_max_chars": args.soft_max_chars,
        "labels": LABELS,
        "hard_rules": [
            "AI may segment and label only; no text rewriting in this stage.",
            "Every source block must be completely covered in original order.",
            "Whitespace-normalized concatenated output must equal source text.",
            "Punctuation is evidence, not ground truth.",
            "A task may contain blocks from only one source video.",
        ],
    }

    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "PROMPT.txt").write_text(PROMPT, encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Output:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
