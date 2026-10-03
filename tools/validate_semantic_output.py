from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

VALID_LABELS = {
    "KEEP_CANDIDATE",
    "DROP_FRAGMENT",
    "BROKEN_TEXT",
    "WEIRD_PHRASE",
    "PARALINGUISTIC",
    "STUTTER_OR_REPEAT",
    "PHONETIC_CROWDING_RISK",
    "NEED_CONTEXT",
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("tasks_dir")
    p.add_argument("results_dir")
    p.add_argument("--out", default="outputs/semantic_validation_v1")
    args = p.parse_args()

    tasks_dir = Path(args.tasks_dir)
    results_dir = Path(args.results_dir)
    out = Path(args.out)

    if not tasks_dir.is_absolute():
        tasks_dir = ROOT / tasks_dir
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    if not out.is_absolute():
        out = ROOT / out

    tasks_dir = tasks_dir.resolve()
    results_dir = results_dir.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    block_rows = []
    segment_rows = []
    missing_results = []

    for task_path in sorted(tasks_dir.glob("semantic_task_*.json")):
        result_path = results_dir / task_path.name
        if not result_path.exists():
            missing_results.append(task_path.name)
            continue

        task = json.loads(task_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))

        source_blocks = {b["block_id"]: b for b in task["blocks"]}
        result_blocks = {
            b.get("block_id"): b
            for b in result.get("blocks", [])
            if isinstance(b, dict) and b.get("block_id")
        }

        for block_id, source in source_blocks.items():
            rb = result_blocks.get(block_id)
            errors = []

            if rb is None:
                segments = []
                errors.append("MISSING_RESULT_BLOCK")
            else:
                segments = rb.get("segments") or []

            out_texts = []

            for idx, seg in enumerate(segments):
                text = str(seg.get("text_exact", ""))
                label = str(seg.get("label", ""))
                reason = str(seg.get("reason_short", ""))

                if label not in VALID_LABELS:
                    errors.append(f"INVALID_LABEL:{idx}:{label}")

                out_texts.append(text)

                segment_rows.append({
                    "task_id": task["task_id"],
                    "block_id": block_id,
                    "video_id": source["video_id"],
                    "segment_index": idx,
                    "text_exact": text,
                    "label": label,
                    "reason_short": reason,
                    "char_count": len(text),
                    "word_count": len(norm(text).split()) if norm(text) else 0,
                })

            source_norm = norm(source["source_text"])
            output_norm = norm(" ".join(out_texts))
            coverage_ok = source_norm == output_norm

            if not coverage_ok:
                errors.append("TEXT_COVERAGE_MISMATCH")

            block_rows.append({
                "task_id": task["task_id"],
                "block_id": block_id,
                "video_id": source["video_id"],
                "source_start_sec": source["source_start_sec"],
                "source_end_sec": source["source_end_sec"],
                "source_duration_sec": source["source_duration_sec"],
                "segment_count": len(segments),
                "coverage_ok": coverage_ok,
                "errors": "|".join(errors),
            })

        for extra in sorted(set(result_blocks) - set(source_blocks)):
            block_rows.append({
                "task_id": task["task_id"],
                "block_id": extra,
                "coverage_ok": False,
                "errors": "EXTRA_RESULT_BLOCK",
            })

    blocks = pd.DataFrame(block_rows)
    segments = pd.DataFrame(segment_rows)

    if not blocks.empty:
        blocks.to_csv(out / "BLOCK_VALIDATION.csv", index=False, encoding="utf-8-sig")
    if not segments.empty:
        segments.to_csv(out / "SEMANTIC_SEGMENTS.csv", index=False, encoding="utf-8-sig")

    ready = bool(
        not missing_results
        and not blocks.empty
        and blocks["coverage_ok"].fillna(False).all()
        and (blocks["errors"].fillna("") == "").all()
    )

    summary = {
        "validated_blocks": int(len(blocks)),
        "coverage_ok_blocks": int(blocks["coverage_ok"].fillna(False).sum()) if not blocks.empty else 0,
        "blocks_with_errors": int((blocks["errors"].fillna("") != "").sum()) if not blocks.empty else 0,
        "segments": int(len(segments)),
        "missing_result_files": missing_results,
        "ready_for_mapping": ready,
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
