from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MOJIBAKE_MARKERS = ("Ã", "Â", "Æ", "Ä", "áº", "á»", "â€œ", "â€", "ðŸ")


def main() -> int:
    p = argparse.ArgumentParser(description="Verify that semantic task JSON text is real UTF-8 Vietnamese, not mojibake.")
    p.add_argument(
        "task",
        nargs="?",
        default="outputs/semantic_tasks_v1/tasks/semantic_task_0001.json",
    )
    args = p.parse_args()

    path = Path(args.task)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()

    obj = json.loads(path.read_text(encoding="utf-8"))
    first = obj["blocks"][0]["source_text"]

    marker_hits = {m: first.count(m) for m in MOJIBAKE_MARKERS if m in first}
    has_common_vi = any(ch in first for ch in "ăâđêôơưĂÂĐÊÔƠƯ")

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("FILE:", path)
    print("TASK_ID:", obj.get("task_id"))
    print("FIRST_BLOCK:", obj["blocks"][0].get("block_id"))
    print()
    print("TEXT:")
    print(first)
    print()
    print("UTF8_VI_CHARS_PRESENT:", has_common_vi)
    print("MOJIBAKE_MARKERS:", marker_hits)
    print()

    if marker_hits:
        print("FAIL: the JSON itself appears mojibaked. Do not send these tasks to an LLM yet.")
        return 2

    print("PASS: JSON text appears to be valid Vietnamese UTF-8.")
    print("If PowerShell Get-Content still shows YÃªn..., that is only console decoding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
