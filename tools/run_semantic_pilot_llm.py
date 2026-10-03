from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

ALLOWED = {
    "KEEP_CANDIDATE",
    "DROP_FRAGMENT",
    "BROKEN_TEXT",
    "WEIRD_PHRASE",
    "PARALINGUISTIC",
    "STUTTER_OR_REPEAT",
    "PHONETIC_CROWDING_RISK",
    "NEED_CONTEXT",
}

SYSTEM = """You are a strict Vietnamese TTS dataset reviewer.

For each item, you may ONLY:
1) split source_text into one or more coherent spoken segments;
2) assign exactly one allowed primary label to each segment;
3) give a very short reason.

You MUST NOT rewrite, correct, normalize, add, remove, paraphrase, or reorder source words.
Punctuation is evidence, not ground truth.
Tiny fragments are not automatically bad: use prev_context and next_context.
PHONETIC_CROWDING_RISK is not an automatic drop.
STUTTER_OR_REPEAT is for suspicious accidental repetition, not legitimate repeated names or expressive language.

The whitespace-normalized concatenation of all text_exact outputs MUST equal the whitespace-normalized source_text exactly.

Return JSON only with keys: pilot_id, segments.
Each segment must contain: text_exact, label, reason_short.
"""


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def validate(item: dict, result: dict) -> list[str]:
    errors = []

    if str(result.get("pilot_id")) != str(item["pilot_id"]):
        errors.append("PILOT_ID_MISMATCH")

    segments = result.get("segments")
    if not isinstance(segments, list) or not segments:
        return errors + ["NO_SEGMENTS"]

    texts = []

    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            errors.append(f"SEGMENT_{i}_NOT_OBJECT")
            continue

        text = str(seg.get("text_exact", ""))
        label = str(seg.get("label", ""))

        if not text:
            errors.append(f"SEGMENT_{i}_EMPTY_TEXT")

        if label not in ALLOWED:
            errors.append(f"SEGMENT_{i}_INVALID_LABEL:{label}")

        texts.append(text)

    if norm(" ".join(texts)) != norm(item["source_text"]):
        errors.append("TEXT_COVERAGE_MISMATCH")

    return errors


def make_user(item: dict, previous_errors: list[str] | None = None) -> str:
    payload = {
        "pilot_id": item["pilot_id"],
        "video_id": item["video_id"],
        "pilot_bucket": item["pilot_bucket"],
        "prev_context": item.get("prev_context", ""),
        "source_text": item["source_text"],
        "next_context": item.get("next_context", ""),
        "hints": item.get("hints", {}),
        "allowed_labels": sorted(ALLOWED),
    }

    msg = "Review this item:\n" + json.dumps(payload, ensure_ascii=False, indent=2)

    if previous_errors:
        msg += (
            "\n\nPrevious output failed validation: "
            + ", ".join(previous_errors)
            + ". Retry and preserve source_text exactly."
        )

    return msg


def call_chat(base_url: str, api_key: str, model: str, user: str, timeout: int) -> str:
    url = base_url.rstrip("/")

    if not url.endswith("/chat/completions"):
        url += "/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
    }

    response = requests.post(
        url,
        headers=headers,
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run semantic pilot through an OpenAI-compatible chat-completions API."
    )
    parser.add_argument(
        "pilot_json",
        nargs="?",
        default="outputs/semantic_pilot_v1/SEMANTIC_PILOT.json",
    )
    parser.add_argument("--out", default="outputs/semantic_pilot_results_v1")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass

    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip()

    if not base_url or not api_key or not model:
        raise SystemExit(
            "Missing LLM_BASE_URL / LLM_API_KEY / LLM_MODEL. "
            "Set them in the shell or create local .env from .env.example. "
            "Do not commit secrets."
        )

    placeholder_values = (
        "YOUR_PROVIDER",
        "YOUR_SECRET_KEY",
        "YOUR_MODEL_NAME",
        "replace_me",
        "example.com",
    )
    combined = f"{base_url}\n{api_key}\n{model}"

    if any(marker.lower() in combined.lower() for marker in placeholder_values):
        raise SystemExit(
            "LLM configuration still contains placeholder values. "
            "Set a real provider BASE_URL, API key, and model before running."
        )

    pilot_path = Path(args.pilot_json)
    if not pilot_path.is_absolute():
        pilot_path = ROOT / pilot_path
    pilot_path = pilot_path.resolve()

    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    items = pilot["items"]

    end = min(len(items), args.offset + args.limit) if args.limit > 0 else len(items)
    selected = items[args.offset:end]

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    item_dir = out / "items"
    item_dir.mkdir(exist_ok=True)

    rows = []

    for n, item in enumerate(selected, start=1):
        pilot_id = str(item["pilot_id"])
        dest = item_dir / f"{pilot_id}.json"

        if dest.exists():
            try:
                existing = json.loads(dest.read_text(encoding="utf-8"))
                existing_result = existing.get("result", {})
                existing_errors = validate(item, existing_result)

                if not existing_errors:
                    print(
                        f"[{n:03d}/{len(selected):03d}] {pilot_id} SKIP_VALID",
                        flush=True,
                    )
                    rows.append(
                        {
                            "pilot_id": pilot_id,
                            "status": "SKIP_VALID",
                            "attempts": existing.get("attempts", 0),
                            "errors": "",
                        }
                    )
                    continue
            except Exception:
                pass

        previous_errors = None
        final_result = None
        final_errors = ["NOT_RUN"]
        raw_text = ""
        attempts = 0

        for attempt in range(1, args.max_retries + 2):
            attempts = attempt

            try:
                raw_text = call_chat(
                    base_url,
                    api_key,
                    model,
                    make_user(item, previous_errors),
                    args.timeout,
                )

                result = json.loads(raw_text.strip())
                errors = validate(item, result)

                final_result = result
                final_errors = errors

                if not errors:
                    break

                previous_errors = errors

            except Exception as exc:
                final_errors = [
                    f"REQUEST_OR_PARSE_ERROR:{type(exc).__name__}:{exc}"
                ]
                previous_errors = final_errors

            if attempt <= args.max_retries:
                time.sleep(max(args.sleep, 0.2))

        status = "PASS" if not final_errors else "FAIL"

        record = {
            "pilot_id": pilot_id,
            "model": model,
            "attempts": attempts,
            "status": status,
            "validation_errors": final_errors,
            "source": item,
            "result": final_result,
            "raw_response": raw_text if final_result is None else None,
        }

        dest.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        rows.append(
            {
                "pilot_id": pilot_id,
                "status": status,
                "attempts": attempts,
                "errors": "|".join(final_errors),
            }
        )

        print(
            f"[{n:03d}/{len(selected):03d}] {pilot_id} "
            f"{status} attempts={attempts} errors={final_errors}",
            flush=True,
        )

        time.sleep(args.sleep)

    summary = pd.DataFrame(rows)
    summary.to_csv(
        out / "RUN_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )

    totals = {
        "selected": len(selected),
        "pass": int(summary["status"].isin(["PASS", "SKIP_VALID"]).sum())
        if not summary.empty
        else 0,
        "fail": int((summary["status"] == "FAIL").sum())
        if not summary.empty
        else 0,
        "model": model,
        "offset": args.offset,
        "limit": args.limit,
    }

    (out / "SUMMARY.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(totals, ensure_ascii=False, indent=2))
    print("Output:", out)

    return 0 if totals["fail"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
