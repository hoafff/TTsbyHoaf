from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

KEEP_16 = [
    "RiAsGTSORZ4",
    "HdbsrkbYzHM",
    "-XC3WibmUKQ",
    "YaxBsFrzwZE",
    "I29SlhGqXgg",
    "0_gFaWnMiCY",
    "Pf9QTip2hqI",
    "S9YN9llAFv4",
    "arAbGLeCUsU",
    "E_AgFYj41Nk",
    "vimmfFJdrYM",
    "hDNesxEPAws",
    "HAe5kdfMP9U",
    "Itad_gcdHHM",
    "0Wjq6gqd1Sc",
    "9XJ78IeFBxY",
]

EXCLUDE_ONLY = "0nJXkX0L3kQ"

TARGET_CHARS = 50_000
SOFT_MIN_CHARS = 40_000
SOFT_MAX_CHARS = 60_000
CONTEXT_SENTENCES = 3

ROOT = Path("/kaggle/working/source16_captions")
RAW_DIR = ROOT / "raw_json3"
TABLE_DIR = ROOT / "tables"
REVIEW_DIR = ROOT / "review_parts"
LOG_DIR = ROOT / "logs"

for p in (ROOT, RAW_DIR, TABLE_DIR, REVIEW_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)

assert len(KEEP_16) == 16
assert len(set(KEEP_16)) == 16
assert EXCLUDE_ONLY not in KEEP_16


def install_ytdlp() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-U",
            "--pre",
            "yt-dlp[default]",
        ]
    )


def hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    sec -= h * 3600
    m = int(sec // 60)
    sec -= m * 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def clean_ws(text: str) -> str:
    text = html.unescape(str(text))
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()


def join_text(a: str, b: str) -> str:
    a = a or ""
    b = b or ""
    if not a:
        return b
    if not b:
        return a
    if a.endswith((" ", "\n")) or b.startswith((" ", "\n", ",", ".", "?", "!", "…", ":", ";", ")", "]", "}")):
        return a + b
    return a + " " + b


TERMINAL_RE = re.compile(r'[.!?…]+(?:["”’\'»)\]]+)?\s*$')


def ends_sentence(text: str) -> bool:
    return bool(TERMINAL_RE.search(text.strip()))


def download_json3(video_id: str, idx: int) -> tuple[Path | None, str]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    tmpl = str(RAW_DIR / "%(id)s.%(ext)s")
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--skip-download",
        "--write-auto-subs",
        "--sub-langs",
        "vi",
        "--sub-format",
        "json3",
        "--no-warnings",
        "-o",
        tmpl,
        url,
    ]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path = LOG_DIR / f"{idx:02d}_{video_id}.log"
    log_path.write_text(p.stdout, encoding="utf-8", errors="ignore")

    matches = sorted(RAW_DIR.glob(f"{video_id}.vi.json3"))
    if p.returncode == 0 and matches:
        return matches[0], p.stdout

    return None, p.stdout


def parse_json3(video_id: str, path: Path) -> pd.DataFrame:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = []

    for event_index, event in enumerate(obj.get("events", [])):
        segs = event.get("segs") or []
        raw_text = "".join(str(seg.get("utf8", "")) for seg in segs)
        text = clean_ws(raw_text)
        if not text:
            continue

        start_ms = float(event.get("tStartMs") or 0)
        dur_ms = float(event.get("dDurationMs") or 0)
        end_ms = start_ms + dur_ms

        rows.append(
            {
                "video_id": video_id,
                "event_index": event_index,
                "start_sec": start_ms / 1000.0,
                "end_sec": end_ms / 1000.0,
                "duration_sec": dur_ms / 1000.0,
                "raw_text": raw_text,
                "text_ws_clean": text,
                "ends_terminal_punctuation": ends_sentence(text),
            }
        )

    return pd.DataFrame(rows)


def build_sentence_units(events: pd.DataFrame) -> pd.DataFrame:
    units = []
    current_text = ""
    start_sec = None
    end_sec = None
    first_event = None
    last_event = None
    source_video = None

    def flush(boundary: str) -> None:
        nonlocal current_text, start_sec, end_sec, first_event, last_event, source_video
        text = clean_ws(current_text)
        if not text:
            current_text = ""
            start_sec = end_sec = first_event = last_event = source_video = None
            return

        units.append(
            {
                "video_id": source_video,
                "sentence_id": f"{source_video}_S{len(units)+1:06d}",
                "start_sec": start_sec,
                "end_sec": end_sec,
                "first_event_index": first_event,
                "last_event_index": last_event,
                "text": text,
                "char_count": len(text),
                "word_count": len(text.split()),
                "boundary_basis": boundary,
            }
        )
        current_text = ""
        start_sec = end_sec = first_event = last_event = source_video = None

    for row in events.itertuples(index=False):
        if start_sec is None:
            start_sec = float(row.start_sec)
            first_event = int(row.event_index)
            source_video = str(row.video_id)

        end_sec = float(row.end_sec)
        last_event = int(row.event_index)
        current_text = join_text(current_text, str(row.text_ws_clean))

        if ends_sentence(current_text):
            flush("terminal_punctuation")

    if clean_ws(current_text):
        flush("end_of_source_without_terminal_punctuation")

    return pd.DataFrame(units)


def make_review_parts(sentences: pd.DataFrame) -> tuple[pd.DataFrame, list[Path]]:
    index_rows = []
    files: list[Path] = []
    sentence_records = sentences.to_dict("records")

    start_i = 0
    part_no = 1
    previous_context: list[dict] = []

    while start_i < len(sentence_records):
        selected = []
        chars = 0
        i = start_i

        while i < len(sentence_records):
            sent = sentence_records[i]
            add_chars = len(sent["text"]) + 1

            if selected and chars >= SOFT_MIN_CHARS and chars + add_chars > TARGET_CHARS:
                break

            selected.append(sent)
            chars += add_chars
            i += 1

            if chars >= SOFT_MAX_CHARS:
                break

        # Never split a sentence. A single long sentence/unit may exceed the soft max.
        if not selected:
            selected = [sentence_records[start_i]]
            i = start_i + 1
            chars = len(selected[0]["text"])

        video_id = str(selected[0]["video_id"])
        part_id = f"{video_id}_part_{part_no:03d}"
        path = REVIEW_DIR / f"{part_id}.txt"

        lines = [
            f"SOURCE_VIDEO_ID: {video_id}",
            f"PART_ID: {part_id}",
            f"TARGET_CHARS: {TARGET_CHARS}",
            "RULE: Never infer that caption punctuation is perfect; preserve it and review it.",
            "RULE: CONTEXT_ONLY lines are context, not duplicate review decisions.",
            "",
        ]

        ctx = previous_context[-CONTEXT_SENTENCES:]
        if ctx:
            lines.append("===== CONTEXT_ONLY =====")
            for s in ctx:
                lines.append(
                    f"[CONTEXT_ONLY][{s['sentence_id']}][{hms(s['start_sec'])} --> {hms(s['end_sec'])}] {s['text']}"
                )
            lines.append("")

        lines.append("===== REVIEW =====")
        for s in selected:
            lines.append(
                f"[{s['sentence_id']}][{hms(s['start_sec'])} --> {hms(s['end_sec'])}] {s['text']}"
            )

        body = "\n".join(lines).rstrip() + "\n"
        path.write_text(body, encoding="utf-8")
        files.append(path)

        index_rows.append(
            {
                "video_id": video_id,
                "part_id": part_id,
                "part_no": part_no,
                "review_sentence_count": len(selected),
                "context_sentence_count": len(ctx),
                "first_sentence_id": selected[0]["sentence_id"],
                "last_sentence_id": selected[-1]["sentence_id"],
                "start_sec": selected[0]["start_sec"],
                "end_sec": selected[-1]["end_sec"],
                "review_text_chars": chars,
                "file_chars": len(body),
                "file": path.name,
            }
        )

        previous_context.extend(selected)
        start_i = i
        part_no += 1

    return pd.DataFrame(index_rows), files


install_ytdlp()

try:
    import yt_dlp
    print("yt-dlp:", yt_dlp.version.__version__)
except Exception:
    pass

audit_rows = []
all_events = []
all_sentences = []
all_parts = []

print("=" * 110)
print("MAYMAY SOURCE16 AUTO-VI CAPTION FETCH + REVIEW PACK BUILD")
print("=" * 110)

for idx, video_id in enumerate(KEEP_16, start=1):
    print(f"[{idx:02d}/16] {video_id}", flush=True)
    raw_path, log = download_json3(video_id, idx)

    if raw_path is None:
        audit_rows.append(
            {
                "index": idx,
                "video_id": video_id,
                "status": "CAPTION_DOWNLOAD_FAIL",
                "raw_json3": "",
            }
        )
        print("  CAPTION_DOWNLOAD_FAIL", flush=True)
        continue

    events = parse_json3(video_id, raw_path)
    if events.empty:
        audit_rows.append(
            {
                "index": idx,
                "video_id": video_id,
                "status": "CAPTION_EMPTY",
                "raw_json3": raw_path.name,
            }
        )
        print("  CAPTION_EMPTY", flush=True)
        continue

    sentences = build_sentence_units(events)
    parts, _ = make_review_parts(sentences)

    events.to_csv(TABLE_DIR / f"{video_id}_EVENTS.csv", index=False, encoding="utf-8-sig")
    sentences.to_csv(TABLE_DIR / f"{video_id}_SENTENCE_UNITS.csv", index=False, encoding="utf-8-sig")
    parts.to_csv(TABLE_DIR / f"{video_id}_REVIEW_INDEX.csv", index=False, encoding="utf-8-sig")

    all_events.append(events)
    all_sentences.append(sentences)
    all_parts.append(parts)

    audit_rows.append(
        {
            "index": idx,
            "video_id": video_id,
            "status": "OK",
            "raw_json3": raw_path.name,
            "event_count": len(events),
            "sentence_unit_count": len(sentences),
            "review_part_count": len(parts),
            "text_chars": int(sentences["char_count"].sum()) if not sentences.empty else 0,
            "unterminated_final_units": int(
                (sentences["boundary_basis"] == "end_of_source_without_terminal_punctuation").sum()
            ) if not sentences.empty else 0,
        }
    )

    print(
        f"  OK | events={len(events)} | sentence_units={len(sentences)} | "
        f"parts={len(parts)} | chars={audit_rows[-1]['text_chars']}",
        flush=True,
    )

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(ROOT / "SOURCE16_CAPTION_FETCH_AUDIT.csv", index=False, encoding="utf-8-sig")

events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
sentences_df = pd.concat(all_sentences, ignore_index=True) if all_sentences else pd.DataFrame()
parts_df = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()

if not events_df.empty:
    events_df.to_csv(ROOT / "ALL_CAPTION_EVENTS.csv", index=False, encoding="utf-8-sig")
if not sentences_df.empty:
    sentences_df.to_csv(ROOT / "ALL_SENTENCE_UNITS.csv", index=False, encoding="utf-8-sig")
if not parts_df.empty:
    parts_df.to_csv(ROOT / "ALL_REVIEW_INDEX.csv", index=False, encoding="utf-8-sig")

summary = {
    "stage": "SOURCE16_CAPTION_FETCH_V1",
    "sources_total": 16,
    "sources_ok": int((audit_df["status"] == "OK").sum()),
    "sources_failed": int((audit_df["status"] != "OK").sum()),
    "caption_events": int(len(events_df)),
    "sentence_units": int(len(sentences_df)),
    "review_parts": int(len(parts_df)),
    "total_text_chars": int(sentences_df["char_count"].sum()) if not sentences_df.empty else 0,
    "target_chars_per_review_part": TARGET_CHARS,
    "soft_min_chars": SOFT_MIN_CHARS,
    "soft_max_chars": SOFT_MAX_CHARS,
    "context_overlap_sentences": CONTEXT_SENTENCES,
    "critical_rules": [
        "Auto-caption punctuation is preserved but not treated as ground truth.",
        "Review packs never split inside a sentence unit merely to hit the character budget.",
        "Raw JSON3 caption events and timestamps remain the provenance source.",
        "No audio is downloaded or cut in this stage.",
        "Exclude only 0nJXkX0L3kQ.",
    ],
}

(ROOT / "SUMMARY.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("=" * 110)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("OUTPUT:", ROOT)
