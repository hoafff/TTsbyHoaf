from __future__ import annotations

import html
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

KEEP_16 = [
    "RiAsGTSORZ4", "HdbsrkbYzHM", "-XC3WibmUKQ", "YaxBsFrzwZE",
    "I29SlhGqXgg", "0_gFaWnMiCY", "Pf9QTip2hqI", "S9YN9llAFv4",
    "arAbGLeCUsU", "E_AgFYj41Nk", "vimmfFJdrYM", "hDNesxEPAws",
    "HAe5kdfMP9U", "Itad_gcdHHM", "0Wjq6gqd1Sc", "9XJ78IeFBxY",
]

ROOT = Path("/kaggle/working/source16_segmentation")
OUT = ROOT / "outputs"
PER_SOURCE = OUT / "sources"
CHECKPOINT_SLUG = "maymay-source16-whisper-checkpoint"
MEDIA_SLUG = "maymay-source16-media"

BUCKET_SEC = 30
MIN_SEC = 3.0
TARGET_SEC = 8.0
MAX_SEC = 14.0
ABS_MAX_SEC = 16.0
MIN_WORDS = 3
SOURCE_CAPTION_TRUST = 0.60
CLIP_CAPTION_REVIEW = 0.55

TERMINAL = re.compile(r'[.!?…]+(?:["”’\'»)\]]+)?$')
CLAUSE = re.compile(r'[,;:]+(?:["”’\'»)\]]+)?$')
NO_SPACE_BEFORE = re.compile(r'^[,.;:!?…%)\]}”’»]')
NO_SPACE_AFTER = re.compile(r'[(\[{“‘«]$')


def clean(text: str) -> str:
    text = html.unescape(str(text)).replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def lexical(text: str) -> str:
    text = clean(text).lower()
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹĐđ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity(a: str, b: str) -> float:
    aa, bb = lexical(a).split(), lexical(b).split()
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb, autojunk=False).ratio()


def join_words(words: list[str]) -> str:
    out = ""
    for raw in words:
        tok = clean(raw)
        if not tok:
            continue
        if not out or NO_SPACE_BEFORE.search(tok) or NO_SPACE_AFTER.search(out):
            out += tok
        else:
            out += " " + tok
    return clean(out)


def find_checkpoint_root() -> Path:
    roots = sorted(
        p.parent for p in Path("/kaggle/input").rglob("state.json")
        if CHECKPOINT_SLUG in str(p).lower()
    )
    if not roots:
        raise FileNotFoundError(f"Mounted checkpoint dataset {CHECKPOINT_SLUG} not found")
    print("CHECKPOINT_INPUT", roots[0], flush=True)
    return roots[0]


def find_media(video_id: str, caption: bool = False) -> Path:
    wanted = f"{video_id}.vi.json3" if caption else f"{video_id}.*"
    matches = [
        p for p in Path("/kaggle/input").rglob(wanted)
        if p.is_file() and MEDIA_SLUG in str(p).lower()
    ]
    if caption:
        if not matches:
            raise FileNotFoundError(f"Missing caption for {video_id}")
        return sorted(matches)[0]

    matches = [
        p for p in matches
        if p.name != f"{video_id}.vi.json3"
        and p.suffix.lower() not in {".json3", ".json", ".txt", ".csv", ".part", ".ytdl"}
    ]
    if not matches:
        raise FileNotFoundError(f"Missing audio for {video_id}")
    return sorted(matches)[0]


def validate_state(root: Path) -> dict:
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    cfg = state.get("config", {})
    errors = []
    if state.get("stage") != "SOURCE16_WHISPER_CONSENSUS_V1":
        errors.append("stage")
    if cfg.get("model") != "large-v3":
        errors.append("model")
    if int(cfg.get("beam_size", -1)) != 1:
        errors.append("beam_size")
    if list(cfg.get("sources", [])) != KEEP_16:
        errors.append("source_list")
    if list(state.get("completed", [])) != KEEP_16:
        errors.append("completed_16")
    if errors:
        raise RuntimeError(f"Incompatible/incomplete Whisper checkpoint: {errors}")
    print("CHECKPOINT_VALIDATED completed=16/16", flush=True)
    return state


def parse_caption(path: Path) -> pd.DataFrame:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for i, event in enumerate(obj.get("events", [])):
        text = clean("".join(str(s.get("utf8", "")) for s in (event.get("segs") or [])))
        if not text:
            continue
        start_ms = float(event.get("tStartMs") or 0)
        dur_ms = float(event.get("dDurationMs") or 0)
        rows.append({
            "event_index": i,
            "start_sec": start_ms / 1000.0,
            "end_sec": (start_ms + dur_ms) / 1000.0,
            "text": text,
        })
    return pd.DataFrame(rows)


def bucket_map(df: pd.DataFrame, text_col: str) -> dict[int, str]:
    if df.empty:
        return {}
    tmp = df.copy()
    mid = (pd.to_numeric(tmp["start_sec"]) + pd.to_numeric(tmp["end_sec"])) / 2.0
    tmp["_bucket"] = (mid // BUCKET_SEC).astype(int)
    return {
        int(k): " ".join(g[text_col].astype(str))
        for k, g in tmp.groupby("_bucket", sort=True)
    }


def build_consensus(segments: pd.DataFrame, captions: pd.DataFrame) -> pd.DataFrame:
    ws, yt = bucket_map(segments, "text"), bucket_map(captions, "text")
    rows = []
    for b in sorted(set(ws) & set(yt)):
        rows.append({
            "bucket": b,
            "start_sec": b * BUCKET_SEC,
            "end_sec": (b + 1) * BUCKET_SEC,
            "token_similarity": similarity(ws[b], yt[b]),
        })
    return pd.DataFrame(rows)


def prep_words(df: pd.DataFrame) -> pd.DataFrame:
    required = {"segment_id", "word_index", "word", "start_sec", "end_sec", "probability"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"WHISPER_WORDS missing columns: {sorted(missing)}")
    out = df.copy()
    for c in ("start_sec", "end_sec", "probability"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["start_sec", "end_sec", "probability"])
    out["word"] = out["word"].map(clean)
    out = out[out["word"].astype(bool)].sort_values(
        ["start_sec", "end_sec", "segment_id", "word_index"], kind="stable"
    ).reset_index(drop=True)
    out["prev_end"] = out["end_sec"].shift(1)
    out["next_start"] = out["start_sec"].shift(-1)
    out["gap_before"] = (out["start_sec"] - out["prev_end"]).clip(lower=0)
    out["gap_after"] = (out["next_start"] - out["end_sec"]).clip(lower=0)
    out["segment_boundary"] = out["segment_id"].astype(str) != out["segment_id"].shift(-1).astype(str)
    if len(out):
        out.loc[out.index[-1], "segment_boundary"] = True
    out["terminal"] = out["word"].map(lambda x: bool(TERMINAL.search(x)))
    out["clause"] = out["word"].map(lambda x: bool(CLAUSE.search(x)))
    return out


def boundary(words: pd.DataFrame, i: int) -> tuple[float, str]:
    if i == len(words) - 1:
        return 10.0, "end_of_source"
    r = words.iloc[i]
    gap = float(r["gap_after"]) if pd.notna(r["gap_after"]) else 0.0
    score, why = 0.0, []
    if bool(r["terminal"]):
        score += 4.0
        why.append("terminal")
    elif bool(r["clause"]):
        score += 1.5
        why.append("clause")
    if gap >= 0.60:
        score += 4.0
        why.append("pause>=0.60")
    elif gap >= 0.35:
        score += 3.0
        why.append("pause>=0.35")
    elif gap >= 0.20:
        score += 2.0
        why.append("pause>=0.20")
    elif gap >= 0.12:
        score += 0.75
        why.append("pause>=0.12")
    if bool(r["segment_boundary"]):
        score += 1.0
        why.append("segment_boundary")
    return score, "+".join(why) if why else "weak_boundary"


def choose_end(words: pd.DataFrame, start: int) -> tuple[int, str, float]:
    t0 = float(words.iloc[start]["start_sec"])
    options = []
    fallback = start
    for i in range(start, len(words)):
        dur = float(words.iloc[i]["end_sec"]) - t0
        if dur <= ABS_MAX_SEC:
            fallback = i
        if dur > MAX_SEC:
            break
        if dur < MIN_SEC:
            continue
        bscore, basis = boundary(words, i)
        duration_fit = max(0.0, 2.5 - abs(dur - TARGET_SEC) * (2.5 / TARGET_SEC))
        word_bonus = 0.5 if i - start + 1 >= MIN_WORDS else -2.0
        options.append((bscore + duration_fit + word_bonus, i, basis))
    if options:
        score, i, basis = max(options, key=lambda x: (
            x[0],
            -abs((float(words.iloc[x[1]]["end_sec"]) - t0) - TARGET_SEC),
        ))
        return i, basis, score
    return fallback, "forced_max_or_short_tail", -1.0


def make_spans(words: pd.DataFrame) -> list[tuple[int, int, str, float]]:
    spans, start = [], 0
    while start < len(words):
        end, basis, score = choose_end(words, start)
        spans.append((start, max(start, end), basis, score))
        start = end + 1
    if len(spans) >= 2:
        s, e, basis, score = spans[-1]
        ps, pe, pbasis, pscore = spans[-2]
        tail = float(words.iloc[e]["end_sec"]) - float(words.iloc[s]["start_sec"])
        merged = float(words.iloc[e]["end_sec"]) - float(words.iloc[ps]["start_sec"])
        if tail < MIN_SEC and merged <= ABS_MAX_SEC:
            spans[-2] = (ps, e, pbasis + "+merged_short_tail", max(pscore, score))
            spans.pop()
    return spans


def overlap_consensus(consensus: pd.DataFrame, start: float, end: float) -> float | None:
    total = weight = 0.0
    for r in consensus.itertuples(index=False):
        overlap = max(0.0, min(end, float(r.end_sec)) - max(start, float(r.start_sec)))
        if overlap:
            weight += overlap
            total += overlap * float(r.token_similarity)
    return total / weight if weight else None


def segment_quality(clip_words: pd.DataFrame, segments: pd.DataFrame) -> tuple[float | None, float | None]:
    ids = set(pd.to_numeric(clip_words["segment_id"], errors="coerce").dropna().astype(int))
    source_ids = pd.to_numeric(segments["segment_id"], errors="coerce")
    sub = segments[source_ids.isin(ids)]
    if sub.empty:
        return None, None
    lp = pd.to_numeric(sub["avg_logprob"], errors="coerce").dropna()
    ns = pd.to_numeric(sub["no_speech_prob"], errors="coerce").dropna()
    return (
        float(lp.median()) if not lp.empty else None,
        float(ns.max()) if not ns.empty else None,
    )


def classify(row: dict, caption_trusted: bool) -> tuple[str, str]:
    drop, review = [], []
    dur, wc = float(row["raw_duration_sec"]), int(row["word_count"])
    mp, p10 = row["mean_word_probability"], row["word_probability_p10"]
    lp, ns = row["median_segment_avg_logprob"], row["max_segment_no_speech_prob"]
    cs = row["clip_consensus_mean"]

    if dur > ABS_MAX_SEC:
        drop.append("duration>absolute_max")
    if wc < 2:
        drop.append("too_few_words")
    if mp is not None and mp < 0.55:
        drop.append("very_low_word_probability")
    if ns is not None and ns > 0.65:
        drop.append("very_high_no_speech")
    if drop:
        return "DROP", ";".join(drop)

    if dur < MIN_SEC:
        review.append("short_duration")
    if dur > MAX_SEC:
        review.append("long_duration")
    if wc < MIN_WORDS:
        review.append("few_words")
    if mp is not None and mp < 0.78:
        review.append("word_probability")
    if p10 is not None and p10 < 0.45:
        review.append("word_probability_p10")
    if lp is not None and lp < -0.90:
        review.append("segment_logprob")
    if ns is not None and ns > 0.35:
        review.append("no_speech")
    if str(row["boundary_basis"]).startswith("forced"):
        review.append("forced_boundary")
    if caption_trusted and cs is not None and cs < CLIP_CAPTION_REVIEW:
        review.append("caption_consensus")
    if review:
        return "REVIEW", ";".join(review)
    if caption_trusted:
        return "KEEP_AUTO", "whisper_confident+caption_reference_trusted"
    return "KEEP_WHISPER_PRIMARY", "whisper_confident+caption_reference_low_trust"


def quality_score(row: dict, caption_trusted: bool) -> float:
    mp = 0.0 if row["mean_word_probability"] is None else float(row["mean_word_probability"])
    lp = 0.5 if row["median_segment_avg_logprob"] is None else max(
        0.0, min(1.0, (float(row["median_segment_avg_logprob"]) + 2.0) / 2.0)
    )
    ns = 0.5 if row["max_segment_no_speech_prob"] is None else 1.0 - max(
        0.0, min(1.0, float(row["max_segment_no_speech_prob"]))
    )
    dur = float(row["raw_duration_sec"])
    ds = max(0.0, 1.0 - abs(dur - TARGET_SEC) / TARGET_SEC)
    cs = row["clip_consensus_mean"]
    if caption_trusted and cs is not None:
        score = 0.45 * mp + 0.18 * lp + 0.12 * ns + 0.15 * float(cs) + 0.10 * ds
    else:
        score = 0.58 * mp + 0.20 * lp + 0.12 * ns + 0.10 * ds
    return round(max(0.0, min(1.0, score)), 6)


def make_candidates(video_id: str, words: pd.DataFrame, segments: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    words = prep_words(words)
    if words.empty:
        raise RuntimeError(f"No valid Whisper words for {video_id}")
    source_consensus = float(consensus["token_similarity"].mean()) if not consensus.empty else None
    caption_trusted = source_consensus is not None and source_consensus >= SOURCE_CAPTION_TRUST

    rows = []
    for n, (s, e, basis, bscore) in enumerate(make_spans(words), start=1):
        cw = words.iloc[s:e + 1]
        raw_start, raw_end = float(cw.iloc[0]["start_sec"]), float(cw.iloc[-1]["end_sec"])
        prev_end = float(words.iloc[s - 1]["end_sec"]) if s else None
        next_start = float(words.iloc[e + 1]["start_sec"]) if e + 1 < len(words) else None
        left_gap = max(0.0, raw_start - prev_end) if prev_end is not None else 0.24
        right_gap = max(0.0, next_start - raw_end) if next_start is not None else 0.30
        pre_pad, post_pad = min(0.12, left_gap / 2), min(0.15, right_gap / 2)
        probs = pd.to_numeric(cw["probability"], errors="coerce").dropna()
        median_lp, max_ns = segment_quality(cw, segments)
        row = {
            "video_id": video_id,
            "clip_id": f"{video_id}_C{n:06d}",
            "raw_start_sec": raw_start,
            "raw_end_sec": raw_end,
            "raw_duration_sec": raw_end - raw_start,
            "cut_start_sec": max(0.0, raw_start - pre_pad),
            "cut_end_sec": raw_end + post_pad,
            "word_count": int(len(cw)),
            "text": join_words(cw["word"].astype(str).tolist()),
            "boundary_basis": basis,
            "boundary_score": float(bscore),
            "gap_before_clip_sec": left_gap,
            "gap_after_clip_sec": right_gap,
            "mean_word_probability": float(probs.mean()) if not probs.empty else None,
            "word_probability_p10": float(probs.quantile(0.10)) if not probs.empty else None,
            "median_segment_avg_logprob": median_lp,
            "max_segment_no_speech_prob": max_ns,
            "source_consensus_mean": source_consensus,
            "clip_consensus_mean": overlap_consensus(consensus, raw_start, raw_end),
            "caption_reference_trusted": bool(caption_trusted),
        }
        row["cut_duration_sec"] = row["cut_end_sec"] - row["cut_start_sec"]
        row["decision"], row["decision_reason"] = classify(row, caption_trusted)
        row["quality_score"] = quality_score(row, caption_trusted)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    for p in (ROOT, OUT, PER_SOURCE):
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("SOURCE16 CPU SEGMENTATION PLANNER V1")
    print("NO AUDIO WILL BE CUT IN THIS STAGE")
    print(f"duration: min={MIN_SEC}s target={TARGET_SEC}s max={MAX_SEC}s absolute={ABS_MAX_SEC}s")
    print("=" * 100)

    cp = find_checkpoint_root()
    state = validate_state(cp)
    all_candidates, source_rows = [], []

    for i, video_id in enumerate(KEEP_16, start=1):
        print("=" * 100)
        print(f"SOURCE [{i:02d}/16] {video_id}", flush=True)
        src = cp / "outputs" / video_id
        for name in ("WHISPER_WORDS.csv", "WHISPER_SEGMENTS.csv", "SUMMARY.json"):
            if not (src / name).exists():
                raise FileNotFoundError(f"Missing checkpoint artifact {src / name}")

        audio = find_media(video_id)
        caption = find_media(video_id, caption=True)
        words = pd.read_csv(src / "WHISPER_WORDS.csv", low_memory=False)
        segments = pd.read_csv(src / "WHISPER_SEGMENTS.csv", low_memory=False)
        captions = parse_caption(caption)
        consensus = build_consensus(segments, captions)
        candidates = make_candidates(video_id, words, segments, consensus)
        candidates["audio_file"] = audio.name

        dst = PER_SOURCE / video_id
        dst.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(dst / "CANDIDATES.csv", index=False, encoding="utf-8-sig")
        consensus.to_csv(dst / "CONSENSUS_BUCKETS_30S.csv", index=False, encoding="utf-8-sig")

        counts = candidates["decision"].value_counts().to_dict()
        row = {
            "video_id": video_id,
            "audio_file": audio.name,
            "candidate_count": int(len(candidates)),
            "source_consensus_mean": (
                float(consensus["token_similarity"].mean()) if not consensus.empty else None
            ),
            "count_KEEP_AUTO": int(counts.get("KEEP_AUTO", 0)),
            "count_KEEP_WHISPER_PRIMARY": int(counts.get("KEEP_WHISPER_PRIMARY", 0)),
            "count_REVIEW": int(counts.get("REVIEW", 0)),
            "count_DROP": int(counts.get("DROP", 0)),
        }
        source_rows.append(row)
        (dst / "SUMMARY.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        all_candidates.append(candidates)
        print(f"SEGMENTATION_SOURCE_DONE {video_id} {counts}", flush=True)

    manifest = pd.concat(all_candidates, ignore_index=True)
    manifest.insert(0, "manifest_schema", 1)
    keep = manifest["decision"].isin(["KEEP_AUTO", "KEEP_WHISPER_PRIMARY"])
    review = manifest["decision"] == "REVIEW"
    drop = manifest["decision"] == "DROP"

    manifest.to_csv(OUT / "CUT_MANIFEST.csv", index=False, encoding="utf-8-sig")
    manifest[keep].to_csv(OUT / "CUT_MANIFEST_KEEP.csv", index=False, encoding="utf-8-sig")
    manifest[review].sort_values(
        ["quality_score", "video_id", "raw_start_sec"]
    ).to_csv(OUT / "REVIEW_QUEUE.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(source_rows).to_csv(OUT / "SOURCE_SUMMARY.csv", index=False, encoding="utf-8-sig")

    summary = {
        "stage": "SOURCE16_CPU_SEGMENTATION_PLANNER_V1",
        "manifest_schema": 1,
        "sources_processed": int(manifest["video_id"].nunique()),
        "candidates_total": int(len(manifest)),
        "decision_counts": {
            "KEEP_AUTO": int((manifest["decision"] == "KEEP_AUTO").sum()),
            "KEEP_WHISPER_PRIMARY": int((manifest["decision"] == "KEEP_WHISPER_PRIMARY").sum()),
            "REVIEW": int(review.sum()),
            "DROP": int(drop.sum()),
        },
        "duration_hours": {
            "KEEP": float(manifest.loc[keep, "cut_duration_sec"].sum()) / 3600,
            "REVIEW": float(manifest.loc[review, "cut_duration_sec"].sum()) / 3600,
            "DROP": float(manifest.loc[drop, "cut_duration_sec"].sum()) / 3600,
        },
        "parameters": {
            "min_sec": MIN_SEC,
            "target_sec": TARGET_SEC,
            "max_sec": MAX_SEC,
            "absolute_max_sec": ABS_MAX_SEC,
            "source_caption_trust": SOURCE_CAPTION_TRUST,
            "clip_caption_review": CLIP_CAPTION_REVIEW,
        },
        "inputs": {
            "media_dataset": "ahndongo/maymay-source16-media",
            "whisper_checkpoint_dataset": "ahndongo/maymay-source16-whisper-checkpoint",
            "checkpoint_stage": state["stage"],
            "model": state["config"]["model"],
            "beam_size": state["config"]["beam_size"],
        },
        "notes": [
            "Planning only: this job does not invoke ffmpeg or cut audio.",
            "Boundaries come from Whisper word timestamps, punctuation, pauses, and segment boundaries.",
            "YouTube captions are only a quality signal; low global caption agreement switches to Whisper-primary mode.",
            "Review CUT_MANIFEST.csv and REVIEW_QUEUE.csv before bulk slicing.",
            "This cheap CPU derivative is intentionally regenerated from durable media + Whisper checkpoints.",
        ],
    }
    (OUT / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("OUTPUT", OUT)
    print("SEGMENTATION_PLANNER_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
