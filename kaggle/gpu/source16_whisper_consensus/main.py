from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
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

MODEL_SIZE = "large-v3"
BUCKET_SEC = 30
BEAM_SIZE = 1
PROGRESS_EVERY_SEGMENTS = 500
ROOT = Path("/kaggle/working/source16_whisper_consensus")
AUDIO_DIR = ROOT / "audio"
CAPTION_DIR = ROOT / "youtube_json3"
OUT_DIR = ROOT / "outputs"
SECRETS_DIR = Path("/kaggle/input/maymay-runtime-secrets")
COOKIE_PATH = SECRETS_DIR / "cookies.txt"
HF_TOKEN_PATH = SECRETS_DIR / "hf_token.txt"
COOKIE_WORK_DIR = ROOT / "runtime_cookie_copies"
DOWNLOAD_LOCK = threading.Lock()

CHECKPOINT_HANDLE = "ahndongo/maymay-source16-whisper-checkpoint"
CHECKPOINT_SCHEMA = 1
CHECKPOINT_DIR = Path("/kaggle/working/source16_whisper_checkpoint")
CHECKPOINT_OUTPUT_DIR = CHECKPOINT_DIR / "outputs"
CHECKPOINT_STATE_PATH = CHECKPOINT_DIR / "state.json"
CHECKPOINT_UPLOAD_RETRIES = 12
CHECKPOINT_UPLOAD_RETRY_SEC = 10


def find_input_file(filename: str, preferred: Path) -> Path | None:
    if preferred.exists():
        return preferred

    root = Path("/kaggle/input")
    if not root.exists():
        print("KAGGLE_INPUT_MISSING", flush=True)
        return None

    # Kaggle can expose attached datasets either directly under /kaggle/input/<slug>
    # or under newer nested layouts such as /kaggle/input/datasets/... .
    matches = sorted(
        p for p in root.rglob(filename)
        if p.is_file()
    )
    if matches:
        for p in matches[:10]:
            print(f"DISCOVERED_INPUT_CANDIDATE {filename}: {p}", flush=True)
        return matches[0]

    mounted = []
    for p in root.rglob("*"):
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) <= 4:
            mounted.append(str(p))
        if len(mounted) >= 100:
            break
    print("KAGGLE_INPUT_TREE_SAMPLE:", mounted, flush=True)
    return None

for p in (ROOT, AUDIO_DIR, CAPTION_DIR, OUT_DIR, COOKIE_WORK_DIR):
    p.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def checkpoint_config() -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": MODEL_SIZE,
        "beam_size": BEAM_SIZE,
        "bucket_seconds": BUCKET_SEC,
        "word_timestamps": True,
        "vad_filter": True,
        "sources": KEEP_16,
    }


def fresh_checkpoint_state() -> dict:
    return {
        "stage": "SOURCE16_WHISPER_CONSENSUS_V1",
        "config": checkpoint_config(),
        "completed": [],
        "summaries": {},
        "failures": {},
        "updated_at_unix": time.time(),
    }


def write_checkpoint_state(state: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at_unix"] = time.time()
    tmp = CHECKPOINT_STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(CHECKPOINT_STATE_PATH)


def persist_checkpoint(state: dict, reason: str) -> None:
    import kagglehub

    write_checkpoint_state(state)
    last_exc: Exception | None = None

    for attempt in range(1, CHECKPOINT_UPLOAD_RETRIES + 1):
        try:
            print(
                f"CHECKPOINT_UPLOAD_STARTED attempt={attempt}/{CHECKPOINT_UPLOAD_RETRIES} reason={reason}",
                flush=True,
            )
            kagglehub.dataset_upload(
                CHECKPOINT_HANDLE,
                str(CHECKPOINT_DIR),
                version_notes=reason,
            )
            print(
                "CHECKPOINT_SAVED",
                f"completed={len(state.get('completed', []))}/{len(KEEP_16)}",
                f"handle={CHECKPOINT_HANDLE}",
                flush=True,
            )
            return
        except Exception as exc:
            last_exc = exc
            print(
                "CHECKPOINT_UPLOAD_RETRY",
                f"attempt={attempt}/{CHECKPOINT_UPLOAD_RETRIES}",
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt < CHECKPOINT_UPLOAD_RETRIES:
                time.sleep(CHECKPOINT_UPLOAD_RETRY_SEC)

    raise RuntimeError(
        f"Checkpoint upload failed after {CHECKPOINT_UPLOAD_RETRIES} attempts: {last_exc}"
    )


def checkpoint_source(state: dict, video_id: str, summary: dict) -> None:
    src = OUT_DIR / video_id
    if not src.exists():
        raise RuntimeError(f"Missing source output before checkpoint: {src}")

    # Keep only GPU-expensive / expensive-to-recompute artifacts.
    # Captions already live in the durable media dataset; pause/consensus tables
    # are deterministic CPU derivatives and are rebuilt after restore.
    keep_files = (
        "SUMMARY.json",
        "WHISPER_SEGMENTS.csv",
        "WHISPER_WORDS.csv",
    )

    dst = CHECKPOINT_OUTPUT_DIR / video_id
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for name in keep_files:
        source_file = src / name
        if not source_file.exists():
            raise RuntimeError(
                f"Missing required expensive artifact before checkpoint: {source_file}"
            )
        shutil.copyfile(source_file, dst / name)

    completed = set(str(x) for x in state.get("completed", []))
    completed.add(video_id)
    state["completed"] = [v for v in KEEP_16 if v in completed]
    state.setdefault("summaries", {})[video_id] = summary
    state.setdefault("failures", {}).pop(video_id, None)

    persist_checkpoint(state, f"completed {video_id}")


def record_checkpoint_failure(state: dict, video_id: str, failure: dict) -> None:
    state.setdefault("failures", {})[video_id] = failure
    persist_checkpoint(state, f"failed {video_id}")


def find_mounted_checkpoint_dir() -> Path | None:
    root = Path("/kaggle/input")
    if not root.exists():
        return None

    candidates = sorted(
        p.parent
        for p in root.rglob("state.json")
        if "maymay-source16-whisper-checkpoint" in str(p).lower()
    )
    return candidates[0] if candidates else None


def restore_checkpoint() -> tuple[dict, bool]:
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)
    CHECKPOINT_DIR.parent.mkdir(parents=True, exist_ok=True)

    print("CHECKPOINT_RESTORE_STARTED", CHECKPOINT_HANDLE, flush=True)
    mounted = find_mounted_checkpoint_dir()
    if mounted is None:
        raise RuntimeError(
            "Checkpoint dataset is not mounted. Bootstrap the private checkpoint "
            "dataset locally and attach ahndongo/maymay-source16-whisper-checkpoint "
            "before running this kernel."
        )

    shutil.copytree(mounted, CHECKPOINT_DIR)
    print("CHECKPOINT_RESTORE_MOUNTED", mounted, flush=True)
    created = False

    if not CHECKPOINT_STATE_PATH.exists():
        raise RuntimeError(
            f"Checkpoint dataset exists but state.json is missing: {CHECKPOINT_HANDLE}"
        )

    state = json.loads(CHECKPOINT_STATE_PATH.read_text(encoding="utf-8"))
    if state.get("config") != checkpoint_config():
        raise RuntimeError(
            "Checkpoint configuration mismatch. Refusing to mix results from a different "
            f"model/beam/source configuration. remote={state.get('config')} "
            f"current={checkpoint_config()}"
        )

    completed = [str(x) for x in state.get("completed", [])]
    unknown = sorted(set(completed) - set(KEEP_16))
    if unknown:
        raise RuntimeError(f"Checkpoint contains unknown completed sources: {unknown}")

    required = {
        "SUMMARY.json",
        "WHISPER_SEGMENTS.csv",
        "WHISPER_WORDS.csv",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for video_id in completed:
        src = CHECKPOINT_OUTPUT_DIR / video_id
        if not src.exists():
            raise RuntimeError(
                f"Checkpoint marks {video_id} complete but output directory is missing."
            )
        missing = sorted(name for name in required if not (src / name).exists())
        if missing:
            raise RuntimeError(
                f"Checkpoint for {video_id} is incomplete; missing files: {missing}"
            )

        dst = OUT_DIR / video_id
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        for name in required:
            shutil.copyfile(src / name, dst / name)

        # Rebuild cheap deterministic artifacts from durable media captions +
        # checkpointed Whisper timestamps. No GPU/Whisper inference is repeated.
        _, caption_path = find_source_input(video_id)
        captions = parse_youtube_json3(caption_path)
        whisper_segments = pd.read_csv(dst / "WHISPER_SEGMENTS.csv")
        words = pd.read_csv(dst / "WHISPER_WORDS.csv")
        pause_features = build_pause_features(words)
        consensus = build_consensus_buckets(whisper_segments, captions)

        captions.to_csv(
            dst / "YOUTUBE_CAPTION_EVENTS.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pause_features.to_csv(
            dst / "PAUSE_FEATURES.csv",
            index=False,
            encoding="utf-8-sig",
        )
        consensus.to_csv(
            dst / "CONSENSUS_BUCKETS_30S.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if not consensus.empty:
            consensus.nsmallest(20, "token_similarity").to_csv(
                dst / "CONSENSUS_BUCKETS_LOWEST_20.csv",
                index=False,
                encoding="utf-8-sig",
            )
        (dst / caption_path.name).write_bytes(caption_path.read_bytes())

        if video_id not in state.get("summaries", {}):
            state.setdefault("summaries", {})[video_id] = json.loads(
                (dst / "SUMMARY.json").read_text(encoding="utf-8")
            )

    print(
        "CHECKPOINT_RESTORED",
        f"completed={len(completed)}/{len(KEEP_16)}",
        f"pending={len(KEEP_16) - len(completed)}",
        flush=True,
    )
    return state, created


def clean_ws(text: str) -> str:
    text = html.unescape(str(text))
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()


def lexical_norm(text: str) -> str:
    text = clean_ws(text).lower()
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹĐđ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    aa = lexical_norm(a).split()
    bb = lexical_norm(b).split()
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb, autojunk=False).ratio()


def install_deps() -> None:
    run([
        sys.executable, "-m", "pip", "install", "-q", "-U",
        "faster-whisper==1.2.1", "av>=11,<19", "kagglehub>=1.0.0"
    ])


def configure_hf_token() -> None:
    token_path = find_input_file("hf_token.txt", HF_TOKEN_PATH)
    if token_path is not None:
        token = token_path.read_text(encoding="utf-8-sig").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            print("HF_AUTH_READY", flush=True)
            return
    print("HF_AUTH_MISSING; continuing with unauthenticated HF access.", flush=True)


def require_cookie() -> Path:
    cookie_path = find_input_file("cookies.txt", COOKIE_PATH)
    if cookie_path is None:
        raise FileNotFoundError(
            "Missing private cookies.txt under /kaggle/input. "
            "Ensure ahndongo/maymay-runtime-secrets is READY before submitting."
        )
    print("COOKIE_AUTH_READY:", cookie_path, flush=True)
    return cookie_path


def find_source_input(video_id: str) -> tuple[Path, Path]:
    root = Path("/kaggle/input")
    if not root.exists():
        raise FileNotFoundError("/kaggle/input is missing")

    caption_matches = sorted(
        p for p in root.rglob(f"{video_id}.vi.json3")
        if p.is_file()
    )
    audio_matches = []
    for p in root.rglob(f"{video_id}.*"):
        if not p.is_file():
            continue
        if p.name == f"{video_id}.vi.json3":
            continue
        if p.suffix.lower() in {".json3", ".json", ".txt", ".csv", ".part", ".ytdl"}:
            continue
        audio_matches.append(p)
    audio_matches = sorted(audio_matches)

    if not audio_matches:
        raise FileNotFoundError(f"Missing staged audio for {video_id} under /kaggle/input")
    if not caption_matches:
        raise FileNotFoundError(f"Missing staged caption for {video_id} under /kaggle/input")

    print(f"STAGED_SOURCE_READY {video_id}: {audio_matches[0]} | {caption_matches[0]}", flush=True)
    return audio_matches[0], caption_matches[0]


def download_source(video_id: str, cookie_path: Path) -> tuple[Path, Path]:
    url = f"https://www.youtube.com/watch?v={video_id}"

    for old in AUDIO_DIR.glob(f"{video_id}.*"):
        old.unlink(missing_ok=True)
    for old in CAPTION_DIR.glob(f"{video_id}.*"):
        old.unlink(missing_ok=True)

    local_cookie = COOKIE_WORK_DIR / f"{video_id}.txt"
    shutil.copyfile(cookie_path, local_cookie)

    audio_tmpl = str(AUDIO_DIR / "%(id)s.%(ext)s")
    common = [
        sys.executable, "-m", "yt_dlp",
        "--cookies", str(local_cookie),
        "--no-playlist",
        "--no-warnings",
        "--retries", "3",
        "--extractor-args", "youtube:player_client=default,web_embedded",
    ]

    try:
        with DOWNLOAD_LOCK:
            print("DOWNLOAD_STARTED", video_id, flush=True)
            run(common + [
                "-f", "bestaudio/best",
                "-o", audio_tmpl,
                url,
            ])

            run(common + [
                "--skip-download",
                "--write-auto-subs",
                "--sub-langs", "vi",
                "--sub-format", "json3",
                "-o", str(CAPTION_DIR / "%(id)s.%(ext)s"),
                url,
            ])
            print("DOWNLOAD_DONE", video_id, flush=True)
    finally:
        local_cookie.unlink(missing_ok=True)

    audio_matches = sorted(
        p for p in AUDIO_DIR.glob(f"{video_id}.*")
        if p.is_file()
    )
    caption_matches = sorted(CAPTION_DIR.glob(f"{video_id}.vi.json3"))

    if not audio_matches:
        raise RuntimeError(f"Audio download failed for {video_id}")
    if not caption_matches:
        raise RuntimeError(f"Vietnamese auto-caption JSON3 missing for {video_id}")

    return audio_matches[0], caption_matches[0]


def youtube_preflight(cookie_path: Path) -> None:
    video_id = KEEP_16[0]
    url = f"https://www.youtube.com/watch?v={video_id}"
    local_cookie = COOKIE_WORK_DIR / "preflight.txt"
    shutil.copyfile(cookie_path, local_cookie)
    try:
        print("YOUTUBE_PREFLIGHT_STARTED", video_id, flush=True)
        run([
            sys.executable, "-m", "yt_dlp",
            "--cookies", str(local_cookie),
            "--no-playlist",
            "--no-warnings",
            "--skip-download",
            "--extractor-args", "youtube:player_client=default,web_embedded",
            "--print", "%(id)s",
            url,
        ])
        print("YOUTUBE_PREFLIGHT_OK", video_id, flush=True)
    finally:
        local_cookie.unlink(missing_ok=True)


def parse_youtube_json3(path: Path) -> pd.DataFrame:
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
        rows.append({
            "event_index": event_index,
            "start_sec": start_ms / 1000.0,
            "end_sec": (start_ms + dur_ms) / 1000.0,
            "duration_sec": dur_ms / 1000.0,
            "text": text,
        })
    return pd.DataFrame(rows)


def detect_gpu_indices() -> list[int]:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        ids = [int(x.strip()) for x in p.stdout.splitlines() if x.strip().isdigit()]
        return ids or [0]
    except Exception:
        return [0]


def load_model(gpu_indices: list[int]):
    from faster_whisper import WhisperModel
    print("MODEL_LOAD_STARTED", MODEL_SIZE, "GPUS", gpu_indices, flush=True)
    model = WhisperModel(
        MODEL_SIZE,
        device="cuda",
        device_index=gpu_indices if len(gpu_indices) > 1 else gpu_indices[0],
        compute_type="float16",
        num_workers=max(1, len(gpu_indices)),
    )
    print("MODEL_LOADED", flush=True)
    return model


def transcribe(model, audio_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    segments_gen, info = model.transcribe(
        str(audio_path),
        language="vi",
        beam_size=BEAM_SIZE,
        word_timestamps=True,
        vad_filter=True,
    )

    segment_rows = []
    word_rows = []

    for seg_no, seg in enumerate(segments_gen, start=1):
        if seg_no == 1:
            print("TRANSCRIBE_FIRST_SEGMENT", audio_path.name, flush=True)
        if seg_no % PROGRESS_EVERY_SEGMENTS == 0:
            print(
                f"TRANSCRIBE_PROGRESS {audio_path.name} segments={seg_no} t={float(seg.end):.1f}s",
                flush=True,
            )

        segment_rows.append({
            "segment_id": int(seg.id),
            "start_sec": float(seg.start),
            "end_sec": float(seg.end),
            "duration_sec": float(seg.end - seg.start),
            "text": clean_ws(seg.text),
            "avg_logprob": float(seg.avg_logprob),
            "no_speech_prob": float(seg.no_speech_prob),
            "compression_ratio": float(seg.compression_ratio),
        })

        prev_end = None
        for word_index, word in enumerate(seg.words or []):
            start = float(word.start)
            end = float(word.end)
            word_rows.append({
                "segment_id": int(seg.id),
                "word_index": int(word_index),
                "word": clean_ws(word.word),
                "start_sec": start,
                "end_sec": end,
                "duration_sec": max(0.0, end - start),
                "gap_before_sec": None if prev_end is None else max(0.0, start - prev_end),
                "probability": float(word.probability),
            })
            prev_end = end

    meta = {
        "language": str(info.language),
        "language_probability": float(info.language_probability),
        "duration_sec": float(info.duration),
        "duration_after_vad_sec": float(info.duration_after_vad),
        "model": MODEL_SIZE,
        "word_timestamps": True,
        "vad_filter": True,
        "beam_size": BEAM_SIZE,
    }
    return pd.DataFrame(segment_rows), pd.DataFrame(word_rows), meta


def bucket_text(df: pd.DataFrame, text_col: str, bucket_sec: int = BUCKET_SEC) -> dict[int, str]:
    if df.empty:
        return {}
    mid = (pd.to_numeric(df["start_sec"]) + pd.to_numeric(df["end_sec"])) / 2.0
    tmp = df.copy()
    tmp["_bucket"] = (mid // bucket_sec).astype(int)
    return {
        int(k): " ".join(g[text_col].astype(str).tolist())
        for k, g in tmp.groupby("_bucket", sort=True)
    }


def build_consensus_buckets(
    whisper_segments: pd.DataFrame,
    captions: pd.DataFrame,
) -> pd.DataFrame:
    yt_b = bucket_text(captions, "text")
    ws_b = bucket_text(whisper_segments, "text")
    common = sorted(set(yt_b) & set(ws_b))

    rows = []
    for bucket in common:
        score = similarity(yt_b[bucket], ws_b[bucket])
        if score >= 0.85:
            band = "HIGH"
        elif score >= 0.65:
            band = "MID"
        else:
            band = "LOW"

        rows.append({
            "bucket": int(bucket),
            "start_sec": float(bucket * BUCKET_SEC),
            "end_sec": float((bucket + 1) * BUCKET_SEC),
            "youtube_text": yt_b[bucket],
            "whisper_text": ws_b[bucket],
            "token_similarity": float(score),
            "match_band": band,
        })
    return pd.DataFrame(rows)


def build_pause_features(words: pd.DataFrame) -> pd.DataFrame:
    if words.empty:
        return words.copy()

    out = words.sort_values(["start_sec", "end_sec"], kind="stable").reset_index(drop=True).copy()
    out["prev_end_sec_global"] = pd.to_numeric(out["end_sec"], errors="coerce").shift(1)
    out["gap_before_global_sec"] = (
        pd.to_numeric(out["start_sec"], errors="coerce") - out["prev_end_sec_global"]
    ).clip(lower=0)
    out["cross_segment_boundary"] = (
        pd.to_numeric(out["segment_id"], errors="coerce")
        != pd.to_numeric(out["segment_id"], errors="coerce").shift(1)
    )

    valid = out["gap_before_global_sec"].dropna()
    if len(valid) >= 5:
        out.loc[valid.index, "gap_before_global_percentile"] = valid.rank(
            method="average", pct=True
        )
    else:
        out["gap_before_global_percentile"] = None

    return out

def quantiles(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "p999": float(s.quantile(0.999)),
        "max": float(s.max()),
    }


def process_one(model, video_id: str) -> dict:
    print("=" * 100, flush=True)
    print("SOURCE:", video_id, flush=True)

    source_dir = OUT_DIR / video_id
    source_dir.mkdir(parents=True, exist_ok=True)

    audio_path, caption_path = find_source_input(video_id)
    captions = parse_youtube_json3(caption_path)
    print("TRANSCRIBE_STARTED", video_id, flush=True)
    whisper_segments, words, whisper_meta = transcribe(model, audio_path)
    print("TRANSCRIBE_DONE", video_id, flush=True)
    consensus = build_consensus_buckets(whisper_segments, captions)
    pause_features = build_pause_features(words)

    captions.to_csv(source_dir / "YOUTUBE_CAPTION_EVENTS.csv", index=False, encoding="utf-8-sig")
    whisper_segments.to_csv(source_dir / "WHISPER_SEGMENTS.csv", index=False, encoding="utf-8-sig")
    words.to_csv(source_dir / "WHISPER_WORDS.csv", index=False, encoding="utf-8-sig")
    pause_features.to_csv(source_dir / "PAUSE_FEATURES.csv", index=False, encoding="utf-8-sig")
    consensus.to_csv(source_dir / "CONSENSUS_BUCKETS_30S.csv", index=False, encoding="utf-8-sig")
    if not consensus.empty:
        consensus.nsmallest(20, "token_similarity").to_csv(
            source_dir / "CONSENSUS_BUCKETS_LOWEST_20.csv",
            index=False,
            encoding="utf-8-sig",
        )
    (source_dir / caption_path.name).write_bytes(caption_path.read_bytes())

    match_counts = (
        consensus["match_band"].value_counts().to_dict()
        if not consensus.empty
        else {}
    )

    summary = {
        "video_id": video_id,
        "audio_file": audio_path.name,
        "youtube_caption_events": int(len(captions)),
        "whisper_segments": int(len(whisper_segments)),
        "whisper_words": int(len(words)),
        "match_band_counts": {str(k): int(v) for k, v in match_counts.items()},
        "bucket_similarity_mean": (
            float(consensus["token_similarity"].mean()) if not consensus.empty else None
        ),
        "bucket_seconds": BUCKET_SEC,
        "gap_before_global_sec_quantiles": quantiles(
            pause_features.get("gap_before_global_sec", pd.Series(dtype=float))
        ),
        "word_duration_sec_quantiles": quantiles(pause_features.get("duration_sec", pd.Series(dtype=float))),
        "whisper": whisper_meta,
        "status": "OK",
    }

    (source_dir / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> int:
    print("=" * 100)
    print("SOURCE16 FASTER-WHISPER CONSENSUS")
    print("sources:", len(KEEP_16))
    print("model:", MODEL_SIZE)
    print("beam_size:", BEAM_SIZE)
    print("=" * 100)

    install_deps()
    configure_hf_token()

    checkpoint_state, checkpoint_created = restore_checkpoint()

    # Prove checkpoint persistence before loading Whisper or spending GPU time.
    persist_checkpoint(
        checkpoint_state,
        "bootstrap checkpoint backend" if checkpoint_created else "checkpoint write preflight",
    )

    completed = set(str(x) for x in checkpoint_state.get("completed", []))
    pending = [video_id for video_id in KEEP_16 if video_id not in completed]
    print(
        "RESUME_PLAN",
        f"completed={len(completed)}/{len(KEEP_16)}",
        f"pending={len(pending)}",
        flush=True,
    )

    # Completed sources no longer need staged media. Fail fast only for pending work.
    for video_id in pending:
        find_source_input(video_id)
    print("STAGED_MEDIA_PREFLIGHT_OK", len(pending), flush=True)

    all_summaries = [
        checkpoint_state["summaries"][video_id]
        for video_id in KEEP_16
        if video_id in completed
    ]

    if pending:
        gpu_indices = detect_gpu_indices()
        print("GPU_INDICES:", gpu_indices, flush=True)
        model = load_model(gpu_indices)

        max_parallel = max(1, len(gpu_indices))
        print("PARALLEL_TRANSCRIPTIONS:", max_parallel, flush=True)

        queue = deque(pending)
        pool = ThreadPoolExecutor(max_workers=max_parallel)
        in_flight: dict = {}

        def submit_next() -> None:
            if not queue:
                return
            video_id = queue.popleft()
            in_flight[pool.submit(process_one, model, video_id)] = video_id

        for _ in range(min(max_parallel, len(queue))):
            submit_next()

        try:
            while in_flight:
                future = next(as_completed(tuple(in_flight)))
                video_id = in_flight.pop(future)

                try:
                    summary = future.result()
                except Exception as exc:
                    failure = {
                        "video_id": video_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    record_checkpoint_failure(checkpoint_state, video_id, failure)
                    print(
                        f"SOURCE_FAILED [{len(checkpoint_state.get('completed', [])):02d}/{len(KEEP_16):02d}] "
                        f"{video_id}: {repr(exc)}",
                        flush=True,
                    )
                else:
                    # Do not launch the next source until this result is safely outside
                    # /kaggle/working in the remote checkpoint dataset.
                    checkpoint_source(checkpoint_state, video_id, summary)
                    all_summaries = [
                        checkpoint_state["summaries"][v]
                        for v in KEEP_16
                        if v in set(checkpoint_state.get("completed", []))
                    ]
                    print(
                        f"SOURCE_COMPLETE [{len(checkpoint_state.get('completed', [])):02d}/{len(KEEP_16):02d}] "
                        f"{video_id}",
                        flush=True,
                    )

                submit_next()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    else:
        gpu_indices = []
        max_parallel = 0
        print("ALL_SOURCES_ALREADY_CHECKPOINTED; skipping Whisper.", flush=True)

    completed = set(str(x) for x in checkpoint_state.get("completed", []))
    failures = [
        checkpoint_state.get("failures", {}).get(video_id)
        for video_id in KEEP_16
        if video_id not in completed
        and checkpoint_state.get("failures", {}).get(video_id) is not None
    ]

    overall = {
        "stage": "SOURCE16_WHISPER_CONSENSUS_V1",
        "sources_total": len(KEEP_16),
        "sources_ok": len(completed),
        "sources_failed": len(failures),
        "model": MODEL_SIZE,
        "word_timestamps": True,
        "beam_size": BEAM_SIZE,
        "bucket_seconds": BUCKET_SEC,
        "gpu_indices": gpu_indices,
        "parallel_transcriptions": max_parallel,
        "media_input": "private Kaggle dataset ahndongo/maymay-source16-media",
        "checkpoint_dataset": CHECKPOINT_HANDLE,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "sources": all_summaries,
        "failures": failures,
    }

    (OUT_DIR / "SUMMARY_ALL.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Keep the aggregate summary in the checkpoint dataset too.
    CHECKPOINT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        OUT_DIR / "SUMMARY_ALL.json",
        CHECKPOINT_OUTPUT_DIR / "SUMMARY_ALL.json",
    )
    persist_checkpoint(checkpoint_state, "aggregate summary")

    print("=" * 100)
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("OUTPUT:", OUT_DIR)
    return 0 if len(completed) == len(KEEP_16) and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
