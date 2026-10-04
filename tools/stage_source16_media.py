from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configs" / "source16.json"
DEFAULT_STAGE = ROOT / "kaggle_stage" / "source16-media"
DEFAULT_COOKIE = ROOT / "cookies.txt"
DATASET_ID = "ahndongo/maymay-source16-media"


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def load_ids() -> list[str]:
    cfg = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    return [str(x) for x in cfg["keep_source_ids"]]


def locate(stage: Path, video_id: str) -> tuple[Path | None, Path | None]:
    caption = stage / f"{video_id}.vi.json3"
    audio = None
    for p in sorted(stage.glob(f"{video_id}.*")):
        if p.name == caption.name:
            continue
        if p.suffix.lower() in {".part", ".ytdl", ".json3"}:
            continue
        if p.is_file():
            audio = p
            break
    return audio, caption if caption.exists() else None


def try_cmd(cmd: list[str]) -> bool:
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode == 0


def preflight(base: list[str], cookie: Path, video_id: str) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"

    print("LOCAL_YOUTUBE_PREFLIGHT_ANON", video_id, flush=True)
    anon_cmd = base + [
        "--no-playlist",
        "--no-warnings",
        "--skip-download",
        "--print", "%(id)s",
        url,
    ]
    if try_cmd(anon_cmd):
        print("LOCAL_YOUTUBE_PREFLIGHT_OK mode=anonymous", video_id, flush=True)
        return "anonymous"

    print("LOCAL_YOUTUBE_PREFLIGHT_COOKIE", video_id, flush=True)
    cookie_cmd = base + [
        "--cookies", str(cookie),
        "--no-playlist",
        "--no-warnings",
        "--skip-download",
        "--extractor-args", "youtube:player_client=default,web_embedded",
        "--print", "%(id)s",
        url,
    ]
    if try_cmd(cookie_cmd):
        print("LOCAL_YOUTUBE_PREFLIGHT_OK mode=cookie", video_id, flush=True)
        return "cookie"

    raise RuntimeError(
        "YouTube preflight failed in both anonymous and cookie modes. "
        "Refresh browser cookies only if anonymous access is also blocked."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=str(DEFAULT_STAGE))
    ap.add_argument("--cookies", default=str(DEFAULT_COOKIE))
    args = ap.parse_args()

    stage = Path(args.stage).resolve()
    cookie = Path(args.cookies).resolve()
    stage.mkdir(parents=True, exist_ok=True)

    if not cookie.exists():
        raise SystemExit(f"Missing cookies file: {cookie}")

    yt = shutil.which("yt-dlp")
    if not yt:
        yt = shutil.which("yt_dlp")
    if not yt:
        # venv normally exposes yt-dlp, but module invocation is the most portable fallback.
        base = [sys.executable, "-m", "yt_dlp"]
    else:
        base = [yt]

    ids = load_ids()
    auth_mode = preflight(base, cookie, ids[0])
    failures: list[dict] = []
    manifest: list[dict] = []

    print("=" * 100)
    print("SOURCE16 LOCAL MEDIA STAGING")
    print("sources:", len(ids))
    print("stage:", stage)
    print("=" * 100)

    for idx, video_id in enumerate(ids, start=1):
        existing_audio, existing_caption = locate(stage, video_id)
        if existing_audio and existing_caption:
            print(f"[{idx:02d}/{len(ids):02d}] SKIP_READY {video_id} {existing_audio.name}", flush=True)
            manifest.append({
                "video_id": video_id,
                "audio": existing_audio.name,
                "caption": existing_caption.name,
                "status": "READY",
            })
            continue

        print(f"[{idx:02d}/{len(ids):02d}] DOWNLOAD {video_id}", flush=True)
        url = f"https://www.youtube.com/watch?v={video_id}"
        cmd = base + [
            "--no-playlist",
            "--retries", "5",
            "--fragment-retries", "5",
        ]
        if auth_mode == "cookie":
            cmd += [
                "--cookies", str(cookie),
                "--extractor-args", "youtube:player_client=default,web_embedded",
            ]
        cmd += [
            "--write-auto-subs",
            "--sub-langs", "vi",
            "--sub-format", "json3",
            "-f", "bestaudio/best",
            "-o", str(stage / "%(id)s.%(ext)s"),
            url,
        ]

        try:
            run(cmd)
            audio, caption = locate(stage, video_id)
            if not audio or not caption:
                raise RuntimeError(
                    f"Incomplete source after yt-dlp: audio={audio}, caption={caption}"
                )
            manifest.append({
                "video_id": video_id,
                "audio": audio.name,
                "caption": caption.name,
                "status": "READY",
            })
            print(f"READY {video_id}: {audio.name} + {caption.name}", flush=True)
        except Exception as exc:
            failures.append({
                "video_id": video_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(f"FAILED {video_id}: {exc}", flush=True)

    (stage / "MANIFEST.json").write_text(
        json.dumps(
            {
                "dataset_id": DATASET_ID,
                "youtube_auth_mode": auth_mode,
                "sources_total": len(ids),
                "sources_ready": len(manifest),
                "sources_failed": len(failures),
                "sources": manifest,
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = {
        "id": DATASET_ID,
        "title": "MayMay Source16 Media",
        "licenses": [{"name": "other"}],
    }
    (stage / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print(f"READY {len(manifest)}/{len(ids)}")
    print(f"FAILED {len(failures)}/{len(ids)}")
    print("MANIFEST:", stage / "MANIFEST.json")
    print("=" * 100)

    return 0 if not failures and len(manifest) == len(ids) else 2


if __name__ == "__main__":
    raise SystemExit(main())
