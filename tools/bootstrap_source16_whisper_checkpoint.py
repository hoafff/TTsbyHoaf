from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configs" / "source16.json"
STAGE = ROOT / "kaggle_stage" / "source16-whisper-checkpoint"
HANDLE = "ahndongo/maymay-source16-whisper-checkpoint"

MODEL_SIZE = "large-v3"
BEAM_SIZE = 1
BUCKET_SEC = 30
CHECKPOINT_SCHEMA = 1


def run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def main() -> int:
    kaggle = shutil.which("kaggle")
    if not kaggle:
        raise SystemExit("Kaggle CLI not found. Activate the project .venv first.")

    cfg = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    sources = [str(x) for x in cfg["keep_source_ids"]]

    probe = run_capture([kaggle, "datasets", "files", HANDLE])
    if probe.returncode == 0:
        print(probe.stdout.strip(), flush=True)
        print("CHECKPOINT_DATASET_ALREADY_EXISTS", HANDLE, flush=True)
        return 0

    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "outputs").mkdir(parents=True, exist_ok=True)

    state = {
        "stage": "SOURCE16_WHISPER_CONSENSUS_V1",
        "config": {
            "schema": CHECKPOINT_SCHEMA,
            "model": MODEL_SIZE,
            "beam_size": BEAM_SIZE,
            "bucket_seconds": BUCKET_SEC,
            "word_timestamps": True,
            "vad_filter": True,
            "sources": sources,
        },
        "completed": [],
        "summaries": {},
        "failures": {},
        "updated_at_unix": time.time(),
    }

    (STAGE / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "id": HANDLE,
        "title": "MayMay Source16 Whisper Checkpoint",
        "licenses": [{"name": "other"}],
    }
    (STAGE / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    create = run_capture([kaggle, "datasets", "create", "-p", str(STAGE)])
    print(create.stdout, end="" if create.stdout.endswith("\n") else "\n", flush=True)
    if create.returncode != 0:
        return create.returncode

    print("WAITING_FOR_CHECKPOINT_DATASET", HANDLE, flush=True)
    for _ in range(60):
        probe = run_capture([kaggle, "datasets", "files", HANDLE])
        if probe.returncode == 0 and "state.json" in (probe.stdout or ""):
            print("CHECKPOINT_DATASET_READY", HANDLE, flush=True)
            return 0
        time.sleep(5)

    raise SystemExit("Checkpoint dataset creation was submitted but state.json did not become visible.")


if __name__ == "__main__":
    raise SystemExit(main())
