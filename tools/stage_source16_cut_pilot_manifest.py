from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT = ROOT / "outputs" / "source16_segmentation_qc" / "PILOT_MANIFEST.csv"
DEFAULT_QC = ROOT / "outputs" / "source16_segmentation_qc" / "QC_SUMMARY.json"
DEFAULT_STAGE = ROOT / "kaggle_stage" / "source16-cut-pilot-manifest-v1"
DATASET_ID = "ahndongo/maymay-source16-cut-pilot-manifest-v1"

KEEP_OR_REVIEW = {"KEEP_AUTO", "KEEP_WHISPER_PRIMARY", "REVIEW"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def kaggle_exe() -> str:
    found = shutil.which("kaggle")
    if found:
        return found

    win = ROOT / ".venv" / "Scripts" / "kaggle.exe"
    if win.exists():
        return str(win)

    posix = ROOT / ".venv" / "bin" / "kaggle"
    if posix.exists():
        return str(posix)

    raise FileNotFoundError("Could not locate Kaggle CLI. Activate .venv or install kaggle.")


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), check=check)


def wait_dataset_ready(kg: str, timeout_sec: int = 300) -> None:
    deadline = time.time() + timeout_sec
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        p = subprocess.run(
            [kg, "datasets", "files", DATASET_ID],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        listing = p.stdout or ""
        if p.returncode == 0 and "PILOT_MANIFEST.csv" in listing:
            print(f"KAGGLE_DATASET_READY attempt={attempt} {DATASET_ID}", flush=True)
            return
        print(f"KAGGLE_DATASET_PROCESSING attempt={attempt}", flush=True)
        time.sleep(5)
    raise RuntimeError(f"Dataset did not become ready within {timeout_sec}s: {DATASET_ID}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Validate and upload the 160-row Source16 listening-pilot manifest as a tiny private Kaggle dataset."
    )
    p.add_argument("--pilot", default=str(DEFAULT_PILOT))
    p.add_argument("--qc", default=str(DEFAULT_QC))
    p.add_argument("--stage", default=str(DEFAULT_STAGE))
    p.add_argument("--no-upload", action="store_true")
    args = p.parse_args()

    pilot = Path(args.pilot).resolve()
    qc_path = Path(args.qc).resolve()
    stage = Path(args.stage).resolve()

    if not pilot.exists():
        raise SystemExit(f"Missing pilot manifest: {pilot}")
    if not qc_path.exists():
        raise SystemExit(f"Missing QC summary: {qc_path}")

    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    if qc.get("qc_pass") is not True:
        raise SystemExit("Refusing to stage pilot because QC_SUMMARY.json does not have qc_pass=true")
    if int(qc.get("pilot_rows", -1)) != 160:
        raise SystemExit(f"Expected qc pilot_rows=160, got {qc.get('pilot_rows')}")

    df = pd.read_csv(pilot, low_memory=False)
    required = {
        "pilot_order", "pilot_reason", "video_id", "clip_id", "decision",
        "cut_start_sec", "cut_end_sec", "cut_duration_sec", "text", "audio_file",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Pilot manifest missing columns: {missing}")

    if len(df) != 160:
        raise SystemExit(f"Expected exactly 160 pilot rows, got {len(df)}")
    if df["clip_id"].astype(str).duplicated().any():
        raise SystemExit("Pilot manifest has duplicate clip_id values")

    decisions = set(df["decision"].astype(str))
    bad_decisions = sorted(decisions - KEEP_OR_REVIEW)
    if bad_decisions:
        raise SystemExit(f"Pilot contains forbidden decisions: {bad_decisions}")

    counts = df.groupby("video_id").size()
    if len(counts) != 16 or not (counts == 10).all():
        raise SystemExit(f"Pilot must contain exactly 10 rows for each of 16 sources: {counts.to_dict()}")

    starts = pd.to_numeric(df["cut_start_sec"], errors="coerce")
    ends = pd.to_numeric(df["cut_end_sec"], errors="coerce")
    durations = pd.to_numeric(df["cut_duration_sec"], errors="coerce")
    invalid = starts.isna() | ends.isna() | durations.isna() | (starts < 0) | (ends <= starts) | (durations <= 0)
    if invalid.any():
        raise SystemExit(f"Pilot contains {int(invalid.sum())} invalid intervals")

    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    staged_pilot = stage / "PILOT_MANIFEST.csv"
    shutil.copy2(pilot, staged_pilot)
    shutil.copy2(qc_path, stage / "QC_SUMMARY.json")

    info = {
        "stage": "SOURCE16_CUT_PILOT_MANIFEST_V1",
        "dataset_id": DATASET_ID,
        "rows": int(len(df)),
        "sources": int(df["video_id"].nunique()),
        "rows_per_source": {str(k): int(v) for k, v in counts.to_dict().items()},
        "decision_counts": {str(k): int(v) for k, v in df["decision"].value_counts().to_dict().items()},
        "pilot_manifest_sha256": sha256_file(staged_pilot),
        "qc_pass": True,
        "rule": "No DROP row is allowed in this pilot dataset.",
    }
    (stage / "MANIFEST_INFO.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "id": DATASET_ID,
        "title": "MayMay Source16 Cut Pilot Manifest V1",
        "licenses": [{"name": "other"}],
    }
    (stage / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(info, ensure_ascii=False, indent=2))
    print("STAGED:", stage)

    if args.no_upload:
        print("UPLOAD_SKIPPED")
        return 0

    kg = kaggle_exe()
    exists = run([kg, "datasets", "files", DATASET_ID], check=False).returncode == 0
    if exists:
        run([
            kg, "datasets", "version",
            "-p", str(stage),
            "-m", "Refresh 160-row Source16 cut/listening pilot manifest after QC pass",
            "--dir-mode", "zip",
        ])
        print("KAGGLE_DATASET_VERSIONED", DATASET_ID)
    else:
        run([
            kg, "datasets", "create",
            "-p", str(stage),
            "--dir-mode", "zip",
        ])
        print("KAGGLE_DATASET_CREATED", DATASET_ID)

    wait_dataset_ready(kg)
    print("PILOT_MANIFEST_DATASET_READY", DATASET_ID)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
