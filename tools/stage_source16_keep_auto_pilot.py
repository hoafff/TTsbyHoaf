from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "kaggle_output" / "source16-segmentation" / "source16_segmentation" / "outputs" / "CUT_MANIFEST.csv"
QC_SUMMARY = ROOT / "outputs" / "source16_segmentation_qc" / "QC_SUMMARY.json"
LOCAL_OUT = ROOT / "outputs" / "source16_keep_auto_pilot"
STAGE = ROOT / "kaggle_stage" / "source16-keep-auto-pilot-manifest-v1"

DATASET_ID = "ahndongo/maymay-source16-keep-auto-pilot-manifest-v1"
TARGET_ROWS = 160
TARGET_SAMPLE_RATE = 22050
EXPECTED_KEEP_AUTO = 17495


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
    raise FileNotFoundError("Kaggle CLI not found")


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), check=check)


def choose_evenly(group: pd.DataFrame, count: int, reason: str) -> list[dict]:
    if group.empty or count <= 0:
        return []
    g = group.sort_values(["quality_score", "raw_start_sec"], kind="stable").reset_index(drop=True)
    if len(g) <= count:
        idxs = list(range(len(g)))
    elif count == 1:
        idxs = [len(g) // 2]
    else:
        idxs = sorted(set(round(i * (len(g) - 1) / (count - 1)) for i in range(count)))
    rows = []
    for i in idxs:
        row = g.iloc[i].to_dict()
        row["pilot_reason"] = reason
        rows.append(row)
    return rows


def build_pilot(df: pd.DataFrame) -> pd.DataFrame:
    keep = df[df["decision"].astype(str) == "KEEP_AUTO"].copy()
    if len(keep) != EXPECTED_KEEP_AUTO:
        raise RuntimeError("Expected %d KEEP_AUTO rows, got %d" % (EXPECTED_KEEP_AUTO, len(keep)))

    sources = list(dict.fromkeys(keep["video_id"].astype(str).tolist()))
    if len(sources) != 11:
        raise RuntimeError("Expected 11 KEEP_AUTO source groups, got %d: %r" % (len(sources), sources))

    base = TARGET_ROWS // len(sources)
    remainder = TARGET_ROWS % len(sources)
    chosen: list[dict] = []

    for source_index, video_id in enumerate(sources):
        quota = base + (1 if source_index < remainder else 0)
        src = keep[keep["video_id"].astype(str) == video_id].copy()
        candidates: list[dict] = []

        candidates.extend(choose_evenly(src, max(1, quota - 3), "keep_auto_quality_spread"))

        shortest = src.sort_values(
            ["cut_duration_sec", "quality_score"], ascending=[True, True], kind="stable"
        ).iloc[0].to_dict()
        shortest["pilot_reason"] = "keep_auto_shortest"
        candidates.append(shortest)

        longest = src.sort_values(
            ["cut_duration_sec", "quality_score"], ascending=[False, True], kind="stable"
        ).iloc[0].to_dict()
        longest["pilot_reason"] = "keep_auto_longest"
        candidates.append(longest)

        weakest = src.sort_values(
            ["boundary_score", "quality_score", "raw_start_sec"], kind="stable"
        ).iloc[0].to_dict()
        weakest["pilot_reason"] = "keep_auto_weakest_boundary"
        candidates.append(weakest)

        seen: set[str] = set()
        selected: list[dict] = []
        for row in candidates:
            clip_id = str(row["clip_id"])
            if clip_id in seen:
                continue
            seen.add(clip_id)
            selected.append(row)
            if len(selected) >= quota:
                break

        if len(selected) < quota:
            fill = src[~src["clip_id"].astype(str).isin(seen)].sort_values(
                ["quality_score", "raw_start_sec"], kind="stable"
            )
            for _, r in fill.iterrows():
                row = r.to_dict()
                row["pilot_reason"] = "keep_auto_fill"
                selected.append(row)
                seen.add(str(row["clip_id"]))
                if len(selected) >= quota:
                    break

        if len(selected) != quota:
            raise RuntimeError("Could not fill quota for %s: %d/%d" % (video_id, len(selected), quota))
        chosen.extend(selected)

    pilot = pd.DataFrame(chosen)
    if len(pilot) != TARGET_ROWS:
        raise RuntimeError("Expected %d pilot rows, got %d" % (TARGET_ROWS, len(pilot)))
    if set(pilot["decision"].astype(str)) != {"KEEP_AUTO"}:
        raise RuntimeError("Pilot is not KEEP_AUTO-only")
    if pilot["clip_id"].astype(str).duplicated().any():
        raise RuntimeError("Pilot contains duplicate clip_id")
    pilot.insert(0, "pilot_order", range(1, len(pilot) + 1))
    return pilot


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
            print("KAGGLE_DATASET_READY attempt=%d %s" % (attempt, DATASET_ID), flush=True)
            return
        print("KAGGLE_DATASET_PROCESSING attempt=%d" % attempt, flush=True)
        time.sleep(5)
    raise RuntimeError("Dataset did not become ready: %s" % DATASET_ID)


def main() -> int:
    if not MANIFEST.exists():
        raise SystemExit("Missing segmentation manifest: %s" % MANIFEST)
    if not QC_SUMMARY.exists():
        raise SystemExit("Missing QC summary: %s" % QC_SUMMARY)

    qc = json.loads(QC_SUMMARY.read_text(encoding="utf-8"))
    if qc.get("qc_pass") is not True:
        raise SystemExit("Refusing pilot because segmentation QC is not PASS")
    if int(qc.get("decision_counts", {}).get("KEEP_AUTO", -1)) != EXPECTED_KEEP_AUTO:
        raise SystemExit("QC summary KEEP_AUTO count does not match expected %d" % EXPECTED_KEEP_AUTO)

    df = pd.read_csv(MANIFEST, low_memory=False)
    pilot = build_pilot(df)

    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    local_manifest = LOCAL_OUT / "PILOT_MANIFEST.csv"
    pilot.to_csv(local_manifest, index=False, encoding="utf-8-sig")

    counts = pilot.groupby("video_id").size().to_dict()
    summary = {
        "stage": "SOURCE16_KEEP_AUTO_PILOT_MANIFEST_V1",
        "target_rows": TARGET_ROWS,
        "rows": int(len(pilot)),
        "sources": int(pilot["video_id"].nunique()),
        "decision_counts": {str(k): int(v) for k, v in pilot["decision"].value_counts().to_dict().items()},
        "rows_per_source": {str(k): int(v) for k, v in counts.items()},
        "target_audio_format": {
            "container": "wav",
            "codec": "pcm_s16le",
            "channels": 1,
            "sample_rate_hz": TARGET_SAMPLE_RATE,
        },
        "pilot_manifest_sha256": sha256_file(local_manifest),
        "purpose": "Validate KEEP_AUTO segmentation at the final 22050 Hz training audio format before bulk cutting.",
    }
    (LOCAL_OUT / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_manifest, STAGE / "PILOT_MANIFEST.csv")
    shutil.copy2(LOCAL_OUT / "SUMMARY.json", STAGE / "PILOT_SUMMARY.json")
    (STAGE / "dataset-metadata.json").write_text(
        json.dumps({
            "id": DATASET_ID,
            "title": "MayMay Source16 KEEP AUTO Pilot Manifest V1",
            "licenses": [{"name": "other"}],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    kg = kaggle_exe()
    exists = run([kg, "datasets", "files", DATASET_ID], check=False).returncode == 0
    if exists:
        run([kg, "datasets", "version", "-p", str(STAGE), "-m",
             "Refresh KEEP_AUTO-only 22050 Hz pilot manifest", "--dir-mode", "zip"])
    else:
        run([kg, "datasets", "create", "-p", str(STAGE), "--dir-mode", "zip"])
    wait_dataset_ready(kg)
    print("KEEP_AUTO_PILOT_MANIFEST_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
