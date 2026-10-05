from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs" / "source16_full_keep_audit" / "MASTER_KEEP_AUDIT.csv"
BUILD_AUDIT = ROOT / "tools" / "build_source16_full_keep_audit.py"
LOCAL_OUT = ROOT / "outputs" / "source16_group_sample"
STAGE = ROOT / "kaggle_stage" / "source16-group-sample-manifest-v1"

DATASET_ID = "ahndongo/maymay-source16-group-sample-manifest-v1"
PER_GROUP = 5
EXPECTED_TOTAL = 15
TARGET_SAMPLE_RATE = 22050

GROUP_ACCEPT = "ACCEPT_CAPTION_CANDIDATE"
GROUP_ALIGN = "REVIEW_ALIGNMENT"
GROUP_WHISPER = "REVIEW_WHISPER_PRIMARY"
GROUPS = [GROUP_ACCEPT, GROUP_ALIGN, GROUP_WHISPER]


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


def ensure_audit() -> None:
    if AUDIT.exists():
        return
    print("MASTER_AUDIT_MISSING -> building 22,150-row audit first", flush=True)
    subprocess.run([sys.executable, str(BUILD_AUDIT)], cwd=str(ROOT), check=True)
    if not AUDIT.exists():
        raise RuntimeError(f"Master audit still missing after build: {AUDIT}")


def choose_spread_distinct_sources(
    df: pd.DataFrame,
    count: int,
    score_col: str,
) -> pd.DataFrame:
    if len(df) < count:
        raise RuntimeError(f"Not enough rows to sample: have={len(df)} need={count}")

    g = df.copy()
    g["_score"] = pd.to_numeric(g[score_col], errors="coerce")
    fallback = float(g["_score"].dropna().median()) if g["_score"].notna().any() else 0.0
    g["_score"] = g["_score"].fillna(fallback)
    g = g.sort_values(["_score", "video_id", "raw_start_sec"], kind="stable").reset_index(drop=True)

    if count == 1:
        targets = [0]
    else:
        targets = [round(i * (len(g) - 1) / (count - 1)) for i in range(count)]

    chosen_idx: list[int] = []
    used_sources: set[str] = set()

    for target in targets:
        order = sorted(range(len(g)), key=lambda i: (abs(i - target), i))
        pick = None
        for i in order:
            if i in chosen_idx:
                continue
            source = str(g.iloc[i]["video_id"])
            if source not in used_sources:
                pick = i
                break
        if pick is None:
            for i in order:
                if i not in chosen_idx:
                    pick = i
                    break
        if pick is None:
            raise RuntimeError("Could not choose deterministic sample row")
        chosen_idx.append(pick)
        used_sources.add(str(g.iloc[pick]["video_id"]))

    out = g.iloc[chosen_idx].drop(columns=["_score"]).copy()
    return out.reset_index(drop=True)


def choose_whisper_primary(df: pd.DataFrame, count: int) -> pd.DataFrame:
    sources = list(dict.fromkeys(df["video_id"].astype(str).tolist()))
    if len(sources) < count:
        raise RuntimeError(
            f"Expected at least {count} KEEP_WHISPER_PRIMARY sources, got {len(sources)}"
        )

    rows = []
    for video_id in sources[:count]:
        src = df[df["video_id"].astype(str) == video_id].copy()
        src["_q"] = pd.to_numeric(src["quality_score"], errors="coerce")
        src = src.sort_values(["_q", "raw_start_sec"], kind="stable").reset_index(drop=True)
        row = src.iloc[len(src) // 2].drop(labels=["_q"]).to_dict()
        rows.append(row)
    return pd.DataFrame(rows)


def build_sample(audit: pd.DataFrame) -> pd.DataFrame:
    accepted = audit[audit["audit_decision"].astype(str) == GROUP_ACCEPT].copy()
    review = audit[audit["audit_decision"].astype(str) == GROUP_ALIGN].copy()
    whisper = audit[audit["audit_decision"].astype(str) == GROUP_WHISPER].copy()

    if len(accepted) != 5746:
        raise RuntimeError(f"Expected 5746 ACCEPT_CAPTION_CANDIDATE rows, got {len(accepted)}")
    if len(review) != 11749:
        raise RuntimeError(f"Expected 11749 REVIEW_ALIGNMENT rows, got {len(review)}")
    if len(whisper) != 4655:
        raise RuntimeError(f"Expected 4655 REVIEW_WHISPER_PRIMARY rows, got {len(whisper)}")

    a = choose_spread_distinct_sources(accepted, PER_GROUP, "proposal_similarity")
    b = choose_spread_distinct_sources(review, PER_GROUP, "proposal_similarity")
    c = choose_whisper_primary(whisper, PER_GROUP)

    batches = []
    for group_name, frame in ((GROUP_ACCEPT, a), (GROUP_ALIGN, b), (GROUP_WHISPER, c)):
        frame = frame.copy()
        frame["audit_group"] = group_name
        batches.append(frame)

    out = pd.concat(batches, ignore_index=True)
    out.insert(0, "sample_order", range(1, len(out) + 1))
    out["whisper_text"] = out["text"].fillna("").astype(str)
    out["caption_candidate_text"] = out["proposal_text"].fillna("").astype(str)
    out["display_text"] = out["whisper_text"]
    mask = out["audit_group"].eq(GROUP_ACCEPT)
    out.loc[mask, "display_text"] = out.loc[mask, "caption_candidate_text"]

    if len(out) != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL} sample rows, got {len(out)}")
    counts = out["audit_group"].value_counts().to_dict()
    for group in GROUPS:
        if int(counts.get(group, 0)) != PER_GROUP:
            raise RuntimeError(f"Wrong sample count for {group}: {counts.get(group, 0)}")
    if out["clip_id"].astype(str).duplicated().any():
        raise RuntimeError("Sample contains duplicate clip_id")
    return out


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
        if p.returncode == 0 and "SAMPLE_MANIFEST.csv" in listing:
            print(f"KAGGLE_DATASET_READY attempt={attempt} {DATASET_ID}", flush=True)
            return
        print(f"KAGGLE_DATASET_PROCESSING attempt={attempt}", flush=True)
        time.sleep(5)
    raise RuntimeError(f"Dataset did not become ready: {DATASET_ID}")


def main() -> int:
    ensure_audit()
    audit = pd.read_csv(AUDIT, low_memory=False)
    if len(audit) != 22150:
        raise SystemExit(f"Expected 22150 master-audit rows, got {len(audit)}")

    sample = build_sample(audit)
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    manifest = LOCAL_OUT / "SAMPLE_MANIFEST.csv"
    sample.to_csv(manifest, index=False, encoding="utf-8-sig")

    summary = {
        "stage": "SOURCE16_GROUP_SAMPLE_MANIFEST_V1",
        "rows": int(len(sample)),
        "per_group": PER_GROUP,
        "group_counts": {
            str(k): int(v) for k, v in sample["audit_group"].value_counts().to_dict().items()
        },
        "sources_by_group": {
            group: sample.loc[sample["audit_group"] == group, "video_id"].astype(str).tolist()
            for group in GROUPS
        },
        "target_audio_format": {
            "container": "wav",
            "codec": "pcm_s16le",
            "channels": 1,
            "sample_rate_hz": TARGET_SAMPLE_RATE,
        },
        "manifest_sha256": sha256_file(manifest),
        "purpose": (
            "Human spot-check of 5 real clips from each current audit group before "
            "further automatic reclassification of the 22,150 original KEEP pool."
        ),
    }
    (LOCAL_OUT / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, STAGE / "SAMPLE_MANIFEST.csv")
    shutil.copy2(LOCAL_OUT / "SUMMARY.json", STAGE / "SAMPLE_SUMMARY.json")
    (STAGE / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": DATASET_ID,
                "title": "MayMay Source16 Group Sample Manifest V1",
                "licenses": [{"name": "other"}],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    kg = kaggle_exe()
    exists = run([kg, "datasets", "files", DATASET_ID], check=False).returncode == 0
    if exists:
        run(
            [
                kg,
                "datasets",
                "version",
                "-p",
                str(STAGE),
                "-m",
                "Refresh 15-clip Source16 group sample manifest",
                "--dir-mode",
                "zip",
            ]
        )
    else:
        run([kg, "datasets", "create", "-p", str(STAGE), "--dir-mode", "zip"])
    wait_dataset_ready(kg)
    print("SOURCE16_GROUP_SAMPLE_MANIFEST_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
