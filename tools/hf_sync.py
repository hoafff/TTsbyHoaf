from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description="Upload a local pipeline output folder to the private HF checkpoint repo.")
    p.add_argument("local_path", help="Folder relative to repo root or absolute path.")
    p.add_argument("hf_path", help="Destination path inside hoa748/TTSbyMayMay.")
    p.add_argument("--repo", default="hoa748/TTSbyMayMay")
    p.add_argument("--message", default="Backup TTsbyHoaf pipeline stage")
    args = p.parse_args()

    local = Path(args.local_path)
    if not local.is_absolute():
        local = ROOT / local
    local = local.resolve()

    if not local.exists() or not local.is_dir():
        raise SystemExit(f"Folder not found: {local}")

    api = HfApi()
    api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=str(local),
        path_in_repo=args.hf_path,
        commit_message=args.message,
    )

    print(f"Uploaded: {local}")
    print(f"HF: {args.repo}/{args.hf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
