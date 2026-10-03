from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "kaggle_jobs.json"
OUTPUT_ROOT = ROOT / "kaggle_output"


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise SystemExit(f"Missing registry: {REGISTRY_PATH}")
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def ensure_kaggle() -> str:
    exe = shutil.which("kaggle")
    if not exe:
        raise SystemExit(
            "Kaggle CLI not found. Activate .venv and run: "
            "pip install -r requirements-control.txt"
        )
    return exe


def run(cmd: list[str], *, cwd: Path | None = None) -> int:
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return proc.returncode


def get_job(name: str) -> tuple[dict, dict]:
    reg = load_registry()
    jobs = reg.get("jobs", {})
    if name not in jobs:
        available = ", ".join(sorted(jobs)) or "(none)"
        raise SystemExit(f"Unknown job '{name}'. Available: {available}")
    return reg, jobs[name]


def cmd_list(_: argparse.Namespace) -> int:
    reg = load_registry()
    jobs = reg.get("jobs", {})
    if not jobs:
        print("No jobs registered.")
        return 0

    for name, job in jobs.items():
        accel = job.get("accelerator") or "CPU"
        print(f"{name:22} {accel:18} {job['handle']}")
        desc = job.get("description")
        if desc:
            print(f"  {desc}")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    kaggle = ensure_kaggle()
    _, job = get_job(args.job)

    job_dir = ROOT / job["path"]
    metadata = job_dir / "kernel-metadata.json"
    if not metadata.exists():
        raise SystemExit(f"Missing kernel metadata: {metadata}")

    cmd = [
        kaggle,
        "kernels",
        "push",
        "-p",
        str(job_dir),
        "--timeout",
        str(args.timeout or job.get("timeout_seconds", 3600)),
    ]

    accelerator = args.accelerator or job.get("accelerator")
    if accelerator:
        cmd += ["--accelerator", accelerator]

    if args.no_run:
        cmd.append("--no-run")

    return run(cmd, cwd=ROOT)


def cmd_status(args: argparse.Namespace) -> int:
    kaggle = ensure_kaggle()
    _, job = get_job(args.job)
    rc = run([kaggle, "kernels", "status", job["handle"]], cwd=ROOT)
    if rc != 0:
        print(
            "\nHint: if this job has never been submitted, run:\n"
            f"  python .\\tools\\kaggle_bridge.py submit {args.job}"
        )
    return rc


def cmd_logs(args: argparse.Namespace) -> int:
    kaggle = ensure_kaggle()
    _, job = get_job(args.job)
    cmd = [kaggle, "kernels", "logs", job["handle"]]
    if args.follow:
        cmd += ["--follow", "--interval", str(args.interval)]
    rc = run(cmd, cwd=ROOT)
    if rc != 0:
        print(
            "\nHint: logs exist only after the job has been submitted. Run:\n"
            f"  python .\\tools\\kaggle_bridge.py submit {args.job}"
        )
    return rc


def cmd_output(args: argparse.Namespace) -> int:
    kaggle = ensure_kaggle()
    _, job = get_job(args.job)

    out_dir = Path(args.path) if args.path else OUTPUT_ROOT / args.job
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        kaggle,
        "kernels",
        "output",
        job["handle"],
        "-p",
        str(out_dir),
        "--force",
    ]

    if args.pattern:
        cmd += ["--file-pattern", args.pattern]

    rc = run(cmd, cwd=ROOT)
    if rc == 0:
        print(f"Output: {out_dir}")
    return rc


def cmd_files(args: argparse.Namespace) -> int:
    kaggle = ensure_kaggle()
    _, job = get_job(args.job)
    return run(
        [kaggle, "kernels", "files", job["handle"], "-v", "--page-size", "200"],
        cwd=ROOT,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TTsbyHoaf Kaggle control bridge (submit/status/logs/output)."
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("list", help="List registered jobs.")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("submit", help="Push and run a registered Kaggle job.")
    sp.add_argument("job")
    sp.add_argument("--timeout", type=int)
    sp.add_argument(
        "--accelerator",
        help="Override accelerator, e.g. NvidiaTeslaT4. Leave unset for CPU.",
    )
    sp.add_argument("--no-run", action="store_true")
    sp.set_defaults(func=cmd_submit)

    sp = sub.add_parser("status", help="Show latest job status.")
    sp.add_argument("job")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("logs", help="Show or follow latest job logs.")
    sp.add_argument("job")
    sp.add_argument("-f", "--follow", action="store_true")
    sp.add_argument("--interval", type=int, default=10)
    sp.set_defaults(func=cmd_logs)

    sp = sub.add_parser("files", help="List output files for the latest run.")
    sp.add_argument("job")
    sp.set_defaults(func=cmd_files)

    sp = sub.add_parser("output", help="Download latest job output.")
    sp.add_argument("job")
    sp.add_argument("-p", "--path")
    sp.add_argument("--pattern")
    sp.set_defaults(func=cmd_output)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
