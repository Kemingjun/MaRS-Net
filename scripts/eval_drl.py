import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
METHOD_DIRS = {
    "marsnet": REPO_ROOT / "marsnet",
    "hdrl": REPO_ROOT / "baselines" / "drl" / "hdrl",
    "tdrl": REPO_ROOT / "baselines" / "drl" / "tdrl",
}


def _resolve_repo_path(raw_path):
    path = Path(raw_path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation entrypoint for MaRS-Net and DRL baselines.")
    parser.add_argument("--method", choices=METHOD_DIRS, default="marsnet")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--decode_strategy", choices=["greedy", "sample", "bs"], default="greedy")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--val_size", type=int, default=10000)
    parser.add_argument("--no_cuda", action="store_true")
    args, passthrough = parser.parse_known_args()

    method_dir = METHOD_DIRS[args.method]
    cmd = [
        sys.executable,
        "eval.py",
        str(_resolve_repo_path(args.dataset)),
        "--model",
        str(_resolve_repo_path(args.model)),
        "--decode_strategy",
        args.decode_strategy,
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--val_size",
        str(args.val_size),
    ]
    if args.width is not None:
        cmd.extend(["--width", str(args.width)])
    if args.no_cuda:
        cmd.append("--no_cuda")
    cmd.extend(passthrough)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=method_dir, check=True)


if __name__ == "__main__":
    main()
