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


def main():
    parser = argparse.ArgumentParser(description="Unified training entrypoint for MaRS-Net and DRL baselines.")
    parser.add_argument("--method", choices=METHOD_DIRS, default="marsnet")
    args, passthrough = parser.parse_known_args()

    method_dir = METHOD_DIRS[args.method]
    cmd = [sys.executable, "run.py", *passthrough]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=method_dir, check=True)


if __name__ == "__main__":
    main()
