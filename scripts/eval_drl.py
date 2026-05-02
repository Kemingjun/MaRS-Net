import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTANCE_ROOT = REPO_ROOT / "Instance"
METHOD_DIRS = {
    "marsnet": REPO_ROOT / "marsnet",
    "hdrl": REPO_ROOT / "baselines" / "drl" / "hdrl",
    "tdrl": REPO_ROOT / "baselines" / "drl" / "tdrl",
}


def _resolve_repo_path(raw_path):
    path = Path(raw_path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _infer_model_size(model_path):
    for part in Path(model_path).parts[::-1]:
        if part.startswith("size_"):
            try:
                return int(part.split("_")[1])
            except (IndexError, ValueError):
                pass
    return None


def _resolve_dataset(raw_dataset, model_path):
    path = Path(raw_dataset)
    candidates = []
    size = _infer_model_size(model_path)

    if raw_dataset in {"Synthetic_Dataset", "Industrial_Dataset"}:
        if size is None:
            raise ValueError(f"Cannot infer checkpoint size from model path: {model_path}")
        dataset_dir = (
            INSTANCE_ROOT / "Synthetic_Dataset" / f"size_{size}_uniform"
            if raw_dataset == "Synthetic_Dataset"
            else INSTANCE_ROOT / "Industrial_Dataset" / f"size_{size}"
        )
        if dataset_dir.exists():
            return dataset_dir.resolve()
        raise FileNotFoundError(f"Dataset family '{raw_dataset}' does not contain instances for size_{size}: {dataset_dir}")

    if size is not None:
        if path.name == f"size_{size}_uniform":
            candidates.append(INSTANCE_ROOT / "Synthetic_Dataset" / path.name)
        elif path.name == f"size_{size}":
            candidates.append(INSTANCE_ROOT / "Industrial_Dataset" / path.name)

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([REPO_ROOT / path, INSTANCE_ROOT / path])

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot resolve dataset '{raw_dataset}'. Use Synthetic_Dataset, Industrial_Dataset, "
        "or an explicit path under Instance/."
    )


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
    model_path = _resolve_repo_path(args.model)
    dataset_path = _resolve_dataset(args.dataset, model_path)
    cmd = [
        sys.executable,
        "eval.py",
        str(dataset_path),
        "--model",
        str(model_path),
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
