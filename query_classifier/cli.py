import argparse
import json
import sys

import torch

from .api import classify


def _auto_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="query-classify",
        description="Run the query classifier on one or more texts.",
    )
    parser.add_argument(
        "texts",
        nargs="+",
        help="Input text(s) to classify.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (cpu, cuda, cuda:0, ...). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output.",
    )

    args = parser.parse_args(argv)

    device = args.device or _auto_device()
    if device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but not available.")

    try:
        results = classify(args.texts, max_length=args.max_length, device=device)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    indent = 2 if args.pretty else None
    for item in results:
        sys.stdout.write(json.dumps(item, ensure_ascii=False, indent=indent) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
