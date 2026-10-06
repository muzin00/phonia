"""Execute one frozen test worker in its pinned model environment."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from test_inference import run_ecapa, run_vowels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("vowels", "ecapa"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    if args.method == "vowels":
        run_vowels(args.run_dir)
    else:
        if args.model_dir is None:
            parser.error("ECAPA requires --model-dir")
        run_ecapa(args.run_dir, args.model_dir)


if __name__ == "__main__":
    main()
