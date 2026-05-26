"""Full sweep over sweep.yaml grid — runs phase1_adapt for every combination.

Usage:
    uv run python scripts/sweep.py --method tent
"""
import argparse
import subprocess
from itertools import product
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",   default="tent", choices=["tent", "bn_adapt"])
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--device",   default=None)
    args = parser.parse_args()

    cfg          = yaml.safe_load(Path("configs/sweep.yaml").read_text())
    corruptions  = cfg["corruption"]
    open_sets    = cfg["open_set"]
    csood_sources= cfg["csood_source"]
    seeds        = cfg["seed"]

    for corruption, open_set, csood_source, seed in product(corruptions, open_sets, csood_sources, seeds):
        if csood_source == "rome32":
            print(f"Skipping rome32 (not yet available): {corruption} open_set={open_set} seed={seed}")
            continue

        cmd = [
            "uv", "run", "python", "scripts/phase1_adapt.py",
            "--method",       args.method,
            "--corruption",   corruption,
            "--open_set",     str(open_set).lower(),
            "--csood_source", csood_source,
            "--seed",         str(seed),
            "--data_dir",     args.data_dir,
        ]
        if args.device:
            cmd += ["--device", args.device]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
