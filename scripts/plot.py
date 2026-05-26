"""Visualise experiment results from JSON files.

Reads from results/, writes figures to figures/.

Usage:
    # Exp 1 — overlay multiple streams (must share same corruption/open_set/seed)
    uv run python scripts/plot.py --exp 1 \
        --streams tent/gaussian_noise_svhn_c_open_seed0 \
                  bn_adapt/gaussian_noise_svhn_c_open_seed0

    # Exp 2 or 3 — single stream
    uv run python scripts/plot.py --exp 2 --streams tent/gaussian_noise_svhn_c_open_seed0
    uv run python scripts/plot.py --exp 3 --streams tent/gaussian_noise_svhn_c_open_seed0
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",     required=True, choices=["1", "2", "3"])
    parser.add_argument("--streams", nargs="+",     required=True)
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args()

    out_root = Path(args.out_dir)

    if args.exp == "1":
        from src.viz.exp1 import plot
        stream_results = {}
        for sid in args.streams:
            p = Path("results") / sid / "exp1_auroc" / "results.json"
            stream_results[sid] = json.loads(p.read_text())
        # Shared stream name = part after method/ in first stream
        stream_name = "/".join(args.streams[0].split("/")[1:])
        out_path = out_root / stream_name / "exp1_auroc.png"
        plot(stream_results, out_path)

    elif args.exp == "2":
        from src.viz.exp2 import plot
        sid = args.streams[0]
        p = Path("results") / sid / "exp2_geometry" / "results.json"
        results = json.loads(p.read_text())
        plot(results, out_root / sid)

    elif args.exp == "3":
        from src.viz.exp3 import plot
        sid = args.streams[0]
        p = Path("results") / sid / "exp3_layerwise" / "results.json"
        results = json.loads(p.read_text())
        plot(results, out_root / sid)


if __name__ == "__main__":
    main()
