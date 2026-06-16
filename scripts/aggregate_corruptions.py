"""Aggregate exp1_auroc results across the 15 CIFAR-10-C corruptions for a
given csOOD source, for Tent and NOVA-TTA.

Prints a plain table to stdout and a LaTeX `tabular` block ready to paste into
the report.

Usage:
    uv run python scripts/aggregate_corruptions.py --csood_source rome32
    uv run python scripts/aggregate_corruptions.py --csood_source svhn_c
"""
import argparse
import json
from pathlib import Path

CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform",
    "fog", "frost", "gaussian_noise", "glass_blur", "impulse_noise",
    "jpeg_compression", "motion_blur", "pixelate", "shot_noise",
    "snow", "zoom_blur",
]


def _read(method: str, corruption: str, csood: str, seed: int) -> dict | None:
    p = Path("results") / method / f"{corruption}_{csood}_open_seed{seed}" / "exp1_auroc" / "results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csood_source", default="rome32")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for c in CORRUPTIONS:
        t = _read("tent", c, args.csood_source, args.seed)
        n = _read("nova-tta", c, args.csood_source, args.seed)
        if t is None or n is None:
            continue
        rows.append({
            "corruption": c,
            "tent_auroc_0": t["auroc"][0],
            "tent_auroc_T": t["auroc"][-1],
            "nova_auroc_0": n["auroc"][0],
            "nova_auroc_T": n["auroc"][-1],
            "tent_acc_0":   t["acc_csid"][0],
            "tent_acc_T":   t["acc_csid"][-1],
            "nova_acc_0":   n["acc_csid"][0],
            "nova_acc_T":   n["acc_csid"][-1],
        })

    if not rows:
        print(f"No results found for csood_source={args.csood_source}")
        return

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / len(rows)

    print(f"\n=== Rome32 sweep ({args.csood_source}, seed={args.seed}, {len(rows)} corruptions) ===\n")
    header = f"{'corruption':<22} {'Tent AUROC':>14} {'NOVA AUROC':>14} {'Tent Acc':>14} {'NOVA Acc':>14}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['corruption']:<22} "
              f"{r['tent_auroc_0']:.2f}->{r['tent_auroc_T']:.2f}     "
              f"{r['nova_auroc_0']:.2f}->{r['nova_auroc_T']:.2f}     "
              f"{r['tent_acc_0']:.2f}->{r['tent_acc_T']:.2f}     "
              f"{r['nova_acc_0']:.2f}->{r['nova_acc_T']:.2f}")
    print("-" * len(header))
    print(f"{'mean':<22} "
          f"{mean('tent_auroc_0'):.2f}->{mean('tent_auroc_T'):.2f}     "
          f"{mean('nova_auroc_0'):.2f}->{mean('nova_auroc_T'):.2f}     "
          f"{mean('tent_acc_0'):.2f}->{mean('tent_acc_T'):.2f}     "
          f"{mean('nova_acc_0'):.2f}->{mean('nova_acc_T'):.2f}")

    # LaTeX block (compatible with the existing tab:corruptions formatting).
    print("\n% --- LaTeX ---")
    def fmt(a: float, b: float, bold: bool) -> str:
        s = f"${a:.2f}\\to{b:.2f}$"
        return f"\\textbf{{{s}}}" if bold else s
    for r in rows:
        tA0, tAT = r["tent_auroc_0"], r["tent_auroc_T"]
        nA0, nAT = r["nova_auroc_0"], r["nova_auroc_T"]
        tC0, tCT = r["tent_acc_0"],   r["tent_acc_T"]
        nC0, nCT = r["nova_acc_0"],   r["nova_acc_T"]
        # Replace the literal underscore in \texttt with \_ for LaTeX safety.
        name = r["corruption"].replace("_", r"\_")
        cells = [
            fmt(tA0, tAT, tAT >= nAT),
            fmt(nA0, nAT, nAT >= tAT),
            fmt(tC0, tCT, tCT >= nCT),
            fmt(nC0, nCT, nCT >= tCT),
        ]
        print(f"    \\texttt{{{name}}} & " + " & ".join(cells) + r" \\")
    mean_cells = [
        fmt(mean("tent_auroc_0"), mean("tent_auroc_T"), mean("tent_auroc_T") >= mean("nova_auroc_T")),
        fmt(mean("nova_auroc_0"), mean("nova_auroc_T"), mean("nova_auroc_T") >= mean("tent_auroc_T")),
        fmt(mean("tent_acc_0"),   mean("tent_acc_T"),   mean("tent_acc_T")   >= mean("nova_acc_T")),
        fmt(mean("nova_acc_0"),   mean("nova_acc_T"),   mean("nova_acc_T")   >= mean("tent_acc_T")),
    ]
    print(r"    \midrule")
    print("    mean & " + " & ".join(mean_cells) + r" \\")


if __name__ == "__main__":
    main()
