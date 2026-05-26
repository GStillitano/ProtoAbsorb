"""Reproduce TENT published results on CIFAR-10-C severity 5.

Protocol (matches github.com/DequanWang/tent):
  - Single pass: 10k images per corruption → T = 10000 // N batches
  - Accuracy measured on the SAME batch used for adaptation (logits from forward_and_adapt)
  - Model reset between corruption types (fresh load each time)
  - Sweep all 15 corruptions, report per-corruption error and mean

Published baseline (WideResNet-40-2, Adam lr=1e-3, severity=5):
  mean corruption error ≈ 11.2% (vs BN adapt ≈ 14.2%, source ≈ 18.1%)
"""
import argparse
from pathlib import Path

import torch
import yaml

from src.model import load_model
from src.data.cifar10c import load_cifar10c_data, CORRUPTIONS
from src.tta import tent
from src.device import get_device


def run_one_corruption(corruption, severity, data_dir, device, lr, N):
    x, y = load_cifar10c_data(corruption, severity, data_dir=data_dir)
    T = len(x) // N

    model = load_model(data_dir=data_dir).to(device)
    tent.configure_model(model)
    params, _ = tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    tent_model = tent.Tent(model, optimizer, steps=1, episodic=False)

    correct = 0
    for t in range(T):
        x_batch = x[t * N : (t + 1) * N].to(device)
        y_batch = y[t * N : (t + 1) * N]
        logits = tent_model(x_batch).detach().cpu()
        correct += (logits.argmax(1) == y_batch).sum().item()

    acc = correct / (T * N)
    return acc, 1.0 - acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--severity",     type=int,   default=5)
    parser.add_argument("--N",            type=int,   default=200)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--corruptions",  nargs="+",  default=None,
                        help="Subset of corruptions to run (default: all 15)")
    parser.add_argument("--data_dir",     default="./data")
    parser.add_argument("--device",       default=None)
    args = parser.parse_args()

    tent_cfg = yaml.safe_load(Path("configs/tent.yaml").read_text())
    lr     = args.lr or tent_cfg["lr"]
    device = args.device or get_device()

    corruptions = args.corruptions or CORRUPTIONS

    print(f"TENT reproduction — severity={args.severity}, N={args.N}, lr={lr}, device={device}")
    print(f"{'Corruption':<25} {'Acc':>8} {'Err':>8}")
    print("-" * 44)

    errors = []
    for corruption in corruptions:
        acc, err = run_one_corruption(
            corruption, args.severity, args.data_dir, device, lr, args.N
        )
        errors.append(err)
        print(f"{corruption:<25} {acc*100:>7.2f}% {err*100:>7.2f}%")

    mean_err = sum(errors) / len(errors)
    print("-" * 44)
    print(f"{'Mean corruption error':<25} {(1-mean_err)*100:>7.2f}% {mean_err*100:>7.2f}%")
    print(f"\nPublished TENT target: ~11.2% mean error (severity=5, Adam lr=1e-3)")


if __name__ == "__main__":
    main()
