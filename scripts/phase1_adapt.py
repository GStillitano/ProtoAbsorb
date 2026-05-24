"""Phase 1: run a TTA method on one stream and save BN affine checkpoints."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from src.model import load_model
from src.bn_affine import extract, save as save_ckpt
from src.data.cifar10c import load_cifar10c_data
from src.data.svhnc import load_svhn_c
from src.data.pools import DataPools
from src.data.stream import build_stream
from src.tta import tent


def ckpt_dir(method: str, corruption: str, csood_source: str, alpha: float, seed: int) -> Path:
    return Path("checkpoints") / method / f"{corruption}_{csood_source}_{alpha:.2f}_seed{seed}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",       required=True, choices=["tent", "bn_adapt"])
    parser.add_argument("--corruption",   default=None)
    parser.add_argument("--severity",     type=int,   default=None)
    parser.add_argument("--alpha",        type=float, default=None)
    parser.add_argument("--csood_source", default=None)
    parser.add_argument("--N",            type=int,   default=None)
    parser.add_argument("--T",            type=int,   default=None)
    parser.add_argument("--seed",         type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--data_dir",     default="./data")
    parser.add_argument("--device",       default=None)
    args = parser.parse_args()

    stream_cfg = yaml.safe_load(Path("configs/stream.yaml").read_text())
    tent_cfg   = yaml.safe_load(Path("configs/tent.yaml").read_text())
    diag_cfg   = yaml.safe_load(Path("configs/diagnostic.yaml").read_text())

    corruption   = args.corruption   or stream_cfg["corruption"]
    severity     = args.severity     or stream_cfg["severity"]
    alpha        = args.alpha        if args.alpha is not None else stream_cfg["alpha"]
    csood_source = args.csood_source or stream_cfg["csood_source"].split()[0]  # strip comment
    N            = args.N            or stream_cfg["N"]
    T            = args.T            or stream_cfg["T"]
    seed         = args.seed         if args.seed is not None else stream_cfg["seed"]
    lr           = args.lr           or tent_cfg["lr"]
    device       = args.device       or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ─────────────────────────────────────────────────────────────
    x_csid, y_csid = load_cifar10c_data(corruption, severity, data_dir=args.data_dir)

    if csood_source == "svhn_c":
        x_csood, _ = load_svhn_c(corruption, severity, data_dir=args.data_dir)
    elif csood_source == "rome32":
        from src.data.rome32 import load_rome32_c
        x_csood, _ = load_rome32_c(
            folder=str(Path(args.data_dir) / "rome32/raw"),
            corruption=corruption, severity=severity,
        )
    else:
        raise ValueError(f"Unknown csood_source: {csood_source}")

    pools = DataPools(
        n_csid=len(x_csid), n_csood=len(x_csood),
        n_diag_csid=diag_cfg["n_csid"], n_diag_csood=diag_cfg["n_csood"],
        seed=seed,
    )
    stream = build_stream(
        x_csid=x_csid, y_csid=y_csid, x_csood=x_csood,
        adapt_csid_indices=pools.csid_adapt,
        adapt_csood_indices=pools.csood_adapt,
        N=N, T=T, alpha=alpha, seed=seed,
    )

    # ── Load and configure model ───────────────────────────────────────────────
    model = load_model(data_dir=args.data_dir).to(device)

    out_dir = ckpt_dir(args.method, corruption, csood_source, alpha, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "base_model.pt")
    save_ckpt(extract(model, t=0), out_dir / "theta_000.pt")

    if args.method == "tent":
        tent.configure_model(model)
        params, _ = tent.collect_params(model)
        optimizer = torch.optim.Adam(params, lr=lr)
        tent_model = tent.Tent(model, optimizer, steps=1, episodic=False)

    # ── Phase 1 loop ──────────────────────────────────────────────────────────
    for t, x_batch in stream:
        x_batch = x_batch.to(device)

        if args.method == "tent":
            tent_model(x_batch)
        else:  # bn_adapt
            with torch.no_grad():
                model(x_batch)

        active_model = tent_model.model if args.method == "tent" else model
        save_ckpt(extract(active_model, t=t), out_dir / f"theta_{t:03d}.pt")

    # ── Write meta.json ───────────────────────────────────────────────────────
    meta = {
        "method": args.method, "corruption": corruption, "severity": severity,
        "alpha": alpha, "N": N, "T": T, "csood_source": csood_source, "seed": seed,
        "batches": stream.to_meta_list(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()
