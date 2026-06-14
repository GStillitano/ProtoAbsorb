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
from src.tta import tent, nova_tta
from src.model import classifier_weights
from src.device import get_device
from src.seed import set_seed


def build_optimizer(name: str, params, lr: float, momentum: float = 0.9):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {name}")


def ckpt_dir(method: str, corruption: str, csood_source: str, open_set: bool, seed: int) -> Path:
    tag = "open" if open_set else "closed"
    return Path("checkpoints") / method / f"{corruption}_{csood_source}_{tag}_seed{seed}"


def _parse_bool(v: str | None) -> bool | None:
    if v is None:
        return None
    return v.lower() in ("true", "1", "yes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",       required=True, choices=["tent", "bn_adapt", "nova-tta"])
    parser.add_argument("--corruption",   default=None)
    parser.add_argument("--severity",     type=int,   default=None)
    parser.add_argument("--open_set",     type=str,   default=None,
                        help="true/false — open-set (balanced ID/OOD) or closed-set (ID only)")
    parser.add_argument("--csood_source", default=None)
    parser.add_argument("--N",            type=int,   default=None)
    parser.add_argument("--T",            type=int,   default=None)
    parser.add_argument("--seed",         type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--data_dir",     default="./data")
    parser.add_argument("--device",       default=None)
    args = parser.parse_args()

    stream_cfg   = yaml.safe_load(Path("configs/stream.yaml").read_text())
    tent_cfg     = yaml.safe_load(Path("configs/tent.yaml").read_text())
    nova_tta_cfg  = yaml.safe_load(Path("configs/nova-tta.yaml").read_text())
    diag_cfg     = yaml.safe_load(Path("configs/diagnostic.yaml").read_text())
    method_cfg   = nova_tta_cfg if args.method == "nova-tta" else tent_cfg

    corruption   = args.corruption   or stream_cfg["corruption"]
    severity     = args.severity     or stream_cfg["severity"]
    open_set     = _parse_bool(args.open_set) if args.open_set is not None else stream_cfg["open_set"]
    csood_source = args.csood_source or stream_cfg["csood_source"].split()[0]
    N            = args.N            or stream_cfg["N"]
    T            = args.T            or stream_cfg["T"]
    seed         = args.seed         if args.seed is not None else stream_cfg["seed"]
    lr           = args.lr           or method_cfg["lr"]
    device       = args.device       or get_device()

    set_seed(seed)

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
        N=N, T=T, open_set=open_set, seed=seed,
    )

    # ── Load and configure model ───────────────────────────────────────────────
    model = load_model(data_dir=args.data_dir).to(device)

    out_dir = ckpt_dir(args.method, corruption, csood_source, open_set, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "base_model.pt")
    save_ckpt(extract(model, t=0), out_dir / "theta_000.pt")

    scorer = optimizer = gmm = W_cpu = None
    if args.method == "tent":
        tent.configure_model(model)
        params, _ = tent.collect_params(model)
        optimizer = torch.optim.Adam(params, lr=lr)
        tent_model = tent.Tent(model, optimizer, steps=1, episodic=False)

    elif args.method == "nova-tta":
        # Adapted model: BN affine trainable.
        nova_tta.configure_model(model)
        params, _ = tent.collect_params(model)
        optimizer = build_optimizer(method_cfg["optimizer"], params, lr,
                                    momentum=method_cfg.get("momentum", 0.9))
        # Frozen scorer: separate model, original weights, BN batch-stat adapt, no grad.
        scorer = load_model(data_dir=args.data_dir).to(device)
        scorer.requires_grad_(False)
        W_cpu  = classifier_weights(scorer).cpu()
        gmm    = nova_tta.GmmScorer(
            components   = nova_tta_cfg["gmm_components"],
            accumulate   = nova_tta_cfg["gmm_accumulate"],
            window       = nova_tta_cfg["gmm_window"],
            warm_start   = nova_tta_cfg["gmm_warm_start"],
            reg_covar    = nova_tta_cfg["gmm_reg_covar"],
            id_component = nova_tta_cfg["id_component"],
            random_state = seed,
        )

    # ── Phase 1 loop ──────────────────────────────────────────────────────────
    for t, x_batch in stream:
        x_batch = x_batch.to(device)

        if args.method == "tent":
            tent_model(x_batch)
        elif args.method == "nova-tta":
            # LR warmup: scale base lr by ramp(t/K), t is 1-indexed.
            f = nova_tta.warmup_factor(t, nova_tta_cfg["warmup_K"],
                                      nova_tta_cfg["warmup_shape"], nova_tta_cfg["warmup_exp_tau"])
            for g in optimizer.param_groups:
                g["lr"] = lr * f
            nova_tta.forward_and_adapt(
                x_batch, model, scorer, W_cpu, gmm, optimizer,
                l1_weight=nova_tta_cfg["l1_weight"], device=device,
            )
        else:  # bn_adapt
            with torch.no_grad():
                model(x_batch)

        active_model = tent_model.model if args.method == "tent" else model
        save_ckpt(extract(active_model, t=t), out_dir / f"theta_{t:03d}.pt")

    # ── Write meta.json ───────────────────────────────────────────────────────
    meta = {
        "method": args.method, "corruption": corruption, "severity": severity,
        "open_set": open_set, "N": N, "T": T, "csood_source": csood_source, "seed": seed,
        "batches": stream.to_meta_list(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()
