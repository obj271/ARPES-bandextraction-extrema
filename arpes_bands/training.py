"""Patch training with unweighted pixel BCE and Adam, as in the paper."""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from torch.nn import functional as F

from .model import BandSRCNN
from .simulation import SimulationConfig, make_bank


def pixel_metrics(probability, target):
    """Pixel average precision and 1-pixel-tolerant ridge precision/recall.

    Anti-aliased targets are binarized at 0.5 for metrics only, never for BCE.
    """
    truth = target >= 0.5
    predicted = probability >= 0.5
    neighborhood = (1,) * (truth.ndim - 2) + (3, 3)
    hit_p = predicted & maximum_filter(truth, size=neighborhood, mode="constant")
    hit_t = truth & maximum_filter(predicted, size=neighborhood, mode="constant")
    precision = float(hit_p.sum() / max(predicted.sum(), 1))
    recall = float(hit_t.sum() / max(truth.sum(), 1))
    # Group tied scores to avoid optimistic AP from arbitrary tie ordering.
    scores = probability.ravel()
    labels = truth.ravel()
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(scores) - 1]
    tp = np.cumsum(labels)[ends]
    p = tp / (ends + 1)
    r = tp / max(labels.sum(), 1)
    ap = float(np.sum(np.diff(np.r_[0, r]) * p))
    return {"average_precision": ap, "precision_1px": precision,
            "recall_1px": recall,
            "f1_1px": 2 * precision * recall / max(precision + recall, 1e-12)}


@torch.no_grad()
def evaluate(model, bank, device):
    model.eval()
    probabilities, targets, losses = [], [], []
    for sample in bank:
        x = torch.from_numpy(sample["intensity"])[None, None].to(device)
        y = torch.from_numpy(sample["label"][6:-6, 6:-6].copy())[None, None].to(device)
        logits = model(x)
        losses.append(float(F.binary_cross_entropy_with_logits(logits, y).item()))
        probabilities.append(logits.sigmoid().cpu().numpy()[0, 0])
        targets.append(y.cpu().numpy()[0, 0])
    metrics = pixel_metrics(np.stack(probabilities), np.stack(targets))
    metrics["bce"] = float(np.mean(losses))
    return metrics


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu.")
    return torch.device(name)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != "srcnn-9-1-5-valid":
        raise ValueError("Unsupported checkpoint architecture.")
    model = BandSRCNN().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def train(args):
    if min(args.steps, args.batch_size, args.eval_every, args.threads) < 1:
        raise ValueError("steps, batch-size, eval-every and threads must be positive.")
    if args.patch_size < 13 or args.patch_size > args.image_size:
        raise ValueError("13 <= patch-size <= image-size required.")
    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Learning rate must be positive and finite.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed + 1)
    config = SimulationConfig(height=args.image_size, width=args.image_size)
    bank = make_bank(args.seed, args.models, args.slices, config)
    validation = make_bank(args.seed + 100_000, args.val_models, args.val_slices, config)
    model = BandSRCNN().to(device)
    if args.init_checkpoint:
        model, _ = load_model(args.init_checkpoint, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    initial = evaluate(model, validation, device)
    prior = np.clip(np.mean([s["label"].mean() for s in bank]), 1e-6, 1 - 1e-6)
    val_targets = np.stack([s["label"][6:-6, 6:-6] for s in validation])
    baseline_bce = float(-(val_targets * np.log(prior) + (1 - val_targets) * np.log(1 - prior)).mean())
    history = [{"step": 0, "validation": initial}]
    print(json.dumps({"device": str(device), "train_images": len(bank),
                      "validation_images": len(validation), "initial": initial,
                      "constant_prior_bce": baseline_bce}), flush=True)
    best = initial["bce"]

    def save(name, step, metrics):
        checkpoint = {"architecture": "srcnn-9-1-5-valid", "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "step": step,
                      "simulation": config.to_dict(), "seed": args.seed,
                      "metrics": metrics, "training_args": vars(args),
                      "normalization": "per-image min-max; no smoothing"}
        # Avoid leaving a partially written checkpoint on interruption.
        temporary = out / (name + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(out / name)

    save("best.pt", 0, initial)
    window_loss = []
    for step in range(1, args.steps + 1):
        model.train()
        xs, ys = [], []
        p = args.patch_size
        for _ in range(args.batch_size):
            sample = bank[int(rng.integers(len(bank)))]
            top, left = rng.integers(0, args.image_size - p + 1, size=2)
            xs.append(sample["intensity"][top:top+p, left:left+p])
            ys.append(sample["label"][top+6:top+p-6, left+6:left+p-6])
        x = torch.from_numpy(np.stack(xs)[:, None]).to(device)
        y = torch.from_numpy(np.stack(ys)[:, None]).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        window_loss.append(loss.item())
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, validation, device)
            row = {"step": step, "train_bce": float(np.mean(window_loss)), "validation": metrics}
            history.append(row)
            print(json.dumps(row), flush=True)
            if metrics["bce"] < best:
                best = metrics["bce"]
                save("best.pt", step, metrics)
            save("last.pt", step, metrics)
            (out / "history.json").write_text(json.dumps({"history": history,
                "constant_prior_bce": baseline_bce, "seed": args.seed,
                "validation_seed": args.seed + 100_000}, indent=2))
            window_loss.clear()
    return out / "best.pt"
