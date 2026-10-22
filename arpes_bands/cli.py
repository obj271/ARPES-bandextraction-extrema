import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .simulation import SimulationConfig, make_bank, simulate


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive.")
    return number


def main():
    parser = argparse.ArgumentParser(description="Physics-supervised ARPES SRCNN")
    commands = parser.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train", help="Random TB spectra -> sharp band confidence")
    t.add_argument("--output", default="outputs/train")
    t.add_argument("--steps", type=positive, default=5000)
    t.add_argument("--models", type=positive, default=20)
    t.add_argument("--slices", type=positive, default=50)
    t.add_argument("--val-models", type=positive, default=4)
    t.add_argument("--val-slices", type=positive, default=5)
    t.add_argument("--image-size", type=positive, default=128)
    t.add_argument("--patch-size", type=positive, default=32)
    t.add_argument("--batch-size", type=positive, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--eval-every", type=positive, default=250)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    t.add_argument("--threads", type=positive, default=2)
    t.add_argument("--init-checkpoint", help="Warm-start weights; optimizer/step counter reset")

    s = commands.add_parser("simulate", help="Generate an independent labeled spectrum")
    s.add_argument("--output", default="outputs/spectrum.npz")
    s.add_argument("--seed", type=int, default=2026)
    s.add_argument("--size", type=positive, default=128)

    i = commands.add_parser("infer", help="Infer on 2D NPY or NPZ (intensity, energy, k)")
    i.add_argument("--checkpoint", required=True)
    i.add_argument("--input", required=True)
    i.add_argument("--output", default="outputs/prediction.npz")
    i.add_argument("--features", action="store_true")
    i.add_argument("--threshold", type=float, default=0.5)
    i.add_argument("--min-distance", type=positive, default=2)
    i.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    i.add_argument("--threads", type=positive, default=2)

    e = commands.add_parser("evaluate", help="Independent synthetic test set (not validation)")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--seed", type=int, default=200_042)
    e.add_argument("--models", type=positive, default=4)
    e.add_argument("--slices", type=positive, default=5)
    e.add_argument("--output", default="outputs/evaluation.json")
    e.add_argument("--threads", type=positive, default=2)

    d = commands.add_parser("demo", help="Predict an unseen simulation and visualize features")
    d.add_argument("--checkpoint", required=True)
    d.add_argument("--output", default="outputs/demo")
    d.add_argument("--seed", type=int, default=2026)
    d.add_argument("--threads", type=positive, default=2)

    args = parser.parse_args()
    if args.command == "train":
        from .training import train
        train(args)
    elif args.command == "simulate":
        out = Path(args.output)
        if out.suffix != ".npz":
            parser.error("--output must end in .npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        sample = simulate(np.random.default_rng(args.seed), SimulationConfig(height=args.size, width=args.size))
        np.savez_compressed(out, **sample)
        print(str(out))
    elif args.command == "infer":
        from .inference import infer_file
        from .training import resolve_device
        torch.set_num_threads(args.threads)
        result, points = infer_file(args.checkpoint, args.input, args.output,
            resolve_device(args.device), args.features, args.threshold, args.min_distance)
        print(json.dumps({"shape": list(result["confidence"].shape), "ridge_points": len(points),
                          "physical_axes": bool(result["physical_axes"]), "output": args.output}))
    elif args.command == "evaluate":
        from .training import evaluate, load_model
        torch.set_num_threads(args.threads)
        model, checkpoint = load_model(args.checkpoint)
        config = SimulationConfig(**checkpoint["simulation"])
        if args.seed in [checkpoint["seed"], checkpoint["seed"] + 100_000]:
            parser.error("Test seed must differ from training and validation seeds.")
        bank = make_bank(args.seed, args.models, args.slices, config)
        metrics = evaluate(model, bank, "cpu")
        report = {"seed": args.seed, "test_images": len(bank), "checkpoint_step": checkpoint["step"],
                  "metrics": metrics, "scope": "synthetic spectra only; no experimental validation"}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
    elif args.command == "demo":
        from .inference import infer_file
        from .plotting import plot_comparison, plot_features
        from .training import load_model
        torch.set_num_threads(args.threads)
        _, checkpoint = load_model(args.checkpoint)
        sample = simulate(np.random.default_rng(args.seed), SimulationConfig(**checkpoint["simulation"]))
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "spectrum.npz", **sample)
        result, points = infer_file(args.checkpoint, out / "spectrum.npz", out / "prediction.npz", features=True)
        plot_comparison(sample, result, points, out / "comparison.png")
        plot_features(result, out)
        print(json.dumps({"output": str(out), "ridge_points": len(points)}))


if __name__ == "__main__":
    main()
