"""Headless scientific plots; all arrays and axes are also saved separately."""

from pathlib import Path
import os
import tempfile

# Cloud containers may have a read-only home; keep plotting caches writable.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "arpes-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "arpes-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(sample, result, points, output):
    panels = [(sample["intensity"], sample["energy"], sample["k"], "Simulated ARPES")]
    if "label" in sample:
        panels.append((sample["label"], sample["energy"], sample["k"], "Target bands"))
    panels.append((result["confidence"], result["energy"], result["k"], "CNN band confidence"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), constrained_layout=True)
    for ax, (image, energy, k, title) in zip(axes, panels):
        extent = [k[0], k[-1], energy[0], energy[-1]]
        im = ax.imshow(image, origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set(title=title, xlabel="kx (radian / lattice constant)", ylabel="Energy relative to EF (eV)")
        ax.set_xlim(sample["k"][0], sample["k"][-1])
        ax.set_ylim(sample["energy"][0], sample["energy"][-1])
        fig.colorbar(im, ax=ax, fraction=0.04)
    if points:
        points = np.asarray(points)
        axes[-1].scatter(points[:, 0], points[:, 1], s=2, color="cyan", alpha=0.7)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_features(result, directory):
    directory = Path(directory)
    for name in ["filters", "layer1", "layer2"]:
        maps = result[name]
        order = np.argsort(maps.var(axis=(1, 2)))[::-1]
        rows = (len(maps) + 7) // 8
        fig, axes = plt.subplots(rows, 8, figsize=(12, rows * 1.5), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, index in zip(axes.ravel(), order):
            ax.imshow(maps[index], origin="lower", aspect="auto", cmap="coolwarm" if name == "filters" else "magma")
            ax.set_title(str(index), fontsize=8)
        fig.suptitle(f"{name}: sorted by variance (channel IDs shown)")
        fig.tight_layout()
        fig.savefig(directory / f"{name}.png", dpi=130)
        plt.close(fig)
