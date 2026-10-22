"""Inference with explicit axes, native valid-convolution crop, and peak export."""

import csv
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

from .simulation import normalize
from .training import load_model


def load_spectrum(path):
    with_or_array = np.load(path, allow_pickle=False)
    physical_axes = False
    if isinstance(with_or_array, np.lib.npyio.NpzFile):
        with with_or_array as data:
            if "intensity" not in data:
                raise ValueError("NPZ requires an 'intensity' array [energy, momentum].")
            image = data["intensity"].copy()
            if image.ndim != 2:
                raise ValueError("Intensity must have exactly two dimensions.")
            if ("energy" in data) != ("k" in data):
                raise ValueError("Provide both 'energy' and 'k' axes, or neither.")
            physical_axes = "energy" in data
            energy = data["energy"].copy() if physical_axes else np.arange(image.shape[0])
            k = data["k"].copy() if physical_axes else np.arange(image.shape[1])
    else:
        image = with_or_array
        if image.ndim != 2:
            raise ValueError("Intensity must have exactly two dimensions.")
        energy, k = np.arange(image.shape[0]), np.arange(image.shape[1])
    image = normalize(image)
    if min(image.shape) < 13:
        raise ValueError("Both dimensions must be >=13.")
    for axis, length in [(energy, image.shape[0]), (k, image.shape[1])]:
        if axis.ndim != 1 or len(axis) != length or not np.isfinite(axis).all():
            raise ValueError("Axes must be finite 1D arrays matching the intensity shape.")
        delta = np.diff(axis)
        if not ((delta > 0).all() or (delta < 0).all()):
            raise ValueError("Axes must be strictly monotonic.")
        if not np.allclose(delta, delta[0], rtol=1e-3, atol=1e-8):
            raise ValueError("CNN requires uniform sampling; resample axes before inference.")
    if energy[1] < energy[0]:
        image, energy = image[::-1].copy(), energy[::-1].copy()
    if k[1] < k[0]:
        image, k = image[:, ::-1].copy(), k[::-1].copy()
    return image, energy, k, physical_axes


@torch.no_grad()
def predict(model, image, device="cpu", features=False):
    x = torch.from_numpy(normalize(image))[None, None].to(device)
    logits, activations = model(x, return_features=True)
    result = {"confidence": logits.sigmoid().cpu().numpy()[0, 0]}
    if features:
        result.update({name: tensor.cpu().numpy()[0] for name, tensor in activations.items()})
        result["filters"] = model.conv1.weight.detach().cpu().numpy()[:, 0]
    return result


def ridge_points(confidence, energy, k, threshold=0.5, min_distance=2):
    """EDC maxima; points are not assigned band identities at crossings."""
    if not 0 <= threshold <= 1 or min_distance < 1:
        raise ValueError("threshold must be in [0,1], min-distance >=1.")
    if confidence.shape != (len(energy), len(k)):
        raise ValueError("Confidence shape must match axes.")
    points = []
    for col in range(len(k)):
        rows, _ = find_peaks(confidence[:, col], height=threshold, distance=min_distance)
        for row in rows:
            # Subpixel parabolic interpolation of the confidence maximum.
            a, b, c = confidence[row-1:row+2, col]
            denom = a - 2 * b + c
            delta = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5)) if denom else 0.0
            e = energy[row] + delta * (energy[row+1] - energy[row])
            points.append((float(k[col]), float(e), float(b)))
    return points


def infer_file(checkpoint, input_path, output, device="cpu", features=False,
               threshold=0.5, min_distance=2):
    model, _ = load_model(checkpoint, device)
    image, energy, k, physical = load_spectrum(input_path)
    result = predict(model, image, device, features)
    result.update(energy=energy[6:-6], k=k[6:-6], physical_axes=np.array(physical))
    if features:
        result.update(feature_energy=energy[4:-4], feature_k=k[4:-4])
    out = Path(output)
    if out.suffix != ".npz":
        raise ValueError("Output must end in .npz.")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **result)
    points = ridge_points(result["confidence"], result["energy"], result["k"], threshold, min_distance)
    with out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["k" if physical else "column_pixel", "energy" if physical else "row_pixel", "confidence"])
        writer.writerows(points)
    return result, points
