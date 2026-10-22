"""Random tight-binding bands, supersampled labels, and ARPES forward model.

The paper does not specify its exact Hamiltonian or noise distributions. The
choices here are explicit, configurable approximations, not author code.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter
from scipy.special import expit


@dataclass
class SimulationConfig:
    height: int = 128
    width: int = 128
    energy_min: float = -0.8
    energy_max: float = 0.1
    supersample: int = 4
    gamma_min: float = 0.010  # eV; |Im Sigma|, Lorentzian HWHM
    gamma_max: float = 0.045
    temperature_min: float = 10.0  # kelvin
    temperature_max: float = 80.0
    resolution_min: float = 0.3  # Gaussian sigma, pixels, for both axes
    resolution_max: float = 1.0
    counts_min: float = 100.0
    counts_max: float = 1000.0
    read_noise_max: float = 0.03  # fraction of peak intensity

    def __post_init__(self):
        if min(self.height, self.width) < 13 or self.supersample < 1:
            raise ValueError("Image dimensions >=13 and supersample >=1 required.")
        for lo, hi in [(self.energy_min, self.energy_max),
                       (self.gamma_min, self.gamma_max),
                       (self.temperature_min, self.temperature_max),
                       (self.resolution_min, self.resolution_max),
                       (self.counts_min, self.counts_max)]:
            if not np.isfinite([lo, hi]).all() or lo > hi:
                raise ValueError("Simulation ranges must be finite and ordered.")
        if self.energy_min == self.energy_max or min(self.gamma_min, self.temperature_min, self.counts_min) <= 0:
            raise ValueError("Energy range and positive linewidth/temperature/counts required.")
        if self.resolution_min < 0 or not np.isfinite(self.read_noise_max) or self.read_noise_max < 0:
            raise ValueError("Resolution and read noise must be nonnegative.")

    def to_dict(self):
        return asdict(self)


def normalize(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Intensity must be a finite 2D array.")
    image = image - image.min()
    span = image.max()
    return image / span if span > 0 else np.zeros_like(image)


def random_hamiltonian(rng):
    """Real Hermitian multiorbital square-lattice Hamiltonian coefficients.

    H(kx,ky) = H0 + Cx cos(kx) + Sx sin(kx) + Cy cos(ky)
               + Cxy cos(kx+ky). Eigenvalues produce crossing/avoided bands.
    """
    n = int(rng.integers(1, 5))
    coeff = []
    for scale in [0.12, 0.18, 0.08, 0.12, 0.05]:
        m = rng.normal(0, scale, (n, n))
        m = np.diag(np.diag(m)) + 0.15 * (m + m.T) / 2
        coeff.append(m)
    coeff[0] -= np.eye(n) * rng.uniform(0.15, 0.5)
    return np.stack(coeff)


def band_energies(coeff, k, ky):
    h = (coeff[0] + np.cos(k)[:, None, None] * coeff[1]
         + np.sin(k)[:, None, None] * coeff[2]
         + np.cos(ky) * coeff[3]
         + np.cos(k + ky)[:, None, None] * coeff[4])
    return np.linalg.eigvalsh(h).T  # [band, kx]


def render_label(coeff, ky, config):
    """Rasterize 1-pixel bands at s-times resolution, then average down."""
    h, w, s = config.height, config.width, config.supersample
    energy = np.linspace(config.energy_min, config.energy_max, h)
    k = np.linspace(-np.pi, np.pi, w)
    de, dk = energy[1] - energy[0], k[1] - k[0]
    fine_k = k[0] - dk / 2 + (np.arange(w * s) + 0.5) * dk / s
    positions = (band_energies(coeff, fine_k, ky) - energy[0]) / de
    high = np.zeros((h * s, w * s), dtype=np.float32)
    # Fill all supersampled pixels covered by a one-native-pixel-wide line.
    for band in positions:
        for offset in (np.arange(s) + 0.5) / s - 0.5:
            rows = np.floor((band + offset + 0.5) * s).astype(int)
            valid = (rows >= 0) & (rows < h * s)
            high[rows[valid], np.flatnonzero(valid)] = 1
    label = high.reshape(h, s, w, s).mean(axis=(1, 3))
    return label, energy, k


def simulate(rng, config=None, coeff=None, ky=None):
    config = config or SimulationConfig()
    coeff = random_hamiltonian(rng) if coeff is None else coeff
    ky = rng.uniform(-np.pi, np.pi) if ky is None else ky
    label, energy, k = render_label(coeff, ky, config)
    gamma = rng.uniform(config.gamma_min, config.gamma_max)
    de = energy[1] - energy[0]
    x = np.arange(-config.height + 1, config.height) * de
    lorentz = gamma / (np.pi * (x * x + gamma * gamma))
    lorentz /= lorentz.sum()
    intensity = convolve1d(label, lorentz, axis=0, mode="constant")
    temperature = rng.uniform(config.temperature_min, config.temperature_max)
    intensity *= expit(-energy[:, None] / (8.617333262e-5 * temperature))
    # Smooth momentum-dependent matrix element; no intensity-free labels added.
    intensity *= (0.7 + 0.3 * np.cos(k + rng.uniform(-np.pi, np.pi)) ** 2)[None, :]
    intensity = normalize(intensity)
    counts = rng.uniform(config.counts_min, config.counts_max)
    intensity = rng.poisson(intensity * counts).astype(np.float32) / counts
    sigma = rng.uniform(config.resolution_min, config.resolution_max, size=2)
    intensity = gaussian_filter(intensity, sigma=sigma, mode="nearest")
    noise = rng.uniform(0, config.read_noise_max)
    intensity += rng.normal(0, noise, intensity.shape)
    intensity = normalize(np.maximum(intensity, 0))
    return {"intensity": intensity, "label": label, "energy": energy, "k": k,
            "gamma": gamma, "temperature": temperature}


def make_bank(seed, models, slices, config):
    """Split whole Hamiltonians, not neighboring slices, between train and val."""
    if models < 1 or slices < 1:
        raise ValueError("models and slices must be positive.")
    rng = np.random.default_rng(seed)
    bank = []
    for _ in range(models):
        coeff = random_hamiltonian(rng)
        for ky in rng.uniform(-np.pi, np.pi, slices):
            bank.append(simulate(rng, config, coeff, ky))
    return bank
