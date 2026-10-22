import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from arpes_bands.inference import infer_file, load_spectrum, predict, ridge_points
from arpes_bands.model import BandSRCNN
from arpes_bands.simulation import SimulationConfig, normalize, render_label, simulate
from arpes_bands.training import load_model, pixel_metrics


torch.set_num_threads(2)


class ModelTests(unittest.TestCase):
    def test_architecture_backward_and_feature_dimensions(self):
        model = BandSRCNN()
        x = torch.randn(2, 1, 32, 40)
        logits, features = model(x, return_features=True)
        self.assertEqual(logits.shape, (2, 1, 20, 28))
        self.assertEqual(features["layer1"].shape, (2, 64, 24, 32))
        self.assertEqual(features["layer2"].shape, (2, 32, 24, 32))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8129)
        loss = F.binary_cross_entropy_with_logits(logits, torch.rand_like(logits))
        loss.backward()
        for p in model.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_valid_convolution_alignment(self):
        model = BandSRCNN()
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.conv1.weight[0, 0, 4, 4] = 1
            model.conv2.weight[0, 0, 0, 0] = 1
            model.conv3.weight[0, 0, 2, 2] = 1
        x = torch.rand(1, 1, 23, 27)
        torch.testing.assert_close(model(x), x[:, :, 6:-6, 6:-6])

    def test_invalid_model_input(self):
        for shape in [(1, 1, 12, 32), (1, 2, 20, 20), (20, 20)]:
            with self.assertRaises(ValueError):
                BandSRCNN()(torch.zeros(shape))

    def test_checkpoint_prediction_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            model = BandSRCNN().eval()
            image = np.random.default_rng(3).random((25, 30)).astype(np.float32)
            torch.save({"architecture": "srcnn-9-1-5-valid", "model": model.state_dict()}, path)
            restored, _ = load_model(path)
            a, b = predict(model, image, features=True), predict(restored, image, features=True)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])


class PhysicsTests(unittest.TestCase):
    def test_flat_band_raster_position_and_antialiasing(self):
        config = SimulationConfig(height=64, width=32)
        coeff = np.zeros((5, 1, 1))
        coeff[0, 0, 0] = -0.403
        label, energy, _ = render_label(coeff, 0, config)
        np.testing.assert_allclose(label.sum(axis=0), 1)
        fitted = (label[:, 0] * energy).sum() / label[:, 0].sum()
        self.assertLess(abs(fitted + 0.403), (energy[1] - energy[0]) / 2)
        self.assertTrue(((label > 0) & (label < 1)).any())

    def test_simulation_reproducibility_and_noise(self):
        config = SimulationConfig(height=48, width=48)
        a = simulate(np.random.default_rng(17), config)
        b = simulate(np.random.default_rng(17), config)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertGreater(a["label"].sum(), 0)
        self.assertTrue(np.isfinite(a["intensity"]).all())
        self.assertEqual(a["intensity"].min(), 0)
        self.assertEqual(a["intensity"].max(), 1)
        self.assertFalse(np.array_equal(a["intensity"], a["label"]))
        self.assertTrue(0.010 <= a["gamma"] <= 0.045)

    def test_bad_simulation_and_intensity_rejected(self):
        with self.assertRaises(ValueError):
            SimulationConfig(gamma_min=-0.1)
        with self.assertRaises(ValueError):
            normalize(np.array([[np.nan]]))
        np.testing.assert_array_equal(normalize(np.ones((13, 13))), np.zeros((13, 13)))


class InferenceTests(unittest.TestCase):
    def test_axis_orientation_and_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.npz"
            intensity = np.arange(20 * 24).reshape(20, 24)
            energy, k = np.linspace(-1, 0, 20), np.linspace(-2, 2, 24)
            np.savez(path, intensity=intensity[::-1, ::-1], energy=energy[::-1], k=k[::-1])
            loaded, e, momentum, physical = load_spectrum(path)
            np.testing.assert_allclose(loaded, normalize(intensity))
            np.testing.assert_allclose(e, energy)
            np.testing.assert_allclose(momentum, k)
            self.assertTrue(physical)

    def test_nonuniform_or_missing_axes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.npz"
            np.savez(path, intensity=np.ones((20, 20)), energy=np.arange(20))
            with self.assertRaises(ValueError):
                load_spectrum(path)
            np.savez(path, intensity=np.ones((20, 20)), energy=np.arange(20)**2, k=np.arange(20))
            with self.assertRaises(ValueError):
                load_spectrum(path)

    def test_peak_coordinates_and_multiple_bands(self):
        energy = np.linspace(-1, 0, 25)
        k = np.linspace(-1, 1, 5)
        confidence = np.zeros((25, 5), dtype=np.float32)
        confidence[6, :] = 0.9
        confidence[16, :] = 0.8
        points = ridge_points(confidence, energy, k)
        self.assertEqual(len(points), 10)
        np.testing.assert_allclose(points[0], [k[0], energy[6], 0.9], atol=1e-6)

    def test_full_file_inference_exports_axes_features_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = BandSRCNN()
            torch.save({"architecture": "srcnn-9-1-5-valid", "model": model.state_dict()}, root / "m.pt")
            np.save(root / "input.npy", np.random.default_rng(1).random((24, 30)))
            result, _ = infer_file(root / "m.pt", root / "input.npy", root / "out.npz", features=True)
            self.assertEqual(result["confidence"].shape, (12, 18))
            self.assertEqual(result["layer1"].shape, (64, 16, 22))
            self.assertEqual(result["energy"][0], 6)
            self.assertEqual(result["feature_energy"][0], 4)
            self.assertFalse(result["physical_axes"])
            self.assertTrue((root / "out.csv").exists())


class MetricTests(unittest.TestCase):
    def test_perfect_and_tied_average_precision(self):
        y = np.zeros((10, 10))
        y[3] = 1
        self.assertEqual(pixel_metrics(y, y)["average_precision"], 1)
        self.assertAlmostEqual(pixel_metrics(np.ones_like(y) * 0.1, y)["average_precision"], 0.1)

    def test_tolerance_does_not_cross_sample_boundary(self):
        y = np.zeros((2, 9, 9))
        p = np.zeros_like(y)
        y[0, 4, 4] = 1
        p[1, 4, 4] = 1
        self.assertEqual(pixel_metrics(p, y)["recall_1px"], 0)


if __name__ == "__main__":
    unittest.main()
