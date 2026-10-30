import unittest

import numpy as np

from isac_spoofing.config import ProjectConfig
from isac_spoofing.evaluation import StreamingDetector
from isac_spoofing.learning_data import FeatureStandardizer
from isac_spoofing.training import DetectorBundle, attack_probabilities_from_logits

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires learning extra")
class StreamingModelTests(unittest.TestCase):
    def make_bundle(self, model_type):
        from isac_spoofing.models.torch_models import build_detector
        torch.set_num_threads(1)
        torch.manual_seed(123)
        model = build_detector(
            model_type, input_dim=2, hidden_dim=6, num_layers=2,
            input_encoder_dim=5 if model_type == "gru" else 0,
        ).eval()
        standardizer = FeatureStandardizer(
            feature_names=("f1", "f0"),
            mean=np.array([0.4, -0.2], dtype=np.float32),
            std=np.array([1.3, 0.8], dtype=np.float32),
        )
        checkpoint = {
            "model_type": model_type,
            "source_dataset_schema_version": "3.0",
            "measurement_timing": {"alarm_availability_offset_steps": 1},
            "source_model_config": ProjectConfig().to_dict(),
        }
        return DetectorBundle(model, standardizer, torch.device("cpu"), checkpoint)

    def test_streaming_matches_batch_with_selection_and_reset(self):
        values = np.random.default_rng(15).normal(size=(12, 3, 3)).astype(np.float32)
        for kind in ("mlp", "gru"):
            with self.subTest(kind=kind):
                bundle = self.make_bundle(kind)
                policy = StreamingDetector(
                    bundle, ("f0", "f1", "f2"), ProjectConfig().to_dict()
                )
                normalized = bundle.standardizer.transform(values[..., [1, 0]])
                with torch.no_grad():
                    logits = bundle.model(torch.as_tensor(normalized.swapaxes(0, 1)))
                    expected = attack_probabilities_from_logits(logits).numpy().T
                actual = np.stack([policy(row) for row in values])
                np.testing.assert_allclose(actual, expected, atol=2e-7)
                policy.reset()
                np.testing.assert_allclose(policy(values[0]), expected[0], atol=2e-7)

    def test_old_and_different_physics_are_not_silently_accepted(self):
        bundle = self.make_bundle("gru")
        bundle.checkpoint.pop("source_dataset_schema_version")
        with self.assertRaisesRegex(ValueError, "schema-3"):
            StreamingDetector(bundle, ("f0", "f1"), ProjectConfig().to_dict())
        policy = StreamingDetector(
            bundle, ("f0", "f1"), ProjectConfig().to_dict(), allow_domain_shift=True
        )
        self.assertEqual(policy.domain_status["status"], "unvalidated_domain_shift")
        bundle = self.make_bundle("gru")
        changed = ProjectConfig().to_dict()
        changed["gps"]["noise_std_m"] = [2.0, 2.0, 3.0]
        with self.assertRaisesRegex(ValueError, "gps configuration"):
            StreamingDetector(bundle, ("f0", "f1"), changed)
