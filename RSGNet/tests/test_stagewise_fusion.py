import unittest
import random

import numpy as np
import torch

from data import canonicalize_hha_channels, degrade_hha_for_evaluation
from data.sunrgbd import SUNDataset, degrade_hha, official_sunrgbd_image_ids
from rsgnet.encoder.fusion import (
    FactorizedPromptRouter,
    TopologyAwareLocalFusion,
    TopologyAwareSemanticAttention,
    GeometryGuidedSemanticAttention,
    GeometryPromptAdapter,
    ReliabilityConditionedLinearAttention,
    ReliabilityConditionedPixelFusion,
    ResidualCrossScaleAligner,
    ShallowGeometryFusion,
    build_stage_fusion,
)
from rsgnet.encoder.geometry import (
    ChannelRoutedPhysicsFactorizedHHAEncoder,
    FactorizedGeometryReliabilityHead,
    GeometryReliabilityHead,
    HHAChannelRelationGate,
    build_factorized_hha_encoder,
    normalized_hha_angle_to_radians,
)
from rsgnet.encoder.layers import LearnableGate
from utils import validate


class RSGNetFusionTests(unittest.TestCase):
    def test_official_validation_split_is_stable_and_disjoint(self):
        train = official_sunrgbd_image_ids("train", validation_fraction=0.10, validation_seed=3407)
        val = official_sunrgbd_image_ids("val", validation_fraction=0.10, validation_seed=3407)
        test = official_sunrgbd_image_ids("test", validation_fraction=0.10, validation_seed=3407)
        self.assertEqual(len(train) + len(val), 5285)
        self.assertEqual(len(val), 528)
        self.assertEqual(len(test), 5050)
        self.assertFalse(set(train) & set(val))
        self.assertFalse((set(train) | set(val)) & set(test))
        self.assertEqual(
            val,
            official_sunrgbd_image_ids("val", validation_fraction=0.10, validation_seed=3407),
        )

    def test_rare_class_crop_contains_target_when_available(self):
        dataset = object.__new__(SUNDataset)
        dataset.crop_h = 16
        dataset.crop_w = 16
        dataset.rare_crop_probability = 1.0
        dataset.rare_crop_class_ids = (19, 27)
        dataset.rare_crop_trials = 32
        label = np.zeros((32, 32), dtype=np.uint8)
        label[24:32, 24:32] = 19
        random.seed(3407)
        y, x = dataset._select_crop_origin(label)
        self.assertGreater(int((label[y:y + 16, x:x + 16] == 19).sum()), 0)

    def test_validation_reports_perfect_interior_and_boundary_metrics(self):
        class PerfectModel(torch.nn.Module):
            def forward(self, _rgb, geometry):
                labels = geometry[:, 0].long()
                logits = geometry.new_full((labels.shape[0], 2, *labels.shape[1:]), -10.0)
                return logits.scatter_(1, labels.unsqueeze(1), 10.0)

        labels = torch.zeros(1, 16, 16, dtype=torch.long)
        labels[:, :, 8:] = 1
        rgb = torch.zeros(1, 3, 16, 16)
        geometry = labels.unsqueeze(1).float().expand(-1, 3, -1, -1).clone()
        metrics = validate(
            PerfectModel(),
            [(rgb, geometry, labels)],
            device=torch.device("cpu"),
            n_classes=2,
            miou_start_class=0,
            return_details=True,
        )
        self.assertAlmostEqual(metrics["miou"], 1.0, places=5)
        self.assertAlmostEqual(metrics["interior_miou"], 1.0, places=5)
        self.assertAlmostEqual(metrics["boundary_f1"], 1.0, places=5)

    def test_learnable_gate_is_bounded(self):
        gate = LearnableGate(init_lo=0.35, init_span=0.75)
        output = gate(torch.linspace(-1.0, 2.0, 100))
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_explicit_invalid_hint_forces_zero_reliability(self):
        estimator = GeometryReliabilityHead().eval()
        hha = torch.randn(2, 3, 24, 24).clamp(-1.0, 1.0)
        invalid = torch.zeros(2, 1, 24, 24)
        invalid[:, :, 4:12, 4:12] = 1.0
        with torch.no_grad():
            reliability = estimator(hha, invalid_hint=invalid)
        self.assertEqual(float(reliability[invalid.bool()].max()), 0.0)
        self.assertGreater(float(reliability[~invalid.bool()].mean()), 0.8)

    def test_ahd_disk_channels_are_canonicalized_to_dha(self):
        hha = np.zeros((3, 4, 3), dtype=np.uint8)
        hha[:, :, 0] = 11
        hha[:, :, 1] = 22
        hha[:, :, 2] = 33
        canonical = canonicalize_hha_channels(hha, "ahd")
        self.assertTrue(np.all(canonical[:, :, 0] == 33))
        self.assertTrue(np.all(canonical[:, :, 1] == 22))
        self.assertTrue(np.all(canonical[:, :, 2] == 11))

    def test_normalized_hha_angle_maps_to_zero_through_ninety_degrees(self):
        normalized = torch.tensor([-1.0, 0.0, 1.0])
        radians = normalized_hha_angle_to_radians(normalized)
        expected = torch.tensor([0.0, torch.pi / 4.0, torch.pi / 2.0])
        self.assertTrue(torch.allclose(radians, expected))

    def test_factorized_degradation_returns_two_reliability_targets(self):
        hha = np.full((32, 40, 3), 100, dtype=np.uint8)
        hha[:3, :3] = 0
        _, target, weight, invalid = degrade_hha(
            hha,
            probability=0.0,
            factorized_target=True,
        )
        self.assertEqual(target.shape, (32, 40, 2))
        self.assertEqual(weight.shape, (32, 40))
        self.assertEqual(invalid.shape, (32, 40))
        self.assertEqual(float(target[:3, :3].max()), 0.0)

    def test_factorized_reliability_masks_both_experts(self):
        estimator = FactorizedGeometryReliabilityHead().eval()
        hha = torch.randn(2, 3, 20, 20).clamp(-1.0, 1.0)
        invalid = torch.zeros(2, 1, 20, 20)
        invalid[:, :, 2:7, 4:9] = 1.0
        with torch.no_grad():
            reliability = estimator(hha, invalid_hint=invalid)
        self.assertEqual(reliability.shape[1], 2)
        invalid_both = invalid.bool().expand(-1, 2, -1, -1)
        self.assertEqual(float(reliability[invalid_both].max()), 0.0)

    def test_stagewise_policy_assigns_modules_by_depth(self):
        layers = [
            build_stage_fusion("stagewise", channels, index, 4, 64, 0.05, 0.05)
            for index, channels in enumerate((16, 32, 64, 64))
        ]
        self.assertIsInstance(layers[0], ShallowGeometryFusion)
        self.assertIsInstance(layers[1], ShallowGeometryFusion)
        self.assertIsInstance(layers[2], GeometryPromptAdapter)
        self.assertIsInstance(layers[3], GeometryGuidedSemanticAttention)

    def test_refined_policy_assigns_linear_cost_modules(self):
        layers = [
            build_stage_fusion(
                "stagewise",
                channels,
                index,
                4,
                64,
                0.10,
                0.10,
                architecture_variant="refined",
            )
            for index, channels in enumerate((16, 32, 64, 64))
        ]
        self.assertIsInstance(layers[0], ReliabilityConditionedPixelFusion)
        self.assertIsInstance(layers[1], ReliabilityConditionedPixelFusion)
        self.assertIsInstance(layers[2], GeometryPromptAdapter)
        self.assertIsInstance(layers[3], ReliabilityConditionedLinearAttention)

    def test_factorized_policy_wraps_linear_cost_modules(self):
        layers = [
            build_stage_fusion(
                "stagewise",
                channels,
                index,
                4,
                64,
                0.10,
                0.10,
                architecture_variant="factorized",
                geometry_encoding="factorized_routed",
            )
            for index, channels in enumerate((16, 32, 64, 64))
        ]
        self.assertTrue(all(isinstance(layer, FactorizedPromptRouter) for layer in layers))
        self.assertIsInstance(layers[0].inner_fusion, ReliabilityConditionedPixelFusion)
        self.assertIsInstance(layers[2].inner_fusion, GeometryPromptAdapter)
        self.assertIsInstance(layers[3].inner_fusion, ReliabilityConditionedLinearAttention)

    def test_topology_policy_uses_stable_local_and_topological_blocks(self):
        layers = [
            build_stage_fusion(
                "topology",
                channels,
                index,
                4,
                64,
                0.05,
                0.05,
                architecture_variant="factorized",
                geometry_encoding="factorized_routed",
            )
            for index, channels in enumerate((16, 32, 64, 64))
        ]
        self.assertIsInstance(layers[0].inner_fusion, TopologyAwareLocalFusion)
        self.assertIsInstance(layers[2].inner_fusion, GeometryPromptAdapter)
        self.assertIsInstance(layers[3].inner_fusion, TopologyAwareSemanticAttention)

    def test_topology_attention_is_finite_and_token_bounded(self):
        module = TopologyAwareSemanticAttention(32, max_attention_tokens=64)
        rgb = torch.randn(2, 32, 16, 16, requires_grad=True)
        prompt = torch.randn_like(rgb, requires_grad=True)
        guide = torch.randn(2, 1, 16, 16)
        reliability = torch.rand(2, 1, 16, 16)
        output = module(rgb, prompt, guide, reliability)
        self.assertEqual(output.shape, rgb.shape)
        output.square().mean().backward()
        gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))

    def test_factorized_router_falls_back_to_rgb_when_both_experts_are_unreliable(self):
        module = build_stage_fusion(
            "stagewise",
            16,
            0,
            4,
            64,
            0.10,
            0.10,
            architecture_variant="factorized",
            geometry_encoding="factorized_routed",
        )
        rgb = torch.randn(2, 16, 12, 12)
        prompts = (torch.randn_like(rgb), torch.randn_like(rgb))
        guide = torch.randn_like(rgb)
        reliability = torch.zeros(2, 2, 12, 12)
        with torch.no_grad():
            output = module(rgb, prompts, guide, reliability)
        self.assertTrue(torch.equal(output, rgb))

    def test_topology_final_soft_router_applies_reliability_once(self):
        module = build_stage_fusion(
            "topology",
            16,
            0,
            4,
            64,
            0.10,
            0.10,
            architecture_variant="factorized",
            geometry_encoding="factorized_final_soft",
        )
        rgb = torch.randn(2, 16, 12, 12)
        prompts = (torch.randn_like(rgb), torch.randn_like(rgb))
        guide = torch.randn_like(rgb)
        reliability = torch.zeros(2, 2, 12, 12)
        with torch.no_grad():
            output = module(rgb, prompts, guide, reliability)
        self.assertTrue(torch.equal(output, rgb))

    def test_channel_relation_gate_is_identity_at_initialization(self):
        gate = HHAChannelRelationGate(prompt_channels=16).eval()
        hha = torch.randn(2, 3, 20, 20)
        reliability = torch.rand(2, 2, 20, 20)
        with torch.no_grad():
            output = gate(hha, reliability)
        self.assertTrue(torch.allclose(output, hha, atol=1e-6, rtol=0.0))
        self.assertTrue(torch.allclose(gate.last_channel_weights, torch.full_like(gate.last_channel_weights, 1.0 / 3.0), atol=1e-6))

    def test_channel_routed_encoder_keeps_factorized_prompt_shapes(self):
        encoder = build_factorized_hha_encoder(
            "factorized_channel_routed",
            out_channels=(16, 32),
            prompt_channels=8,
        )
        self.assertIsInstance(encoder, ChannelRoutedPhysicsFactorizedHHAEncoder)
        hha = torch.randn(2, 3, 24, 24).clamp(-1.0, 1.0)
        reliability = torch.rand(2, 2, 24, 24)
        layout, boundary, boundary_map = encoder(hha, reliability, ((12, 12), (6, 6)))
        self.assertEqual([tuple(item.shape) for item in layout], [(2, 16, 12, 12), (2, 32, 6, 6)])
        self.assertEqual([tuple(item.shape) for item in boundary], [(2, 16, 12, 12), (2, 32, 6, 6)])
        self.assertEqual(tuple(boundary_map.shape), (2, 1, 24, 24))
        self.assertEqual(tuple(encoder.channel_relation.last_channel_weights.shape), (2, 3, 24, 24))
        self.assertEqual(tuple(encoder.last_boundary_weights.shape), (2, 3, 24, 24))
        self.assertTrue(
            torch.allclose(
                encoder.last_boundary_weights.sum(dim=1),
                torch.ones(2, 24, 24),
                atol=1e-5,
            )
        )

    def test_correct_and_swapped_stage_priors_are_opposites(self):
        correct = build_stage_fusion(
            "prompt", 16, 0, 4, 64, 0.10, 0.10,
            architecture_variant="factorized", geometry_encoding="factorized_static",
        )
        swapped = build_stage_fusion(
            "prompt", 16, 0, 4, 64, 0.10, 0.10,
            architecture_variant="factorized", geometry_encoding="factorized_swapped",
        )
        correct_prior = torch.softmax(correct.route_prior_logits, dim=1)
        swapped_prior = torch.softmax(swapped.route_prior_logits, dim=1)
        self.assertTrue(torch.allclose(correct_prior.flip(1), swapped_prior))

    def test_physics_factorized_encoder_keeps_layout_and_boundary_separate(self):
        encoder = build_factorized_hha_encoder(
            "factorized_routed",
            out_channels=(16, 32),
            prompt_channels=8,
        )
        hha = torch.randn(2, 3, 24, 24).clamp(-1.0, 1.0)
        reliability = torch.rand(2, 2, 24, 24)
        layout, boundary, boundary_map = encoder(hha, reliability, ((12, 12), (6, 6)))
        self.assertEqual([tuple(item.shape) for item in layout], [(2, 16, 12, 12), (2, 32, 6, 6)])
        self.assertEqual([tuple(item.shape) for item in boundary], [(2, 16, 12, 12), (2, 32, 6, 6)])
        self.assertEqual(tuple(boundary_map.shape), (2, 1, 24, 24))
        self.assertFalse(torch.equal(layout[0], boundary[0]))

    def test_refined_pixel_fusion_falls_back_to_rgb_when_unreliable(self):
        module = ReliabilityConditionedPixelFusion(16)
        rgb = torch.randn(2, 16, 12, 12)
        prompt = torch.randn_like(rgb)
        guide = torch.randn_like(rgb)
        reliability = torch.zeros(2, 1, 12, 12)
        with torch.no_grad():
            output = module(rgb, prompt, guide, reliability)
        self.assertTrue(torch.equal(output, rgb))

    def test_residual_cross_scale_aligner_has_identity_initialization_option(self):
        module = ResidualCrossScaleAligner(16, 32)
        module.residual_scale.data.zero_()
        shallow = torch.randn(2, 16, 16, 16)
        deep = torch.randn(2, 32, 8, 8)
        with torch.no_grad():
            output = module(shallow, deep)
        self.assertTrue(torch.equal(output, shallow))

    def test_deep_geometry_attention_has_finite_gradients(self):
        module = GeometryGuidedSemanticAttention(32, max_attention_tokens=64)
        rgb = torch.randn(2, 32, 8, 8, requires_grad=True)
        prompt = torch.randn(2, 32, 8, 8, requires_grad=True)
        guide = torch.randn(2, 32, 8, 8, requires_grad=True)
        reliability = torch.rand(2, 1, 8, 8)
        module(rgb, prompt, guide, reliability).square().mean().backward()
        gradients = [parameter.grad for parameter in module.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_linear_geometry_attention_has_finite_gradients(self):
        module = ReliabilityConditionedLinearAttention(32, max_attention_tokens=64)
        rgb = torch.randn(2, 32, 8, 8, requires_grad=True)
        prompt = torch.randn(2, 32, 8, 8, requires_grad=True)
        guide = torch.randn(2, 32, 8, 8, requires_grad=True)
        reliability = torch.rand(2, 1, 8, 8)
        module(rgb, prompt, guide, reliability).square().mean().backward()
        gradients = [parameter.grad for parameter in module.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_robustness_corruptions_are_deterministic(self):
        hha = np.full((64, 64, 3), 100, dtype=np.uint8)
        for mode, severity in (("dropout", 0.3), ("noise", 16), ("shift", 8)):
            first = degrade_hha_for_evaluation(hha, mode=mode, severity=severity, seed=9)
            second = degrade_hha_for_evaluation(hha, mode=mode, severity=severity, seed=9)
            self.assertTrue(np.array_equal(first, second))
            self.assertFalse(np.array_equal(first, hha))


if __name__ == "__main__":
    unittest.main()
