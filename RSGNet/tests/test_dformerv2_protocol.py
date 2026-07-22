import os
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from data.protocols import SUNRGBD37_CLASS_NAMES, apply_protocol
from data.sunrgbd import pad_bottom_right
from utils import get_scaled_size, sliding_window_inference


class IdentitySegmentation(nn.Module):
    def forward(self, rgb, hha):
        return torch.cat([rgb[:, :1], hha[:, :1]], dim=1)


class DFormerV2ProtocolTests(unittest.TestCase):
    def test_protocol_sets_the_official_evaluation_contract(self):
        args = SimpleNamespace(
            protocol="sunrgbd37_dformerv2",
            dataset_name="SUNRGBD",
            n_classes=41,
            ignore_index=0,
            miou_start_class=1,
            allow_custom_input_size=False,
            input_height=480,
            input_width=640,
            split_policy="file",
            eval_native_size=False,
            sliding_eval=False,
            eval_crop_height=512,
            eval_crop_width=512,
            eval_stride_rate=1.0,
            tta_scales="1.0",
            label_map="",
            eval_pad_height=0,
            eval_pad_width=0,
            tta_fusion="logits",
            tta_align_corners=False,
            tta_size_mode="legacy",
            tta_small_image_mode="pad",
        )
        apply_protocol(args)

        self.assertEqual(args.n_classes, 37)
        self.assertEqual(args.ignore_index, 255)
        self.assertEqual(args.miou_start_class, 0)
        self.assertEqual(args.split_policy, "sunrgbd_official")
        self.assertEqual((args.eval_pad_height, args.eval_pad_width), (531, 730))
        self.assertEqual(args.tta_scales, "0.5,0.75,1.0,1.25,1.5")
        self.assertEqual(args.tta_fusion, "probabilities")
        self.assertTrue(args.tta_align_corners)
        self.assertEqual(args.tta_size_mode, "dformerv2")
        self.assertEqual(args.tta_small_image_mode, "resize")
        self.assertEqual(len(SUNRGBD37_CLASS_NAMES), 37)
        self.assertTrue(os.path.isfile(args.label_map))

    def test_scaled_sizes_match_dformerv2_val_mm(self):
        expected = {
            0.5: (288, 384),
            0.75: (416, 576),
            1.0: (544, 736),
            1.25: (672, 928),
            1.5: (800, 1120),
        }
        for scale, size in expected.items():
            self.assertEqual(
                get_scaled_size(531, 730, scale, size_mode="dformerv2"),
                size,
            )

    def test_eval_canvas_padding_is_bottom_right_and_ignored(self):
        image = np.full((2, 3, 3), 17, dtype=np.uint8)
        label = np.full((2, 3), 4, dtype=np.uint8)
        padded_image = pad_bottom_right(image, (4, 6), 0)
        padded_label = pad_bottom_right(label, (4, 6), 255)

        self.assertTrue(np.array_equal(padded_image[:2, :3], image))
        self.assertTrue(np.array_equal(padded_label[:2, :3], label))
        self.assertEqual(int(padded_image[-1, -1, 0]), 0)
        self.assertEqual(int(padded_label[-1, -1]), 255)

    def test_small_scale_sliding_uses_dformerv2_resize_behavior(self):
        model = IdentitySegmentation()
        rgb = torch.zeros(1, 3, 288, 384)
        hha = torch.zeros_like(rgb)
        logits = sliding_window_inference(
            model,
            rgb,
            hha,
            crop_size=(480, 480),
            align_corners=True,
            small_image_mode="resize",
        )
        self.assertEqual(tuple(logits.shape), (1, 2, 480, 480))


if __name__ == "__main__":
    unittest.main()
