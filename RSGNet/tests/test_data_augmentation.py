import unittest

import numpy as np

from data.sunrgbd import (
    augmentation_scale_range,
    pad_to_minimum_size,
    scaled_size,
)


class DataAugmentationTests(unittest.TestCase):
    def test_lsj_changes_only_the_scale_range_contract(self):
        self.assertEqual(augmentation_scale_range("base"), (0.75, 1.75))
        self.assertEqual(augmentation_scale_range("lsj"), (0.5, 2.0))
        self.assertEqual(augmentation_scale_range("strong"), (0.5, 2.0))

    def test_scaled_size_does_not_clamp_axes_independently(self):
        new_h, new_w = scaled_size(470, 630, 0.5)
        self.assertEqual((new_h, new_w), (235, 315))
        self.assertAlmostEqual(new_h / new_w, 470 / 630, places=3)

    def test_padding_preserves_resized_content_and_label_void(self):
        image = np.full((2, 3, 3), 17, dtype=np.uint8)
        padded_image = pad_to_minimum_size(
            image,
            (6, 8),
            (1, 2, 3),
            top=1,
            left=2,
        )
        self.assertEqual(padded_image.shape, (6, 8, 3))
        self.assertTrue(np.array_equal(padded_image[1:3, 2:5], image))
        self.assertTrue(np.array_equal(padded_image[0, 0], np.asarray([1, 2, 3])))

        label = np.full((2, 3), 5, dtype=np.uint8)
        padded_label = pad_to_minimum_size(
            label,
            (6, 8),
            255,
            top=1,
            left=2,
        )
        self.assertEqual(padded_label.shape, (6, 8))
        self.assertTrue(np.array_equal(padded_label[1:3, 2:5], label))
        self.assertEqual(int(padded_label[0, 0]), 255)


if __name__ == "__main__":
    unittest.main()
