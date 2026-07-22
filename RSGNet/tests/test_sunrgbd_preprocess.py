import importlib.util
import pathlib
import unittest

import numpy as np


PREPROCESSOR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "getDATA"
    / "prepare_sunrgbd.py"
    / "SUN2HHA.py"
)


def _load_preprocessor():
    spec = importlib.util.spec_from_file_location("sun2hha_preprocessor", PREPROCESSOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(PREPROCESSOR.is_file(), "SUN RGB-D preprocessor is not checked out")
class SUNRGBDPreprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preprocessor = _load_preprocessor()

    def test_toolbox_coordinate_conversion(self):
        camera = np.asarray(
            [[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]],
            dtype=np.float32,
        )
        expected = np.asarray(
            [[1.0, 3.0, -2.0], [-4.0, 6.0, -5.0]],
            dtype=np.float32,
        )
        converted = self.preprocessor.camera_to_sunrgbd_coordinates(camera)
        self.assertTrue(np.array_equal(converted, expected))

    def test_depth_decode_matches_toolbox_bit_rotation(self):
        raw = np.asarray([0x0008, 0x0001, 0x1234, 0xFFFF], dtype=np.uint16)
        expected = np.bitwise_and(
            np.bitwise_or(np.right_shift(raw.astype(np.uint32), 3), np.left_shift(raw.astype(np.uint32), 13)),
            0xFFFF,
        ).astype(np.uint16)
        decoded = self.preprocessor.decode_sunrgbd_depth(raw)
        self.assertTrue(np.array_equal(decoded, expected))


if __name__ == "__main__":
    unittest.main()
