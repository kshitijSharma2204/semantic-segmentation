import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROOT
from make_splits import old_split, read_split

IMG_DIR = os.path.join(ROOT, "data/data_semantics/training/image_2")


@unittest.skipUnless(os.path.isdir(IMG_DIR), "KITTI data not found")
class SplitTest(unittest.TestCase):
    def test_saved_split_matches_old_code(self):
        train_names, val_names = old_split(sorted(os.listdir(IMG_DIR)), seed=42)
        self.assertEqual(read_split(os.path.join(ROOT, "splits/val.txt")), val_names)
        self.assertEqual(read_split(os.path.join(ROOT, "splits/train.txt")), train_names)
        self.assertEqual(len(val_names), 40)
        self.assertFalse(set(train_names) & set(val_names))


if __name__ == "__main__":
    unittest.main()
