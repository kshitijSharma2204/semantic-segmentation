import os
import sys
import unittest
from collections import namedtuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import id_to_trainid_lut, new_confusion, scores, update_confusion

Label = namedtuple("Label", ["name", "id", "trainId"])


class MetricTest(unittest.TestCase):
    def test_ground_truth_as_prediction_scores_one(self):
        torch.manual_seed(0)
        gt = torch.randint(0, 19, (2, 64, 96))
        gt[:, :5] = 255
        conf = update_confusion(new_confusion(19), gt, gt.clone())
        s = scores(conf)
        self.assertAlmostEqual(s["miou"], 1.0)
        self.assertAlmostEqual(s["pixel_acc"], 1.0)
        self.assertEqual(s["n_classes_averaged"], 19)

    def test_hand_made_example(self):
        # 3 classes; the last pixel has gt 255 and must be ignored
        gt = torch.tensor([0, 0, 0, 1, 1, 2, 255])
        pred = torch.tensor([0, 0, 1, 1, 255, 2, 0])
        conf = update_confusion(new_confusion(3), gt, pred)
        expected = torch.tensor([[2, 1, 0, 0],
                                 [0, 1, 0, 1],
                                 [0, 0, 1, 0]])
        self.assertTrue(torch.equal(conf, expected))
        s = scores(conf)
        # class 0: TP 2, FN 1, FP 0 -> 2/3
        # class 1: TP 1, FN 1 (the void prediction), FP 1 -> 1/3
        # class 2: TP 1 -> 1
        self.assertAlmostEqual(s["iou"][0], 2 / 3)
        self.assertAlmostEqual(s["iou"][1], 1 / 3)
        self.assertAlmostEqual(s["iou"][2], 1.0)
        self.assertAlmostEqual(s["miou"], 2 / 3)
        self.assertAlmostEqual(s["pixel_acc"], 4 / 6)

    def test_class_without_ground_truth_is_left_out(self):
        # class 2 is predicted once but never appears in gt
        gt = torch.tensor([0, 0, 1, 1])
        pred = torch.tensor([0, 2, 1, 1])
        s = scores(update_confusion(new_confusion(3), gt, pred))
        self.assertEqual(s["classes_without_gt"], [2])
        self.assertEqual(s["n_classes_averaged"], 2)
        self.assertAlmostEqual(s["miou"], (1 / 2 + 1) / 2)

    def test_lut(self):
        labels = [Label("unlabeled", 0, 255), Label("road", 7, 0), Label("car", 26, 13),
                  Label("license plate", -1, -1)]
        lut = id_to_trainid_lut(labels)
        self.assertEqual(lut[7], 0)
        self.assertEqual(lut[26], 13)
        self.assertEqual(lut[0], 255)
        self.assertEqual(lut[200], 255)


if __name__ == "__main__":
    unittest.main()
