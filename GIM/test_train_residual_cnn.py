from datetime import datetime, timedelta
import unittest

import numpy as np
import torch

from train_residual_cnn import ResidualCNN, make_splits


class ResidualCNNTests(unittest.TestCase):
    def test_model_preserves_map_shape(self):
        model = ResidualCNN(channels=8, blocks=1)
        source = torch.randn(2, 1, 71, 72)
        output = model(source)
        self.assertEqual(tuple(output.shape), (2, 1, 71, 72))

    def test_last_three_natural_days_are_test(self):
        start = datetime(2021, 6, 25)
        times = [start + timedelta(minutes=20 * index) for index in range(6 * 72)]
        train, validation, test, test_start = make_splits(
            times, test_days=3, val_fraction=0.1, seed=42
        )
        self.assertEqual(test_start, datetime(2021, 6, 28))
        self.assertEqual(test.size, 3 * 72)
        self.assertEqual(train.size + validation.size, 3 * 72)
        self.assertEqual(validation.size, round(3 * 72 * 0.1))
        self.assertTrue(np.all(test >= 3 * 72))


if __name__ == "__main__":
    unittest.main()

