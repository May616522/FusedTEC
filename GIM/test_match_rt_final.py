from datetime import datetime, timedelta, timezone
import unittest

import numpy as np

from match_rt_final import final_bracket, interpolate_rotated_final, rotate_global_map


UTC = timezone.utc


class InterpolationTests(unittest.TestCase):
    def setUp(self):
        self.lon = np.arange(-180.0, 181.0, 5.0, dtype=np.float32)

    def test_periodic_rotation_crosses_seam(self):
        unique = np.arange(72, dtype=np.float32)[None, :]
        source = np.concatenate([unique, unique[:, :1]], axis=1)
        rotated = rotate_global_map(source, self.lon, 5.0)
        np.testing.assert_allclose(rotated[0, :-1], np.roll(unique[0], -1))
        self.assertEqual(rotated[0, 0], rotated[0, -1])

    def test_half_grid_rotation_is_linear(self):
        unique = np.arange(72, dtype=np.float32)[None, :]
        source = np.concatenate([unique, unique[:, :1]], axis=1)
        rotated = rotate_global_map(source, self.lon, 2.5)
        self.assertAlmostEqual(float(rotated[0, 10]), 10.5)

    def test_cross_day_bracket_and_rotated_interpolation(self):
        t0 = datetime(2021, 4, 1, 22, tzinfo=UTC)
        t1 = datetime(2021, 4, 2, 0, tzinfo=UTC)
        target = datetime(2021, 4, 1, 23, 40, tzinfo=UTC)
        bracket = final_bracket(target, [t0, t1], timedelta(hours=2.01))
        self.assertIsNotNone(bracket)
        self.assertAlmostEqual(bracket[2], 5.0 / 6.0)
        base = np.ones((1, 73), dtype=np.float32)
        maps = np.stack([base * 10.0, base * 22.0])
        result = interpolate_rotated_final(
            target, bracket, [t0, t1], maps, self.lon, 15.0
        )
        np.testing.assert_allclose(result, 20.0, atol=1e-6)

    def test_exact_final_epoch_needs_no_future_map(self):
        epoch = datetime(2021, 4, 2, 0, tzinfo=UTC)
        self.assertEqual(final_bracket(epoch, [epoch], timedelta(hours=2.01)), (0, 0, 0.0))


if __name__ == "__main__":
    unittest.main()

