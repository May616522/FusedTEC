"""Small deterministic tests for the Sentinel-6 TEC batch pipeline."""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import sentinel6_tec_batch as batch


class Sentinel6TecBatchTests(unittest.TestCase):
    def test_ku_correction_conversion(self):
        correction = np.array([-40.308e16 / batch.F_KU_HZ**2])
        np.testing.assert_allclose(batch.correction_to_tecu(correction), [1.0])

    def test_metric_bias_and_r2(self):
        metric = batch.MetricAccumulator()
        metric.update([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        result = metric.result()
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["bias"], 1.0)
        self.assertAlmostEqual(result["rmse"], 1.0)
        self.assertAlmostEqual(result["r"], 1.0)
        self.assertAlmostEqual(result["r2"], -0.5)

    def test_external_gim_exact_grid_node(self):
        gim_root = batch.PROJECT_ROOT / "Data" / "GIM2021" / "JPL" / "FINAL"
        if not gim_root.is_dir():
            self.skipTest("Local JPL FINAL IONEX data are unavailable")
        interpolator = batch.IonexInterpolator(gim_root)
        times, lats, lons, cube = interpolator.load_day(pd.Timestamp("2021-03-01"))
        ti, yi, xi = 3, 20, 30
        frame = pd.DataFrame(
            {
                "time": [pd.Timestamp(times[ti])],
                "lat": [lats[yi]],
                "lon": [lons[xi]],
            }
        )
        actual = interpolator.interpolate_day(frame)[0]
        self.assertAlmostEqual(actual, cube[ti, yi, xi], places=10)

    def test_longitude_normalization_for_both_ionex_conventions(self):
        values = np.array([-190.0, -180.0, -170.0, 170.0, 180.0, 190.0, 550.0])
        negative_grid = np.arange(-180.0, 180.1, 5.0)
        zero_grid = np.arange(0.0, 360.1, 5.0)
        np.testing.assert_allclose(
            batch.IonexInterpolator.normalize_longitudes(values, negative_grid),
            [170.0, -180.0, -170.0, 170.0, -180.0, -170.0, -170.0],
        )
        np.testing.assert_allclose(
            batch.IonexInterpolator.normalize_longitudes(values, zero_grid),
            [170.0, 180.0, 190.0, 170.0, 180.0, 190.0, 190.0],
        )

    def test_interpolation_on_zero_to_360_grid_is_periodic(self):
        interpolator = batch.IonexInterpolator(Path("."))
        times = pd.to_datetime(["2021-03-01T00:00:00", "2021-03-01T02:00:00"])
        times = times.to_numpy(dtype="datetime64[ns]").astype("int64")
        lats = np.array([-10.0, 10.0])
        lons = np.array([0.0, 180.0, 360.0])
        one_map = np.array([[0.0, 18.0, 0.0], [0.0, 18.0, 0.0]])
        cube = np.stack((one_map, one_map))
        interpolator.load_day = lambda day: (times, lats, lons, cube)
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(["2021-03-01T01:00:00"] * 3),
                "lat": [0.0, 0.0, 0.0],
                "lon": [-90.0, 270.0, 630.0],
            }
        )
        np.testing.assert_allclose(interpolator.interpolate_day(frame), [9.0, 9.0, 9.0])

    def test_read_one_netcdf(self):
        source = batch.DEFAULT_INPUT
        files = sorted(Path(source).glob("*.nc"))
        if not files:
            self.skipTest("Local Sentinel-6 data are unavailable")
        frame = batch.read_sentinel6_file(files[0])
        self.assertGreater(len(frame), 0)
        # External GIM columns are added later by the interpolators, not by
        # the single-NetCDF reader.
        self.assertTrue(set(batch.TEC_COLUMNS[:-2]).issubset(frame.columns))
        self.assertTrue(frame["time"].is_monotonic_increasing)

    def test_nr_fraction_0881_matches_official_better_than_0900(self):
        source = batch.DEFAULT_INPUT
        files = sorted(Path(source).glob("*.nc"))
        if not files:
            self.skipTest("Local Sentinel-6 data are unavailable")
        frame = batch.apply_qc(batch.read_sentinel6_file(files[0]), "basic")
        below = frame["tec_below_filtered_tecu"].to_numpy(dtype=float)
        official = frame["tec_official_tecu"].to_numpy(dtype=float)
        valid = np.isfinite(below) & np.isfinite(official)
        error_0881 = below[valid] / 0.881 - official[valid]
        error_0900 = below[valid] / 0.900 - official[valid]
        rmse_0881 = np.sqrt(np.mean(error_0881**2))
        rmse_0900 = np.sqrt(np.mean(error_0900**2))
        self.assertLess(rmse_0881, 0.1)
        self.assertLess(rmse_0881, rmse_0900)


if __name__ == "__main__":
    unittest.main()
