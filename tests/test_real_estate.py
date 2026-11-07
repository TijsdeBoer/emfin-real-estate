"""Tests for toolbox/real_estate.py. Validation uses synthetic data with known truth."""

import numpy as np
import pytest

from emfin_capstone.toolbox import real_estate as re_


def _smooth(r, c):
    o = np.empty(len(r) + 1)
    o[0] = 0.0
    for t in range(1, len(r) + 1):
        o[t] = (1 - c) * r[t - 1] + c * o[t - 1]
    return o


class TestUnsmoothing:
    def test_geltner_recovers_truth_from_ar1_smoothing(self):
        r = np.random.default_rng(0).standard_normal(400) * 0.03
        rec = re_.geltner_unsmooth(_smooth(r, 0.6), 0.6)
        np.testing.assert_allclose(rec[1:], r[1:], atol=1e-10)

    def test_fgw_zeroes_lag1_autocorr(self):
        r = np.random.default_rng(1).standard_normal(500) * 0.03
        unsmoothed, c_hat = re_.fgw_unsmooth(_smooth(r, 0.5))
        assert abs(np.corrcoef(unsmoothed[1:], unsmoothed[:-1])[0, 1]) < 0.05
        assert c_hat == pytest.approx(0.5, abs=0.15)

    def test_unsmoothing_raises_vol(self):
        r = np.random.default_rng(2).standard_normal(500) * 0.03
        o = _smooth(r, 0.6)
        assert re_.geltner_unsmooth(o, 0.6).std() > o[1:].std() * 1.5

    def test_invalid_smoothing_raises(self):
        with pytest.raises(ValueError):
            re_.geltner_unsmooth(np.arange(10.0), 1.0)


class TestStateSpace:
    def test_kalman_recovers_latent_path(self):
        obs, truth = re_.simulate_smoothed_observations(
            240, 0.01, 0.3, 0.0004, (0.6, 0.3), (1e-5, 1e-5), seed=3)
        model = re_.LatentReturnModel(0.01, 0.3, 0.0004, (0.6, 0.3), (1e-5, 1e-5))
        latent = re_.kalman_smooth(obs, model)["latent_smoothed"]
        corr = np.corrcoef(latent, truth)[0, 1]
        assert corr > 0.9
        assert corr > np.corrcoef(obs[:, 0], truth)[0, 1]

    def test_missing_observations_handled(self):
        obs, _ = re_.simulate_smoothed_observations(
            120, 0.01, 0.2, 0.0004, (0.5, 0.2), (1e-5, 1e-5), seed=4)
        obs[::7, 1] = np.nan
        model = re_.LatentReturnModel(0.01, 0.2, 0.0004, (0.5, 0.2), (1e-5, 1e-5))
        assert np.isfinite(re_.kalman_smooth(obs, model)["latent_smoothed"]).all()

    def test_moment_fit_plausible(self):
        obs, _ = re_.simulate_smoothed_observations(400, 0.01, 0.0, 0.0004, (0.55,), (1e-5,), seed=5)
        model = re_.moment_fit(obs)
        assert model.mu == pytest.approx(0.01, abs=0.01)
        assert 0.2 < model.smoothing[0] < 0.9

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            re_.LatentReturnModel(0.0, 1.0, 0.1, (0.5,), (0.1,))


class TestDiagnostics:
    def test_iid_residual_passes_ljung_box(self):
        x = np.random.default_rng(6).standard_normal(400)
        assert re_.ljung_box(x, 10)["p_value"] > 0.01

    def test_autocorrelated_series_fails_ljung_box(self):
        e = np.random.default_rng(7).standard_normal(400)
        assert re_.ljung_box(np.convolve(e, [0.8, 0.5])[:400], 10)["p_value"] < 0.01

    def test_arch_lm_detects_vol_clustering(self):
        rng = np.random.default_rng(8)
        vol = 1.0 + np.arange(400) % 50 / 25
        assert re_.arch_lm(rng.standard_normal(400) * vol, 5)["p_value"] < 0.01
        assert re_.arch_lm(rng.standard_normal(400), 5)["p_value"] > 0.01


class TestBootstrap:
    def test_shape_and_dependence(self):
        rng = np.random.default_rng(9)
        common = rng.standard_normal(300)
        x = np.column_stack([common + 0.1 * rng.standard_normal(300),
                             common + 0.1 * rng.standard_normal(300)])
        scen = re_.block_bootstrap_with_parameter_uncertainty(x, 4, 5000, 12, seed=10)
        assert scen.shape == (5000, 2)
        assert np.corrcoef(scen[:, 0], scen[:, 1])[0, 1] > 0.8

    def test_parameter_uncertainty_changes_scenarios(self):
        base = np.random.default_rng(11).standard_normal((300, 1)) * 0.02
        scen = re_.block_bootstrap_with_parameter_uncertainty(
            base, 4, 3000, 12, 12, param_draws=(np.array([0.2, 0.6, 0.8]),),
            refit_fn=lambda c: base * (1 + c))
        assert scen.shape == (3000, 1)
        assert scen.std() > base.std() * 2

    def test_bad_block_length_raises(self):
        with pytest.raises(ValueError):
            re_.block_bootstrap_with_parameter_uncertainty(np.ones((10, 1)), 2, 10, 50, 0)
