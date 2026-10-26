"""Pure-math tests for python.data.emos.emos_probability."""
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from python.data.emos import EmosCoefficients, emos_probability


def _coeffs(a=0.0, b=1.0, c=1.0, d=0.0) -> EmosCoefficients:
    return EmosCoefficients(
        a=a, b=b, c=c, d=d, n_train=100, low_n=False, ci_warnings=[],
        leads_pooled=[0, 1, 2], by_lead=None, ens_std_p99=5.0,
    )


class TestBetween:
    def test_basic_between(self):
        coeffs = _coeffs()
        prob, cdf_floor, cdf_cap = emos_probability(
            coeffs, ens_mean=70.0, ens_std=2.0,
            strike_type="between", floor=68.0, cap=72.0,
        )
        # μ = 70, σ = sqrt(c + d*std^2) = sqrt(1 + 0) = 1
        # Φ((72-70)/1) − Φ((68-70)/1) ≈ 0.9772 − 0.0228 ≈ 0.9544
        assert 0.0 <= prob <= 1.0
        assert math.isclose(prob, 0.9544997, rel_tol=1e-4)
        assert cdf_floor < cdf_cap

    def test_between_requires_both_strikes(self):
        coeffs = _coeffs()
        with pytest.raises(ValueError):
            emos_probability(coeffs, 70.0, 2.0, strike_type="between", floor=68.0, cap=None)
        with pytest.raises(ValueError):
            emos_probability(coeffs, 70.0, 2.0, strike_type="between", floor=None, cap=72.0)


class TestGreater:
    def test_greater_basic(self):
        coeffs = _coeffs()
        prob, cdf_floor, cdf_cap = emos_probability(
            coeffs, 70.0, 2.0, strike_type="greater", floor=70.0, cap=None,
        )
        assert math.isclose(prob, 0.5, abs_tol=1e-6)
        assert math.isclose(cdf_floor, 0.5, abs_tol=1e-6)
        assert cdf_cap is None  # missing-by-design, NOT NaN

    def test_greater_rejects_cap(self):
        coeffs = _coeffs()
        with pytest.raises(ValueError):
            emos_probability(coeffs, 70.0, 2.0, strike_type="greater", floor=70.0, cap=72.0)


class TestLess:
    def test_less_basic(self):
        coeffs = _coeffs()
        prob, cdf_floor, cdf_cap = emos_probability(
            coeffs, 70.0, 2.0, strike_type="less", floor=None, cap=70.0,
        )
        assert math.isclose(prob, 0.5, abs_tol=1e-6)
        assert cdf_floor is None
        assert math.isclose(cdf_cap, 0.5, abs_tol=1e-6)

    def test_less_rejects_floor(self):
        coeffs = _coeffs()
        with pytest.raises(ValueError):
            emos_probability(coeffs, 70.0, 2.0, strike_type="less", floor=68.0, cap=70.0)


class TestSigmaFloor:
    def test_extremely_small_c_d_does_not_produce_zero_sigma(self):
        coeffs = _coeffs(c=0.0, d=0.0)
        prob, _, _ = emos_probability(
            coeffs, 70.0, 0.0, strike_type="between", floor=69.99, cap=70.01,
        )
        assert math.isfinite(prob)
        assert 0.0 <= prob <= 1.0


class TestPropertyBounds:
    @given(
        a=st.floats(min_value=-50.0, max_value=50.0),
        b=st.floats(min_value=0.3, max_value=1.5),
        c=st.floats(min_value=1e-6, max_value=5.0),
        d=st.floats(min_value=0.0, max_value=2.0),
        ens_mean=st.floats(min_value=-50.0, max_value=150.0),
        ens_std=st.floats(min_value=0.0, max_value=20.0),
    )
    @settings(max_examples=200, deadline=None)
    def test_between_outputs_in_unit_interval(self, a, b, c, d, ens_mean, ens_std):
        coeffs = _coeffs(a=a, b=b, c=c, d=d)
        prob, cdf_floor, cdf_cap = emos_probability(
            coeffs, ens_mean, ens_std, strike_type="between", floor=ens_mean - 5, cap=ens_mean + 5,
        )
        assert 0.0 <= cdf_floor <= cdf_cap <= 1.0
        assert 0.0 <= prob <= 1.0
        assert math.isfinite(prob)

    @given(
        ens_mean=st.floats(min_value=50.0, max_value=90.0),
        ens_std=st.floats(min_value=1.0, max_value=10.0),
        floor=st.floats(min_value=40.0, max_value=80.0),
        cap=st.floats(min_value=80.0, max_value=120.0),
    )
    @settings(max_examples=200, deadline=None)
    def test_between_equals_greater_minus_greater(self, ens_mean, ens_std, floor, cap):
        coeffs = _coeffs()
        p_between, _, _ = emos_probability(
            coeffs, ens_mean, ens_std, strike_type="between", floor=floor, cap=cap,
        )
        p_floor, _, _ = emos_probability(
            coeffs, ens_mean, ens_std, strike_type="greater", floor=floor, cap=None,
        )
        p_cap, _, _ = emos_probability(
            coeffs, ens_mean, ens_std, strike_type="greater", floor=cap, cap=None,
        )
        assert math.isclose(p_between, p_floor - p_cap, abs_tol=1e-10)
