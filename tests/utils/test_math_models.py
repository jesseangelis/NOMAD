"""Unit tests for nomad.utils.math_models."""

import numpy as np
import pytest

from nomad.utils.math_models import LogisticModel, get_sam_relevance_score


# ---------------------------------------------------------------------------
# LogisticModel.core
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_logistic_model_midpoint():
    """Verifies the 4PL model value at x=a equals the midpoint (c+d)/2."""
    a, b, c, d = 2.0, 1.0, 0.0, 1.0
    x = np.array([a])

    result = LogisticModel.core(x, a, b, c, d)

    # f(a) = c + (d-c) / (1 + 10^0) = (c + d) / 2
    assert np.isclose(result[0], (c + d) / 2.0, atol=1e-5)


@pytest.mark.unit
def test_logistic_model_converges_to_d_for_large_x():
    """Verifies the 4PL model converges to d for very large x.

    With b > 0, 10^(b(a-x)) → 0 as x → +∞, so the denominator → 1 and
    f(x) → c + (d-c) / 1 = d.
    """
    a, b, c, d = 0.0, 1.0, 0.1, 1.0
    x = np.array([100.0])  # far right of inflection point

    result = LogisticModel.core(x, a, b, c, d)

    assert np.isclose(result[0], d, atol=1e-3)


@pytest.mark.unit
def test_logistic_model_converges_to_c_for_small_x():
    """Verifies the 4PL model converges to c for very small x.

    With b > 0, 10^(b(a-x)) → +∞ as x → -∞, so the denominator → ∞ and
    f(x) → c + (d-c) / ∞ = c.
    """
    a, b, c, d = 0.0, 1.0, 0.1, 1.0
    x = np.array([-100.0])  # far left of inflection point

    result = LogisticModel.core(x, a, b, c, d)

    assert np.isclose(result[0], c, atol=1e-3)


@pytest.mark.unit
def test_logistic_model_returns_array():
    """Verifies that LogisticModel.core returns a numpy array."""
    x = np.linspace(-5, 5, 10)
    result = LogisticModel.core(x, 0.0, 1.0, 0.0, 1.0)

    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape


@pytest.mark.unit
def test_logistic_model_monotone_increasing():
    """Verifies that the model is monotone increasing for positive hill slope.

    With b > 0, as x increases, the exponent b(a-x) becomes more negative,
    so 10^(b(a-x)) decreases, the denominator decreases, and f(x) increases.
    """
    a, b, c, d = 0.0, 1.0, 0.0, 1.0
    x = np.linspace(-3, 3, 50)
    result = LogisticModel.core(x, a, b, c, d)

    assert np.all(np.diff(result) >= 0)


# ---------------------------------------------------------------------------
# get_sam_relevance_score
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_sam_relevance_score_low_n_eff_returns_zero():
    """Verifies that n_eff ≤ 4 returns a relevance score of 0 and p-value of 1."""
    rs, p_val = get_sam_relevance_score(
        rss_null=10.0, rss_alt=1.0, n_eff=4.0, lfc=2.0
    )

    assert rs == 0.0
    assert p_val == 1.0


@pytest.mark.unit
def test_get_sam_relevance_score_significant_result():
    """Verifies that a large F-statistic yields a high relevance score and low p-value."""
    rs, p_val = get_sam_relevance_score(
        rss_null=1000.0, rss_alt=1.0, n_eff=100.0, lfc=5.0
    )

    assert rs > 0.0
    assert p_val < 0.05


@pytest.mark.unit
def test_get_sam_relevance_score_null_equals_alt_is_nonsignificant():
    """Verifies that rss_null == rss_alt yields near-zero relevance score."""
    rs, p_val = get_sam_relevance_score(
        rss_null=10.0, rss_alt=10.0, n_eff=50.0, lfc=0.1
    )

    assert rs < 1.0
    assert p_val > 0.05


@pytest.mark.unit
def test_get_sam_relevance_score_returns_floats():
    """Verifies that get_sam_relevance_score returns a tuple of two floats."""
    result = get_sam_relevance_score(
        rss_null=5.0, rss_alt=1.0, n_eff=20.0, lfc=1.5
    )

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(v, float) for v in result)
