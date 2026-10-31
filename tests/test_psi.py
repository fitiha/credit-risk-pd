import numpy as np

from credit_risk.psi import categorical_psi, label, numeric_psi, psi_from_shares


def test_identical_distributions_have_zero_psi():
    x = np.random.default_rng(0).normal(size=5000)
    assert numeric_psi(x, x) == 0
    assert psi_from_shares([.5, .5], [.5, .5]) == 0


def test_a_shifted_distribution_has_large_psi_and_a_small_shift_is_stable():
    rng = np.random.default_rng(1)
    a, same, shifted = rng.normal(size=8000), rng.normal(size=8000), rng.normal(1.0, 1, 8000)
    assert label(numeric_psi(a, same)) == "stable"
    assert label(numeric_psi(a, shifted)) == "major shift"


def test_known_textbook_value():
    assert abs(psi_from_shares([.4, .6], [.5, .5]) - 0.0407) < 1e-3


def test_categorical_and_missing_values():
    assert categorical_psi(["a"] * 80 + ["b"] * 20, ["a"] * 50 + ["b"] * 50) > 0.25
    assert numeric_psi([1, 2, 3, np.nan] * 500, [1, 2, 3, np.nan] * 500) == 0
