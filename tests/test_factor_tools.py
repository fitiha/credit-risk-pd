import numpy as np
from sklearn.decomposition import FactorAnalysis

from credit_risk.factor_tools import bartlett_sphericity, kmo, parallel_analysis, promax, tucker_congruence


def simulated(n=4000, corr=0.0, seed=3):
    rng = np.random.default_rng(seed)
    cov = np.array([[1, corr, 0], [corr, 1, 0], [0, 0, 1]])
    f = rng.multivariate_normal(np.zeros(3), cov, n)
    L = np.array([[.8, 0, 0], [.7, .2, 0], [0, .8, 0], [0, .6, .3], [0, 0, .9], [.3, 0, .7], [.5, .5, 0], [0, 0, .5]])
    return f @ L.T + rng.standard_normal((n, 8)) * .5


def test_bartlett_and_kmo_match_the_values_from_the_reference_library():
    Z = simulated()
    chi, p = bartlett_sphericity(Z)
    assert abs(chi - 13929.966) < 1  # value from factor_analyzer on this data
    per_field, overall = kmo(Z)
    assert abs(overall - 0.7066) < 1e-3 and per_field.shape == (8,) and p < 1e-6


def test_noise_is_not_suitable_for_factor_analysis():
    Z = np.random.default_rng(0).standard_normal((3000, 8))
    assert bartlett_sphericity(Z)[1] > 0.01
    assert kmo(Z)[1] < 0.55


def test_parallel_analysis_finds_the_true_number_of_factors():
    keep, observed, noise = parallel_analysis(simulated(), n_sim=30)
    assert keep == 3 and observed[3] <= noise[3]


def test_promax_recovers_the_true_factor_correlation():
    Z = simulated(corr=0.5)
    load = FactorAnalysis(n_components=3, rotation="varimax", random_state=0).fit(Z).components_.T
    rotated, phi = promax(load)
    assert np.allclose(np.diag(phi), 1, atol=1e-6)
    assert np.isclose(np.sort(np.abs(phi[np.triu_indices(3, 1)]))[-1], 0.5, atol=0.12)  # one pair is truly correlated at 0.5
    assert rotated.shape == (8, 3)


def test_tucker_congruence_is_one_for_the_same_factors_in_any_order():
    A = np.random.default_rng(1).standard_normal((10, 3))
    assert np.allclose(tucker_congruence(A, -A[:, [2, 0, 1]]), 1)
