"""Small, standard factor-analysis helpers: suitability tests, parallel analysis, promax rotation, congruence."""
import numpy as np
from scipy.stats import chi2


def bartlett_sphericity(Z):
    """Bartlett's test that the correlation matrix is the identity (no shared structure). Returns (chi-square, p-value)."""
    n, p = Z.shape
    R = np.corrcoef(Z, rowvar=False)
    stat = -(n - 1 - (2 * p + 5) / 6) * np.linalg.slogdet(R)[1]
    return float(stat), float(chi2.sf(stat, p * (p - 1) / 2))


def kmo(Z):
    """Kaiser-Meyer-Olkin measure per field and overall: how much of the correlation is shared rather than pairwise noise.
    Above 0.6 is the usual minimum for factor analysis."""
    R = np.corrcoef(Z, rowvar=False)
    inv = np.linalg.inv(R)
    d = np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    partial = -inv / d
    np.fill_diagonal(partial, 0)
    r2 = R.copy()
    np.fill_diagonal(r2, 0)
    per_field = (r2 ** 2).sum(axis=0) / ((r2 ** 2).sum(axis=0) + (partial ** 2).sum(axis=0))
    overall = (r2 ** 2).sum() / ((r2 ** 2).sum() + (partial ** 2).sum())
    return per_field, float(overall)


def reduced_eigenvalues(R):
    """Eigenvalues of the correlation matrix with each diagonal replaced by the field's squared multiple correlation."""
    smc = 1 - 1 / np.diag(np.linalg.inv(R))
    R2 = R.copy()
    np.fill_diagonal(R2, smc)
    return np.sort(np.linalg.eigvalsh(R2))[::-1]


def parallel_analysis(Z, n_sim=100, percentile=95, seed=0):
    """Number of factors to keep: those whose eigenvalue beats the same eigenvalue of pure noise of the same shape."""
    n, p = Z.shape
    observed = reduced_eigenvalues(np.corrcoef(Z, rowvar=False))
    rng = np.random.default_rng(seed)
    sims = np.array([reduced_eigenvalues(np.corrcoef(rng.standard_normal((n, p)), rowvar=False)) for _ in range(n_sim)])
    noise = np.percentile(sims, percentile, axis=0)
    keep = int(np.argmax(observed <= noise)) if (observed <= noise).any() else p
    return keep, observed, noise


def promax(loadings, power=4):
    """Oblique (promax) rotation of varimax loadings. Returns (rotated loadings, factor correlation matrix)."""
    A = np.asarray(loadings, float)
    target = A * np.abs(A) ** (power - 1)
    U = np.linalg.solve(A.T @ A, A.T @ target)
    U = U @ np.diag(np.sqrt(np.diag(np.linalg.inv(U.T @ U))))
    rotated = A @ U
    Ui = np.linalg.inv(U)
    return rotated, Ui @ Ui.T


def tucker_congruence(A, B):
    """For each column of A, the best |cosine| with any column of B (1 = the same factor)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    cos = np.abs(A.T @ B) / np.sqrt(np.outer((A ** 2).sum(axis=0), (B ** 2).sum(axis=0)))
    return cos.max(axis=1)
