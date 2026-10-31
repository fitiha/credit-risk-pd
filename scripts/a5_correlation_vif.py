"""A6, part 1: Spearman correlation and multicollinearity (VIF) of the treated numeric fields.

Uses loans up to the fit cutoff. Writes reports/a5_correlation_vif/{figures,tables}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from statsmodels.stats.outliers_influence import variance_inflation_factor

from credit_risk.factor_fields import FACTOR_FIELDS
from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.reporting import fresh_dirs
from credit_risk.transform import ApplicationTransformer

FIG, TAB = fresh_dirs("a5_correlation_vif")
raw = load_applications()
d = ApplicationTransformer().fit(raw).transform(raw)
fit = d[d.issue_d <= FIT_CUTOFF]
X = fit[FACTOR_FIELDS]

corr = X.corr(method="spearman")
dist = 1 - corr.abs().to_numpy()
np.fill_diagonal(dist, 0)
order = leaves_list(linkage(squareform((dist + dist.T) / 2, checks=False), method="average"))
corr_ordered = corr.iloc[order, order]
corr.round(3).to_csv(TAB / "spearman_correlation.csv")

Z = (X - X.mean()) / X.std()
vif = pd.Series([variance_inflation_factor(Z.to_numpy(), i) for i in range(Z.shape[1])], index=FACTOR_FIELDS, name="vif").sort_values(ascending=False)
vif.round(2).to_csv(TAB / "vif.csv", header=True, index_label="field")

pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack().rename("spearman").reset_index()
pairs.columns = ["field_a", "field_b", "spearman"]
high = pairs[pairs.spearman.abs() >= 0.7].sort_values("spearman", key=abs, ascending=False)
high.round(3).to_csv(TAB / "highly_correlated_pairs.csv", index=False)

fig, ax = plt.subplots(figsize=(9.5, 8.5))
im = ax.imshow(corr_ordered, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(order)), corr_ordered.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(order)), corr_ordered.index, fontsize=7)
fig.colorbar(im, label="Spearman correlation")
ax.set_title("Spearman correlation of the treated numeric fields (loans to 2014; fields ordered by similarity)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "spearman_correlation.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 6))
v = vif.iloc[::-1]
ax.barh(v.index, v.values, color=["#c1440e" if x > 10 else "#e6a26a" if x > 5 else "#2a6f97" for x in v.values])
ax.axvline(5, color="grey", ls=":", lw=1); ax.axvline(10, color="grey", ls=":", lw=1)
ax.set_xlabel("Variance inflation factor (dotted: 5 and 10)"); ax.tick_params(axis="y", labelsize=7)
ax.set_title("How much each field repeats the others (VIF)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "vif.png", dpi=140); plt.close(fig)

off = corr.to_numpy()[np.triu_indices(len(corr), 1)]
print(f"{len(FACTOR_FIELDS)} fields, {len(fit):,} loans | share of field pairs with |rho| >= 0.3: {np.mean(np.abs(off) >= .3):.0%}, >= 0.5: {np.mean(np.abs(off) >= .5):.0%}, >= 0.7: {np.mean(np.abs(off) >= .7):.1%}")
print("VIF > 5:", int((vif > 5).sum()), "| VIF > 10:", int((vif > 10).sum()), "| max:", round(vif.max(), 1))
print("\nVIF top 10:\n", vif.head(10).round(1).to_string())
print("\npairs with |rho| >= 0.7:\n", high.round(2).to_string(index=False))
