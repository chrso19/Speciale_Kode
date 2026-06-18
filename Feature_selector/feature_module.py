"""
feature_selector_suite.py
─────────────────────────
Runs multiple feature selection methods on a time-series dataset and produces
a rank-aggregation table that can be used to pick a consensus feature set for
both shallow and deep learners.

Methods included
────────────────
  1. Mutual Information       – model-free, captures non-linear dependencies
  2. Lasso / ElasticNet       – linear model with automatic sparsity (L1/L2)
  3. Tree MDI + permutation   – non-linear, interaction-aware
  4. SHAP (TreeExplainer)     – faithful model-based importance via Shapley values
  5. RFECV (optional)         – wrapper method; respects temporal order via TimeSeriesSplit

Usage
─────
    from feature_selector_suite import FeatureSelectorSuite

    suite = FeatureSelectorSuite(
        X_train=X_train,   # pd.DataFrame, already scaled if needed
        y_train=y_train,   # pd.Series
        n_top=20,          # how many features to show in plots
    )
    results = suite.run_all()
    suite.plot_comparison()
    consensus = suite.consensus_features(top_k=15)

Notes
─────
- All selectors are fitted on the *training* fold only. Never pass the full
  walk-forward dataset or any validation/test data.
- If your X_train uses float32 (e.g. from a PyTorch pipeline), the suite
  converts it to float64 for sklearn compatibility.
- RFE is slow for large feature sets; set run_rfe=False to skip it.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.feature_selection import mutual_info_regression, RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import shap


# ──────────────────────────────────────────────────────────────────────────────

class FeatureSelectorSuite:
    """
    Runs multiple feature selectors on training data and aggregates their
    rankings into a single consensus score.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix (training fold only). Shape (n_samples, n_features).
    y_train : pd.Series or np.ndarray
        Target vector (training fold only).
    n_top : int
        Number of top features to highlight in plots.
    n_cv_splits : int
        Number of time-series CV splits used inside selectors that use CV.
    run_rfe : bool
        Whether to run RFECV (slow for large feature sets).
    random_state : int
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train,
        n_top: int = 20,
        n_cv_splits: int = 3,
        run_rfe: bool = False,
        random_state: int = 42,
    ):
        self.feature_names = list(X_train.columns)
        self.n_features = len(self.feature_names)
        self.n_top = min(n_top, self.n_features)
        self.n_cv_splits = n_cv_splits
        self.run_rfe = run_rfe
        self.random_state = random_state

        # Convert to float64 for sklearn compatibility
        self.X = X_train.astype(np.float64).values
        self.y = np.asarray(y_train, dtype=np.float64).ravel()

        # Scale X for linear methods (does not affect tree/MI methods)
        self._scaler = StandardScaler()
        self.X_scaled = self._scaler.fit_transform(self.X)

        # Results store: method → pd.Series(index=feature_names, values=score)
        self.scores: dict[str, pd.Series] = {}
        # Ranks store: method → pd.Series(index=feature_names, values=rank 1=best)
        self.ranks: dict[str, pd.Series] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run_all(self) -> pd.DataFrame:
        """
        Run all enabled selectors and return the rank-aggregation table.

        Returns
        -------
        pd.DataFrame
            Columns: one per method + 'mean_rank' + 'consensus_score'.
            Rows: one per feature. Sorted by mean_rank ascending.
        """
        print("Running feature selectors …")
        self._run_mutual_info()
        self._run_lasso()
        self._run_tree_mdi()
        self._run_tree_permutation()
        self._run_shap()
        if self.run_rfe:
            self._run_rfecv()

        # Build rank table
        rank_df = pd.DataFrame(self.ranks, index=self.feature_names)
        rank_df["mean_rank"] = rank_df.mean(axis=1)
        # Consensus score: inverse of mean rank, normalised 0-1
        inv = 1.0 / rank_df["mean_rank"]
        rank_df["consensus_score"] = inv / inv.sum()
        rank_df.sort_values("mean_rank", inplace=True)

        self.rank_table = rank_df
        print(f"\nDone. Top {self.n_top} features by consensus rank:")
        print(rank_df.head(self.n_top)[["mean_rank", "consensus_score"]].to_string())
        return rank_df

    def consensus_features(self, top_k: int = 15) -> list[str]:
        """Return the top-k features by mean rank across all selectors."""
        if not hasattr(self, "rank_table"):
            raise RuntimeError("Call run_all() first.")
        return list(self.rank_table.head(top_k).index)

    def plot_comparison(
        self,
        save_path: Optional[str] = None,
        figsize: tuple = (16, 10),
    ) -> None:
        """
        Plot a comparison grid: one bar chart per selector + one consensus bar.
        """
        if not hasattr(self, "rank_table"):
            raise RuntimeError("Call run_all() first.")

        methods = [m for m in self.scores]
        n_methods = len(methods)
        top_features = list(self.rank_table.head(self.n_top).index)

        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            "Feature selector comparison — electricity price forecasting",
            fontsize=13, fontweight="bold", y=0.98,
        )

        # Grid: n_methods panels + 1 consensus panel
        n_cols = min(3, n_methods + 1)
        n_rows = int(np.ceil((n_methods + 1) / n_cols))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.4)

        colours = {
            "mutual_info":        "#4A86C8",
            "lasso":              "#E87B44",
            "elasticnet":         "#E8A844",
            "tree_mdi":           "#5AAB61",
            "tree_permutation":   "#3D8B57",
            "shap":               "#9B59B6",
            "rfecv":              "#C0392B",
        }

        for idx, method in enumerate(methods):
            ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
            scores = self.scores[method].reindex(top_features).fillna(0)
            # Normalise scores to [0,1] for comparability
            if scores.max() > 0:
                scores = scores / scores.max()
            colour = colours.get(method, "#888888")
            bars = ax.barh(
                range(len(top_features)), scores.values,
                color=colour, alpha=0.75, height=0.65,
            )
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features, fontsize=7.5)
            ax.invert_yaxis()
            ax.set_xlabel("Normalised score", fontsize=8)
            ax.set_title(method.replace("_", " ").title(), fontsize=9, fontweight="bold")
            ax.tick_params(axis="x", labelsize=7.5)
            ax.set_xlim(0, 1.1)
            # Add thin grid
            ax.xaxis.grid(True, linewidth=0.4, alpha=0.5)
            ax.set_axisbelow(True)

        # Consensus panel (last slot)
        ax_c = fig.add_subplot(gs[(n_methods) // n_cols, (n_methods) % n_cols])
        cs = self.rank_table.loc[top_features, "consensus_score"]
        if cs.max() > 0:
            cs = cs / cs.max()
        ax_c.barh(
            range(len(top_features)), cs.values,
            color="#2C3E50", alpha=0.85, height=0.65,
        )
        ax_c.set_yticks(range(len(top_features)))
        ax_c.set_yticklabels(top_features, fontsize=7.5)
        ax_c.invert_yaxis()
        ax_c.set_xlabel("Consensus score", fontsize=8)
        ax_c.set_title("Consensus (mean rank)", fontsize=9, fontweight="bold",
                        color="#2C3E50")
        ax_c.tick_params(axis="x", labelsize=7.5)
        ax_c.set_xlim(0, 1.1)
        ax_c.xaxis.grid(True, linewidth=0.4, alpha=0.5)
        ax_c.set_axisbelow(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        plt.show()

    # ──────────────────────────────────────────────────────────────────────────
    # Individual selectors
    # ──────────────────────────────────────────────────────────────────────────

    def _run_mutual_info(self):
        print("  [1/5] Mutual information …", end=" ", flush=True)
        mi = mutual_info_regression(
            self.X, self.y,
            discrete_features=False,
            random_state=self.random_state,
        )
        self._store("mutual_info", mi)
        print("done")

    def _run_lasso(self):
        print("  [2/5] Lasso / ElasticNet …", end=" ", flush=True)
        tss = TimeSeriesSplit(n_splits=self.n_cv_splits)

        # Lasso
        lasso = LassoCV(cv=tss, max_iter=5000, random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso.fit(self.X_scaled, self.y)
        self._store("lasso", np.abs(lasso.coef_))

        # ElasticNet (handles correlated lags better)
        enet = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
            cv=tss, max_iter=5000, random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enet.fit(self.X_scaled, self.y)
        self._store("elasticnet", np.abs(enet.coef_))
        print("done")

    def _run_tree_mdi(self):
        print("  [3/5] Random Forest MDI …", end=" ", flush=True)
        rf = RandomForestRegressor(
            n_estimators=200,
            max_features="sqrt",
            n_jobs=-1,
            random_state=self.random_state,
        )
        rf.fit(self.X, self.y)
        self._store("tree_mdi", rf.feature_importances_)
        self._rf = rf  # reuse for permutation importance
        print("done")

    def _run_tree_permutation(self):
        print("  [4/5] Permutation importance …", end=" ", flush=True)
        if not hasattr(self, "_rf"):
            self._run_tree_mdi()
        result = permutation_importance(
            self._rf, self.X, self.y,
            n_repeats=10,
            random_state=self.random_state,
            n_jobs=-1,
        )
        # Use mean importance; clip negatives to 0 (feature worse than random)
        imp = np.clip(result.importances_mean, 0, None)
        self._store("tree_permutation", imp)
        print("done")

    def _run_shap(self):
        print("  [5/5] SHAP (LightGBM TreeExplainer) …", end=" ", flush=True)
        lgb = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=-1,
        )
        lgb.fit(self.X, self.y)
        explainer = shap.TreeExplainer(lgb)
        # Use a sample of up to 2000 rows to keep this fast
        n_sample = min(2000, len(self.X))
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(self.X), size=n_sample, replace=False)
        shap_values = explainer.shap_values(self.X[idx])
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        self._store("shap", mean_abs_shap)
        print("done")

    def _run_rfecv(self):
        print("  [6/6] RFECV (Lasso estimator, TimeSeriesSplit) …", end=" ", flush=True)
        from sklearn.linear_model import Lasso
        tss = TimeSeriesSplit(n_splits=self.n_cv_splits)
        estimator = Lasso(alpha=0.01, max_iter=5000)
        rfe = RFECV(
            estimator=estimator,
            step=1,
            cv=tss,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rfe.fit(self.X_scaled, self.y)
        # Ranking: 1 = selected, higher = eliminated earlier
        # Invert: lower rank number = more important
        rfe_score = 1.0 / rfe.ranking_.astype(float)
        self._store("rfecv", rfe_score)
        print("done")

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _store(self, name: str, raw_scores: np.ndarray):
        """Store a score array and derive its rank series."""
        s = pd.Series(raw_scores, index=self.feature_names, name=name)
        self.scores[name] = s
        # Rank: 1 = highest score (most important). Ties get average rank.
        self.ranks[name] = s.rank(ascending=False, method="average")


