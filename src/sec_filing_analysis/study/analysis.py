"""Statistical tests for the text-price study.

Two questions are kept apart:

  Association  Does the text measure move together with the abnormal return?
               Reported as rank correlations with confidence intervals.
  Prediction   Does a model fitted on earlier years tell the direction of the
               abnormal return in later years better than a naive guess?

Filings of one earnings season share the same market mood, so observations
are not independent. All uncertainty estimates therefore resample whole
calendar quarters rather than single filings.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

TEXT_FEATURES = [
    "kw_tone", "finbert_tone", "llm_tone",
    "d_kw_tone", "d_finbert_tone", "d_llm_tone",
    "mdna_change", "risk_change",
]


def winsorize(values: pd.Series, tail: float = 0.01) -> pd.Series:
    """Clip the most extreme values so single outliers do not drive results."""
    return values.clip(values.quantile(tail), values.quantile(1 - tail))


def standardize_within(data: pd.DataFrame, columns: list[str], by: str = "ticker") -> pd.DataFrame:
    """z-score each column within a company.

    Companies differ in how they write: a bank's MD&A reads more cautiously
    than a software company's in every quarter. What can carry information is
    how a filing compares with the same company's other filings.
    """
    data = data.copy()
    grouped = data.groupby(by)[columns]
    data[columns] = (data[columns] - grouped.transform("mean")) / grouped.transform("std")
    return data


def residualize(values: pd.Series, controls: pd.DataFrame) -> pd.Series:
    """What is left of `values` after removing the part explained by `controls`.

    Works on ranks, to match the rank correlations used elsewhere: both sides
    are ranked, a linear fit of the controls is subtracted, and the residual is
    returned. Rows with missing data are NaN.
    """
    keep = values.notna() & controls.notna().all(axis=1)
    y = rankdata(values[keep])
    design = np.column_stack([np.ones(keep.sum())] + [rankdata(controls.loc[keep, column]) for column in controls])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)

    residual = pd.Series(np.nan, index=values.index)
    residual[keep] = y - design @ coefficients
    return residual


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def _cluster_rows(clusters: pd.Series) -> list[np.ndarray]:
    codes = pd.factorize(clusters)[0]
    return [np.flatnonzero(codes == code) for code in np.unique(codes)]


def spearman_with_ci(x: pd.Series, y: pd.Series, clusters: pd.Series, n_boot: int = 2000, seed: int = 0) -> dict:
    """Spearman correlation with a cluster-bootstrap 95% interval and p-value."""
    keep = (x.notna() & y.notna()).to_numpy()
    x, y, clusters = x.to_numpy()[keep], y.to_numpy()[keep], clusters[keep]
    rho = _rank_correlation(x, y)

    rows = _cluster_rows(clusters)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        sample = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        draws[b] = _rank_correlation(x[sample], y[sample])

    p_value = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {
        "n": int(keep.sum()),
        "rho": rho,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p": float(min(max(p_value, 1 / n_boot), 1.0)),
    }


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Adjust p-values for the number of tests (false discovery rate)."""
    order = np.argsort(p_values.to_numpy())
    ranked = p_values.to_numpy()[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    result = np.empty(len(p_values))
    result[order] = adjusted
    return pd.Series(result, index=p_values.index)


def association_table(data: pd.DataFrame, features: list[str], targets: list[str], clusters: str = "quarter") -> pd.DataFrame:
    """Correlation of every feature with every target, adjusted for multiple testing."""
    rows = []
    for target in targets:
        for feature in features:
            result = spearman_with_ci(data[feature], data[target], data[clusters])
            rows.append({"target": target, "feature": feature, **result})

    table = pd.DataFrame(rows)
    table["q"] = benjamini_hochberg(table["p"])
    return table


def tercile_means(data: pd.DataFrame, feature: str, target: str, by: str = "ticker") -> pd.DataFrame:
    """Mean target by low / middle / high third of a feature, ranked within company."""
    rank = data.groupby(by)[feature].rank(pct=True)
    tercile = pd.cut(rank, [0, 1 / 3, 2 / 3, 1], labels=["low", "middle", "high"])
    grouped = data.groupby(tercile, observed=True)[target]
    return pd.DataFrame({"mean": grouped.mean(), "sem": grouped.sem(), "n": grouped.count()})


def walk_forward(data: pd.DataFrame, features: list[str], target: str, test_years: list[int]) -> pd.DataFrame:
    """Out-of-sample predictions of the sign of `target`.

    For each test year a logistic regression is fitted on all earlier years
    only. Features are z-scored within company using the training years'
    mean and spread, so nothing from the test year leaks into the model.
    Returns one row per test observation with the predicted probability of a
    positive abnormal return and the naive prediction (the direction that was
    more frequent in the training years).
    """
    data = data.dropna(subset=features + [target])
    predictions = []

    for year in test_years:
        train, test = data[data["year"] < year].copy(), data[data["year"] == year].copy()
        if len(train) < 50 or test.empty:
            continue

        company = train.groupby("ticker")[features]
        mean, spread = company.mean(), company.std().replace(0, np.nan)
        for part in (train, test):
            part[features] = (
                (part[features] - mean.reindex(part["ticker"]).to_numpy())
                / spread.reindex(part["ticker"]).to_numpy()
            ).fillna(0.0)

        scaler = StandardScaler().fit(train[features])
        y_train = (train[target] > 0).astype(int)
        model = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(train[features]), y_train)

        predictions.append(pd.DataFrame({
            "year": year,
            "quarter": test["quarter"].to_numpy(),
            "actual": (test[target] > 0).astype(int).to_numpy(),
            "probability": model.predict_proba(scaler.transform(test[features]))[:, 1],
            "naive": int(y_train.mean() >= 0.5),
        }))

    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def score_predictions(predictions: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Accuracy and AUC of walk-forward predictions with cluster-bootstrap intervals.

    AUC is the probability that a filing followed by a positive abnormal
    return received a higher score than one followed by a negative return;
    0.5 is chance. `accuracy_gain` is accuracy minus the naive guess.
    """
    actual = predictions["actual"].to_numpy()
    probability = predictions["probability"].to_numpy()
    correct = ((probability >= 0.5).astype(int) == actual).astype(float)
    naive_correct = (predictions["naive"].to_numpy() == actual).astype(float)

    rows = _cluster_rows(predictions["quarter"])
    rng = np.random.default_rng(seed)
    auc_draws, gain_draws = [], []
    for _ in range(n_boot):
        sample = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        if 0 < actual[sample].mean() < 1:
            auc_draws.append(roc_auc_score(actual[sample], probability[sample]))
        gain_draws.append(correct[sample].mean() - naive_correct[sample].mean())

    return {
        "n": int(len(actual)),
        "share_positive": float(actual.mean()),
        "accuracy": float(correct.mean()),
        "naive_accuracy": float(naive_correct.mean()),
        "accuracy_gain": float(correct.mean() - naive_correct.mean()),
        "accuracy_gain_ci": [float(np.percentile(gain_draws, 2.5)), float(np.percentile(gain_draws, 97.5))],
        "auc": float(roc_auc_score(actual, probability)),
        "auc_ci": [float(np.percentile(auc_draws, 2.5)), float(np.percentile(auc_draws, 97.5))],
    }
