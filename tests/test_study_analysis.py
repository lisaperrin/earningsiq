import numpy as np
import pandas as pd
import pytest

from earningsiq.study.analysis import (
    benjamini_hochberg,
    score_predictions,
    spearman_with_ci,
    standardize_within,
    tercile_means,
    walk_forward,
)
from earningsiq.study.features import add_change_features, keyword_tone, split_sentences


def test_split_sentences_drops_border_fragments_tables_and_duplicates():
    whole = "Net sales increased during the quarter because demand was strong."
    chunks = [
        f"{whole} Revenue | 100 | 90 | 11 %. The gross margin improved compared with the prior year perio",
        f"prior year period. {whole} Operating expenses grew more slowly than revenue in the same period.",
    ]

    sentences = split_sentences(chunks)

    assert sentences == [
        whole,
        "Operating expenses grew more slowly than revenue in the same period.",
    ]


def test_keyword_tone_counts_word_forms():
    assert keyword_tone(["Sales growth was strong and margins improved."]) == 1.0
    assert keyword_tone(["Sales declined and risks increased."]) == pytest.approx(-1 / 3)
    assert np.isnan(keyword_tone(["The company sells phones."]))


def test_change_features_compare_with_previous_filing_of_same_type():
    features = pd.DataFrame({
        "ticker": ["AAA"] * 4,
        "filing_type": ["10-Q", "10-K", "10-Q", "10-K"],
        "accession": ["q1", "k1", "q2", "k2"],
        "filing_accepted": pd.to_datetime(["2023-05-01", "2023-11-01", "2023-08-01", "2024-11-01"]),
        "kw_tone": [0.1, 0.5, 0.4, 0.2],
    })
    embeddings = pd.DataFrame({
        "accession": ["q1", "q2", "k1", "k2"],
        "section": ["MD&A"] * 4,
        "embedding": [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])],
    })

    result = add_change_features(features, embeddings).set_index("accession")

    assert np.isnan(result.loc["q1", "d_kw_tone"]) and np.isnan(result.loc["k1", "mdna_change"])
    assert result.loc["q2", "d_kw_tone"] == pytest.approx(0.3)   # vs q1, not vs the 10-K in between
    assert result.loc["k2", "d_kw_tone"] == pytest.approx(-0.3)
    assert result.loc["q2", "mdna_change"] == pytest.approx(1.0)  # orthogonal
    assert result.loc["k2", "mdna_change"] == pytest.approx(0.0)  # same direction
    assert result["risk_change"].isna().all()                     # no Risk Factors embeddings given


def test_standardize_within_removes_company_level():
    data = pd.DataFrame({"ticker": ["A", "A", "A", "B", "B", "B"], "tone": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]})

    z = standardize_within(data, ["tone"])["tone"]

    assert z.tolist() == pytest.approx([-1, 0, 1, -1, 0, 1])


def make_sample(effect, n=600, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({
        "x": x,
        "y": effect * x + rng.normal(size=n),
        "quarter": rng.integers(0, 24, n),
        "ticker": rng.integers(0, 20, n),
        "year": 2019 + rng.integers(0, 6, n),
    })


def test_spearman_detects_a_real_relation_and_not_noise():
    related = make_sample(effect=0.5)
    result = spearman_with_ci(related["x"], related["y"], related["quarter"], n_boot=300)
    assert result["ci_low"] > 0.2 and result["p"] < 0.01

    unrelated = make_sample(effect=0.0)
    result = spearman_with_ci(unrelated["x"], unrelated["y"], unrelated["quarter"], n_boot=300)
    assert result["ci_low"] < 0 < result["ci_high"]


def test_benjamini_hochberg():
    adjusted = benjamini_hochberg(pd.Series([0.01, 0.04, 0.03, 0.5]))

    assert adjusted.tolist() == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])


def test_tercile_means_orders_by_feature():
    data = make_sample(effect=1.0)

    means = tercile_means(data, "x", "y")["mean"]

    assert means["low"] < means["middle"] < means["high"]


def test_walk_forward_uses_only_earlier_years_and_scores_above_chance():
    data = make_sample(effect=1.0)

    predictions = walk_forward(data, ["x"], "y", test_years=[2022, 2023, 2024])
    scores = score_predictions(predictions, n_boot=200)

    assert set(predictions["year"]) == {2022, 2023, 2024}
    assert len(predictions) == (data["year"] >= 2022).sum()
    assert scores["auc"] > 0.7 and scores["auc_ci"][0] > 0.5

    noise = score_predictions(walk_forward(make_sample(effect=0.0), ["x"], "y", [2022, 2023, 2024]), n_boot=200)
    assert noise["auc_ci"][0] < 0.5 < noise["auc_ci"][1]
