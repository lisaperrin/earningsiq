"""
09_earnings_release_study.py - What in an earnings release shows up in the stock?

Script 08 found that the 10-K/10-Q itself is not what the market reacts to.
This script looks at the earnings release that precedes it:

  Numbers  earnings surprise against analyst estimates (Yahoo Finance) and
           against the company's own history (SEC structured data)
  Text     tone of the press release (exhibit 99.1 of the 8-K), measured with
           keywords, FinBERT and the LLM
  Outcome  direction of the abnormal return, the month after, and the size of
           the move and the change in volatility

Usage:
    python scripts/09_earnings_release_study.py            # cached where possible
    python scripts/09_earnings_release_study.py --no-llm   # skip the LLM tone measure
    python scripts/09_earnings_release_study.py --refresh  # download and score again

Run script 08 first: it dates the releases and caches the prices.
"""

import sys
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earningsiq.config import settings
from earningsiq.study import analysis, fundamentals, plots, press_releases
from earningsiq.study.returns import EventStudy
from earningsiq.utils.logger import log

NUMBER_FEATURES = ["analyst_surprise", "income_surprise", "revenue_surprise"]
TONE_FEATURES = ["pr_kw_tone", "pr_finbert_tone", "pr_llm_tone", "d_pr_kw_tone", "d_pr_finbert_tone", "d_pr_llm_tone"]
PATH_WINDOW = (-5, 21)
TEST_YEARS = [2022, 2023, 2024]


def cached(path: Path, refresh: bool, build):
    if path.exists() and not refresh:
        log.info(f"Using cached {path.name}")
        return pd.read_parquet(path)
    frame = build()
    frame.to_parquet(path)
    return frame


def build_dataset(study_dir: Path, refresh: bool, use_llm: bool):
    filings_path = study_dir / "study_dataset.parquet"
    if not filings_path.exists():
        raise SystemExit("Run scripts/08_text_price_study.py first")

    # One row per earnings release
    filings = pd.read_parquet(filings_path)
    data = (
        filings.dropna(subset=["release_accession", "release_session"])
        .sort_values("filing_accepted")
        .drop_duplicates("release_accession")
        [["ticker", "cik", "release_accession", "release_accepted", "release_session", "period_end",
          "car_announce", "car_announce_simple", "move_release"]]
        .reset_index(drop=True)
    )
    tickers = sorted(data["ticker"].unique())

    prices = pd.read_parquet(study_dir / "prices.parquet")
    study = EventStudy(prices)
    events = list(zip(data["ticker"], data["release_session"]))
    data["car_release_drift"] = [study.car(ticker, session, 2, 21) for ticker, session in events]
    # Robustness check: stock minus market, without an estimated trend
    simple = EventStudy(prices, market_adjusted=True)
    data["car_release_drift_simple"] = [simple.car(ticker, session, 2, 21) for ticker, session in events]
    # Volatility in the month after the release relative to the month before
    data["vol_change"] = np.log([
        study.volatility(ticker, session, 2, 21) / study.volatility(ticker, session, -21, -2)
        for ticker, session in events
    ])

    paths = {}
    for accession, (ticker, session) in zip(data["release_accession"], events):
        abnormal = study.abnormal_returns(ticker, session, *PATH_WINDOW)
        if abnormal is not None:
            paths[accession] = abnormal
    paths = pd.DataFrame(paths).T

    # Numbers
    history = cached(study_dir / "surprises.parquet", refresh, lambda: fundamentals.fetch_surprises(tickers))
    analyst = cached(study_dir / "analyst_surprises.parquet", refresh, lambda: fundamentals.fetch_analyst_surprises(tickers))
    data = fundamentals.attach_surprises(data, history)
    data = fundamentals.attach_analyst_surprises(data, analyst)

    # Text
    texts = cached(study_dir / "press_releases.parquet", refresh, lambda: press_releases.fetch_press_releases(data))
    tone = cached(
        study_dir / "press_release_features.parquet", refresh,
        lambda: press_releases.build_release_features(texts, data, use_llm=use_llm),
    )
    data = data.merge(tone, on="release_accession", how="left")

    data["year"] = pd.DatetimeIndex(data["release_session"]).year
    data["quarter"] = pd.DatetimeIndex(data["release_session"]).to_period("Q").astype(str)
    return data, paths


def run_analysis(data: pd.DataFrame, suffix: str = "") -> dict:
    """`suffix` selects the return definition ("" = market model, "_simple" = market-adjusted)."""
    tone = [feature for feature in TONE_FEATURES if feature in data]
    features = NUMBER_FEATURES + tone
    targets = ["car_announce", "car_release_drift"]

    data = data.copy()
    for column in targets:
        data[column] = data[f"{column}{suffix}"]

    clean = data.copy()
    for column in targets + NUMBER_FEATURES:
        clean[column] = analysis.winsorize(clean[column])
    # Tone is compared within a company; surprises are already relative to
    # the company's own estimate or history.
    clean = analysis.standardize_within(clean, tone)

    association = analysis.association_table(clean, features, targets)

    # Does the text say anything the numbers do not? Remove the part of the
    # reaction that the surprises explain and correlate tone with the rest.
    clean["car_announce_beyond_numbers"] = analysis.residualize(clean["car_announce"], clean[NUMBER_FEATURES])
    beyond = analysis.association_table(clean, tone, ["car_announce_beyond_numbers"])
    association = pd.concat([association, beyond], ignore_index=True)
    association["q"] = analysis.benjamini_hochberg(association["p"])

    # Size of the move rather than its direction
    size = clean.assign(**{f"abs_{feature}": clean[feature].abs() for feature in features})
    magnitude = pd.concat([
        analysis.association_table(size, [f"abs_{feature}" for feature in features], ["move_release"]),
        analysis.association_table(size, features, ["vol_change"]),
    ], ignore_index=True)
    magnitude["q"] = analysis.benjamini_hochberg(magnitude["p"])

    feature_sets = {
        "own history": ["income_surprise", "revenue_surprise"],
        "analyst surprise": ["analyst_surprise"],
        "tone": tone,
        "analyst surprise + tone": ["analyst_surprise"] + tone,
    }
    complete = data.dropna(subset=features + targets).copy()
    for column in NUMBER_FEATURES:
        complete[column] = analysis.winsorize(complete[column])
    prediction = []
    for target in targets:
        for name, columns in feature_sets.items():
            scores = analysis.score_predictions(analysis.walk_forward(complete, columns, target, TEST_YEARS))
            prediction.append({"target": target, "features": name, **scores})

    return {"data": clean, "association": association, "magnitude": magnitude, "prediction": pd.DataFrame(prediction)}


def describe_sample(data: pd.DataFrame) -> dict:
    surprise_third = pd.qcut(data["analyst_surprise"], 3, labels=["lowest", "middle", "highest"])
    by_third = data.groupby(surprise_third, observed=True)
    tone = [feature for feature in ("pr_kw_tone", "pr_finbert_tone", "pr_llm_tone") if feature in data]
    return {
        "releases": int(len(data)),
        "companies": int(data["ticker"].nunique()),
        "with_analyst_surprise": int(data["analyst_surprise"].notna().sum()),
        "with_history_surprise": int(data["income_surprise"].notna().sum()),
        "with_press_release_tone": int(data["pr_finbert_tone"].notna().sum()),
        "beat_estimate_share": float((data["analyst_surprise"] > 0).mean()),
        "by_analyst_surprise_third": {
            "median_surprise_pct": by_third["analyst_surprise"].median().round(2).to_dict(),
            "mean_car_announce": by_third["car_announce"].mean().round(4).to_dict(),
            "share_positive_reaction": by_third["car_announce"].apply(lambda values: float((values > 0).mean())).round(3).to_dict(),
            "mean_move_size": by_third["move_release"].mean().round(2).to_dict(),
            "n": by_third.size().to_dict(),
        },
        "tone_correlations": data[tone].corr(method="spearman").round(3).to_dict(),
        "analyst_vs_history_surprise": float(data[["analyst_surprise", "income_surprise"]].corr(method="spearman").iloc[0, 1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Event study of earnings releases")
    parser.add_argument("--no-llm", action="store_true", help="Skip the LLM tone measure")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached downloads and scores")
    args = parser.parse_args()

    study_dir = settings.processed_data_dir / "study"
    figure_dir = settings.project_root / "docs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    data, paths = build_dataset(study_dir, args.refresh, use_llm=not args.no_llm)
    results = run_analysis(data)
    check = run_analysis(data, suffix="_simple")
    sample = describe_sample(data)

    data.to_parquet(study_dir / "release_dataset.parquet")
    results["association"].to_csv(study_dir / "release_association.csv", index=False)
    results["magnitude"].to_csv(study_dir / "release_magnitude.csv", index=False)
    results["prediction"].to_csv(study_dir / "release_prediction.csv", index=False)
    check["association"].to_csv(study_dir / "release_association_market_adjusted.csv", index=False)
    check["prediction"].to_csv(study_dir / "release_prediction_market_adjusted.csv", index=False)
    (study_dir / "release_sample.json").write_text(json.dumps(sample, indent=2))

    plots.association_figure(
        results["association"], figure_dir / "release_association.png",
        title="Earnings releases: numbers, tone and abnormal returns",
    )
    plots.prediction_figure(
        results["prediction"], figure_dir / "release_prediction.png",
        labels={"own history": "Surprise vs. own history", "analyst surprise": "Surprise vs. analysts",
                "tone": "Press release tone", "analyst surprise + tone": "Analyst surprise and tone"},
    )
    plots.car_path_figure(
        data, paths, "analyst_surprise", "Beat estimates most", "Missed or barely met",
        figure_dir / "release_car_paths.png", key="release_accession", event="earnings release", within_company=False,
    )

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)
    print("\nSAMPLE")
    print(json.dumps(sample, indent=2))
    print("\nASSOCIATION WITH DIRECTION (Spearman rho)")
    print(results["association"].round(3).to_string(index=False))
    print("\nASSOCIATION WITH SIZE (move on the release day; change in volatility afterwards)")
    print(results["magnitude"].round(3).to_string(index=False))
    print("\nPREDICTION (walk-forward, test years {})".format(", ".join(map(str, TEST_YEARS))))
    print(results["prediction"].round(3).to_string(index=False))
    print("\nROBUSTNESS: direction results with market-adjusted returns (stock minus market)")
    print(check["association"].round(3).to_string(index=False))
    print(check["prediction"].round(3).to_string(index=False))
    print(f"\nResults: {study_dir}\nFigures: {figure_dir}")


if __name__ == "__main__":
    main()
