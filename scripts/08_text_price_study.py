"""
08_text_price_study.py - Does the language of a filing relate to the stock's reaction?

An event study over all downloaded filings:
  1. Event dates: when each 10-K/10-Q and its earnings release became public
  2. Abnormal returns around those dates (market model against SPY)
  3. Text features: MD&A tone (keywords, FinBERT, LLM) and change vs. the previous filing
  4. Association tests and out-of-sample prediction
  5. Figures

Usage:
    python scripts/08_text_price_study.py              # full run; intermediate results are cached
    python scripts/08_text_price_study.py --no-llm     # skip the LLM tone measure (the slowest step)
    python scripts/08_text_price_study.py --refresh    # ignore cached events, prices and features

Needs the output of scripts 01 and 02. Results go to data/processed/study/,
figures to docs/figures/.
"""

import sys
from pathlib import Path
import argparse
import json

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_filing_analysis.config import settings
from sec_filing_analysis.study import analysis, events, features, plots
from sec_filing_analysis.study.returns import EventStudy, download_prices
from sec_filing_analysis.utils.logger import log

PRICE_START, PRICE_END = "2017-10-01", "2025-03-31"
RELEASE_START, RELEASE_END = "2018-11-01", "2025-01-31"

# (name, anchor column, first day, last day) relative to the anchor session
RETURN_WINDOWS = [
    ("car_announce", "release_session", 0, 1),   # reaction to the earnings release
    ("car_filing", "filing_session", 0, 1),      # reaction to the filing
    ("car_drift", "filing_session", 2, 21),      # the following month
    ("car_pre", "filing_session", -5, -1),       # the week before the filing
]
PATH_WINDOW = (-5, 21)
TEST_YEARS = [2022, 2023, 2024]

# Filings published at least this many sessions after the earnings release:
# their [0, +1] window no longer contains the release itself.
LAGGED_MIN_SESSIONS = 2


def cached(path: Path, refresh: bool, build):
    if path.exists() and not refresh:
        log.info(f"Using cached {path.name}")
        return pd.read_parquet(path)
    frame = build()
    frame.to_parquet(path)
    return frame


def build_dataset(study_dir: Path, refresh: bool, use_llm: bool):
    filing_index = pd.read_parquet(settings.processed_data_dir / "filing_index.parquet")
    tickers = sorted(filing_index["ticker"].unique())

    filing_events = events.build_filing_events(filing_index)
    releases = cached(
        study_dir / "earnings_releases.parquet", refresh,
        lambda: events.fetch_earnings_releases(tickers, RELEASE_START, RELEASE_END),
    )
    data = events.attach_earnings_releases(filing_events, releases)

    prices = cached(study_dir / "prices.parquet", refresh, lambda: download_prices(tickers, PRICE_START, PRICE_END))
    sessions = prices.index
    study = EventStudy(prices)
    simple = EventStudy(prices, market_adjusted=True)

    data["filing_session"] = [events.event_session(t, sessions) for t in data["filing_accepted"]]
    data["release_session"] = [events.event_session(t, sessions) for t in data["release_accepted"]]
    position = {session: i for i, session in enumerate(sessions)}
    data["lag_sessions"] = [
        position[filing] - position[release] if pd.notna(filing) and pd.notna(release) else None
        for filing, release in zip(data["filing_session"], data["release_session"])
    ]

    for name, anchor, first_day, last_day in RETURN_WINDOWS:
        data[name] = [study.car(ticker, session, first_day, last_day) for ticker, session in zip(data["ticker"], data[anchor])]
        # Robustness check: stock minus market, without an estimated trend
        data[f"{name}_simple"] = [simple.car(ticker, session, first_day, last_day) for ticker, session in zip(data["ticker"], data[anchor])]

    # How large is the move on the event day itself, whatever its direction?
    data["move_release"] = [study.move_size(ticker, session) for ticker, session in zip(data["ticker"], data["release_session"])]
    data["move_filing"] = [study.move_size(ticker, session) for ticker, session in zip(data["ticker"], data["filing_session"])]

    paths = {}
    for accession, ticker, session in zip(data["accession"], data["ticker"], data["filing_session"]):
        abnormal = study.abnormal_returns(ticker, session, *PATH_WINDOW)
        if abnormal is not None:
            paths[accession] = abnormal
    paths = pd.DataFrame(paths).T

    def build_features():
        chunks = pd.read_parquet(settings.processed_data_dir / "chunked_documents.parquet")
        return features.build_text_features(chunks, data, use_llm=use_llm)

    text = cached(study_dir / "text_features.parquet", refresh, build_features)
    if use_llm and "llm_tone" not in text:
        log.warning("Cached text features have no LLM tone; run with --refresh to compute it")

    data = data.merge(text.drop(columns=["ticker", "filing_type", "filing_accepted"]), on="accession", how="left")
    data["year"] = pd.DatetimeIndex(data["filing_session"]).year
    data["quarter"] = pd.DatetimeIndex(data["filing_session"]).to_period("Q").astype(str)
    return data, paths


def run_analysis(data: pd.DataFrame, suffix: str = "") -> dict:
    """Association and prediction results. `suffix` selects the return definition
    ("" = market model, "_simple" = market-adjusted)."""
    text_features = [feature for feature in analysis.TEXT_FEATURES if feature in data]
    targets = ["car_announce", "car_filing", "car_drift"]

    clean = data.copy()
    for column in targets + ["car_pre"]:
        clean[column] = analysis.winsorize(clean[f"{column}{suffix}"])

    # Association: features compared within each company
    standardized = analysis.standardize_within(clean, text_features)
    lagged = standardized[standardized["lag_sessions"] >= LAGGED_MIN_SESSIONS].rename(columns={"car_filing": "car_filing_lagged"})
    association = pd.concat([
        analysis.association_table(standardized, text_features, targets),
        analysis.association_table(lagged, text_features, ["car_filing_lagged"]),
    ], ignore_index=True)
    association["q"] = analysis.benjamini_hochberg(association["p"])

    # Prediction: only information available when the filing is published
    # Risk Factors are missing from many 10-Qs ("no material changes"), so
    # the prediction models use the MD&A measures only to keep the sample.
    keyword = ["kw_tone", "d_kw_tone"]
    model_features = [feature for feature in text_features if feature != "risk_change"]
    feature_sets = {
        "car_filing": {"keywords": keyword, "all text": model_features, "past return": ["car_pre"], "text + past return": model_features + ["car_pre"]},
        "car_drift": {"keywords": keyword, "all text": model_features, "past return": ["car_pre", "car_filing"], "text + past return": model_features + ["car_pre", "car_filing"]},
    }
    # Every model is scored on the same filings: those with all features present
    complete = clean.dropna(subset=model_features + ["car_pre", "car_filing", "car_drift"])
    prediction = []
    for target, sets in feature_sets.items():
        for name, columns in sets.items():
            scores = analysis.score_predictions(analysis.walk_forward(complete, columns, target, TEST_YEARS))
            prediction.append({"target": target, "features": name, **scores})

    return {
        "data": standardized,
        "association": association,
        "prediction": pd.DataFrame(prediction),
        "text_features": text_features,
    }


def describe_sample(data: pd.DataFrame, text_features: list[str]) -> dict:
    later = data[data["lag_sessions"] >= LAGGED_MIN_SESSIONS]

    def move_summary(moves: pd.Series) -> dict:
        return {"mean": float(moves.mean()), "median": float(moves.median()), "share_above_2": float((moves > 2).mean())}

    return {
        # In units of the stock's ordinary daily move; about 0.8 on an ordinary day
        "move_on_release_day": move_summary(data["move_release"]),
        "move_on_later_filing_day": move_summary(later["move_filing"]),
        "filings": int(len(data)),
        "companies": int(data["ticker"].nunique()),
        "first_filing": str(data["filing_accepted"].min().date()),
        "last_filing": str(data["filing_accepted"].max().date()),
        "filed_after_close_share": float((data["filing_accepted"].dt.time >= events.MARKET_CLOSE).mean()),
        "with_earnings_release": int(data["release_accepted"].notna().sum()),
        "same_session_as_release_share": float((data["lag_sessions"] == 0).mean()),
        "lagged_filings": int((data["lag_sessions"] >= LAGGED_MIN_SESSIONS).sum()),
        "median_lag_sessions": float(data["lag_sessions"].median()),
        "with_mdna_tone": int(data["finbert_tone"].notna().sum()),
        "companies_with_mdna_tone": int(data.loc[data["finbert_tone"].notna(), "ticker"].nunique()),
        "with_all_features": int(data.dropna(subset=text_features).shape[0]),
        "car_std": {name: float(data[name].std()) for name, *_ in RETURN_WINDOWS},
        "car_mean": {name: float(data[name].mean()) for name, *_ in RETURN_WINDOWS},
        "tone_correlations": data[[f for f in features.TONE_FEATURES if f in data]].corr(method="spearman").round(3).to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Event study: filing text vs. stock reaction")
    parser.add_argument("--no-llm", action="store_true", help="Skip the LLM tone measure")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached intermediate results")
    args = parser.parse_args()

    study_dir = settings.processed_data_dir / "study"
    study_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = settings.project_root / "docs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    data, paths = build_dataset(study_dir, args.refresh, use_llm=not args.no_llm)
    results = run_analysis(data)
    check = run_analysis(data, suffix="_simple")
    sample = describe_sample(data, results["text_features"])

    data.to_parquet(study_dir / "study_dataset.parquet")
    results["association"].to_csv(study_dir / "association.csv", index=False)
    results["prediction"].to_csv(study_dir / "prediction.csv", index=False)
    check["association"].to_csv(study_dir / "association_market_adjusted.csv", index=False)
    check["prediction"].to_csv(study_dir / "prediction_market_adjusted.csv", index=False)
    (study_dir / "sample.json").write_text(json.dumps(sample, indent=2))

    plots.association_figure(results["association"], figure_dir / "association.png")
    plots.prediction_figure(results["prediction"], figure_dir / "prediction.png")
    plots.car_path_figure(
        results["data"], paths, "finbert_tone", "Most positive tone", "Most negative tone",
        figure_dir / "car_paths.png",
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print("\nSAMPLE")
    print(json.dumps({k: v for k, v in sample.items() if k != "tone_correlations"}, indent=2))
    print("\nASSOCIATION (Spearman rho of within-company text measure with abnormal return)")
    print(results["association"].round(3).to_string(index=False))
    print("\nPREDICTION (walk-forward, test years {})".format(", ".join(map(str, TEST_YEARS))))
    print(results["prediction"].round(3).to_string(index=False))
    print("\nROBUSTNESS: the same with market-adjusted returns (stock minus market)")
    print(check["association"].round(3).to_string(index=False))
    print(check["prediction"].round(3).to_string(index=False))
    print(f"\nResults: {study_dir}\nFigures: {figure_dir}")


if __name__ == "__main__":
    main()
