"""Abnormal returns around an event, using the market model.

For each event the stock's normal relation to the market is estimated over a
window that ends before the event:

    r_stock = alpha + beta * r_market + error

The abnormal return on a day is the actual return minus what this relation
predicts. Summing abnormal returns over a window gives the cumulative
abnormal return (CAR), the part of the price move not explained by the
market that day.

The estimated alpha is the stock's own recent trend. Over a window of a month
it adds up, so a stock that did well before the event gets a lower CAR for
the same price path. The simpler "market-adjusted" return (stock minus
market, no estimation) does not have this property and serves as a check.
"""

from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from ..utils.logger import log

MARKET_TICKER = "SPY"


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Daily closing prices adjusted for splits and dividends, one column per ticker."""
    symbols = sorted(set(tickers) | {MARKET_TICKER})
    data = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    prices = data["Close"]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    missing = [symbol for symbol in symbols if symbol not in prices or prices[symbol].isna().all()]
    if missing:
        raise RuntimeError(f"No price data for: {', '.join(missing)}")

    log.info(f"Prices for {len(symbols)} symbols, {len(prices)} sessions")
    return prices.dropna(subset=[MARKET_TICKER])


class EventStudy:
    def __init__(
        self,
        prices: pd.DataFrame,
        estimation_days: int = 250,
        gap_days: int = 21,
        min_estimation_days: int = 120,
        market_adjusted: bool = False,
    ):
        self.sessions = prices.index
        self.returns = np.log(prices).diff()
        self.estimation_days = estimation_days
        self.gap_days = gap_days
        self.min_estimation_days = min_estimation_days
        self.market_adjusted = market_adjusted

    def abnormal_returns(
        self,
        ticker: str,
        session: pd.Timestamp,
        first_day: int,
        last_day: int,
    ) -> Optional[pd.Series]:
        """Daily abnormal returns from first_day to last_day relative to the event session (day 0).

        Returns None if the event lies outside the price history or there is
        too little history to estimate the market model.
        """
        if session is None or pd.isna(session) or session not in self.sessions:
            return None

        position = self.sessions.get_loc(session)
        window_start, window_end = position + first_day, position + last_day
        estimation_end = position - self.gap_days
        estimation_start = estimation_end - self.estimation_days
        if estimation_start < 1 or window_start < 1 or window_end >= len(self.sessions):
            return None

        stock = self.returns[ticker]
        market = self.returns[MARKET_TICKER]

        estimation = pd.DataFrame({
            "stock": stock.iloc[estimation_start:estimation_end],
            "market": market.iloc[estimation_start:estimation_end],
        }).dropna()
        if len(estimation) < self.min_estimation_days:
            return None

        if self.market_adjusted:
            beta, alpha = 1.0, 0.0
        else:
            beta, alpha = np.polyfit(estimation["market"], estimation["stock"], 1)

        window = slice(window_start, window_end + 1)
        abnormal = stock.iloc[window] - (alpha + beta * market.iloc[window])
        if abnormal.isna().any():
            return None

        abnormal.index = range(first_day, last_day + 1)
        # Spread of the stock's abnormal return on ordinary days
        abnormal.attrs["normal_std"] = float((estimation["stock"] - alpha - beta * estimation["market"]).std())
        return abnormal

    def move_size(self, ticker: str, session: pd.Timestamp) -> float:
        """Size of the event-day move in units of the stock's ordinary daily moves.

        The absolute abnormal return on day 0 divided by the standard deviation
        of abnormal returns in the estimation window. On an ordinary day this
        averages about 0.8; NaN if unavailable.
        """
        abnormal = self.abnormal_returns(ticker, session, 0, 0)
        if abnormal is None:
            return np.nan
        return float(abs(abnormal.iloc[0]) / abnormal.attrs["normal_std"])

    def car(self, ticker: str, session: pd.Timestamp, first_day: int, last_day: int) -> float:
        """Cumulative abnormal return over [first_day, last_day]; NaN if unavailable."""
        abnormal = self.abnormal_returns(ticker, session, first_day, last_day)
        return float(abnormal.sum()) if abnormal is not None else np.nan
