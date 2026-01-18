from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import yfinance as yf
import pandas as pd
from ..config import settings
from ..utils.logger import log


@dataclass
class PriceReaction:
    ticker: str
    earnings_date: datetime
    pre_price: float
    post_price: float
    price_change_pct: float
    volume_change_pct: float
    direction: str


class PriceDataFetcher:
    def __init__(self):
        self.cache_dir = settings.processed_data_dir / "price_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._price_cache: dict[str, pd.DataFrame] = {}

    def get_stock_data(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None,
        period: str = "5y"
    ) -> pd.DataFrame:
        try:
            stock = yf.Ticker(ticker)

            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date)
            else:
                df = stock.history(period=period)

            df.reset_index(inplace=True)
            df["ticker"] = ticker

            return df

        except Exception as e:
            log.error(f"Failed to fetch price data for {ticker}: {e}")
            return pd.DataFrame()

    def get_earnings_dates(self, ticker: str) -> pd.DataFrame:
        try:
            stock = yf.Ticker(ticker)
            earnings_dates = stock.earnings_dates

            if earnings_dates is None or earnings_dates.empty:
                log.warning(f"No earnings dates found for {ticker}")
                return pd.DataFrame()

            df = earnings_dates.reset_index()
            df["ticker"] = ticker
            df.rename(columns={"index": "earnings_date"}, inplace=True)

            return df

        except Exception as e:
            log.error(f"Failed to fetch earnings dates for {ticker}: {e}")
            return pd.DataFrame()

    def _get_cached_price_data(self, ticker: str) -> pd.DataFrame:
        """Fetch price history once and cache it. Uses 10y to cover 2019-2024 filing range."""
        if ticker not in self._price_cache:
            df = self.get_stock_data(ticker, period="10y")
            if df.empty:
                log.warning(f"No price data returned for {ticker}")
            else:
                log.info(f"Cached {len(df)} price records for {ticker}")
            self._price_cache[ticker] = df
        return self._price_cache[ticker]

    def calculate_price_reaction(
        self,
        ticker: str,
        earnings_date: datetime,
        window_days: int = 2
    ) -> Optional[PriceReaction]:
        try:
            price_data = self._get_cached_price_data(ticker).copy()

            if price_data.empty:
                return None

            price_data["Date"] = pd.to_datetime(price_data["Date"]).dt.tz_localize(None)
            earnings_dt = pd.to_datetime(earnings_date).tz_localize(None)

            pre_data = price_data[price_data["Date"] < earnings_dt].tail(window_days)
            post_data = price_data[price_data["Date"] >= earnings_dt].head(window_days)

            if pre_data.empty or post_data.empty:
                return None

            pre_price = pre_data["Close"].mean()
            post_price = post_data["Close"].mean()
            price_change_pct = ((post_price - pre_price) / pre_price) * 100

            pre_volume = pre_data["Volume"].mean()
            post_volume = post_data["Volume"].mean()
            volume_change_pct = ((post_volume - pre_volume) / pre_volume) * 100

            direction = "up" if price_change_pct > 0 else "down"

            return PriceReaction(
                ticker=ticker,
                earnings_date=earnings_date,
                pre_price=pre_price,
                post_price=post_price,
                price_change_pct=price_change_pct,
                volume_change_pct=volume_change_pct,
                direction=direction
            )

        except Exception as e:
            log.error(f"Failed to calculate price reaction for {ticker} on {earnings_date}: {e}")
            return None

    def build_price_reaction_dataset(
        self,
        filing_index: pd.DataFrame,
        window_days: int = 2
    ) -> pd.DataFrame:
        reactions = []

        for ticker in filing_index["ticker"].unique():
            try:
                ticker_filings = filing_index[filing_index["ticker"] == ticker]

                for _, row in ticker_filings.iterrows():
                    filing_date = pd.to_datetime(row["filing_date"])

                    reaction = self.calculate_price_reaction(ticker, filing_date, window_days)

                    if reaction:
                        reactions.append({
                            "ticker": reaction.ticker,
                            "earnings_date": reaction.earnings_date,
                            "filing_type": row["filing_type"],
                            "pre_price": reaction.pre_price,
                            "post_price": reaction.post_price,
                            "price_change_pct": reaction.price_change_pct,
                            "volume_change_pct": reaction.volume_change_pct,
                            "direction": reaction.direction,
                        })

                log.info(f"Calculated price reactions for {ticker}: {len(reactions)} data points")

            except Exception as e:
                log.error(f"Failed to process {ticker}: {e}")

        df = pd.DataFrame(reactions)

        output_path = settings.processed_data_dir / "price_reactions.parquet"
        df.to_parquet(output_path, index=False)

        log.info(f"Built price reaction dataset with {len(df)} entries")
        return df

    def get_fundamentals(self, ticker: str) -> dict:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            fundamentals = {
                "ticker": ticker,
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "revenue": info.get("totalRevenue"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "roe": info.get("returnOnEquity"),
                "debt_to_equity": info.get("debtToEquity"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }

            return fundamentals

        except Exception as e:
            log.error(f"Failed to fetch fundamentals for {ticker}: {e}")
            return {}

    def build_fundamentals_dataset(self, tickers: list[str]) -> pd.DataFrame:
        fundamentals_list = []

        for ticker in tickers:
            fundamentals = self.get_fundamentals(ticker)
            if fundamentals:
                fundamentals_list.append(fundamentals)

        df = pd.DataFrame(fundamentals_list)

        output_path = settings.processed_data_dir / "fundamentals.parquet"
        df.to_parquet(output_path, index=False)

        log.info(f"Built fundamentals dataset with {len(df)} tickers")
        return df
