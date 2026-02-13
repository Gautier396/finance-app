from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


def get_clean_data(
    tickers: Iterable[str],
    days: int = 252,
    seed: int = 42,
    use_yahoo: bool = True,
) -> pd.DataFrame:
    """Return a clean price DataFrame indexed by business day."""
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        raise ValueError("tickers must contain at least one symbol")

    if use_yahoo:
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=int(days * 1.6))
        data = yf.download(
            tickers=tickers,
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            interval="1d",
            auto_adjust=True,
            group_by="column",
            progress=False,
            threads=True,
        )

        if data is None or data.empty:
            raise ValueError("Yahoo Finance returned no data for tickers")

        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                prices = data["Close"]
            elif "Adj Close" in data.columns.get_level_values(0):
                prices = data["Adj Close"]
            else:
                prices = data.xs("Close", level=0, axis=1, drop_level=True)
        else:
            prices = data[["Close"]] if "Close" in data.columns else data

        prices = prices.dropna(how="all")
        if prices.empty:
            raise ValueError("No usable price data after cleaning")

        prices = prices.tail(days)
        prices.columns = [c.upper() for c in prices.columns]
        return prices

    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    rng = np.random.default_rng(seed)

    # Simulate geometric random walks with mild drift/volatility.
    drift = 0.0004
    vol = 0.015
    steps = rng.normal(loc=drift, scale=vol, size=(len(index), len(tickers)))
    start_prices = rng.uniform(80, 220, size=len(tickers))
    prices = start_prices * np.exp(np.cumsum(steps, axis=0))

    df = pd.DataFrame(prices, index=index, columns=tickers)
    return df.dropna(how="all")


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from a price DataFrame."""
    if prices is None or prices.empty:
        raise ValueError("prices DataFrame is empty")

    clean_prices = prices.sort_index().dropna(how="all")
    returns = np.log(clean_prices / clean_prices.shift(1))
    return returns.dropna(how="all")
