from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def get_clean_data(tickers: Iterable[str], days: int = 252, seed: int = 42) -> pd.DataFrame:
    """Return a clean price DataFrame indexed by business day.

    This local deterministic generator avoids network dependency while giving
    realistic-looking price series for API/testing flows.
    """
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        raise ValueError("tickers must contain at least one symbol")

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
