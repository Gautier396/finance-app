from __future__ import annotations

import numpy as np
import pandas as pd


def compute_beta(returns: pd.DataFrame, market_col: str) -> pd.Series:
    """Compute CAPM beta of each asset against a market return series."""
    if returns is None or returns.empty:
        raise ValueError("returns DataFrame is empty")
    if market_col not in returns.columns:
        raise ValueError(f"market_col '{market_col}' not found in returns")

    aligned = returns.dropna(how="any")
    if aligned.shape[0] < 2:
        raise ValueError("not enough data points after dropping NaN values")

    market_var = aligned[market_col].var(ddof=0)
    if np.isclose(market_var, 0.0):
        raise ValueError("market variance is zero; beta is undefined")

    cov_with_market = aligned.cov(ddof=0)[market_col]
    beta = cov_with_market / market_var

    return beta.drop(labels=[market_col], errors="ignore")


def compute_beta_from_prices(prices: pd.DataFrame, market_col: str) -> pd.Series:
    """Compute CAPM beta from price data by first converting to log returns."""
    if prices is None or prices.empty:
        raise ValueError("prices DataFrame is empty")

    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    return compute_beta(returns, market_col)
