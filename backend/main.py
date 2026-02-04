from fastapi import FastAPI

from backend.data.data_loader import compute_log_returns, get_clean_data
from backend.quant.beta import compute_beta_from_prices

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Backend is running"}


@app.get("/returns")
def returns():
    tickers = ["AAPL", "MSFT", "GOOGL"]
    prices = get_clean_data(tickers)
    asset_returns = compute_log_returns(prices)

    return {
        "columns": list(asset_returns.columns),
        "rows": asset_returns.tail().to_dict(),
    }


@app.get("/test-returns")
def test_returns():
    tickers = ["AAPL", "MSFT", "GOOGL"]
    prices = get_clean_data(tickers)
    asset_returns = compute_log_returns(prices)

    return {
        "columns": list(asset_returns.columns),
        "last_rows": asset_returns.tail(5).to_dict(),
    }


@app.get("/beta")
def beta(tickers: str = "AAPL,MSFT,GOOGL", market: str = "SPY"):
    assets = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    market_ticker = market.strip().upper()

    if not assets:
        return {"error": "No valid asset tickers provided"}

    all_tickers = [market_ticker] + [t for t in assets if t != market_ticker]
    prices = get_clean_data(all_tickers)
    betas = compute_beta_from_prices(prices, market_ticker)

    return {
        "market": market_ticker,
        "tickers": assets,
        "beta": {k: v for k, v in betas.to_dict().items() if k in assets},
    }
