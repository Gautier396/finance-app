from fastapi import FastAPI
from fastapi.responses import Response

from backend.data.data_loader import compute_log_returns, get_clean_data
from backend.quant.beta import calculate_metrics, compute_beta_from_prices
from backend.quant.visualization import create_risk_chart

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


@app.get("/risk-chart")
def risk_chart(tickers: str = "AAPL,MSFT,GOOGL", market: str = ""):
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not tickers_list:
        return {"error": "No valid tickers provided"}

    prices = get_clean_data(tickers_list)
    returns = compute_log_returns(prices)
    metrics = calculate_metrics(returns, market_col=market)
    fig = create_risk_chart(metrics)

    return Response(content=fig.to_json(), media_type="application/json")
