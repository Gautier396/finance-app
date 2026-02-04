from fastapi import FastAPI
from backend.data.data_loader import get_clean_data, compute_log_returns

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Backend is running"}


@app.get("/returns")
def returns():
    tickers = ["AAPL", "MSFT", "GOOGL"]
    prices = get_clean_data(tickers)

    returns = compute_log_returns(prices)

    return {
        "columns": list(returns.columns),
        "rows": returns.tail().to_dict()
    }

@app.get("/test-returns")
def test_returns():
    tickers = ["AAPL", "MSFT", "GOOGL"]

    prices = get_clean_data(tickers)
    returns = compute_log_returns(prices)

    return {
        "columns": list(returns.columns),
        "last_rows": returns.tail(5).to_dict()
    }
