from fastapi import FastAPI
from backend.data.data_loader import get_clean_data, compute_log_returns
from backend.quant.beta import calculate_metrics
from backend.quant.visualization import create_risk_chart

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Backend is running"}

@app.get("/returns")
def returns():
    tickers = ["AAPL", "MSFT", "GOOGL"]
    prices = get_clean_data(tickers)
    returns = compute_log_returns(prices)
    return {"columns": list(returns.columns), "rows": returns.tail().to_dict()}


@app.get("/metrics")
def beta(
    tickers: str = "AAPL,MSFT,GOOGL",
    market: str = "^GSPC"
):
    # 1. Nettoyage des tickers
    assets = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    market_ticker = market.strip().upper()

    # 2. Liste complète (actifs + marché)
    all_tickers = list(set(assets + [market_ticker]))

    # 3. Récupération des prix
    prices = get_clean_data(all_tickers)

    # 4. Calcul des rendements log
    returns = compute_log_returns(prices)

    # 5. Calcul des métriques (TA fonction)
    metrics = calculate_metrics(returns)

    # 6. On ne renvoie que les actifs demandés
    filtered_metrics = {
        k: v for k, v in metrics.items() if k in assets
    }

    return {
        "market": market_ticker,
        "tickers": assets,
        "metrics": filtered_metrics
    }

@app.get("/risk-chart")
def risk_chart(tickers: str):
    # 1. Transformation des tickers
    tickers_list = tickers.split(",")

    # 2. Chargement des données
    prices = get_clean_data(tickers_list)

    # 3. Calcul des rendements
    returns = compute_log_returns(prices)

    # 4. Calcul des métriques (beta + risk)
    metrics = calculate_metrics(returns)

    # 5. Création du graphique Plotly
    fig = create_risk_chart(metrics)

    # 6. Conversion du graphique en JSON (format web)
    return JSONResponse(content=fig.to_dict())
