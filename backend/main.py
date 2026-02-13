from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from backend.data.data_loader import get_clean_data, compute_log_returns
from backend.quant.beta import calculate_metrics
from backend.quant.visualization import create_risk_chart
from backend.quant.monte_carlo import GarchMonteCarlo
from backend.quant.clustering import (
    prepare_clustering_data,
    apply_kmeans_expert,
    build_correlation_dendrogram,
    apply_gmm_expert,
    get_gmm_analysis_table,
    apply_kmeans_risk,
    apply_gmm_risk,
)
import plotly.figure_factory as ff
import numpy as np
import pandas as pd

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


@app.get("/beta")
def beta(
    tickers: str = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,NFLX,JPM,V,MA,UNH,HD,XOM,BAC,AVGO,COST,ORCL,ADBE,CRM",
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
def risk_chart(tickers: str = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,NFLX,JPM,V,MA,UNH,HD,XOM,BAC,AVGO,COST,ORCL,ADBE,CRM"):
    # 1. Transformation des tickers
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    # 2. Chargement des données
    prices = get_clean_data(tickers_list)

    # 3. Calcul des rendements
    returns = compute_log_returns(prices)

    # 4. Calcul des métriques (beta + risk)
    metrics = calculate_metrics(returns)

    # 5. Création du graphique Plotly
    fig = create_risk_chart(metrics)

    # 6. Conversion du graphique en JSON (format web)
    # Plotly fournit un JSON déjà sérialisé (gère numpy)
    fig_json = fig.to_json()
    return Response(content=fig_json, media_type="application/json")


@app.get("/monte-carlo")
def monte_carlo(
    ticker: str = "AAPL",
    days_forecast: int = 252,
    num_simulations: int = 5000,
    lookback_years: int = 2,
):
    simulator = GarchMonteCarlo(
        ticker=ticker,
        days_forecast=days_forecast,
        num_simulations=num_simulations,
    )
    simulator.run_simulation()
    return simulator.get_analysis()


@app.get("/clusters")
def clusters(
    tickers: str = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,NFLX,JPM,V,MA",
    n_clusters: int = 4,
):
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    prices = get_clean_data(tickers_list)
    returns = compute_log_returns(prices)

    metrics = calculate_metrics(returns)
    metrics_dict = {
        k: {"beta": float(v[0]), "specific_risk_pct": float(v[1])}
        for k, v in metrics.items()
    }

    df_result = apply_kmeans_risk(metrics_dict, n_clusters)

    payload = []
    for ticker, row in df_result.iterrows():
        payload.append(
            {
                "ticker": ticker,
                "cluster": row["cluster"],
                "beta": float(row["beta"]),
                "specific_risk_pct": float(row["specific_risk_pct"]),
            }
        )

    return {
        "tickers": tickers_list,
        "clusters": payload,
        "n_clusters": n_clusters,
    }


@app.get("/dendrogram")
def dendrogram(
    tickers: str = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,NFLX,JPM,V,MA",
    n_clusters: int = 4,
):
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    prices = get_clean_data(tickers_list)
    returns = compute_log_returns(prices)

    corr_matrix = returns.corr()
    Z, threshold, labels = build_correlation_dendrogram(corr_matrix, n_clusters)

    fig = ff.create_dendrogram(
        corr_matrix.values,
        labels=labels,
        linkagefun=lambda _: Z,
        color_threshold=threshold,
    )
    fig.update_layout(
        title="Dendrogramme : Proximité des Mouvements (Pearson)",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font=dict(color="#cbd5f5"),
        margin=dict(t=60, r=20, b=60, l=60),
    )
    fig.update_xaxes(tickangle=90)

    fig_json = fig.to_json()
    return Response(content=fig_json, media_type="application/json")


@app.get("/gmm")
def gmm(
    tickers: str = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,NFLX,JPM,V,MA",
    n_components: int = 4,
):
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    prices = get_clean_data(tickers_list)
    returns = compute_log_returns(prices)
    metrics = calculate_metrics(returns)
    metrics_dict = {
        k: {"beta": float(v[0]), "specific_risk_pct": float(v[1])}
        for k, v in metrics.items()
    }

    df_result = apply_gmm_risk(metrics_dict, n_components)
    df_table = get_gmm_analysis_table(df_result)

    clusters_payload = []
    for ticker, row in df_result.iterrows():
        clusters_payload.append(
            {
                "ticker": ticker,
                "cluster": row["cluster"],
                "certitude_ia_pct": float(row["certitude_ia_pct"]),
                "beta": float(row["beta"]),
                "specific_risk_pct": float(row["specific_risk_pct"]),
            }
        )

    table_payload = []
    for ticker, row in df_table.iterrows():
        table_payload.append(
            {
                "ticker": ticker,
                "cluster": row["cluster"],
                "certitude_ia_pct": float(row["certitude_ia_pct"]),
                "beta": float(row["beta"]),
                "specific_risk_pct": float(row["specific_risk_pct"]),
            }
        )

    return {
        "tickers": tickers_list,
        "clusters": clusters_payload,
        "ambiguous": table_payload,
        "n_components": n_components,
    }


@app.get("/correlation")
def correlation(
    tickers: str = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,NFLX,JPM,V,MA",
):
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    prices = get_clean_data(tickers_list)
    returns = compute_log_returns(prices)
    corr = returns.corr()
    return {
        "tickers": list(corr.columns),
        "matrix": corr.values.tolist(),
    }


@app.get("/portfolio-analytics")
def portfolio_analytics(
    tickers: str = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,NFLX,JPM,V,MA",
    market: str = "^GSPC",
):
    assets = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    market_ticker = market.strip().upper()
    all_tickers = list(dict.fromkeys(assets + [market_ticker]))

    prices = get_clean_data(all_tickers)
    returns = compute_log_returns(prices)

    # Equal-weight portfolio on selected assets
    asset_returns = returns[assets]
    portfolio_returns = asset_returns.mean(axis=1)

    benchmark_returns = returns[market_ticker] if market_ticker in returns else None

    def equity_curve(ret_series, start=100.0):
        return start * (1 + ret_series).cumprod()

    portfolio_curve = equity_curve(portfolio_returns)
    benchmark_curve = (
        equity_curve(benchmark_returns) if benchmark_returns is not None else None
    )

    # KPIs
    daily_rf = 0.02 / 252
    excess = portfolio_returns - daily_rf
    vol = portfolio_returns.std(ddof=0) * np.sqrt(252)
    ret = portfolio_returns.mean() * 252
    sharpe = (excess.mean() * 252) / vol if vol != 0 else 0
    cum = portfolio_curve
    drawdown = (cum / cum.cummax()) - 1
    max_dd = float(drawdown.min())

    allocation = [
        {"ticker": t, "weight_pct": round(100 / len(assets), 2)} for t in assets
    ]

    return {
        "allocation": allocation,
        "equity": {
            "dates": [d.strftime("%Y-%m-%d") for d in portfolio_curve.index],
            "portfolio": portfolio_curve.values.tolist(),
            "benchmark": benchmark_curve.values.tolist() if benchmark_curve is not None else [],
            "benchmark_ticker": market_ticker if benchmark_curve is not None else None,
        },
        "kpis": {
            "annual_return_pct": round(ret * 100, 2),
            "annual_volatility_pct": round(vol * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
        },
    }

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
