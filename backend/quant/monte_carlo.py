import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from datetime import timedelta


class GarchMonteCarlo:
    def __init__(self, ticker, days_forecast=252, num_simulations=1000):
        """
        Moteur de simulation Monte Carlo basé sur GARCH(1,1).
        """
        self.ticker = ticker
        self.days = days_forecast
        self.sims = num_simulations
        self.data = None
        self.params = None
        self.last_vol = None
        self.last_shock = None
        self.results = None
        self.risk_free_rate = 0.02  # Taux sans risque (ex: 2% pour Sharpe Ratio)

    def fit_garch(self):
        """
        Calibre un modèle GARCH(1,1) sur l'historique.
        """
        df = yf.download(self.ticker, period="5y", progress=False, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            try:
                self.data = df.xs(self.ticker, axis=1, level=1)["Close"]
            except KeyError:
                self.data = df.iloc[:, 0]
        else:
            self.data = df["Close"]

        self.data = self.data.dropna()

        # Rendements en % (meilleure convergence GARCH)
        returns = 100 * self.data.pct_change().dropna()

        model = arch_model(
            returns, vol="Garch", p=1, q=1, mean="Zero", dist="Normal"
        )
        res = model.fit(disp="off")

        self.params = res.params
        self.last_vol = res.conditional_volatility.iloc[-1]
        self.last_shock = res.resid.iloc[-1]

    def run_simulation(self):
        if self.params is None:
            self.fit_garch()

        omega = self.params["omega"]
        alpha = self.params["alpha[1]"]
        beta = self.params["beta[1]"]

        sim_variances = np.zeros((self.days, self.sims))
        sim_returns = np.zeros((self.days, self.sims))

        sim_variances[0, :] = self.last_vol**2
        last_resid_sq = self.last_shock**2

        for t in range(1, self.days):
            z = np.random.normal(0, 1, self.sims)

            if t == 1:
                prev_resid_sq = np.full(self.sims, last_resid_sq)
                prev_var = np.full(self.sims, self.last_vol**2)
            else:
                prev_resid_sq = sim_returns[t - 1, :] ** 2
                prev_var = sim_variances[t - 1, :]

            sim_variances[t, :] = omega + alpha * prev_resid_sq + beta * prev_var
            sim_returns[t, :] = np.sqrt(sim_variances[t, :]) * z

        sim_returns_pct = sim_returns / 100
        last_price = self.data.iloc[-1]

        price_paths = last_price * np.cumprod(1 + sim_returns_pct, axis=0)
        self.results = price_paths
        return price_paths

    def get_analysis(self):
        """
        Calcule les indicateurs de risque avancés et formate la sortie JSON.
        """
        if self.results is None:
            return {"error": "Run simulation first"}

        # --- CALCUL DES MÉTRIQUES FINANCIÈRES ---

        # On prend la distribution des prix finaux (dernier jour)
        final_prices = self.results[-1]
        start_price = self.results[0, 0]

        # Rendements totaux pour chaque scénario
        total_returns = (final_prices / start_price) - 1

        # 1. Probabilité de Gain (Win Rate)
        # Pourcentage de scénarios où l'on finit positif
        win_prob = np.mean(final_prices > start_price)

        # 2. VaR 95% (Value at Risk)
        # "Dans les 5% des pires cas, on ne perdra pas plus que X%"
        # On cherche le percentile 5 (le seuil du pire)
        p5_price = np.percentile(final_prices, 5)
        var_95_percent = (p5_price / start_price) - 1
        var_95_amount = p5_price - start_price

        # 3. CVaR 95% (Conditional VaR / Expected Shortfall)
        # "Si on tape dans les 5% pires, quelle est la perte MOYENNE ?"
        # C'est une mesure de risque plus honnête que la VaR simple
        worst_5_percent_prices = final_prices[final_prices <= p5_price]
        cvar_95_percent = (worst_5_percent_prices.mean() / start_price) - 1

        # 4. Ratio de Sharpe Implicite (Annualisé)
        # (Rendement Moyen - Taux sans risque) / Volatilité des scénarios
        avg_return = np.mean(total_returns)
        std_return = np.std(total_returns)
        sharpe_ratio = (avg_return - self.risk_free_rate) / std_return if std_return != 0 else 0

        # 5. Volatilité attendue (Annualisée)
        volatility_ann = std_return  # Sur la période projetée (simplifié)

        # --- PRÉPARATION DES DONNÉES GRAPHIQUES ---
        # On réduit les données pour le web (Percentiles uniquement)
        p10 = np.percentile(self.results, 10, axis=1)
        p50 = np.percentile(self.results, 50, axis=1)  # Médiane
        p90 = np.percentile(self.results, 90, axis=1)

        last_date = self.data.index[-1]
        future_dates = [
            (last_date + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(1, self.days + 1)
        ]

        return {
            "meta": {
                "ticker": self.ticker,
                "current_price": round(start_price, 2),
                "forecast_days": self.days,
                "simulations_count": self.sims
            },
            "kpis": {
                "win_probability": round(win_prob * 100, 1),  # En %
                "expected_return": round(avg_return * 100, 2),  # En %
                "sharpe_ratio": round(sharpe_ratio, 2),
                "volatility": round(volatility_ann * 100, 2),  # En %
                "risk_analysis": {
                    "VaR_95_percent": round(var_95_percent * 100, 2),  # ex: -15.4%
                    "VaR_95_value": round(var_95_amount, 2),  # ex: -25.50$
                    "CVaR_95_percent": round(cvar_95_percent * 100, 2),  # ex: -22.1%
                    "interpretation": f"Il y a 95% de chances que la perte ne dépasse pas {abs(round(var_95_percent*100, 1))}%."
                }
            },
            "chart_data": {
                "dates": future_dates,
                "p10": np.round(p10, 2).tolist(),
                "median": np.round(p50, 2).tolist(),
                "p90": np.round(p90, 2).tolist()
            }
        }
