import { useMemo, useState } from "react";
import Plot from "react-plotly.js";
import RiskChart from "./components/RiskChart";
import "./App.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("risk");
  const [tickers, setTickers] = useState(
    "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,NFLX,JPM,V,MA"
  );
  const [riskFigure, setRiskFigure] = useState(null);
  const [gmmData, setGmmData] = useState([]);
  const [gmmTable, setGmmTable] = useState([]);
  const [loading, setLoading] = useState(false);
  const [gmmLoading, setGmmLoading] = useState(false);
  const [error, setError] = useState(null);
  const [gmmError, setGmmError] = useState(null);

  const selectedTickers = useMemo(() => {
    return tickers
      .split(",")
      .map((t) => t.trim().toUpperCase())
      .filter(Boolean);
  }, [tickers]);

  const fetchRisk = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `http://127.0.0.1:8000/risk-chart?tickers=${encodeURIComponent(
          tickers
        )}`
      );
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const text = await res.text();
      const fig = JSON.parse(text);
      setRiskFigure(fig);
    } catch {
      setError("Erreur lors de l'appel a l'API");
    } finally {
      setLoading(false);
    }
  };

  const fetchGmm = async () => {
    setGmmLoading(true);
    setGmmError(null);
    try {
      const res = await fetch(
        `http://127.0.0.1:8000/gmm?tickers=${encodeURIComponent(tickers)}`
      );
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      setGmmData(data?.clusters ?? []);
      setGmmTable(data?.ambiguous ?? []);
    } catch {
      setGmmError("Erreur lors du clustering GMM");
    } finally {
      setGmmLoading(false);
    }
  };

  const gmmTraces = useMemo(() => {
    if (!gmmData.length) return [];
    const groups = new Map();
    gmmData.forEach((item) => {
      const key = item.cluster || "Groupe";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(item);
    });
    return Array.from(groups.entries()).map(([cluster, items]) => ({
      type: "scatter",
      mode: "markers+text",
      name: cluster,
      x: items.map((i) => i.beta),
      y: items.map((i) => i.specific_risk_pct),
      text: items.map((i) => i.ticker),
      customdata: items.map((i) => i.certitude_ia_pct),
      textposition: "top center",
      marker: {
        size: items.map((i) => 8 + (i.certitude_ia_pct || 0) * 0.2),
        opacity: 0.85,
      },
      hovertemplate:
        "<b>%{text}</b><br>Bêta: %{x}<br>Risque spécifique: %{y}%<br>Certitude: %{customdata:.2f}%<extra></extra>",
    }));
  }, [gmmData]);

  return (
    <div
      style={{
        padding: "32px",
        maxWidth: "1100px",
        margin: "0 auto",
        background: "#ffffff",
        borderRadius: "14px",
        border: "1px solid #d1d5db",
        marginTop: "32px",
      }}
    >
      <h1 style={{ marginTop: 0 }}>Quant Risk Dashboard</h1>

      <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
        <button
          onClick={() => setActiveTab("risk")}
          style={{
            padding: "8px 14px",
            borderRadius: 8,
            border: "1px solid #d1d5db",
            background: activeTab === "risk" ? "#111827" : "#f3f4f6",
            color: activeTab === "risk" ? "#fff" : "#111827",
          }}
        >
          Analyse du Risque
        </button>
        <button
          onClick={() => setActiveTab("gmm")}
          style={{
            padding: "8px 14px",
            borderRadius: 8,
            border: "1px solid #d1d5db",
            background: activeTab === "gmm" ? "#111827" : "#f3f4f6",
            color: activeTab === "gmm" ? "#fff" : "#111827",
          }}
        >
          Clustering GMM
        </button>
      </div>

      <div style={{ marginBottom: "16px" }}>
        <input
          value={tickers}
          onChange={(e) => setTickers(e.target.value)}
          style={{ width: "70%" }}
        />
        {activeTab === "risk" ? (
          <button onClick={fetchRisk} style={{ marginLeft: "10px" }}>
            {loading ? "Calcul..." : "Calculer"}
          </button>
        ) : (
          <button onClick={fetchGmm} style={{ marginLeft: "10px" }}>
            {gmmLoading ? "Clustering..." : "Lancer GMM"}
          </button>
        )}
      </div>

      {activeTab === "risk" && (
        <>
          {loading && <p>Calcul en cours...</p>}
          {error && <p style={{ color: "red" }}>{error}</p>}
          <RiskChart figure={riskFigure} />
        </>
      )}

      {activeTab === "gmm" && (
        <>
          {gmmError && <p style={{ color: "red" }}>{gmmError}</p>}
          {gmmTraces.length > 0 ? (
            <Plot
              data={gmmTraces}
              layout={{
                autosize: true,
                height: 520,
                margin: { t: 40, r: 20, b: 60, l: 60 },
                xaxis: { title: "Bêta" },
                yaxis: { title: "Risque spécifique (%)" },
              }}
              config={{ displaylogo: false, responsive: true }}
              style={{ width: "100%" }}
            />
          ) : (
            <p>Lance le clustering GMM pour afficher le graphique.</p>
          )}

          {gmmTable.length > 0 && (
            <table
              style={{
                width: "100%",
                marginTop: 20,
                borderCollapse: "collapse",
              }}
            >
              <thead>
                <tr>
                  <th style={{ textAlign: "left", borderBottom: "1px solid #e5e7eb" }}>
                    Ticker
                  </th>
                  <th style={{ textAlign: "left", borderBottom: "1px solid #e5e7eb" }}>
                    Cluster
                  </th>
                  <th style={{ textAlign: "left", borderBottom: "1px solid #e5e7eb" }}>
                    Certitude (%)
                  </th>
                  <th style={{ textAlign: "left", borderBottom: "1px solid #e5e7eb" }}>
                    Bêta
                  </th>
                  <th style={{ textAlign: "left", borderBottom: "1px solid #e5e7eb" }}>
                    Risque spécifique
                  </th>
                </tr>
              </thead>
              <tbody>
                {gmmTable.map((row) => (
                  <tr key={row.ticker}>
                    <td style={{ padding: "6px 4px" }}>{row.ticker}</td>
                    <td style={{ padding: "6px 4px" }}>{row.cluster}</td>
                    <td style={{ padding: "6px 4px" }}>{row.certitude_ia_pct}</td>
                    <td style={{ padding: "6px 4px" }}>{row.beta}</td>
                    <td style={{ padding: "6px 4px" }}>{row.specific_risk_pct}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  );
}
