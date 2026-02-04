import { useState } from "react";
import RiskChart from "./components/RiskChart";

export default function App() {
  const [tickers, setTickers] = useState("AAPL,MSFT,GOOGL");
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchMetrics = async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`http://127.0.0.1:8000/metrics?tickers=${encodeURIComponent(tickers)}`);

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json();
      setMetrics(data.metrics ?? null);
    } catch {
      setError("Erreur lors de l'appel a l'API");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        padding: "40px",
        maxWidth: "1100px",
        margin: "0 auto",
        background: "#ffffff",
        borderRadius: "14px",
        border: "1px solid #d1d5db",
        marginTop: "32px",
      }}
    >
      <h1 style={{ marginTop: 0 }}>Quant Risk Dashboard</h1>

      <div style={{ marginBottom: "20px" }}>
        <input value={tickers} onChange={(e) => setTickers(e.target.value)} />
        <button onClick={fetchMetrics} style={{ marginLeft: "10px" }}>
          Calculer
        </button>
      </div>

      {loading && <p>Calcul en cours...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      <RiskChart data={metrics} />
    </div>
  );
}
