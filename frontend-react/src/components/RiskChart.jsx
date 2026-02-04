export default function RiskChart({ data }) {
  if (!data || Object.keys(data).length === 0) {
    return <p>Aucune metrique a afficher. Cliquez sur "Calculer".</p>;
  }

  const rows = Object.entries(data).map(([ticker, values]) => ({
    ticker,
    beta: Number(values?.beta ?? 0),
    volatility: Number(values?.volatility ?? 0),
  }));

  return (
    <div>
      <h2>Resultats</h2>
      <table style={{ borderCollapse: "collapse", width: "100%" }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", borderBottom: "1px solid #ccc", padding: "8px" }}>Ticker</th>
            <th style={{ textAlign: "left", borderBottom: "1px solid #ccc", padding: "8px" }}>Beta</th>
            <th style={{ textAlign: "left", borderBottom: "1px solid #ccc", padding: "8px" }}>Volatility</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.ticker}>
              <td style={{ borderBottom: "1px solid #eee", padding: "8px" }}>{row.ticker}</td>
              <td style={{ borderBottom: "1px solid #eee", padding: "8px" }}>{row.beta.toFixed(3)}</td>
              <td style={{ borderBottom: "1px solid #eee", padding: "8px" }}>{row.volatility.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
