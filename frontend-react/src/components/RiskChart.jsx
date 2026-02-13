import Plot from "react-plotly.js";

export default function RiskChart({ figure }) {
  const hasData = Array.isArray(figure?.data) && figure.data.length > 0;

  if (!hasData) {
    return <p>Aucun graphique a afficher. Cliquez sur "Calculer".</p>;
  }

  const layout = {
    ...figure.layout,
    autosize: true,
    height: 520,
    margin: { t: 60, r: 20, b: 60, l: 60 },
  };

  return (
    <Plot
      data={figure.data}
      layout={layout}
      frames={figure.frames ?? []}
      config={{ responsive: true, displaylogo: false }}
      style={{ width: "100%" }}
    />
  );
}
