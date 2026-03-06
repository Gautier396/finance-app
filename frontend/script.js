const BASE_URL = 'http://127.0.0.1:8000';

// State management
const state = {
    tickers: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
    market: '^GSPC',
    data: {}
};

// DOM Elements
const tickersInput = document.getElementById('tickers-input');
const marketInput = document.getElementById('market-input');
const loadDataBtn = document.getElementById('load-data-btn');
const loadingSpinner = document.getElementById('loading');
const errorMessage = document.getElementById('error-message');
const navTabs = document.querySelectorAll('.nav-tab');
const sections = document.querySelectorAll('.section');

// Event Listeners
loadDataBtn.addEventListener('click', handleLoadData);
navTabs.forEach(tab => {
    tab.addEventListener('click', handleNavTab);
});

document.getElementById('run-clustering-btn').addEventListener('click', handleClustering);
document.getElementById('run-mc-btn').addEventListener('click', handleMonteCarlo);

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadInitialData();
});

// ==================== Main Functions ====================

async function handleLoadData() {
    state.tickers = tickersInput.value
        .split(',')
        .map(t => t.trim().toUpperCase())
        .filter(t => t);
    state.market = marketInput.value.trim().toUpperCase();

    if (state.tickers.length === 0) {
        showError('Please enter at least one ticker');
        return;
    }

    showLoading(true);
    hideError();

    try {
        await Promise.all([
            loadPortfolioData(),
            loadBetaData(),
            loadRiskChart(),
            loadCorrelationData()
        ]);
        showLoading(false);
    } catch (err) {
        const msg = err.message || JSON.stringify(err);
        showError(`Error loading data: ${msg}`);
        showLoading(false);
    }
}

async function loadInitialData() {
    showLoading(true);
    try {
        await Promise.all([
            loadPortfolioData(),
            loadBetaData(),
            loadRiskChart(),
            loadCorrelationData()
        ]);
    } catch (err) {
        console.error('Initial load error:', err);
    }
    showLoading(false);
}

// ==================== Portfolio ====================

async function loadPortfolioData() {
    try {
        const tickers = state.tickers.join(',');
        const market = state.market;
        const url = `${BASE_URL}/portfolio-analytics?tickers=${tickers}&market=${market}`;
        
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to load portfolio data');
        
        const data = await response.json();
        state.data.portfolio = data;

        // Update KPIs
        document.getElementById('annual-return').textContent = 
            data.kpis.annual_return_pct.toFixed(2);
        document.getElementById('annual-volatility').textContent = 
            data.kpis.annual_volatility_pct.toFixed(2);
        document.getElementById('sharpe-ratio').textContent = 
            data.kpis.sharpe_ratio.toFixed(2);
        document.getElementById('max-drawdown').textContent = 
            data.kpis.max_drawdown_pct.toFixed(2);

        // Plot equity curve
        plotEquityCurve(data);

        // Update allocation
        updateAllocation(data.allocation);
    } catch (err) {
        console.error('Portfolio error:', err);
        throw err;
    }
}

function plotEquityCurve(data) {
    const portfolioTrace = {
        x: data.equity.dates,
        y: data.equity.portfolio,
        name: 'Portfolio',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#6b8fa3', width: 2 }
    };

    const traces = [portfolioTrace];

    if (data.equity.benchmark && data.equity.benchmark.length > 0) {
        traces.push({
            x: data.equity.dates,
            y: data.equity.benchmark,
            name: data.equity.benchmark_ticker || 'Benchmark',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#5a7a99', width: 2, dash: 'dash' }
        });
    }

    const layout = {
        title: 'Equity Curve: Portfolio vs Benchmark',
        xaxis: { title: 'Date', gridcolor: '#4a6fa5' },
        yaxis: { title: 'Value ($)', gridcolor: '#4a6fa5' },
        paper_bgcolor: '#131820',
        plot_bgcolor: '#0a0e1a',
        font: { color: '#e8e8f0' },
        hovermode: 'x unified',
        margin: { t: 60, r: 20, b: 60, l: 60 }
    };

    Plotly.newPlot('portfolio-chart', traces, layout, { responsive: true });
}

function updateAllocation(allocation) {
    const container = document.getElementById('allocation-table');
    container.innerHTML = allocation
        .map(item => `
            <div class="allocation-item">
                <div class="ticker">${item.ticker}</div>
                <div class="weight">${item.weight_pct.toFixed(1)}%</div>
            </div>
        `)
        .join('');
}

// ==================== Beta Analysis ====================

async function loadBetaData() {
    try {
        const tickers = state.tickers.join(',');
        const market = state.market;
        const url = `${BASE_URL}/beta?tickers=${tickers}&market=${market}`;
        
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to load beta data');
        
        const data = await response.json();
        state.data.beta = data;

        // Update table
        const tbody = document.getElementById('beta-tbody');
        tbody.innerHTML = Object.entries(data.metrics)
            .map(([ticker, metrics]) => `
                <tr>
                    <td><strong>${ticker}</strong></td>
                    <td>${metrics[0].toFixed(3)}</td>
                    <td>${metrics[1].toFixed(2)}</td>
                </tr>
            `)
            .join('');
    } catch (err) {
        console.error('Beta error:', err);
        throw err;
    }
}

// ==================== Risk Chart ====================

async function loadRiskChart() {
    try {
        const tickers = state.tickers.join(',');
        const url = `${BASE_URL}/risk-chart?tickers=${tickers}`;
        
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to load risk chart');
        
        const data = await response.json();
        console.log('risk-chart response', data);
        state.data.riskChart = data;

        // response.json() already returns an object with `data` and `layout`
        Plotly.newPlot('risk-chart', data.data, data.layout, { responsive: true });
    } catch (err) {
        console.error('Risk chart error:', err);
        throw err;
    }
}

// ==================== Clustering ====================

async function handleClustering() {
    const nClusters = parseInt(document.getElementById('n-clusters').value);
    showLoading(true);
    hideError();

    try {
        const tickers = state.tickers.join(',');
        const url = `${BASE_URL}/clusters?tickers=${tickers}&n_clusters=${nClusters}`;
        
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to load clustering data');
        
        const data = await response.json();
        console.log('clustering response', data);
        state.data.clustering = data;

        // Plot clustering
        plotClustering(data);

        // Update table
        const tbody = document.getElementById('clustering-tbody');
        tbody.innerHTML = data.clusters
            .map(item => {
                // extract numeric part of label (e.g. "Groupe 2")
                const num = parseInt((item.cluster + '').replace(/\D/g, ''), 10);
                const label = isNaN(num) ? item.cluster : num;
                return `
                <tr>
                    <td><strong>${item.ticker}</strong></td>
                    <td><span class="cluster-badge">Cluster ${label}</span></td>
                    <td>${parseFloat(item.beta).toFixed(3)}</td>
                    <td>${parseFloat(item.specific_risk_pct).toFixed(2)}</td>
                </tr>
            `;
            })
            .join('');
        
        showLoading(false);
    } catch (err) {
        showError(`Clustering error: ${err.message}`);
        showLoading(false);
    }
}

function plotClustering(data) {
    const colors = ['#6b8fa3', '#7a9fba', '#5a7a99', '#8aafca', '#6b8fa3'];
    
    const traces = [];
    for (let i = 0; i < data.n_clusters; i++) {
        // convert item.cluster string to numeric if possible
        const clusterItems = data.clusters.filter(item => {
            const num = parseInt((item.cluster + '').replace(/\D/g, ''), 10);
            return num === i;
        });
        
        traces.push({
            x: clusterItems.map(item => parseFloat(item.beta)),
            y: clusterItems.map(item => parseFloat(item.specific_risk_pct)),
            text: clusterItems.map(item => item.ticker),
            mode: 'markers+text',
            type: 'scatter',
            name: `Cluster ${i}`,
            marker: {
                size: 12,
                color: colors[i % colors.length]
            },
            textposition: 'top center'
        });
    }

    const layout = {
        title: `K-Means Clustering (k=${data.n_clusters})`,
        xaxis: { title: 'Beta', gridcolor: '#4a6fa5' },
        yaxis: { title: 'Specific Risk (%)', gridcolor: '#4a6fa5' },
        paper_bgcolor: '#131820',
        plot_bgcolor: '#0a0e1a',
        font: { color: '#e8e8f0' },
        hovermode: 'closest',
        margin: { t: 60, r: 20, b: 60, l: 60 }
    };

    Plotly.newPlot('clustering-chart', traces, layout, { responsive: true });
}

// ==================== Correlation ====================

async function loadCorrelationData() {
    try {
        const tickers = state.tickers.join(',');
        const url = `${BASE_URL}/correlation?tickers=${tickers}`;
        
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to load correlation data');
        
        const data = await response.json();
        state.data.correlation = data;

        plotCorrelationHeatmap(data);
    } catch (err) {
        console.error('Correlation error:', err);
        throw err;
    }
}

function plotCorrelationHeatmap(data) {
    const trace = {
        z: data.matrix,
        x: data.tickers,
        y: data.tickers,
        type: 'heatmap',
        colorscale: 'Viridis',
        text: data.matrix.map(row => 
            row.map(val => val.toFixed(2))
        ),
        texttemplate: '%{text}',
        textfont: { size: 10 },
        hoverongaps: false
    };

    const layout = {
        title: 'Correlation Matrix',
        xaxis: { tickangle: 45 },
        paper_bgcolor: '#131820',
        plot_bgcolor: '#0a0e1a',
        font: { color: '#e8e8f0' },
        margin: { t: 60, r: 20, b: 100, l: 100 }
    };

    Plotly.newPlot('correlation-heatmap', [trace], layout, { responsive: true });
}

// ==================== Monte Carlo ====================

async function handleMonteCarlo() {
    const ticker = document.getElementById('mc-ticker').value.trim().toUpperCase();
    const days = parseInt(document.getElementById('mc-days').value);
    const sims = parseInt(document.getElementById('mc-sims').value);

    if (!ticker) {
        showError('Please enter a ticker');
        return;
    }

    showLoading(true);
    hideError();

    try {
        const url = `${BASE_URL}/monte-carlo?ticker=${ticker}&days_forecast=${days}&num_simulations=${sims}`;
        
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to run Monte Carlo simulation');
        
        const data = await response.json();
        state.data.monteCarlo = data;

        plotMonteCarloResults(data);
        displayMCResults(data);
        
        showLoading(false);
    } catch (err) {
        showError(`Monte Carlo error: ${err.message}`);
        showLoading(false);
    }
}

function plotMonteCarloResults(data) {
    const chartData = data.chart_data || {};
    const dates = chartData.dates || [];
    const p10 = chartData.p10 || [];
    const median = chartData.median || [];
    const p90 = chartData.p90 || [];

    const traces = [];

    // Add median path (95% confidence)
    if (median && median.length > 0) {
        traces.push({
            x: dates,
            y: median,
            type: 'scatter',
            mode: 'lines',
            name: 'Median Path',
            line: { color: '#6b8fa3', width: 3 },
            fill: 'tonexty',
            fillcolor: 'rgba(107, 143, 163, 0.15)',
            hovertemplate: '<b>Median</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        });
    }

    // Add 90th percentile
    if (p90 && p90.length > 0) {
        traces.push({
            x: dates,
            y: p90,
            type: 'scatter',
            mode: 'lines',
            name: '90th Percentile',
            line: { color: '#7a9fba', width: 2, dash: 'dash' },
            hovertemplate: '<b>90th %ile</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        });
    }

    // Add 10th percentile
    if (p10 && p10.length > 0) {
        traces.push({
            x: dates,
            y: p10,
            type: 'scatter',
            mode: 'lines',
            name: '10th Percentile',
            line: { color: '#5a7a99', width: 2, dash: 'dash' },
            fill: 'tonexty',
            fillcolor: 'rgba(107, 143, 163, 0.1)',
            hovertemplate: '<b>10th %ile</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        });
    }

    const layout = {
        title: `Monte Carlo Simulation: ${data.meta?.ticker || 'Asset'} (${data.meta?.simulations_count || '?'} simulations, ${data.meta?.forecast_days || '?'} days)`,
        xaxis: { title: 'Date', gridcolor: '#4a6fa5' },
        yaxis: { title: 'Price ($)', gridcolor: '#4a6fa5' },
        paper_bgcolor: '#131820',
        plot_bgcolor: '#0a0e1a',
        font: { color: '#e8e8f0' },
        hovermode: 'x unified',
        margin: { t: 60, r: 20, b: 60, l: 60 }
    };

    Plotly.newPlot('mc-chart', traces, layout, { responsive: true });
}

function displayMCResults(data) {
    const resultsDiv = document.getElementById('mc-results');
    const kpis = data.kpis || {};
    const risk = kpis.risk_analysis || {};
    const meta = data.meta || {};
    
    resultsDiv.innerHTML = `
        <h3>Simulation Results for ${meta.ticker}</h3>
        <div class="result-grid">
            <div class="result-card">
                <div class="label">Current Price</div>
                <div class="value">$${meta.current_price?.toFixed(2) || 'N/A'}</div>
            </div>
            <div class="result-card">
                <div class="label">Expected Return</div>
                <div class="value">${kpis.expected_return?.toFixed(2) || 'N/A'}%</div>
            </div>
            <div class="result-card">
                <div class="label">Win Probability</div>
                <div class="value">${kpis.win_probability?.toFixed(1) || 'N/A'}%</div>
            </div>
            <div class="result-card">
                <div class="label">Sharpe Ratio</div>
                <div class="value">${kpis.sharpe_ratio?.toFixed(2) || 'N/A'}</div>
            </div>
            <div class="result-card">
                <div class="label">Volatility (Annualized)</div>
                <div class="value">${kpis.volatility?.toFixed(2) || 'N/A'}%</div>
            </div>
            <div class="result-card">
                <div class="label">VaR 95% (Return)</div>
                <div class="value">${risk.VaR_95_percent?.toFixed(2) || 'N/A'}%</div>
            </div>
            <div class="result-card">
                <div class="label">VaR 95% (Amount)</div>
                <div class="value">$${risk.VaR_95_value?.toFixed(2) || 'N/A'}</div>
            </div>
            <div class="result-card">
                <div class="label">CVaR 95%</div>
                <div class="value">${risk.CVaR_95_percent?.toFixed(2) || 'N/A'}%</div>
            </div>
        </div>
        ${risk.interpretation ? `<div style="margin-top: 20px; padding: 15px; background: rgba(107, 143, 163, 0.1); border-radius: 8px; color: #a0a8c0;"><strong>Risk Interpretation:</strong> ${risk.interpretation}</div>` : ''}
    `;
}

// ==================== Navigation ====================

function handleNavTab(e) {
    const sectionId = e.target.dataset.section;
    
    // Update active tab
    navTabs.forEach(tab => tab.classList.remove('active'));
    e.target.classList.add('active');

    // Update active section
    sections.forEach(section => section.classList.remove('active'));
    document.getElementById(sectionId).classList.add('active');
}

// ==================== Utilities ====================

function showLoading(show) {
    loadingSpinner.style.display = show ? 'flex' : 'none';
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.add('show');
}

function hideError() {
    errorMessage.classList.remove('show');
    errorMessage.textContent = '';
}

// Add cluster badge styles dynamically
const style = document.createElement('style');
style.textContent = `
    .cluster-badge {
        background: linear-gradient(135deg, #6b8fa3, #7a9fba);
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
`;
document.head.appendChild(style);
