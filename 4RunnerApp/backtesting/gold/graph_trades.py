# graph_backtest.py
import pandas as pd
import numpy as np
import json
import os

# create graphs directory
os.makedirs("graphs", exist_ok=True)

# load backtest results
df = pd.read_csv("data/backtest_results.csv")

# parse expiration dates for grouping
df["expiry_date"] = df["ticker"].apply(lambda t: t.split("-")[1][:7] if len(t.split("-")) >= 2 else "")

# get unique expirations
expirations = sorted(df["expiry_date"].unique())

# get tickers grouped by expiration, sorted by strike
ticker_by_expiry = {}
for exp in expirations:
    sub = df[df["expiry_date"] == exp]
    tickers = sub["ticker"].unique().tolist()
    # sort by strike
    def get_strike(t):
        if "-T" in t:
            try: return float(t.split("-T")[1])
            except: return 0
        elif "-B" in t:
            try: return float(t.split("-B")[1])
            except: return 0
        return 0
    tickers.sort(key=get_strike)
    ticker_by_expiry[exp] = tickers

# convert full dataframe to JSON
records = df.to_dict(orient="records")

chart_data = json.dumps({
    "trades": records,
    "expirations": expirations,
    "ticker_by_expiry": ticker_by_expiry,
})

html = """<!DOCTYPE html>
<html><head>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap');
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0e17; color: #c8cdd5; font-family: 'DM Sans', sans-serif; }
.container { max-width: 1600px; margin: 0 auto; padding: 1.5rem; }
h1 { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 500; color: #e8ecf1; margin-bottom: 0.75rem; }
.controls { display: flex; gap: 1rem; margin-bottom: 1.25rem; align-items: flex-end; flex-wrap: wrap; }
.control-group { display: flex; flex-direction: column; gap: 4px; }
label { font-size: 0.7rem; color: #5a6270; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.05em; }
select { background: #141923; color: #c8cdd5; border: 1px solid #1e2736; padding: 0.5rem 0.75rem; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; cursor: pointer; outline: none; min-width: 90px; }
select:hover { border-color: #2a3a52; }
select:focus { border-color: #3b82f6; }
.divider { width: 1px; height: 36px; background: #1e2736; align-self: flex-end; margin-bottom: 4px; }
.ticker-label { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #3b82f6; background: #141923; border: 1px solid #1e2736; border-radius: 6px; padding: 0.5rem 1rem; display: inline-block; }
.stats { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
.stat { background: #141923; border: 1px solid #1e2736; border-radius: 8px; padding: 0.4rem 0.7rem; }
.stat-label { font-size: 0.55rem; color: #5a6270; text-transform: uppercase; letter-spacing: 0.05em; font-family: 'JetBrains Mono', monospace; }
.stat-value { font-size: 0.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.green { color: #22c55e; }
.red { color: #ef4444; }
.blue { color: #3b82f6; }
.amber { color: #f59e0b; }
.purple { color: #a855f7; }
.teal { color: #2dd4bf; }
.chart { margin-bottom: 0.5rem; }
</style>
</head><body>
<div class="container">
<h1>KXGOLDMON backtest explorer</h1>
<div class="controls">
    <div class="control-group">
        <label>Expiration</label>
        <select id="expirySelect"></select>
    </div>
    <div class="control-group">
        <label>Contract</label>
        <select id="tickerSelect" style="min-width:200px;"></select>
    </div>
    <div class="divider"></div>
    <div class="control-group">
        <label>Min edge</label>
        <select id="edgeSelect">
            <option value="0.01">0.01</option>
            <option value="0.02">0.02</option>
            <option value="0.03">0.03</option>
            <option value="0.05" selected>0.05</option>
            <option value="0.07">0.07</option>
            <option value="0.10">0.10</option>
            <option value="0.15">0.15</option>
            <option value="0.20">0.20</option>
        </select>
    </div>
    <div class="control-group">
        <label>Min DTE</label>
        <select id="dteSelect">
            <option value="0">0</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3" selected>3</option>
            <option value="5">5</option>
            <option value="7">7</option>
            <option value="10">10</option>
            <option value="14">14</option>
        </select>
    </div>
    <div class="divider"></div>
    <div class="control-group">
        <label>Theo min</label>
        <select id="theoMinSelect">
            <option value="0.00" selected>0.00</option>
            <option value="0.05">0.05</option>
            <option value="0.10">0.10</option>
            <option value="0.15">0.15</option>
            <option value="0.20">0.20</option>
            <option value="0.25">0.25</option>
            <option value="0.30">0.30</option>
            <option value="0.35">0.35</option>
            <option value="0.40">0.40</option>
            <option value="0.45">0.45</option>
        </select>
    </div>
    <div class="control-group">
        <label>Theo max</label>
        <select id="theoMaxSelect">
            <option value="1.00" selected>1.00</option>
            <option value="0.95">0.95</option>
            <option value="0.90">0.90</option>
            <option value="0.85">0.85</option>
            <option value="0.80">0.80</option>
            <option value="0.75">0.75</option>
            <option value="0.70">0.70</option>
            <option value="0.65">0.65</option>
            <option value="0.60">0.60</option>
            <option value="0.55">0.55</option>
        </select>
    </div>
    <div class="divider"></div>
    <div class="control-group">
        <label>Mkt min</label>
        <select id="mktMinSelect">
            <option value="0.00" selected>0.00</option>
            <option value="0.05">0.05</option>
            <option value="0.10">0.10</option>
            <option value="0.15">0.15</option>
            <option value="0.20">0.20</option>
            <option value="0.25">0.25</option>
            <option value="0.30">0.30</option>
            <option value="0.35">0.35</option>
            <option value="0.40">0.40</option>
            <option value="0.45">0.45</option>
        </select>
    </div>
    <div class="control-group">
        <label>Mkt max</label>
        <select id="mktMaxSelect">
            <option value="1.00" selected>1.00</option>
            <option value="0.95">0.95</option>
            <option value="0.90">0.90</option>
            <option value="0.85">0.85</option>
            <option value="0.80">0.80</option>
            <option value="0.75">0.75</option>
            <option value="0.70">0.70</option>
            <option value="0.65">0.65</option>
            <option value="0.60">0.60</option>
            <option value="0.55">0.55</option>
        </select>
    </div>
</div>
<div id="tickerInfo" style="margin-bottom: 1rem;"></div>
<div id="statsRow" class="stats"></div>
<div id="priceChart" class="chart"></div>
<div id="volChart" class="chart"></div>
<div id="edgeChart" class="chart"></div>
<div id="pnlChart" class="chart"></div>
</div>
<script>
const ALL_DATA = """ + chart_data + """;

const expirySelect = document.getElementById('expirySelect');
ALL_DATA.expirations.forEach(exp => {
    const count = ALL_DATA.ticker_by_expiry[exp].length;
    const opt = document.createElement('option');
    opt.value = exp;
    opt.text = exp + ' (' + count + ' contracts)';
    expirySelect.appendChild(opt);
});

function updateTickerDropdown() {
    const expiry = expirySelect.value;
    const tickerSelect = document.getElementById('tickerSelect');
    tickerSelect.innerHTML = '';
    const tickers = ALL_DATA.ticker_by_expiry[expiry] || [];
    tickers.forEach(t => {
        const count = ALL_DATA.trades.filter(r => r.ticker === t).length;
        let strike = '';
        let prefix = '';
        if (t.includes('-T')) { strike = t.split('-T')[1]; prefix = 'above '; }
        else if (t.includes('-B')) { strike = t.split('-B')[1]; prefix = 'range '; }
        const opt = document.createElement('option');
        opt.value = t;
        opt.text = prefix + '$' + strike + ' (' + count + ')';
        tickerSelect.appendChild(opt);
    });
    update();
}

function update() {
    const ticker = document.getElementById('tickerSelect').value;
    const minEdge = parseFloat(document.getElementById('edgeSelect').value);
    const minDte = parseFloat(document.getElementById('dteSelect').value);
    const theoMin = parseFloat(document.getElementById('theoMinSelect').value);
    const theoMax = parseFloat(document.getElementById('theoMaxSelect').value);
    const mktMin = parseFloat(document.getElementById('mktMinSelect').value);
    const mktMax = parseFloat(document.getElementById('mktMaxSelect').value);

    let trades = ALL_DATA.trades.filter(r => r.ticker === ticker);
    trades.sort((a, b) => a.created_time.localeCompare(b.created_time));

    if (trades.length === 0) {
        document.getElementById('tickerInfo').innerHTML = '';
        document.getElementById('statsRow').innerHTML = '';
        return;
    }

    const strike = trades[0].strike;
    const settlement = trades[0].settlement;
    const settleVal = trades[0].settle_val;
    const contractType = trades[0].contract_type;
    const yesSub = trades[0].yes_sub_title || '';

    document.getElementById('tickerInfo').innerHTML =
        '<span class="ticker-label">' + ticker + '</span>' +
        '<span style="margin-left:1rem;font-size:0.8rem;color:#5a6270;">' + yesSub + '</span>';

    const times = trades.map(t => t.created_time);
    const marketPrices = trades.map(t => t.market_price);
    const theos = trades.map(t => t.theo);
    const edges = trades.map(t => t.edge);
    const goldPrices = trades.map(t => t.gold_price);
    const vols = trades.map(t => t.gvz_vol);
    const dtes = trades.map(t => t.days_to_exp);

    const hoverMarket = trades.map(t =>
        '<b>Market: $' + t.market_price.toFixed(4) + '</b>' +
        '<br>Theo: $' + t.theo.toFixed(4) +
        '<br>Edge: ' + (t.edge >= 0 ? '+' : '') + t.edge.toFixed(4) +
        '<br>Gold: $' + (t.gold_price ? t.gold_price.toFixed(2) : '?') +
        '<br>Vol: ' + (t.gvz_vol ? t.gvz_vol.toFixed(4) : '?') +
        '<br>DTE: ' + t.days_to_exp.toFixed(1) + 'd'
    );

    const hoverTheo = trades.map(t =>
        '<b>Theo: $' + t.theo.toFixed(4) + '</b>' +
        '<br>Market: $' + t.market_price.toFixed(4) +
        '<br>Edge: ' + (t.edge >= 0 ? '+' : '') + t.edge.toFixed(4) +
        '<br>Gold: $' + (t.gold_price ? t.gold_price.toFixed(2) : '?') +
        '<br>Vol: ' + (t.gvz_vol ? t.gvz_vol.toFixed(4) : '?') +
        '<br>DTE: ' + t.days_to_exp.toFixed(1) + 'd'
    );

    const buyIdx = [];
    const sellIdx = [];
    const strategyPnl = [];

    trades.forEach((t, i) => {
        const hasSettlement = t.settle_val !== null && t.settle_val !== '' && !isNaN(t.settle_val) && String(t.settle_val) !== 'nan';
        const inTheoRange = t.theo >= theoMin && t.theo <= theoMax;
        const inMktRange = t.market_price >= mktMin && t.market_price <= mktMax;
        const meetsEdge = Math.abs(t.edge) >= minEdge;
        const meetsDte = t.days_to_exp >= minDte;

        if (meetsEdge && meetsDte && inTheoRange && inMktRange) {
            if (t.edge > 0) {
                buyIdx.push(i);
                if (hasSettlement) {
                    strategyPnl.push({ time: t.created_time, pnl: t.settle_val - t.market_price });
                }
            } else {
                sellIdx.push(i);
                if (hasSettlement) {
                    strategyPnl.push({ time: t.created_time, pnl: t.market_price - t.settle_val });
                }
            }
        }
    });

    let cumPnl = [];
    let cumTimes = [];
    let runningPnl = 0;
    strategyPnl.forEach(s => {
        runningPnl += s.pnl;
        cumTimes.push(s.time);
        cumPnl.push(Math.round(runningPnl * 10000) / 10000);
    });

    const avgVol = vols.length > 0 ? (vols.reduce((a, b) => a + b, 0) / vols.length).toFixed(4) : '?';
    const avgGold = goldPrices.length > 0 ? Math.round(goldPrices.reduce((a, b) => a + b, 0) / goldPrices.length) : '?';
    const totalPnl = strategyPnl.reduce((a, b) => a + b.pnl, 0).toFixed(4);
    const nTrades = buyIdx.length + sellIdx.length;
    const avgTradePnl = strategyPnl.length > 0 ? (strategyPnl.reduce((a, b) => a + b.pnl, 0) / strategyPnl.length).toFixed(4) : 'N/A';
    const winRate = strategyPnl.length > 0
        ? (strategyPnl.filter(s => s.pnl > 0).length / strategyPnl.length * 100).toFixed(1) + '%'
        : 'N/A';

    let dteCutoff = '';
    for (let i = 0; i < trades.length; i++) {
        if (trades[i].days_to_exp < minDte) { dteCutoff = trades[i].created_time; break; }
    }

    let settleDisplay = 'pending';
    let settleClass = 'amber';
    if (settlement === 'yes') { settleDisplay = 'YES'; settleClass = 'green'; }
    else if (settlement === 'no') { settleDisplay = 'NO'; settleClass = 'red'; }

    document.getElementById('statsRow').innerHTML = `
        <div class="stat"><div class="stat-label">Type</div><div class="stat-value">${contractType}</div></div>
        <div class="stat"><div class="stat-label">Strike</div><div class="stat-value">$${strike}</div></div>
        <div class="stat"><div class="stat-label">Settled</div><div class="stat-value ${settleClass}">${settleDisplay}</div></div>
        <div class="stat"><div class="stat-label">Trades</div><div class="stat-value blue">${trades.length}</div></div>
        <div class="stat"><div class="stat-label">Signals</div><div class="stat-value amber">${nTrades}</div></div>
        <div class="stat"><div class="stat-label">Buys</div><div class="stat-value green">${buyIdx.length}</div></div>
        <div class="stat"><div class="stat-label">Sells</div><div class="stat-value red">${sellIdx.length}</div></div>
        <div class="stat"><div class="stat-label">Total PnL</div><div class="stat-value ${parseFloat(totalPnl) >= 0 ? 'green' : 'red'}">$${totalPnl}</div></div>
        <div class="stat"><div class="stat-label">Avg PnL</div><div class="stat-value ${avgTradePnl !== 'N/A' && parseFloat(avgTradePnl) >= 0 ? 'green' : 'red'}">$${avgTradePnl}</div></div>
        <div class="stat"><div class="stat-label">Win Rate</div><div class="stat-value">${winRate}</div></div>
        <div class="stat"><div class="stat-label">Theo</div><div class="stat-value purple">${theoMin}-${theoMax}</div></div>
        <div class="stat"><div class="stat-label">Mkt</div><div class="stat-value teal">${mktMin}-${mktMax}</div></div>
        <div class="stat"><div class="stat-label">Avg Vol</div><div class="stat-value purple">${avgVol}</div></div>
        <div class="stat"><div class="stat-label">Avg Gold</div><div class="stat-value">$${avgGold}</div></div>
    `;

    const layoutBase = {
        paper_bgcolor: '#0a0e17',
        plot_bgcolor: '#141923',
        font: { color: '#c8cdd5', family: 'DM Sans' },
        xaxis: { gridcolor: '#1e2736', linecolor: '#1e2736' },
        yaxis: { gridcolor: '#1e2736', linecolor: '#1e2736' },
        margin: { l: 60, r: 60, t: 40, b: 40 },
        legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } },
        hovermode: 'closest',
    };

    const dteShapes = dteCutoff ? [{
        type: 'line', x0: dteCutoff, x1: dteCutoff, y0: 0, y1: 1, yref: 'paper',
        line: { color: '#f59e0b', width: 2, dash: 'dashdot' },
    }] : [];
    const dteAnnotations = dteCutoff ? [{
        x: dteCutoff, y: 1, yref: 'paper', text: 'DTE cutoff (' + minDte + 'd)',
        showarrow: false, font: { color: '#f59e0b', size: 10 }, yanchor: 'bottom',
    }] : [];

    // theo range shading (purple)
    const theoRangeShapes = (theoMin > 0 || theoMax < 1) ? [
        { type: 'rect', y0: theoMin, y1: theoMax, x0: 0, x1: 1, xref: 'paper',
          fillcolor: 'rgba(168,85,247,0.06)', line: { width: 0 } },
        { type: 'line', y0: theoMin, y1: theoMin, x0: 0, x1: 1, xref: 'paper',
          line: { color: '#a855f7', width: 1, dash: 'dot' } },
        { type: 'line', y0: theoMax, y1: theoMax, x0: 0, x1: 1, xref: 'paper',
          line: { color: '#a855f7', width: 1, dash: 'dot' } },
    ] : [];

    // market range shading (teal)
    const mktRangeShapes = (mktMin > 0 || mktMax < 1) ? [
        { type: 'rect', y0: mktMin, y1: mktMax, x0: 0, x1: 1, xref: 'paper',
          fillcolor: 'rgba(45,212,191,0.06)', line: { width: 0 } },
        { type: 'line', y0: mktMin, y1: mktMin, x0: 0, x1: 1, xref: 'paper',
          line: { color: '#2dd4bf', width: 1, dash: 'dot' } },
        { type: 'line', y0: mktMax, y1: mktMax, x0: 0, x1: 1, xref: 'paper',
          line: { color: '#2dd4bf', width: 1, dash: 'dot' } },
    ] : [];

    const settleShapes = [];
    if (settlement === 'yes') {
        settleShapes.push({ type: 'line', y0: 1.0, y1: 1.0, x0: 0, x1: 1, xref: 'paper',
            line: { color: '#22c55e', width: 2, dash: 'dash' } });
    } else if (settlement === 'no') {
        settleShapes.push({ type: 'line', y0: 0.0, y1: 0.0, x0: 0, x1: 1, xref: 'paper',
            line: { color: '#ef4444', width: 2, dash: 'dash' } });
    }

    // panel 1
    const priceTraces = [
        { x: times, y: marketPrices, mode: 'markers+lines', name: 'market',
          marker: { size: 5, color: '#3b82f6' }, line: { width: 0.5, color: 'rgba(59,130,246,0.3)' },
          text: hoverMarket, hoverinfo: 'text+x' },
        { x: times, y: theos, mode: 'markers+lines', name: 'theo',
          marker: { size: 5, color: '#f59e0b' }, line: { width: 0.5, color: 'rgba(245,158,11,0.3)' },
          text: hoverTheo, hoverinfo: 'text+x' },
    ];
    if (buyIdx.length > 0) {
        const buyHover = buyIdx.map(i =>
            '<b>BUY</b><br>Market: $' + marketPrices[i].toFixed(4) +
            '<br>Theo: $' + theos[i].toFixed(4) +
            '<br>Edge: +' + edges[i].toFixed(4) +
            '<br>Gold: $' + goldPrices[i].toFixed(2) +
            '<br>DTE: ' + dtes[i].toFixed(1) + 'd'
        );
        priceTraces.push({
            x: buyIdx.map(i => times[i]), y: buyIdx.map(i => marketPrices[i]),
            mode: 'markers', name: 'buy (' + buyIdx.length + ')',
            marker: { size: 11, symbol: 'triangle-up', color: '#22c55e', line: { width: 1, color: 'white' } },
            text: buyHover, hoverinfo: 'text+x',
        });
    }
    if (sellIdx.length > 0) {
        const sellHover = sellIdx.map(i =>
            '<b>SELL</b><br>Market: $' + marketPrices[i].toFixed(4) +
            '<br>Theo: $' + theos[i].toFixed(4) +
            '<br>Edge: ' + edges[i].toFixed(4) +
            '<br>Gold: $' + goldPrices[i].toFixed(2) +
            '<br>DTE: ' + dtes[i].toFixed(1) + 'd'
        );
        priceTraces.push({
            x: sellIdx.map(i => times[i]), y: sellIdx.map(i => marketPrices[i]),
            mode: 'markers', name: 'sell (' + sellIdx.length + ')',
            marker: { size: 11, symbol: 'triangle-down', color: '#ef4444', line: { width: 1, color: 'white' } },
            text: sellHover, hoverinfo: 'text+x',
        });
    }

    Plotly.react('priceChart', priceTraces, {
        ...layoutBase,
        title: { text: 'Market price vs Theo', font: { size: 14 } },
        yaxis: { ...layoutBase.yaxis, title: 'Price ($)' },
        height: 420,
        shapes: [...settleShapes, ...dteShapes, ...theoRangeShapes, ...mktRangeShapes],
        annotations: dteAnnotations,
    });

    // panel 2
    Plotly.react('volChart', [
        { x: times, y: vols, mode: 'lines', name: 'GVZ vol', yaxis: 'y',
          line: { color: '#a855f7', width: 2 } },
        { x: times, y: goldPrices, mode: 'lines', name: 'gold price', yaxis: 'y2',
          line: { color: '#22c55e', width: 2 } },
    ], {
        ...layoutBase,
        title: { text: 'Volatility & Gold Price', font: { size: 14 } },
        yaxis: { ...layoutBase.yaxis, title: 'Vol', titlefont: { color: '#a855f7' }, tickfont: { color: '#a855f7' } },
        yaxis2: { title: 'Gold ($)', titlefont: { color: '#22c55e' }, tickfont: { color: '#22c55e' },
            overlaying: 'y', side: 'right', gridcolor: '#1e2736' },
        height: 250,
        shapes: [
            { type: 'line', y0: strike, y1: strike, x0: 0, x1: 1, xref: 'paper', yref: 'y2',
              line: { color: '#ef4444', width: 1, dash: 'dot' } },
            ...dteShapes,
        ],
    });

    // panel 3
    const edgeColors = edges.map(e => e > 0 ? '#22c55e' : '#ef4444');
    Plotly.react('edgeChart', [
        { x: times, y: edges, type: 'bar', name: 'edge', marker: { color: edgeColors } },
    ], {
        ...layoutBase,
        title: { text: 'Edge (theo - market)', font: { size: 14 } },
        yaxis: { ...layoutBase.yaxis, title: 'Edge' },
        height: 200,
        shapes: [
            { type: 'line', y0: minEdge, y1: minEdge, x0: 0, x1: 1, xref: 'paper',
              line: { color: '#22c55e', width: 1, dash: 'dash' } },
            { type: 'line', y0: -minEdge, y1: -minEdge, x0: 0, x1: 1, xref: 'paper',
              line: { color: '#ef4444', width: 1, dash: 'dash' } },
            ...dteShapes,
        ],
    });

    // panel 4
    if (cumTimes.length > 0) {
        const finalPnl = cumPnl[cumPnl.length - 1].toFixed(4);
        Plotly.react('pnlChart', [
            { x: cumTimes, y: cumPnl, mode: 'lines', name: 'cumulative pnl',
              line: { color: '#f59e0b', width: 2 }, fill: 'tozeroy', fillcolor: 'rgba(245,158,11,0.1)' },
        ], {
            ...layoutBase,
            title: { text: 'Cumulative PnL: $' + finalPnl + ' (' + cumPnl.length + ' trades)', font: { size: 14 } },
            yaxis: { ...layoutBase.yaxis, title: 'PnL ($)' },
            height: 250,
            shapes: [{ type: 'line', y0: 0, y1: 0, x0: 0, x1: 1, xref: 'paper',
                line: { color: '#5a6270', width: 1 } }],
        });
    } else {
        document.getElementById('pnlChart').innerHTML =
            '<div style="text-align:center;padding:2rem;color:#5a6270;font-family:JetBrains Mono,monospace;">No settled trades matching filters</div>';
    }
}

expirySelect.addEventListener('change', updateTickerDropdown);
document.getElementById('tickerSelect').addEventListener('change', update);
document.getElementById('edgeSelect').addEventListener('change', update);
document.getElementById('dteSelect').addEventListener('change', update);
document.getElementById('theoMinSelect').addEventListener('change', update);
document.getElementById('theoMaxSelect').addEventListener('change', update);
document.getElementById('mktMinSelect').addEventListener('change', update);
document.getElementById('mktMaxSelect').addEventListener('change', update);

updateTickerDropdown();
</script>
</body></html>"""

filename = "graphs/backtest_explorer.html"
with open(filename, "w") as f:
    f.write(html)

print(f"Saved to {filename}")
print(f"Open in browser: file://{os.path.abspath(filename)}")