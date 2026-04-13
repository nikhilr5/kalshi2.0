"""
Graph recorded market data from SQLite database.
Creates an interactive HTML page with dropdowns for:
    - Expiration event
    - Market range (specific markets or all)
    - Show type (bid, ask, both)
    - BTC spot overlay

Usage:
    python3 graph_recorded_data.py market_data_2026-04-12.db
    python3 graph_recorded_data.py  # defaults to most recent .db file
"""

import sqlite3
import pandas as pd
import json
import os
import sys
import glob

os.makedirs("graphs", exist_ok=True)

# find database file
if len(sys.argv) > 1:
    db_path = sys.argv[1]
else:
    # find most recent .db file
    db_files = sorted(glob.glob("market_data_*.db"))
    if not db_files:
        db_files = sorted(glob.glob("data/market_data_*.db"))
    if not db_files:
        print("No database files found. Pass path as argument: python3 graph_recorded_data.py market_data_2026-04-12.db")
        sys.exit(1)
    db_path = db_files[-1]

print(f"Loading {db_path}...")
db_date = db_path.split("market_data_")[-1].replace(".db", "")

conn = sqlite3.connect(db_path)

# load orderbook data
df = pd.read_sql("""
    SELECT timestamp, ticker, event_ticker, yes_sub_title, yes_bid, yes_ask, btc_price, strike
    FROM orderbook
    ORDER BY timestamp
""", conn)

# load BTC prices
btc = pd.read_sql("SELECT timestamp, price FROM btc_price ORDER BY timestamp", conn)
conn.close()

print(f"Loaded {len(df):,} orderbook rows, {len(btc):,} BTC rows")

# get unique events
events = sorted(df["event_ticker"].unique())

# build data grouped by event, then by ticker
event_data = {}
for event in events:
    sub = df[df["event_ticker"] == event]
    tickers = sub.sort_values("strike")["ticker"].unique()

    markets = []
    for ticker in tickers:
        t_data = sub[sub["ticker"] == ticker]
        if len(t_data) < 2:
            continue
        label = t_data["yes_sub_title"].iloc[0] or ticker
        strike = float(t_data["strike"].iloc[0])
        markets.append({
            "ticker": ticker,
            "label": label,
            "strike": strike,
            "timestamps": t_data["timestamp"].tolist(),
            "yes_bid": t_data["yes_bid"].tolist(),
            "yes_ask": t_data["yes_ask"].tolist(),
            "btc_price": t_data["btc_price"].tolist(),
        })
    event_data[event] = markets

chart_data = json.dumps({
    "events": events,
    "event_data": event_data,
    "btc": {
        "timestamps": btc["timestamp"].tolist(),
        "prices": btc["price"].tolist(),
    },
    "db_date": db_date,
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
.controls { display: flex; gap: 1rem; margin-bottom: 1rem; align-items: flex-end; flex-wrap: wrap; }
select { background: #141923; color: #c8cdd5; border: 1px solid #1e2736; padding: 0.5rem 1rem; border-radius: 6px;
         font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; cursor: pointer; outline: none; }
select:hover { border-color: #2a3a52; }
select:focus { border-color: #3b82f6; }
select[multiple] { height: 200px; min-width: 250px; }
label { font-size: 0.7rem; color: #5a6270; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.05em; }
.control-group { display: flex; flex-direction: column; gap: 4px; }
.stats { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #5a6270; margin-bottom: 0.5rem; }
.hint { font-size: 0.65rem; color: #3b4555; margin-top: 2px; }
button { background: #3b82f6; color: white; border: none; padding: 0.5rem 1rem; border-radius: 6px;
         font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; cursor: pointer; }
button:hover { background: #2563eb; }
.btn-row { display: flex; gap: 0.5rem; align-items: flex-end; }
</style>
</head><body>
<div class="container">
<h1>Recorded Market Data</h1>
<div class="controls">
    <div class="control-group">
        <label>Expiration Event</label>
        <select id="eventSelect"></select>
    </div>
    <div class="control-group">
        <label>Markets (Ctrl/Cmd+click to multi-select)</label>
        <select id="marketSelect" multiple></select>
        <div class="hint">Leave empty = show all</div>
    </div>
    <div class="control-group">
        <label>Show</label>
        <select id="showSelect">
            <option value="bid" selected>Yes Bid</option>
            <option value="ask">Yes Ask</option>
            <option value="both">Bid & Ask</option>
            <option value="mid">Mid Price</option>
        </select>
    </div>
    <div class="control-group btn-row">
        <button onclick="selectAll()">Select All</button>
        <button onclick="selectNone()">Clear</button>
        <button onclick="selectNearBTC()">Near BTC</button>
    </div>
</div>
<div id="stats" class="stats"></div>
<div id="priceChart"></div>
<div id="btcChart"></div>
</div>
<script>
const ALL_DATA = """ + chart_data + """;

// populate event dropdown
const eventSelect = document.getElementById('eventSelect');
ALL_DATA.events.forEach(ev => {
    const markets = ALL_DATA.event_data[ev] || [];
    const opt = document.createElement('option');
    opt.value = ev;
    opt.text = ev + ' (' + markets.length + ' markets)';
    eventSelect.appendChild(opt);
});

function populateMarketSelect() {
    const event = eventSelect.value;
    const markets = ALL_DATA.event_data[event] || [];
    const sel = document.getElementById('marketSelect');
    sel.innerHTML = '';

    markets.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.ticker;
        opt.text = m.label + ' (strike: ' + m.strike + ')';
        sel.appendChild(opt);
    });
}

function getSelectedMarkets() {
    const sel = document.getElementById('marketSelect');
    const selected = Array.from(sel.selectedOptions).map(o => o.value);
    // if nothing selected, return all
    if (selected.length === 0) {
        return Array.from(sel.options).map(o => o.value);
    }
    return selected;
}

function selectAll() {
    const sel = document.getElementById('marketSelect');
    Array.from(sel.options).forEach(o => o.selected = true);
    update();
}

function selectNone() {
    const sel = document.getElementById('marketSelect');
    Array.from(sel.options).forEach(o => o.selected = false);
    update();
}

function selectNearBTC() {
    // select markets whose strike is within $3000 of latest BTC price
    const btcPrices = ALL_DATA.btc.prices;
    const latestBTC = btcPrices.length > 0 ? btcPrices[btcPrices.length - 1] : 0;
    const event = eventSelect.value;
    const markets = ALL_DATA.event_data[event] || [];
    const sel = document.getElementById('marketSelect');

    Array.from(sel.options).forEach((opt, i) => {
        const m = markets.find(m => m.ticker === opt.value);
        if (m) {
            const dist = Math.abs(m.strike - latestBTC);
            opt.selected = dist <= 3000;
        }
    });
    update();
}

function update() {
    const event = eventSelect.value;
    const showType = document.getElementById('showSelect').value;
    const allMarkets = ALL_DATA.event_data[event] || [];
    const selectedTickers = getSelectedMarkets();

    // filter to selected markets
    const markets = allMarkets.filter(m => selectedTickers.includes(m.ticker));

    if (markets.length === 0) {
        document.getElementById('stats').textContent = 'No markets selected';
        Plotly.react('priceChart', [], {});
        return;
    }

    const totalPoints = markets.reduce((sum, m) => sum + m.timestamps.length, 0);
    document.getElementById('stats').textContent =
        markets.length + ' / ' + allMarkets.length + ' markets shown | ' +
        totalPoints.toLocaleString() + ' data points';

    // color palette
    const colors = [
        '#ef4444','#f59e0b','#22c55e','#3b82f6','#a855f7','#ec4899',
        '#14b8a6','#f97316','#06b6d4','#8b5cf6','#10b981','#e11d48',
        '#0ea5e9','#d946ef','#84cc16','#facc15','#fb923c','#34d399',
        '#60a5fa','#c084fc','#f472b6','#2dd4bf','#fbbf24','#4ade80',
        '#818cf8','#fb7185','#38bdf8','#a3e635','#fde68a','#c4b5fd',
    ];

    const traces = [];
    markets.forEach((m, i) => {
        const color = colors[i % colors.length];

        if (showType === 'bid' || showType === 'both') {
            traces.push({
                x: m.timestamps, y: m.yes_bid,
                mode: 'lines', name: m.label,
                line: { color: color, width: 1.5 },
                hovertemplate: '<b>' + m.label + '</b><br>Bid: $%{y:.4f}<br>BTC: $' +
                    (m.btc_price.length > 0 ? m.btc_price[m.btc_price.length-1].toLocaleString() : '?') +
                    '<br>%{x}<extra></extra>',
            });
        }
        if (showType === 'ask' || showType === 'both') {
            traces.push({
                x: m.timestamps, y: m.yes_ask,
                mode: 'lines',
                name: m.label + (showType === 'both' ? ' (ask)' : ''),
                line: { color: color, width: 1, dash: showType === 'both' ? 'dot' : 'solid' },
                hovertemplate: '<b>' + m.label + '</b><br>Ask: $%{y:.4f}<br>%{x}<extra></extra>',
            });
        }
        if (showType === 'mid') {
            const midPrices = m.yes_bid.map((b, j) => {
                const a = m.yes_ask[j] || 0;
                if (b > 0 && a > 0) return (b + a) / 2;
                if (b > 0) return b;
                if (a > 0) return a;
                return 0;
            });
            traces.push({
                x: m.timestamps, y: midPrices,
                mode: 'lines', name: m.label,
                line: { color: color, width: 1.5 },
                hovertemplate: '<b>' + m.label + '</b><br>Mid: $%{y:.4f}<br>%{x}<extra></extra>',
            });
        }
    });

    const layout = {
        paper_bgcolor: '#0a0e17',
        plot_bgcolor: '#141923',
        font: { color: '#c8cdd5', family: 'DM Sans' },
        title: { text: showType.charAt(0).toUpperCase() + showType.slice(1) + ' — ' + event, font: { size: 14 } },
        xaxis: { gridcolor: '#1e2736', linecolor: '#1e2736', title: 'Time' },
        yaxis: { gridcolor: '#1e2736', linecolor: '#1e2736', title: 'Price ($)' },
        height: 500,
        margin: { l: 60, r: 60, t: 50, b: 50 },
        legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 9 } },
        hovermode: 'closest',
    };

    Plotly.react('priceChart', traces, layout);

    // BTC chart
    Plotly.react('btcChart', [{
        x: ALL_DATA.btc.timestamps,
        y: ALL_DATA.btc.prices,
        mode: 'lines',
        name: 'BTC/USD',
        line: { color: '#f59e0b', width: 2 },
        hovertemplate: '<b>BTC</b><br>$%{y:,.2f}<br>%{x}<extra></extra>',
    }], {
        paper_bgcolor: '#0a0e17',
        plot_bgcolor: '#141923',
        font: { color: '#c8cdd5', family: 'DM Sans' },
        title: { text: 'BTC/USD Spot', font: { size: 14 } },
        xaxis: { gridcolor: '#1e2736', linecolor: '#1e2736', title: 'Time' },
        yaxis: { gridcolor: '#1e2736', linecolor: '#1e2736', title: 'BTC ($)' },
        height: 300,
        margin: { l: 60, r: 60, t: 50, b: 50 },
        hovermode: 'closest',
    });
}

// event listeners
eventSelect.addEventListener('change', () => { populateMarketSelect(); update(); });
document.getElementById('marketSelect').addEventListener('change', update);
document.getElementById('showSelect').addEventListener('change', update);

// initial render
populateMarketSelect();
update();
</script>
</body></html>"""

output_file = f"graphs/recorded_markets_{db_date}.html"
with open(output_file, "w") as f:
    f.write(html)

print(f"Saved to {output_file}")
print(f"Open with: open {output_file}")