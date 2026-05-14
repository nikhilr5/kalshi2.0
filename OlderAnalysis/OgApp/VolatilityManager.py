from ImpliedVolatilityCalculator import ImpliedVolatilityCalculator
from VolatilitySmile import VolatilitySmile
from Contract import Contract
import math
import json
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np


# manage how often we recalculate implied vols and refit smile
# all strikes should have same expiration
class VolatilityManager:
    def __init__(self, serve_plot: bool = True):
        self.call_impliedVol = {}
        self.put_impliedVol = {}
        self.rate = 0.05
        self.vol_calculator = ImpliedVolatilityCalculator()
        self.call_smile = VolatilitySmile()
        self.put_smile = VolatilitySmile()
        self.call_fitted = False
        self.put_fitted = False
        self.underlying_price = 0
        self.plot_count = 0

        # latest plot data as JSON for the browser
        self.plot_json = "{}"

        # start local web server for live chart
        if serve_plot:
            self._start_server()

    def _start_server(self):
        manager = self

        class Handler(SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/data":
                    # serve latest plot data
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(manager.plot_json.encode())
                elif self.path == "/" or self.path == "/index.html":
                    # serve the chart page
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(CHART_HTML.encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            # suppress request logs
            def log_message(self, format, *args):
                pass

        server = HTTPServer(("localhost", 8050), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print("Vol smile chart live at http://localhost:8050")

    # add or update a contract's market data
    def update_contract(self, contract: Contract):
        if contract.type == 'call':
            self.call_impliedVol[contract.strike] = contract
        else:
            self.put_impliedVol[contract.strike] = contract

    # recalculate all implied vols and refit smiles
    def recalculate(self, underlying_price):
        self.underlying_price = underlying_price

        # update call implied vols
        for strike, contract in self.call_impliedVol.items():
            iv = self.vol_calculator.solve_quadratic(
                underlying_price, contract.get_midpoint(),
                contract.strike, contract.getDaysTilExpiration(), contract.type
            )
            self.call_impliedVol[strike].implied_vol = iv

        # update put implied vols
        for strike, contract in self.put_impliedVol.items():
            iv = self.vol_calculator.solve_quadratic(
                underlying_price, contract.get_midpoint(),
                contract.strike, contract.getDaysTilExpiration(), contract.type
            )
            self.put_impliedVol[strike].implied_vol = iv

        # refit call smile (need at least 3 points)
        valid_calls = [(c.strike, c.implied_vol) for c in self.call_impliedVol.values() if c.implied_vol is not None]
        if len(valid_calls) >= 3:
            call_strikes, call_IVs = zip(*valid_calls)
            self.call_smile.fit(call_strikes, call_IVs)
            self.call_fitted = True

        # refit put smile
        valid_puts = [(p.strike, p.implied_vol) for p in self.put_impliedVol.values() if p.implied_vol is not None]
        if len(valid_puts) >= 3:
            put_strikes, put_IVs = zip(*valid_puts)
            self.put_smile.fit(put_strikes, put_IVs)
            self.put_fitted = True

        self.has_vol = True

        # update live chart data
        self._update_plot()

    # get vol estimate for a given contract from fitted smile
    def get_volatility(self, contract):
        if contract.type == "call":
            if self.call_fitted:
                vol = self.call_smile.solve(contract.strike)
                return max(vol, 0.01)  # floor at 1%

    # build plot data as JSON for the browser
    def _update_plot(self):
        data = {"underlying": self.underlying_price, "traces": []}

        if self.call_fitted:
            valid_calls = [(c.strike, c.implied_vol) for c in self.call_impliedVol.values() if c.implied_vol is not None]
            if valid_calls:
                strikes, ivs = zip(*sorted(valid_calls))
                # scatter points
                data["traces"].append({
                    "x": list(strikes), "y": list(ivs),
                    "mode": "markers", "name": "Call IV",
                    "marker": {"color": "blue", "size": 8}
                })
                # fitted curve
                s_min, s_max = min(strikes), max(strikes)
                x_fit = np.linspace(s_min, s_max, 100).tolist()
                y_fit = [self.call_smile.solve(s) for s in x_fit]
                data["traces"].append({
                    "x": x_fit, "y": y_fit,
                    "mode": "lines", "name": "Call Smile",
                    "line": {"color": "blue"}
                })

        if self.put_fitted:
            valid_puts = [(p.strike, p.implied_vol) for p in self.put_impliedVol.values() if p.implied_vol is not None]
            if valid_puts:
                strikes, ivs = zip(*sorted(valid_puts))
                data["traces"].append({
                    "x": list(strikes), "y": list(ivs),
                    "mode": "markers", "name": "Put IV",
                    "marker": {"color": "red", "size": 8}
                })
                s_min, s_max = min(strikes), max(strikes)
                x_fit = np.linspace(s_min, s_max, 100).tolist()
                y_fit = [self.put_smile.solve(s) for s in x_fit]
                data["traces"].append({
                    "x": x_fit, "y": y_fit,
                    "mode": "lines", "name": "Put Smile",
                    "line": {"color": "red"}
                })

        self.plot_json = json.dumps(data)
        self.plot_count += 1


# HTML page that polls /data and renders with plotly
CHART_HTML = """
<!DOCTYPE html>
<html><head>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>body { margin: 0; font-family: sans-serif; background: #1a1a2e; }</style>
</head><body>
<div id="chart" style="width:100vw;height:100vh;"></div>
<script>
const layout = {
    title: { text: 'Vol Smile', font: { color: '#eee' } },
    paper_bgcolor: '#1a1a2e',
    plot_bgcolor: '#16213e',
    xaxis: { title: 'Strike', color: '#eee', gridcolor: '#333' },
    yaxis: { title: 'Implied Vol', color: '#eee', gridcolor: '#333' },
    legend: { font: { color: '#eee' } },
    shapes: []
};

Plotly.newPlot('chart', [], layout);

async function update() {
    try {
        const resp = await fetch('/data');
        const data = await resp.json();
        if (data.traces) {
            layout.shapes = data.underlying > 0 ? [{
                type: 'line', x0: data.underlying, x1: data.underlying,
                y0: 0, y1: 1, yref: 'paper',
                line: { color: 'green', width: 2, dash: 'dash' }
            }] : [];
            layout.title.text = 'Vol Smile | UND: ' + (data.underlying || 0).toFixed(2);
            Plotly.react('chart', data.traces, layout);
        }
    } catch(e) {}
}

setInterval(update, 2000);
update();
</script>
</body></html>
"""