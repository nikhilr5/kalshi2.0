"""Turn the raw forecast-error history into the model's sigma lookup + report.

Reads cache/sigma_history_errors.csv (built by build_sigma_table.py) and emits:
  cache/sigma_recommended.csv  - station x lead x season -> n, bias, sigma, rmse
  cache/sigma_model.json       - nested lookup for the live model to import
  SIGMA_REPORT.md              - human summary

Recommended model per lead: NBS (NBM short) for leads 0-2, NBE (NBM extended)
for leads 3-7 -- the same NBM family the live feed will use, and the tightest at
every lead in the cross-model comparison.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ERR = pd.read_csv(HERE / "cache" / "sigma_history_errors.csv")

# which model to trust at each lead (NBM family; tightest in the bake-off)
LEAD_MODEL = {0: "NBS", 1: "NBS", 2: "NBS", 3: "NBE", 4: "NBE", 5: "NBE", 6: "NBE", 7: "NBE"}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]


def stats(e):
    e = np.asarray(e, float)
    return dict(n=int(len(e)),
                bias=round(float(e.mean()), 2),
                sigma=round(float(e.std(ddof=1)), 2) if len(e) > 1 else None,
                rmse=round(float(np.sqrt((e ** 2).mean())), 2))


def main():
    # pick the recommended model's rows for each lead
    keep = [ERR[(ERR.model == m) & (ERR.lead_days == L)] for L, m in LEAD_MODEL.items()]
    rec = pd.concat(keep, ignore_index=True)

    rows, model_json = [], {"_meta": {
        "built_from": "IEM MOS archive (NBS/NBE) vs IEM ASOS daily max_temp_f",
        "window": "2024-06-19 .. 2026-06-19 (2y)",
        "lead_model": LEAD_MODEL,
        "truth_caveat": "ASOS daily max ~1F vs CLI settlement; station map unverified for terse series",
        "usage": "high ~ Normal(forecast - bias, sigma); look up by station,lead,season",
    }, "stations": {}}

    for st in sorted(rec.station.unique()):
        sj = {"by_lead": {}, "by_lead_season": {}}
        for L in sorted(rec.lead_days.unique()):
            g = rec[(rec.station == st) & (rec.lead_days == L)]
            if g.empty:
                continue
            s_all = stats(g.error)
            sj["by_lead"][int(L)] = {**s_all, "model": LEAD_MODEL[L]}
            rows.append(dict(station=st, lead=int(L), season="ALL", model=LEAD_MODEL[L], **s_all))
            seas = {}
            for sea in SEASON_ORDER:
                gs = g[g.season == sea]
                if len(gs) >= 30:
                    ss = stats(gs.error)
                    seas[sea] = ss
                    rows.append(dict(station=st, lead=int(L), season=sea,
                                     model=LEAD_MODEL[L], **ss))
            sj["by_lead_season"][int(L)] = seas
        model_json["stations"][st] = sj

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "cache" / "sigma_recommended.csv", index=False)
    (HERE / "cache" / "sigma_model.json").write_text(json.dumps(model_json, indent=2))
    write_report(out, model_json)
    print("wrote cache/sigma_recommended.csv, cache/sigma_model.json, SIGMA_REPORT.md")
    return out


def write_report(out, mj):
    allrows = out[out.season == "ALL"]
    piv_s = allrows.pivot(index="station", columns="lead", values="sigma")
    piv_b = allrows.pivot(index="station", columns="lead", values="bias")
    nbs2 = out[(out.season != "ALL") & (out.lead <= 2)]
    seas_s = (nbs2.groupby(["station", "season"]).sigma.mean().round(2)
              .reset_index().pivot(index="station", columns="season", values="sigma")
              .reindex(columns=SEASON_ORDER))
    L = []
    L.append("# Historical forecast-error σ table (2026-06-19)\n")
    L.append("Empirically-measured daily-high forecast error for the five modeled "
             "settlement stations: **σ** (distribution width) and **bias** "
             "(systematic offset, forecast − observed) by **station × lead × season**.\n")
    L.append("## Method\n")
    L.append("- **Forecast** — IEM MOS archive. Daily high = max over the local "
             "calendar day of the model's temp / max-min line. One run per day "
             "(NBS 13Z, NBE/MEX/GFS 12Z) → independent daily samples. Truncated "
             "horizon-edge days (no afternoon coverage) dropped.\n")
    L.append("- **Truth** — IEM ASOS daily `max_temp_f` at each station "
             "(METAR-derived; ~1°F vs the CLI value Kalshi settles on).\n")
    L.append("- **error = forecast_high − observed_high**, per local day. "
             "σ = std(error), bias = mean(error). Window 2024-06-19 → 2026-06-19 "
             "(2y); n ≈ 684/station/lead for NBS, 728 for NBE.\n")
    L.append("- **Recommended model per lead** — NBS (NBM short) leads 0–2, "
             "NBE (NBM extended) leads 3–7. NBM was tightest at every lead vs "
             "GFS-MOS; NBE and the independent GFS-extended (MEX) agreed within "
             "~0.2°F, so the σ curve is cross-validated.\n")
    L.append("\n## σ (°F) by station × lead — pooled seasons\n")
    L.append(piv_s.to_markdown())
    L.append("\n\n## bias (°F, forecast − observed) by station × lead\n")
    L.append(piv_b.to_markdown())
    L.append("\n\n## σ (°F) by station × season — trading leads (0–2)\n")
    L.append(seas_s.to_markdown())
    L.append("\n\n## Key findings\n")
    L.append("1. **σ grows ~linearly with lead**: ~2.5°F day-of → ~3.4°F at 2 days "
             "→ ~6–7°F at 7 days. Use the lead-matched σ, never a global one.\n")
    L.append("2. **Lead-0 cold bias ≈ −2°F** at every station: the same-day morning "
             "(13Z) run systematically under-forecasts the afternoon high. "
             "Subtract it — it's the single most actionable correction.\n")
    L.append("3. **Season dominates σ** (swings 2.0–4.5°F within one station): "
             "Dallas/OKC are *easiest in summer* (stable heat ridge, σ≈2.5) and "
             "hardest in winter (front timing, σ≈3.6–4.1); Boston worst in spring "
             "(σ4.5); LA worst in autumn (Santa Ana, σ3.8). Naive 'summer=hard' is "
             "wrong for the southern plains.\n")
    L.append("4. **NBM beats GFS-MOS** at every lead (e.g. lead-1 σ 2.98 vs 3.30) — "
             "confirms NBM as the right live feed.\n")
    L.append("5. **LA is most predictable far out** (σ 3.9→5.1 over leads 3–7) vs "
             "OKC/BOS/NYC blowing out to ~7°F — LA carries tradable edge to longer "
             "leads; continental stations decay fast.\n")
    L.append("\n## Caveats\n")
    L.append("- Truth is ASOS daily max, not the CLI settlement value (~1°F gap "
             "that flips near-the-money buckets). Re-fit on CLI once the recorder "
             "accrues settled highs.\n")
    L.append("- Station map unverified for terse series (esp. **DAL** KDAL vs KDFW, "
             "and LA KLAX vs downtown USC) — a wrong station injects a fixed offset "
             "that masquerades as bias.\n")
    L.append("- 2-year window: ~2 samples/season/year. Extend to 5–10y (IEM has "
             "MOS to 2003) for tighter seasonal σ.\n")
    L.append("- **Model handoff (lead 2→3, NBS→NBE)** can make σ non-monotonic — "
             "e.g. LA's NBE σ (~2.9) sits below NBS's lead-2 σ (3.45) because NBM-"
             "extended is genuinely tighter for LA than NBM-short's hourly max. "
             "Real, but don't read the splice as a smooth curve.\n")
    L.append("\n## Files\n")
    L.append("- `cache/sigma_recommended.csv` — station × lead × season → n, bias, σ, rmse\n")
    L.append("- `cache/sigma_model.json` — nested lookup for the live model to import\n")
    L.append("- `cache/sigma_table.csv` — full station × model × season × lead grid\n")
    L.append("- `cache/sigma_history_errors.csv` — every matched day-forecast (48,280 rows)\n")
    L.append("- `build_sigma_table.py` (pull+align), `finalize_sigma.py` (this)\n")
    (HERE / "SIGMA_REPORT.md").write_text("".join(L))


if __name__ == "__main__":
    main()
