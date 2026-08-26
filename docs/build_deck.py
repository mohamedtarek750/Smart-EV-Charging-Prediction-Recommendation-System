"""Build the 5-slide deck HTML from the project's real numbers."""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
d1 = json.loads((HERE / "deck_data1.json").read_text())
d2 = json.loads((HERE / "deck_data2.json").read_text())

# --------------------------------------------------------------- palette ---
BG = "#080D16"
PANEL = "#0F1826"
LINE = "#1E2C42"
TEXT = "#E9F0FA"
MUTED = "#8A9BB2"
GREEN = "#34D399"
CYAN = "#38BDF8"
AMBER = "#FBBF24"
RED = "#F87171"

HEAT_STOPS = [
    (0.00, (0x10, 0x1B, 0x2E)),
    (0.25, (0x1B, 0x5A, 0x86)),
    (0.50, (0x2E, 0xA8, 0x8C)),
    (0.75, (0xE0, 0xB0, 0x3C)),
    (1.00, (0xE2, 0x55, 0x4A)),
]


def heat_colour(t: float) -> str:
    t = max(0.0, min(1.0, t))
    for i in range(len(HEAT_STOPS) - 1):
        a, ca = HEAT_STOPS[i]
        b, cb = HEAT_STOPS[i + 1]
        if a <= t <= b:
            f = (t - a) / (b - a) if b > a else 0.0
            r = round(ca[0] + (cb[0] - ca[0]) * f)
            g = round(ca[1] + (cb[1] - ca[1]) * f)
            bl = round(ca[2] + (cb[2] - ca[2]) * f)
            return f"#{r:02x}{g:02x}{bl:02x}"
    return "#000000"


# ------------------------------------------------- SVG: utilisation heatmap -
def heatmap_svg() -> str:
    h = d1["heatmap"]
    names, vals, vmax = h["stations"], h["values"], h["vmax"]
    cw, ch, gap = 29, 18, 3
    lab_w = 222
    top = 26
    width = lab_w + 24 * (cw + gap)
    height = top + len(names) * (ch + gap) + 34

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
             f'xmlns="http://www.w3.org/2000/svg" role="img" '
             f'aria-label="Mean charger utilisation by station and hour of day">']
    for hh in range(0, 24, 3):
        x = lab_w + hh * (cw + gap) + cw / 2
        parts.append(f'<text x="{x:.0f}" y="14" fill="{MUTED}" font-size="12" '
                     f'font-family="ubuntu-mono, monospace" text-anchor="middle">{hh:02d}</text>')
    for r, name in enumerate(names):
        y = top + r * (ch + gap)
        short = name if len(name) <= 26 else name[:25] + "…"
        parts.append(f'<text x="{lab_w - 12}" y="{y + ch - 5}" fill="{MUTED}" font-size="12" '
                     f'font-family="roboto, sans-serif" text-anchor="end">{short}</text>')
        for c in range(24):
            v = vals[r][c] / vmax
            parts.append(f'<rect x="{lab_w + c * (cw + gap)}" y="{y}" width="{cw}" height="{ch}" '
                         f'rx="2" fill="{heat_colour(v)}"/>')
    # legend
    ly = height - 20
    parts.append(f'<text x="{lab_w}" y="{ly + 12}" fill="{MUTED}" font-size="13" '
                 f'font-family="roboto, sans-serif">idle</text>')
    for i in range(34):
        parts.append(f'<rect x="{lab_w + 42 + i * 9}" y="{ly}" width="9" height="12" '
                     f'fill="{heat_colour(i / 33)}"/>')
    parts.append(f'<text x="{lab_w + 42 + 34 * 9 + 10}" y="{ly + 12}" fill="{MUTED}" '
                 f'font-size="13" font-family="roboto, sans-serif">saturated '
                 f'({vmax * 100:.0f}% of chargers busy)</text>')
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------ SVG: load-profile compare -
def profiles_svg() -> str:
    p = d2["profiles"]
    cap = d2["cap_kw"]
    series = [("uncontrolled", p["uncontrolled"], RED),
              ("greedy", p["greedy"], AMBER),
              ("optimised", p["optimised"], GREEN)]
    W, H = 980, 400
    ml, mr, mt, mb = 74, 20, 26, 46
    pw, ph = W - ml - mr, H - mt - mb
    ymax = 2800.0

    def X(i): return ml + i * pw / 23
    def Y(v): return mt + ph - (v / ymax) * ph

    parts = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
             f'xmlns="http://www.w3.org/2000/svg" role="img" '
             f'aria-label="Aggregate charging load across 24 hours under three strategies">']
    for gv in range(0, 2801, 700):
        parts.append(f'<line x1="{ml}" y1="{Y(gv):.1f}" x2="{W - mr}" y2="{Y(gv):.1f}" '
                     f'stroke="{LINE}" stroke-width="1"/>')
        parts.append(f'<text x="{ml - 12}" y="{Y(gv) + 5:.1f}" fill="{MUTED}" font-size="14" '
                     f'font-family="ubuntu-mono, monospace" text-anchor="end">{gv:,}</text>')
    for hh in range(0, 24, 3):
        parts.append(f'<text x="{X(hh):.0f}" y="{H - mb + 26}" fill="{MUTED}" font-size="14" '
                     f'font-family="ubuntu-mono, monospace" text-anchor="middle">{hh:02d}</text>')

    parts.append(f'<line x1="{ml}" y1="{Y(cap):.1f}" x2="{W - mr}" y2="{Y(cap):.1f}" '
                 f'stroke="{CYAN}" stroke-width="2" stroke-dasharray="8 6"/>')
    parts.append(f'<text x="{W - mr}" y="{Y(cap) - 10:.1f}" fill="{CYAN}" font-size="14" '
                 f'font-family="roboto, sans-serif" font-weight="500" text-anchor="end">'
                 f'grid ceiling {cap:,.0f} kW</text>')

    for label, vals, colour in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                     f'stroke-width="3.4" stroke-linejoin="round" stroke-linecap="round"/>')

    lx = ml + 14
    for label, vals, colour in series:
        parts.append(f'<rect x="{lx}" y="{mt + 6}" width="13" height="13" rx="3" fill="{colour}"/>')
        parts.append(f'<text x="{lx + 20}" y="{mt + 17}" fill="{TEXT}" font-size="15" '
                     f'font-family="roboto, sans-serif">{label} · peak {max(vals):,} kW</text>')
        lx += 22 + len(label) * 9 + 150
    parts.append(f'<text x="{ml - 12}" y="{mt - 8}" fill="{MUTED}" font-size="13" '
                 f'font-family="roboto, sans-serif" text-anchor="end">kW</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- content ---
FONT_CSS = '<link rel="stylesheet" href="https://use.typekit.net/bnj0vsp.css">'

CSS = f"""
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #05080E; font-family: "roboto", sans-serif; }}

.slide {{
  position: relative; width: 1920px; height: 1080px; overflow: hidden;
  background: {BG}; color: {TEXT}; padding: 74px 96px 66px;
  page-break-after: always; break-after: page;
}}
.slide:last-child {{ page-break-after: auto; break-after: auto; }}

.eyebrow {{
  font-family: "ubuntu-mono", monospace; font-size: 21px; letter-spacing: .22em;
  text-transform: uppercase; color: {GREEN}; font-weight: 700;
}}
.h1 {{ font-family: "roboto", sans-serif; font-weight: 900; font-size: 116px;
      line-height: .96; letter-spacing: -.028em; color: {TEXT}; }}
.h2 {{ font-family: "roboto", sans-serif; font-weight: 900; font-size: 54px;
      line-height: 1.04; letter-spacing: -.022em; color: {TEXT}; }}
.lede {{ font-size: 30px; line-height: 1.5; color: {MUTED}; font-weight: 400; }}
.body {{ font-size: 24px; line-height: 1.62; color: #C2CFE0; font-weight: 400; }}
.label {{ font-family: "ubuntu-mono", monospace; font-size: 17px; letter-spacing: .16em;
         text-transform: uppercase; color: {MUTED}; font-weight: 700; }}
.num {{ font-family: "ubuntu-mono", monospace; font-weight: 700; }}

.slide-no {{
  position: absolute; right: 96px; bottom: 46px; font-family: "ubuntu-mono", monospace;
  font-size: 19px; color: #5A6B82; font-weight: 700; letter-spacing: .1em;
}}
.foot {{
  position: absolute; left: 96px; bottom: 46px; font-family: "ubuntu-mono", monospace;
  font-size: 18px; color: #5A6B82; letter-spacing: .04em;
}}
.rule {{ height: 4px; width: 116px; background: {GREEN}; border-radius: 2px; }}

.panel {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 16px; }}
.stat-v {{ font-family: "ubuntu-mono", monospace; font-weight: 700; font-size: 52px;
          line-height: 1; color: {TEXT}; letter-spacing: -.02em; }}
.stat-l {{ font-size: 19px; color: {MUTED}; margin-top: 12px; line-height: 1.35; }}

table.data {{ border-collapse: collapse; width: 100%; }}
table.data th {{
  font-family: "ubuntu-mono", monospace; font-size: 17px; letter-spacing: .1em;
  text-transform: uppercase; color: {MUTED}; font-weight: 700; text-align: left;
  padding: 0 20px 14px 0; border-bottom: 1px solid {LINE};
}}
table.data td {{ font-size: 23px; color: #C8D5E6; padding: 12px 20px 12px 0;
                border-bottom: 1px solid #16223A; }}
table.data td.mono {{ font-family: "ubuntu-mono", monospace; font-weight: 700; color: {TEXT}; }}
table.data tr:last-child td {{ border-bottom: none; }}

.dot {{ display: inline-block; width: 13px; height: 13px; border-radius: 50%;
       margin-right: 11px; vertical-align: 1px; }}
ul.ticks {{ list-style: none; }}
ul.ticks li {{ font-size: 23px; line-height: 1.48; color: #C2CFE0; padding-left: 34px;
              position: relative; margin-bottom: 14px; }}
ul.ticks li::before {{ content: ""; position: absolute; left: 0; top: 13px; width: 15px;
                      height: 3px; background: {GREEN}; border-radius: 2px; }}
ul.ticks li b {{ color: {TEXT}; font-weight: 700; }}
"""


def slide1() -> str:
    return f"""
<div class="slide" data-canvas-width="1920" data-canvas-height="1080">
  <div style="position:absolute;inset:0;background:
       radial-gradient(1100px 640px at 78% 8%, rgba(52,211,153,.13), transparent 62%),
       radial-gradient(900px 560px at 8% 96%, rgba(56,189,248,.10), transparent 60%);"></div>

  <div style="position:relative;">
    <div class="eyebrow">New Administrative Capital &nbsp;·&nbsp; Egypt</div>
    <div class="h1" style="margin-top:26px;">Smart EV Charging</div>
    <div class="h1" style="color:{GREEN};">Prediction &amp; Recommendation</div>
    <div class="rule" style="margin:38px 0 30px;"></div>
    <div class="lede" style="max-width:1180px;">
      Three machine-learning models forecast a 20-station charging network one hour ahead.
      A linear program then decides how to spread that load across the grid.
    </div>
  </div>

  <div style="position:absolute;left:96px;right:96px;bottom:170px;display:flex;gap:26px;">
    {"".join(f'''<div class="panel" style="flex:1;padding:32px 34px;">
      <div class="stat-v">{v}</div><div class="stat-l">{l}</div></div>'''
      for v, l in [("20", "charging stations across the city"),
                   ("168", "chargers · 11.1 MW installed"),
                   ("175,200", "station-hours of history"),
                   ("297,856", "charging sessions simulated")])}
  </div>

  <div style="position:absolute;left:96px;right:96px;bottom:104px;display:flex;gap:44px;">
    <div style="font-size:23px;color:{MUTED};">
      <span class="dot" style="background:{GREEN};"></span><b style="color:{TEXT};">Driver</b>
      &nbsp;— where do I charge, and how long will it really take?
    </div>
    <div style="font-size:23px;color:{MUTED};">
      <span class="dot" style="background:{CYAN};"></span><b style="color:{TEXT};">City operator</b>
      &nbsp;— what will the network and the grid do tonight?
    </div>
  </div>

  <div class="foot">github.com/mohamedtarek750/Smart-EV-Charging-Prediction-Recommendation-System</div>
  <div class="slide-no">01 / 05</div>
</div>"""


def slide2() -> str:
    return f"""
<div class="slide" data-canvas-width="1920" data-canvas-height="1080">
  <div class="eyebrow">01 &nbsp;·&nbsp; The data</div>
  <div class="h2" style="margin-top:20px;max-width:1500px;">
    No public dataset exists — so the city was simulated
  </div>
  <div class="rule" style="margin:28px 0 34px;"></div>

  <div style="display:flex;gap:56px;align-items:flex-start;">
    <div style="width:660px;flex:none;">
      <div class="body" style="margin-bottom:26px;">
        A <b style="color:{TEXT};">discrete-event simulator</b> — not random numbers.
        Every car is an event with an arrival time, a queue and a charging session.
      </div>
      <ul class="ticks">
        <li><b>Eight venue types</b>, each with its own hourly demand shape — government,
            business, residential, retail, transit, leisure, hotel, education</li>
        <li><b>Friday–Saturday weekend</b> and 17 Egyptian public holidays</li>
        <li><b>Charging physics</b> — pack size, state of charge, power taper above 80 %,
            thermal penalty in cold and heat</li>
        <li><b>M/M/c queue with abandonment</b> — 6.9 % of drivers give up and leave</li>
        <li><b>Weather, venue events, +45 % adoption growth</b> across the year</li>
      </ul>
      <div class="panel" style="margin-top:26px;padding:20px 26px;display:flex;gap:38px;">
        <div><div class="stat-v" style="font-size:38px;">11,082</div>
             <div class="stat-l">MWh delivered</div></div>
        <div><div class="stat-v" style="font-size:38px;">32.1 %</div>
             <div class="stat-l">mean utilisation</div></div>
        <div><div class="stat-v" style="font-size:38px;">3,439</div>
             <div class="stat-l">kW city peak load</div></div>
      </div>
    </div>

    <div style="flex:1;min-width:0;">
      <div class="label" style="margin-bottom:16px;">
        Mean charger utilisation — 20 stations × 24 hours
      </div>
      {heatmap_svg()}
      <div class="body" style="margin-top:22px;font-size:22px;color:{MUTED};">
        The structure the models must learn: a commuter double-peak, a retail-led weekend,
        and stations that saturate while their neighbours sit idle.
      </div>
    </div>
  </div>

  <div class="foot">src/data/simulate.py &nbsp;→&nbsp; data/raw/</div>
  <div class="slide-no">02 / 05</div>
</div>"""


def slide3() -> str:
    cards = [
        ("①", "arrivals", "How many EVs plug in next hour?", "0.746", "24 %", GREEN),
        ("②", "avg_wait_min", "How long will a driver queue?", "0.618", "41 %", AMBER),
        ("③", "energy_kwh", "How much grid energy is drawn?", "0.927", "58 %", CYAN),
    ]
    card_html = "".join(f"""
      <div class="panel" style="flex:1;padding:24px 30px 22px;border-top:4px solid {c};">
        <div style="display:flex;align-items:baseline;gap:14px;">
          <div style="font-size:30px;color:{c};font-weight:900;">{n}</div>
          <div class="num" style="font-size:27px;color:{TEXT};">{t}</div>
        </div>
        <div class="stat-l" style="margin:12px 0 20px;font-size:20px;height:46px;">{q}</div>
        <div style="display:flex;align-items:flex-end;gap:34px;">
          <div><div class="num" style="font-size:50px;color:{c};line-height:1;">{r2}</div>
               <div class="stat-l" style="font-size:17px;">R² held out</div></div>
          <div><div class="num" style="font-size:36px;color:{TEXT};line-height:1;">−{up}</div>
               <div class="stat-l" style="font-size:17px;">MAE vs baseline</div></div>
        </div>
      </div>""" for n, t, q, r2, up, c in cards)

    return f"""
<div class="slide" data-canvas-width="1920" data-canvas-height="1080">
  <div class="eyebrow">02 &nbsp;·&nbsp; The models</div>
  <div class="h2" style="margin-top:20px;">Three models, one hour ahead</div>
  <div class="rule" style="margin:28px 0 34px;"></div>

  <div style="display:flex;gap:26px;">{card_html}</div>

  <div style="display:flex;gap:52px;margin-top:30px;align-items:flex-start;">
    <div style="width:900px;flex:none;">
      <div class="label" style="margin-bottom:20px;">Validation — time split, last 2 months held out</div>
      <table class="data">
        <tr><th>target</th><th>MAE</th><th>RMSE</th><th>R²</th><th>baseline</th><th>improvement</th></tr>
        <tr><td class="mono">arrivals</td><td class="mono">0.980</td><td class="mono">1.505</td>
            <td class="mono">0.746</td><td>same hour last week</td>
            <td class="mono" style="color:{GREEN};">−24.0 %</td></tr>
        <tr><td class="mono">avg_wait_min</td><td class="mono">1.105</td><td class="mono">4.125</td>
            <td class="mono">0.618</td><td>historical profile</td>
            <td class="mono" style="color:{GREEN};">−41.3 %</td></tr>
        <tr><td class="mono">energy_kwh</td><td class="mono">14.52</td><td class="mono">23.48</td>
            <td class="mono">0.927</td><td>same hour last week</td>
            <td class="mono" style="color:{GREEN};">−57.7 %</td></tr>
      </table>
      <div class="body" style="margin-top:20px;font-size:21px;color:{MUTED};">
        A random split would leak the future through the lag features. Training stops at
        31 October; scoring runs on the 29,280 station-hours after it.
      </div>
      <div class="panel" style="margin-top:26px;padding:24px 32px;">
        <div style="display:flex;gap:56px;align-items:flex-end;">
          <div><div class="num" style="font-size:48px;color:{GREEN};line-height:1;">93.7 %</div>
               <div class="stat-l" style="font-size:19px;">traffic-light accuracy
               <span style="color:{GREEN};">●</span><span style="color:{AMBER};">●</span><span style="color:{RED};">●</span></div></div>
          <div><div class="num" style="font-size:48px;color:{TEXT};line-height:1;">93.0 %</div>
               <div class="stat-l" style="font-size:19px;">predictions within 5 minutes</div></div>
        </div>
      </div>
    </div>

    <div style="flex:1;">
      <div class="panel" style="padding:22px 30px;">
        <div class="label" style="margin-bottom:14px;">51 features, six families</div>
        <div class="body" style="font-size:20px;line-height:1.62;">
          calendar &amp; cyclical encodings · weather &amp; scheduled events · station capacity ·
          lags at 1/2/3/24/48/<b style="color:{TEXT};">168</b> h · rolling means over 3/24/168 h ·
          leak-free expanding (station × weekday × hour) profile · city-wide network state
        </div>
      </div>
      <div class="panel" style="padding:22px 30px;margin-top:16px;border-left:4px solid {AMBER};">
        <div class="label" style="margin-bottom:16px;color:{AMBER};">What the queue model learned</div>
        <div class="body" style="font-size:20px;line-height:1.56;">
          Its three strongest features are <span class="num" style="color:{TEXT};">arrivals</span>,
          <span class="num" style="color:{TEXT};">chargers per arrival</span> and
          <span class="num" style="color:{TEXT};">session length</span> — it rediscovered
          ρ = λ ⁄ (c·μ) from queueing theory on its own.
        </div>
      </div>
    </div>
  </div>

  <div class="foot">HistGradientBoosting · scikit-learn · cascade scored end-to-end (wait R² 0.506, energy R² 0.888)</div>
  <div class="slide-no">03 / 05</div>
</div>"""


def slide4() -> str:
    t = {r["strategy"]: r for r in d2["table"]}
    return f"""
<div class="slide" data-canvas-width="1920" data-canvas-height="1080">
  <div class="eyebrow">03 &nbsp;·&nbsp; The decision</div>
  <div class="h2" style="margin-top:20px;max-width:1560px;">
    Forecasting says what will happen.<br/>
    <span style="color:{GREEN};">Optimisation decides what to do about it.</span>
  </div>
  <div class="rule" style="margin:24px 0 26px;"></div>

  <div style="display:flex;gap:52px;align-items:flex-start;">
    <div style="width:700px;flex:none;">
      <div class="body" style="margin-bottom:26px;">
        Once you know 30 cars arrive between 17:00 and 19:00, powering them is
        <b style="color:{TEXT};">not a prediction problem</b> — it is constrained allocation.
        So it is solved as a linear program, not another regressor.
      </div>
      <div class="panel" style="padding:24px 30px;font-family:'ubuntu-mono',monospace;
           font-size:18px;line-height:1.78;color:#B9C9DC;">
        <span style="color:{GREEN};">minimise</span>&nbsp;&nbsp; Σ price[t]·x[v,t]
          <span style="color:{MUTED};">&nbsp; energy bill</span><br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; + w · P_peak
          <span style="color:{MUTED};">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; flatten the profile</span><br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; + 500 · Σ u[v]
          <span style="color:{MUTED};">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; unmet energy</span><br/><br/>
        <span style="color:{CYAN};">s.t.</span>&nbsp;&nbsp;&nbsp;&nbsp; Σ x[v,t] + u[v] = E_v
          <span style="color:{MUTED};">&nbsp; every car gets its charge</span><br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0 ≤ x[v,t] ≤ P_v
          <span style="color:{MUTED};">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; charger + vehicle limit</span><br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; x = 0 outside [arrive, leave]<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Σ_v x[v,t] ≤ cap[t]
          <span style="color:{MUTED};">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; grid ceiling</span>
      </div>
      <div class="body" style="margin-top:24px;font-size:21px;color:{MUTED};">
        Solved with HiGHS via <span class="num">scipy.optimize.linprog</span>.
        {d2["n_vehicles"]:,} vehicles · {d2["total_energy_kwh"]:,} kWh requested.
        The unmet-energy slack keeps it always feasible.
      </div>
    </div>

    <div style="flex:1;min-width:0;">
      <div class="label" style="margin-bottom:14px;">Aggregate city load over 24 hours</div>
      {profiles_svg()}
      <div style="display:flex;gap:22px;margin-top:22px;">
        <div class="panel" style="flex:1;padding:22px 26px;border-top:4px solid {GREEN};">
          <div class="num" style="font-size:52px;color:{GREEN};line-height:1;">−28.3 %</div>
          <div class="stat-l">peak demand<br/>2,568 → 1,840 kW</div>
        </div>
        <div class="panel" style="flex:1;padding:22px 26px;">
          <div class="num" style="font-size:52px;line-height:1;">92.7 %</div>
          <div class="stat-l">energy delivered<br/>on time — unchanged</div>
        </div>
        <div class="panel" style="flex:1;padding:22px 26px;">
          <div class="num" style="font-size:52px;line-height:1;">4 → 0</div>
          <div class="stat-l">hours breaching<br/>the grid ceiling</div>
        </div>
      </div>
    </div>
  </div>

  <div class="foot">src/optimization/scheduler.py · same cars, same energy, same deadlines</div>
  <div class="slide-no">04 / 05</div>
</div>"""


def slide5() -> str:
    rows = [
        ("Iconic Tower CBD", "1.64", "5.4", "1.5", "25.8", "32.7", GREEN, True),
        ("Financial District North", "1.03", "3.9", "1.4", "43.0", "48.3", GREEN, False),
        ("Government District Hub", "1.34", "4.7", "2.2", "43.0", "49.9", GREEN, False),
        ("Downtown Mall Level B1", "1.06", "4.0", "21.3", "43.0", "68.3", RED, False),
        ("Council of Ministers Garage", "0.74", "3.3", "0.0", "117.2", "120.5", GREEN, False),
    ]
    tr = "".join(f"""<tr style="{'background:rgba(52,211,153,.09);' if best else ''}">
        <td style="padding-left:{'14px' if best else '0'};">
          <span class="dot" style="background:{c};"></span>{n}</td>
        <td class="mono">{km}</td><td class="mono">{dr}</td>
        <td class="mono" style="color:{c};">{w}</td><td class="mono">{ch}</td>
        <td class="mono" style="font-size:26px;color:{TEXT};">{tot}</td></tr>"""
        for n, km, dr, w, ch, tot, c, best in rows)

    return f"""
<div class="slide" data-canvas-width="1920" data-canvas-height="1080">
  <div class="eyebrow">04 &nbsp;·&nbsp; The product</div>
  <div class="h2" style="margin-top:20px;">Two products from one forecasting core</div>
  <div class="rule" style="margin:28px 0 34px;"></div>

  <div style="display:flex;gap:34px;align-items:stretch;">
    <div class="panel" style="flex:1.55;padding:36px 38px;">
      <div style="display:flex;align-items:baseline;gap:16px;margin-bottom:8px;">
        <div class="label" style="color:{GREEN};">Driver app</div>
        <div style="font-size:21px;color:{MUTED};">near Downtown · 18:00 · battery 22 % → 85 %</div>
      </div>
      <div class="body" style="margin:16px 0 26px;">
        Ranked by <b style="color:{TEXT};">total time</b> = drive + queue + charge — never by distance.
      </div>
      <table class="data">
        <tr><th>station</th><th>km</th><th>drive</th><th>queue</th><th>charge</th><th>total min</th></tr>
        {tr}
      </table>
      <div class="body" style="margin-top:26px;font-size:22px;">
        The system <b style="color:{TEXT};">rejects the two nearest stations</b>: the mall is 1 km
        away but carries a 21-minute predicted queue, and the ministries garage is empty but
        its 22 kW chargers would take two hours.
      </div>
    </div>

    <div style="flex:1;display:flex;flex-direction:column;gap:22px;">
      <div class="panel" style="flex:1;padding:34px 36px;">
        <div class="label" style="color:{CYAN};margin-bottom:22px;">City operator dashboard</div>
        <ul class="ticks" style="font-size:22px;">
          <li style="margin-bottom:15px;"><b>24-hour outlook</b> — expected EVs, grid load,
              headroom against the ceiling</li>
          <li style="margin-bottom:15px;"><b>Congestion heatmap</b> — every station, every hour</li>
          <li style="margin-bottom:15px;"><b>Live network map</b> colour-coded by service level</li>
          <li style="margin-bottom:0;"><b>Overload alerts</b> before the peak arrives</li>
        </ul>
      </div>
      <div class="panel" style="padding:30px 36px;">
        <div class="label" style="margin-bottom:18px;">Built with</div>
        <div class="body" style="font-size:21px;line-height:1.7;">
          Python · pandas · scikit-learn · SciPy&nbsp;HiGHS · Streamlit · FastAPI
        </div>
        <div style="display:flex;gap:36px;margin-top:26px;">
          <div><div class="num" style="font-size:38px;color:{GREEN};line-height:1;">40</div>
               <div class="stat-l" style="font-size:17px;">invariant tests passing</div></div>
          <div><div class="num" style="font-size:38px;line-height:1;">7</div>
               <div class="stat-l" style="font-size:17px;">REST endpoints</div></div>
          <div><div class="num" style="font-size:38px;line-height:1;">4</div>
               <div class="stat-l" style="font-size:17px;">dashboard views</div></div>
        </div>
      </div>
    </div>
  </div>

  <div class="foot">github.com/mohamedtarek750/Smart-EV-Charging-Prediction-Recommendation-System</div>
  <div class="slide-no">05 / 05</div>
</div>"""


HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Smart EV Charging — Project Deck</title>
<meta name="hz:slide-selector" content=".slide"/>
<meta name="hz:canvas-width" content="1920"/>
<meta name="hz:canvas-height" content="1080"/>
{FONT_CSS}
<style>
@page {{ size: 20in 11.25in; margin: 0; }}
{CSS}
</style>
</head>
<body>
{slide1()}
{slide2()}
{slide3()}
{slide4()}
{slide5()}
</body>
</html>
"""

out = HERE / "deck.html"
out.write_text(HTML, encoding="utf-8")
print(f"wrote {out}  ({len(HTML):,} bytes)")
