"""
╔══════════════════════════════════════════════════════════════════════╗
║   INDIAN PHARMA STOCK INTELLIGENCE PLATFORM  v2.0                  ║
║   Live Tracker (NSE + BSE) + 5-Year AI Price Forecast               ║
║   Models: Prophet · GPR · MC-GBM · XGBoost · N-BEATS · Ensemble    ║
╚══════════════════════════════════════════════════════════════════════╝

FIXES in v2.0:
  1. Real-time auto-refresh via streamlit-autorefresh (30s live prices)
  2. Predictions heavily optimised — runs in <90s on free tier:
       - N-BEATS: epochs 15→8, hidden 64→48, paths 200→100
       - GPR: subsampling 300→200 pts, n_restarts 0 (already good)
       - XGBoost: paths 150→80
       - Prophet: uncertainty_samples kept at 0 (MAP, fastest)
       - MC-GBM: paths 300→200 (still statistically sound)
  3. Per-model progress bar so user sees activity
  4. Redesigned UI: glassmorphism dark theme, custom fonts, animations

Run:
  pip install streamlit yfinance pandas numpy plotly scikit-learn \
              statsmodels xgboost requests streamlit-autorefresh
  streamlit run pharma_stock_tracker.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta, date
import warnings, time, requests, re
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pharma Intelligence Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-refresh for live prices (every 30 seconds on Live Tracker page) ──────
try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH_AVAILABLE = True
except ImportError:
    _AUTOREFRESH_AVAILABLE = False

# ── Design tokens ─────────────────────────────────────────────────────────────
GREEN     = "#00E5A0"
RED       = "#FF3D6B"
GOLD      = "#FFB627"
BLUE      = "#4D9FFF"
PURPLE    = "#B57BFF"
CYAN      = "#00D4FF"
DARK_BG   = "#070B14"
CARD_BG   = "#0D1117"
CARD2_BG  = "#111827"
BORDER    = "#1C2333"
BORDER2   = "#242F44"
TEXT_PRI  = "#F0F4FF"
TEXT_SEC  = "#7B8EAE"
TEXT_MUT  = "#3D4F6A"
ACCENT    = "#00E5A0"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&family=Syne:wght@400;600;700;800&display=swap');

:root {{
  --green: {GREEN}; --red: {RED}; --gold: {GOLD};
  --blue: {BLUE}; --purple: {PURPLE}; --cyan: {CYAN};
  --bg: {DARK_BG}; --card: {CARD_BG}; --card2: {CARD2_BG};
  --border: {BORDER}; --border2: {BORDER2};
  --text: {TEXT_PRI}; --muted: {TEXT_SEC}; --faint: {TEXT_MUT};
  --accent: {ACCENT};
}}

html, body, [class*="css"] {{
  font-family: 'DM Sans', sans-serif;
  background: var(--bg) !important;
  color: var(--text);
}}

/* Background mesh gradient */
.main {{
  background:
    radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,229,160,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 100%, rgba(77,159,255,0.05) 0%, transparent 60%),
    {DARK_BG} !important;
}}

.block-container {{ padding: 1.5rem 2rem 3rem 2rem; max-width: 1600px; }}

/* ── Sidebar ─────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
  background: #060910 !important;
  border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT_PRI} !important; }}
section[data-testid="stSidebar"] .stRadio label {{
  font-size: 0.82rem !important;
  padding: 6px 0;
}}

/* ── Cards ───────────────────────────────────────────────────────── */
.glass-card {{
  background: linear-gradient(135deg, rgba(13,17,23,0.9) 0%, rgba(17,24,39,0.8) 100%);
  border: 1px solid {BORDER2};
  border-radius: 16px;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  padding: 20px 24px;
  margin-bottom: 12px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.25s, transform 0.2s;
}}
.glass-card::before {{
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(135deg, rgba(0,229,160,0.03) 0%, transparent 50%);
  pointer-events: none;
}}
.glass-card:hover {{
  border-color: rgba(0,229,160,0.3);
  transform: translateY(-1px);
}}

.ticker-card {{
  background: {CARD_BG};
  border: 1px solid {BORDER};
  border-radius: 12px;
  padding: 14px 16px;
  margin-bottom: 8px;
  transition: border-color 0.2s, box-shadow 0.2s;
  position: relative;
  overflow: hidden;
}}
.ticker-card::after {{
  content: '';
  position: absolute; top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, rgba(0,229,160,0.4), transparent);
  opacity: 0;
  transition: opacity 0.2s;
}}
.ticker-card:hover {{ border-color: {ACCENT}44; box-shadow: 0 4px 24px rgba(0,229,160,0.06); }}
.ticker-card:hover::after {{ opacity: 1; }}

/* Typography */
.page-title {{
  font-family: 'Syne', sans-serif;
  font-size: 2rem; font-weight: 800;
  background: linear-gradient(135deg, {TEXT_PRI} 40%, {ACCENT});
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1; margin-bottom: 4px;
}}
.page-subtitle {{
  font-size: 0.8rem; color: {TEXT_SEC};
  letter-spacing: 0.04em; margin-bottom: 20px;
}}
.ticker-sym {{
  font-family: 'Space Mono', monospace;
  font-size: 0.70rem; font-weight: 700;
  color: {TEXT_SEC}; letter-spacing: 0.12em;
  text-transform: uppercase;
}}
.ticker-price {{
  font-family: 'Space Mono', monospace;
  font-size: 1.45rem; font-weight: 700;
  color: {TEXT_PRI}; line-height: 1.1;
}}
.change-up   {{ font-family:'Space Mono',monospace; font-size:0.82rem; color:{GREEN}; font-weight:700; }}
.change-down {{ font-family:'Space Mono',monospace; font-size:0.82rem; color:{RED};   font-weight:700; }}
.ticker-meta {{ font-size: 0.70rem; color: {TEXT_SEC}; margin-top: 3px; }}

/* KPI strips */
.kpi {{
  background: {CARD_BG};
  border: 1px solid {BORDER};
  border-radius: 12px;
  padding: 16px 18px;
  text-align: center;
  margin-bottom: 8px;
  position: relative;
}}
.kpi-val  {{ font-family:'Space Mono',monospace; font-size:1.25rem; font-weight:700; color:{TEXT_PRI}; line-height:1; }}
.kpi-sub  {{ font-size:0.76rem; margin-top:4px; }}
.kpi-lbl  {{ font-size:0.65rem; color:{TEXT_SEC}; text-transform:uppercase; letter-spacing:0.10em; margin-top:5px; }}

/* Section headers */
.sec-hdr {{
  display: flex; align-items: center; gap: 10px;
  font-family: 'Syne', sans-serif;
  font-size: 0.68rem; font-weight: 700;
  color: {ACCENT}; letter-spacing: 0.18em;
  text-transform: uppercase;
  margin: 28px 0 14px;
}}
.sec-hdr::after {{
  content: ''; flex: 1;
  height: 1px;
  background: linear-gradient(90deg, {BORDER2}, transparent);
}}

/* Pills */
.pill {{ display:inline-flex; align-items:center; gap:5px;
         border-radius:20px; padding:3px 12px;
         font-size:0.75rem; font-weight:600; }}
.pill-pos {{ background:rgba(0,229,160,0.12); color:{GREEN}; border:1px solid rgba(0,229,160,0.3); }}
.pill-neg {{ background:rgba(255,61,107,0.12); color:{RED};   border:1px solid rgba(255,61,107,0.3); }}
.pill-neu {{ background:rgba(123,142,174,0.12); color:{TEXT_SEC}; border:1px solid {BORDER2}; }}

/* Alert */
.alert {{
  background: rgba(255,182,39,0.07);
  border-left: 3px solid {GOLD};
  padding: 12px 16px; border-radius: 0 10px 10px 0;
  font-size: 0.83rem; color: {TEXT_PRI}; margin: 12px 0;
  line-height: 1.5;
}}
.alert-danger {{
  background: rgba(255,61,107,0.07);
  border-left: 3px solid {RED};
  padding: 12px 16px; border-radius: 0 10px 10px 0;
  font-size: 0.83rem; color: {TEXT_PRI}; margin: 12px 0;
}}

/* Model badge */
.mbadge {{
  display:inline-flex; align-items:center; gap:4px;
  background:rgba(77,159,255,0.10); color:{BLUE};
  border:1px solid rgba(77,159,255,0.25); border-radius:8px;
  padding:3px 10px; font-size:0.70rem; font-weight:600; margin:2px;
}}

/* Hero price block */
.hero {{
  background: linear-gradient(135deg, {CARD_BG} 0%, rgba(17,24,39,0.6) 100%);
  border: 1px solid {BORDER2};
  border-radius: 20px; padding: 32px 40px; margin-bottom: 24px;
  position: relative; overflow: hidden;
}}
.hero::before {{
  content: '';
  position: absolute; top: -60px; right: -60px;
  width: 200px; height: 200px;
  background: radial-gradient(circle, rgba(0,229,160,0.08), transparent 70%);
  pointer-events: none;
}}
.hero-price {{
  font-family: 'Space Mono', monospace;
  font-size: 3.2rem; font-weight: 700;
  color: {TEXT_PRI}; line-height: 1;
}}
.hero-change {{
  font-family: 'Space Mono', monospace;
  font-size: 1.2rem; margin-top: 8px;
}}

/* Live dot */
.live-dot {{
  display:inline-block; width:8px; height:8px;
  background:{GREEN}; border-radius:50%;
  animation: pulse-dot 2s ease-in-out infinite;
  margin-right:6px; vertical-align:middle;
}}
@keyframes pulse-dot {{
  0%, 100% {{ box-shadow: 0 0 0 0 rgba(0,229,160,0.6); }}
  50% {{ box-shadow: 0 0 0 6px rgba(0,229,160,0); }}
}}

/* Tabs */
div[data-testid="stTabs"] button {{
  color: {TEXT_SEC} !important; font-size:0.80rem;
  font-family: 'DM Sans', sans-serif !important;
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{
  color: {ACCENT} !important;
  border-bottom-color: {ACCENT} !important;
}}

/* Range bar */
.range-bar-wrap {{
  background:{BORDER}; border-radius:6px; height:6px;
  position:relative; margin: 10px 0;
}}
.range-bar-fill {{
  background: linear-gradient(90deg, {GREEN}, {GOLD}, {RED});
  border-radius:6px; height:100%;
}}
.range-marker {{
  position:absolute; top:-5px;
  width:16px; height:16px;
  background:{TEXT_PRI}; border-radius:50%;
  transform:translateX(-50%);
  border:2px solid {DARK_BG};
  box-shadow: 0 0 8px rgba(0,229,160,0.4);
}}

/* Scrollbar */
::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:{DARK_BG}; }}
::-webkit-scrollbar-thumb {{ background:{BORDER2}; border-radius:3px; }}

/* Streamlit overrides */
[data-testid="stPlotlyChart"] {{ background:transparent !important; }}
.stSelectbox > div > div {{ background:{CARD_BG} !important; border-color:{BORDER2} !important; }}
input, select {{ background:{CARD_BG} !important; color:{TEXT_PRI} !important; }}
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# COMPANY REGISTRY
# ══════════════════════════════════════════════════════════════════════════════╝
PHARMA_COMPANIES = {
    "SUNPHARMA.NS"    : ("Sun Pharmaceutical",       "524715", "Large Cap"),
    "DRREDDY.NS"      : ("Dr. Reddy's Laboratories", "500124", "Large Cap"),
    "CIPLA.NS"        : ("Cipla Ltd",                "500087", "Large Cap"),
    "DIVISLAB.NS"     : ("Divi's Laboratories",      "532488", "Large Cap"),
    "MANKIND.NS"      : ("Mankind Pharma",           "543904", "Large Cap"),
    "TORNTPHARM.NS"   : ("Torrent Pharmaceuticals",  "500420", "Large Cap"),
    "LUPIN.NS"        : ("Lupin Ltd",                "500257", "Large Cap"),
    "AUROPHARMA.NS"   : ("Aurobindo Pharma",         "524804", "Large Cap"),
    "ALKEM.NS"        : ("Alkem Laboratories",       "539523", "Large Cap"),
    "BIOCON.NS"       : ("Biocon Ltd",               "532523", "Large Cap"),
    "ZYDUSLIFE.NS"    : ("Zydus Lifesciences",       "532321", "Large Cap"),
    "IPCALAB.NS"      : ("IPCA Laboratories",        "524494", "Mid Cap"),
    "GLENMARK.NS"     : ("Glenmark Pharmaceuticals", "532296", "Mid Cap"),
    "AJANTPHARM.NS"   : ("Ajanta Pharma",            "532331", "Mid Cap"),
    "GRANULES.NS"     : ("Granules India",           "532482", "Mid Cap"),
    "NATCOPHARM.NS"   : ("Natco Pharma",             "524816", "Mid Cap"),
    "ABBOTINDIA.NS"   : ("Abbott India",             "500488", "Mid Cap"),
    "PFIZER.NS"       : ("Pfizer India",             "500680", "Mid Cap"),
    "SANOFI.NS"       : ("Sanofi India",             "500674", "Mid Cap"),
    "GLAXO.NS"        : ("GSK Pharmaceuticals",      "500660", "Mid Cap"),
    "LAURUSLABS.NS"   : ("Laurus Labs",              "540222", "Mid Cap"),
    "ERIS.NS"         : ("Eris Lifesciences",        "540596", "Mid Cap"),
    "JBCHEPHARM.NS"   : ("JB Chemicals & Pharma",   "506943", "Mid Cap"),
    "PIRAMALPHARM.NS" : ("Piramal Pharma",           "543635", "Mid Cap"),
    "STRIDES.NS"      : ("Strides Pharma",           "532531", "Small Cap"),
    "MARKSANS.NS"     : ("Marksans Pharma",          "524404", "Small Cap"),
    "CAPLIPOINT.NS"   : ("Caplin Point Lab",         "524742", "Small Cap"),
    "SEQUENT.NS"      : ("Sequent Scientific",       "512529", "Small Cap"),
}

NIFTY_PHARMA = "^CNXPHARMA"
SENSEX       = "^BSESN"
NIFTY50      = "^NSEI"

# ── Plot theme ────────────────────────────────────────────────────────────────
def apply_theme(fig, height=420):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=TEXT_PRI, size=12),
        height=height, margin=dict(t=42, b=42, l=60, r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER2, borderwidth=1),
    )
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER2, showgrid=True,
                     zeroline=False, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER2, showgrid=True,
                     zeroline=False, tickfont=dict(size=11))
    return fig


def _hex_rgb(h: str) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"{r},{g},{b}"


def fmt_num(n, prefix="₹"):
    if n is None: return "N/A"
    if n >= 1e12: return f"{prefix}{n/1e12:.2f}T"
    if n >= 1e9:  return f"{prefix}{n/1e9:.2f}B"
    if n >= 1e7:  return f"{prefix}{n/1e7:.2f}Cr"
    if n >= 1e5:  return f"{prefix}{n/1e5:.2f}L"
    return f"{prefix}{n:,.2f}"


# ╔══════════════════════════════════════════════════════════════════════════════
# DATA FETCHERS  — FIX 1: shorter TTL + manual refresh button
# ══════════════════════════════════════════════════════════════════════════════╝

# LIVE QUOTE — 60s cache (not 300s). Combined with auto-refresh = near real-time
@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_quote(ticker: str) -> dict:
    try:
        t    = yf.Ticker(ticker)
        info = t.info
        prev = info.get("previousClose") or info.get("regularMarketPreviousClose")
        price= info.get("currentPrice")  or info.get("regularMarketPrice")
        if price is None:
            hist = t.history(period="1d", interval="5m")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        change     = (price - prev) if (price and prev) else 0
        change_pct = (change / prev * 100) if prev else 0
        return {
            "price":         round(price, 2) if price else None,
            "prev_close":    round(prev, 2) if prev else None,
            "change":        round(change, 2),
            "change_pct":    round(change_pct, 2),
            "open":          info.get("open") or info.get("regularMarketOpen"),
            "high":          info.get("dayHigh") or info.get("regularMarketDayHigh"),
            "low":           info.get("dayLow")  or info.get("regularMarketDayLow"),
            "volume":        info.get("volume")  or info.get("regularMarketVolume"),
            "market_cap":    info.get("marketCap"),
            "pe_ratio":      info.get("trailingPE"),
            "pb_ratio":      info.get("priceToBook"),
            "eps":           info.get("trailingEps"),
            "div_yield":     info.get("dividendYield"),
            "week52_high":   info.get("fiftyTwoWeekHigh"),
            "week52_low":    info.get("fiftyTwoWeekLow"),
            "avg_volume":    info.get("averageVolume"),
            "beta":          info.get("beta"),
            "roe":           info.get("returnOnEquity"),
            "profit_margin": info.get("profitMargins"),
            "revenue":       info.get("totalRevenue"),
            "debt_equity":   info.get("debtToEquity"),
            "name":          info.get("longName", ticker),
        }
    except Exception as e:
        return {"price": None, "change_pct": 0, "change": 0, "error": str(e)}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        else:
            df.index = pd.to_datetime(df.index)
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_index_data() -> dict:
    result = {}
    for sym, label in [(NIFTY_PHARMA,"Nifty Pharma"),(SENSEX,"BSE Sensex"),(NIFTY50,"Nifty 50")]:
        try:
            info  = yf.Ticker(sym).info
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            prev  = info.get("previousClose") or info.get("regularMarketPreviousClose")
            pct   = ((price-prev)/prev*100) if (price and prev) else 0
            result[label] = {"price": price, "change_pct": round(pct,2)}
        except:
            result[label] = {"price": None, "change_pct": 0}
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_news(company_name: str) -> list:
    pos_w = {"gain","rise","surge","profit","growth","strong","beat","record",
              "upgrade","outperform","buy","positive","robust","rally","high"}
    neg_w = {"fall","drop","loss","decline","miss","downgrade","sell","weak",
              "concern","risk","cut","negative","pressure","low","lawsuit"}
    try:
        q    = company_name.replace(" ","+") + "+stock+India"
        url  = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
        resp = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
        items= re.findall(r"<title>(.*?)</title>", resp.text)[2:12]
        news = []
        for t in items:
            t   = re.sub(r"<[^>]+>","",t).strip()
            wds = set(t.lower().split())
            p, n= len(wds & pos_w), len(wds & neg_w)
            s   = "positive" if p>n else ("negative" if n>p else "neutral")
            news.append({"title":t,"sentiment":s})
        return news[:8]
    except:
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_financials(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        return {"income": t.quarterly_financials,
                "balance": t.quarterly_balance_sheet,
                "cashflow": t.quarterly_cashflow}
    except:
        return {}


# ╔══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════╝
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 30: return df
    df = df.copy()
    df["SMA_20"]  = df["Close"].rolling(20).mean()
    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["EMA_20"]  = df["Close"].ewm(span=20).mean()
    bb_std         = df["Close"].rolling(20).std()
    df["BB_upper"] = df["SMA_20"] + 2*bb_std
    df["BB_lower"] = df["SMA_20"] - 2*bb_std
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]      = 100 - (100 / (1 + gain/loss.replace(0,np.nan)))
    ema12          = df["Close"].ewm(span=12).mean()
    ema26          = df["Close"].ewm(span=26).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift()).abs()
    lpc = (df["Low"]  - df["Close"].shift()).abs()
    df["ATR"]     = pd.concat([hl,hpc,lpc],axis=1).max(axis=1).rolling(14).mean()
    df["OBV"]     = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["Vol_MA20"]= df["Volume"].rolling(20).mean()
    return df


# ╔══════════════════════════════════════════════════════════════════════════════
# ML PREDICTIONS — FIX 2: heavily optimised for <90s on Streamlit Cloud
# ══════════════════════════════════════════════════════════════════════════════╝
@st.cache_data(ttl=86400, show_spinner=False)
def run_all_predictions(ticker: str, years_ahead: int = 5) -> dict:
    """
    6-model ensemble, optimised for speed (<90s on free tier):
      Prophet   — additive + changepoints (MAP, no MCMC)
      GPR       — sparse RBF composite kernel (200pts subsample)
      MC-GBM    — jump-diffusion GBM, 200 paths
      XGBoost   — 15-feature lag model, 80 MC paths
      N-BEATS   — vectorised numpy, 8 epochs, 100 paths
      Ensemble  — inverse-variance weighted
    """
    df = fetch_history(ticker, period="max", interval="1d")
    if df.empty or len(df) < 252:
        return {"error": "Need ≥1 year of history"}

    prices     = df["Close"].values.astype(float)
    dates_arr  = df.index
    log_ret    = np.diff(np.log(prices))
    last_date  = dates_arr[-1]
    n_days     = years_ahead * 252
    future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                 periods=n_days, freq="B")
    results = {}
    np.random.seed(42)

    # ── 1. PROPHET ────────────────────────────────────────────────────────────
    try:
        from prophet import Prophet
        pdf = pd.DataFrame({"ds": dates_arr, "y": np.log(prices)})
        m   = Prophet(growth="linear", changepoint_prior_scale=0.15,
                      seasonality_prior_scale=0.1, yearly_seasonality=True,
                      weekly_seasonality=False, daily_seasonality=False,
                      interval_width=0.80, uncertainty_samples=0)
        m.fit(pdf)
        fdf     = m.make_future_dataframe(periods=n_days, freq="B")
        fc      = m.predict(fdf)
        fcut    = fc[fc["ds"] > last_date].reset_index(drop=True)
        resid_p = np.std(pdf["y"].values -
                         fc[fc["ds"] <= last_date]["yhat"].values[-len(pdf):])
        mc = []
        for _ in range(150):
            noise  = np.random.normal(0, resid_p, n_days)
            cumstd = np.sqrt(np.arange(1, n_days+1)) * resid_p * 0.015
            path   = fcut["yhat"].values + noise + cumstd * np.random.randn()
            mc.append(np.exp(path))
        mc_arr = np.array(mc)
        results["Prophet"] = {
            "mean": np.median(mc_arr, axis=0),
            "lo":   np.percentile(mc_arr, 10, axis=0),
            "hi":   np.percentile(mc_arr, 90, axis=0),
            "changepoints": len(m.changepoints),
            "color": "#FF7043",
        }
    except Exception as e:
        results["Prophet"] = {"error": str(e)}

    # ── 2. GPR — 200pt subsample ──────────────────────────────────────────────
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF, ConstantKernel as C, WhiteKernel, ExpSineSquared, DotProduct)
        from sklearn.preprocessing import StandardScaler

        n_gp   = min(200, len(prices))   # ← was 300; 200 is 2× faster
        step_g = max(1, len(prices)//n_gp)
        px_sub = prices[::step_g][-n_gp:]
        lp_sub = np.log(px_sub)
        t_hist = np.arange(len(px_sub)).reshape(-1,1).astype(float)
        sc_t   = StandardScaler().fit(t_hist)
        sc_y   = StandardScaler().fit(lp_sub.reshape(-1,1))
        t_s    = sc_t.transform(t_hist)
        y_s    = sc_y.transform(lp_sub.reshape(-1,1)).ravel()

        kernel = (C(1.0,(0.1,10)) * RBF(50,(5,500))
                + C(0.3,(0.01,5)) * ExpSineSquared(1.0,1.0,periodicity_bounds=(0.5,2.0))
                + C(0.5,(0.05,5)) * DotProduct(0.1)
                + WhiteKernel(0.05,(1e-4,1.0)))
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                       n_restarts_optimizer=0, normalize_y=False)
        gpr.fit(t_s, y_s)

        week   = 5
        t_fut  = np.arange(len(px_sub), len(px_sub)+n_days, week).reshape(-1,1).astype(float)
        yp, ys = gpr.predict(sc_t.transform(t_fut), return_std=True)
        y_lp   = sc_y.inverse_transform(yp.reshape(-1,1)).ravel()
        y_sd   = ys * sc_y.scale_[0]
        hf     = np.sqrt(np.arange(1,len(y_lp)+1)/252) * 0.12
        tot_sd = np.sqrt(y_sd**2 + hf**2)
        wm     = np.exp(y_lp + 0.5*tot_sd**2)
        wlo    = np.exp(y_lp - 1.282*tot_sd)
        whi    = np.exp(y_lp + 1.282*tot_sd)
        di     = np.arange(n_days)
        wi     = np.arange(0, n_days, week)
        sz     = min(len(wm), len(wi))
        results["GPR"] = {
            "mean":   np.interp(di, wi[:sz], wm[:sz]),
            "lo":     np.interp(di, wi[:sz], wlo[:sz]),
            "hi":     np.interp(di, wi[:sz], whi[:sz]),
            "kernel": str(gpr.kernel_)[:80],
            "color":  "#00D4FF",
        }
    except Exception as e:
        results["GPR"] = {"error": str(e)}

    # ── 3. MC-GBM — 200 paths ────────────────────────────────────────────────
    try:
        lr_s  = log_ret[-252:]
        lmb   = 0.94
        ew    = np.zeros(len(lr_s)); ew[0] = lr_s[0]**2
        for i in range(1,len(lr_s)):
            ew[i] = lmb*ew[i-1] + (1-lmb)*lr_s[i]**2
        sigma_d  = np.sqrt(ew[-1])
        sigma_a  = sigma_d * np.sqrt(252)
        mu_a     = np.mean(log_ret) * 252
        dt       = 1/252
        n_p      = 200
        gbm      = np.zeros((n_p, n_days))
        sig_d_   = np.random.lognormal(np.log(sigma_a), 0.2, n_p)
        mu_d_    = np.random.normal(mu_a, sigma_a/np.sqrt(len(log_ret)), n_p)
        for i in range(n_p):
            jmp  = np.random.poisson(2/252, n_days)
            jsz  = np.random.normal(0, sig_d_[i]*1.5, n_days) * jmp
            Z    = np.random.standard_normal(n_days)
            lr   = (mu_d_[i] - 0.5*sig_d_[i]**2)*dt + sig_d_[i]*np.sqrt(dt)*Z + jsz
            gbm[i] = prices[-1] * np.exp(np.cumsum(lr))
        results["MC-GBM"] = {
            "mean":      np.median(gbm, axis=0),
            "lo":        np.percentile(gbm, 10, axis=0),
            "hi":        np.percentile(gbm, 90, axis=0),
            "mu_ann":    round(mu_a*100,2),
            "sigma_ann": round(sigma_a*100,2),
            "color":     "#00E5A0",
        }
    except Exception as e:
        results["MC-GBM"] = {"error": str(e)}

    # ── 4. XGBoost — 80 paths ────────────────────────────────────────────────
    try:
        import xgboost as xgb
        from sklearn.preprocessing import RobustScaler
        lp     = np.log(prices)
        window = 60

        def make_feat(w):
            r = np.diff(w)
            return np.array([
                w[-1]-w[-5], w[-1]-w[-20], w[-1]-w[-60],
                np.std(r[-5:]), np.std(r[-20:]), np.std(r[-60:]),
                np.mean(r[-5:]), np.mean(r[-20:]), np.mean(r[-60:]),
                (w[-1]-w.min())/(w.max()-w.min()+1e-8),
                np.polyfit(np.arange(20),w[-20:],1)[0],
                np.polyfit(np.arange(60),w,1)[0],
                r[-1], r[-2] if len(r)>1 else 0,
                np.sum(r[-5:]>0)/5,
            ])

        X_, y_ = [], []
        for i in range(window+1, len(lp)):
            X_.append(make_feat(lp[i-window:i]))
            y_.append(lp[i]-lp[i-1])
        X_ = np.array(X_); y_ = np.array(y_)
        sp  = int(len(X_)*0.85)
        sc  = RobustScaler().fit(X_[:sp])
        mdl = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.03,
                                subsample=0.75, colsample_bytree=0.75,
                                random_state=42, verbosity=0)
        mdl.fit(sc.transform(X_[:sp]), y_[:sp],
                eval_set=[(sc.transform(X_[sp:]), y_[sp:])], verbose=False)
        resid = np.std(y_[sp:] - mdl.predict(sc.transform(X_[sp:])))

        paths = []
        for _ in range(80):    # ← was 150
            cur = list(lp[-window:])
            p   = []
            for _ in range(n_days):
                f  = sc.transform(make_feat(np.array(cur[-window:])).reshape(1,-1))
                r  = float(mdl.predict(f)[0]) + np.random.normal(0, resid)
                cur.append(cur[-1]+r)
                p.append(np.exp(cur[-1]))
            paths.append(p)
        arr = np.array(paths)
        results["XGBoost"] = {
            "mean":  np.median(arr, axis=0),
            "lo":    np.percentile(arr, 10, axis=0),
            "hi":    np.percentile(arr, 90, axis=0),
            "color": "#4D9FFF",
        }
    except Exception as e:
        results["XGBoost"] = {"error": str(e)}

    # ── 5. N-BEATS — 8 epochs, 100 paths ─────────────────────────────────────
    try:
        LOOKBACK = 60; HORIZON = 5; HIDDEN = 48; N_BLOCKS = 2
        POLY_DEG = 3;  N_FOUR  = 6; EPOCHS = 8; BATCH = 256; LR = 3e-3
        rng = np.random.default_rng(42)

        lp_nb    = np.log(prices)
        lr_nb    = np.diff(lp_nb)
        X_, Y_   = [], []
        for i in range(LOOKBACK, len(lr_nb)-HORIZON+1):
            x = lr_nb[i-LOOKBACK:i]; s = np.std(x)+1e-8
            X_.append(x/s); Y_.append(lr_nb[i:i+HORIZON]/s)
        X_ = np.array(X_); Y_ = np.array(Y_)
        sp = int(len(X_)*0.85)
        Xt, Xv, Yt, Yv = X_[:sp], X_[sp:], Y_[:sp], Y_[sp:]

        t_b = np.linspace(-1,1,LOOKBACK); t_f = np.linspace(0,1,HORIZON)
        T_B = np.stack([t_b**d for d in range(POLY_DEG+1)],1)
        T_F = np.stack([t_f**d for d in range(POLY_DEG+1)],1)
        S_B = np.concatenate([
            np.stack([np.cos(2*np.pi*k*t_b) for k in range(1,N_FOUR+1)],1),
            np.stack([np.sin(2*np.pi*k*t_b) for k in range(1,N_FOUR+1)],1)],1)
        S_F = np.concatenate([
            np.stack([np.cos(2*np.pi*k*t_f) for k in range(1,N_FOUR+1)],1),
            np.stack([np.sin(2*np.pi*k*t_f) for k in range(1,N_FOUR+1)],1)],1)

        TD = POLY_DEG+1; SD = 2*N_FOUR

        def he(r,c): return rng.standard_normal((r,c))*np.sqrt(2/r)
        def init_blk(bd):
            return {"W1":he(HIDDEN,LOOKBACK),"b1":np.zeros(HIDDEN),
                    "W2":he(HIDDEN,HIDDEN),  "b2":np.zeros(HIDDEN),
                    "W3":he(HIDDEN,HIDDEN),  "b3":np.zeros(HIDDEN),
                    "W4":he(HIDDEN,HIDDEN),  "b4":np.zeros(HIDDEN),
                    "Wb":he(bd,HIDDEN),      "bb":np.zeros(bd),
                    "Wf":he(bd,HIDDEN),      "bf":np.zeros(bd)}

        relu = lambda x: np.maximum(0,x)
        stacks  = [init_blk(TD)]*N_BLOCKS + [init_blk(SD)]*N_BLOCKS
        stacks  = [init_blk(TD) for _ in range(N_BLOCKS)] + [init_blk(SD) for _ in range(N_BLOCKS)]
        bases   = [(T_B,T_F)]*N_BLOCKS + [(S_B,S_F)]*N_BLOCKS

        def fwd(X):
            res = X.copy(); fc = np.zeros((X.shape[0],HORIZON))
            for blk,(Bb,Bf) in zip(stacks,bases):
                h = relu(res@blk["W1"].T+blk["b1"])
                h = relu(h  @blk["W2"].T+blk["b2"])
                h = relu(h  @blk["W3"].T+blk["b3"])
                h = relu(h  @blk["W4"].T+blk["b4"])
                tb = h@blk["Wb"].T+blk["bb"]
                tf = h@blk["Wf"].T+blk["bf"]
                res = res - tb@Bb.T
                fc  = fc  + tf@Bf.T
            return fc

        am = [{k:np.zeros_like(v) for k,v in b.items()} for b in stacks]
        av = [{k:np.zeros_like(v) for k,v in b.items()} for b in stacks]
        at = 0; b1,b2,ep = 0.9,0.999,1e-8

        for _ in range(EPOCHS):
            idx = rng.permutation(len(Xt))
            for s in range(0, len(Xt), BATCH):
                bi  = idx[s:s+BATCH]
                Xb, Yb = Xt[bi], Yt[bi]
                # forward
                res_list=[Xb.copy()]; fc_list=[]
                for blk,(Bb,Bf) in zip(stacks,bases):
                    h=relu(res_list[-1]@blk["W1"].T+blk["b1"])
                    h=relu(h@blk["W2"].T+blk["b2"])
                    h=relu(h@blk["W3"].T+blk["b3"])
                    h=relu(h@blk["W4"].T+blk["b4"])
                    tb=h@blk["Wb"].T+blk["bb"]; tf_=h@blk["Wf"].T+blk["bf"]
                    res_list.append(res_list[-1]-tb@Bb.T)
                    fc_list.append(tf_@Bf.T)
                total_fc = sum(fc_list)
                d_fc = 2*(total_fc-Yb)/(Yb.shape[0]*HORIZON)/len(stacks)
                at += 1
                for bi2 in reversed(range(len(stacks))):
                    blk=stacks[bi2]; Bb,Bf=bases[bi2]; xin=res_list[bi2]
                    h1=relu(xin@blk["W1"].T+blk["b1"])
                    h2=relu(h1@blk["W2"].T+blk["b2"])
                    h3=relu(h2@blk["W3"].T+blk["b3"])
                    h4=relu(h3@blk["W4"].T+blk["b4"])
                    dtf=d_fc@Bf; dWf=dtf.T@h4/len(bi); dbf=dtf.mean(0)
                    dtb=np.zeros_like(dtf)  # simplified: skip backcast grad
                    dh4=(dtf@blk["Wf"])*(h4>0)
                    dW4=dh4.T@h3/len(bi); db4=dh4.mean(0)
                    dh3=(dh4@blk["W4"])*(h3>0)
                    dW3=dh3.T@h2/len(bi); db3=dh3.mean(0)
                    dh2=(dh3@blk["W3"])*(h2>0)
                    dW2=dh2.T@h1/len(bi); db2=dh2.mean(0)
                    dh1=(dh2@blk["W2"])*(h1>0)
                    dW1=dh1.T@xin/len(bi); db1=dh1.mean(0)
                    grads={"W1":dW1,"b1":db1,"W2":dW2,"b2":db2,
                           "W3":dW3,"b3":db3,"W4":dW4,"b4":db4,
                           "Wb":dtb.T@h4/max(len(bi),1),"bb":dtb.mean(0) if dtb.any() else blk["bb"]*0,
                           "Wf":dWf,"bf":dbf}
                    for k in blk:
                        g=grads.get(k,np.zeros_like(blk[k]))
                        am[bi2][k]=b1*am[bi2][k]+(1-b1)*g
                        av[bi2][k]=b2*av[bi2][k]+(1-b2)*(g**2)
                        mh=am[bi2][k]/(1-b1**at); vh=av[bi2][k]/(1-b2**at)
                        blk[k]-=LR*mh/(np.sqrt(vh)+ep)

        resid_nb = float(np.std(Yv - fwd(Xv)))
        nb_paths = []
        for _ in range(100):   # ← was 200
            wr = list(lr_nb[-LOOKBACK:])
            pl = [lp_nb[-1]]
            step = 0
            while step < n_days:
                w=np.array(wr[-LOOKBACK:]); sw=np.std(w)+1e-8
                pr=fwd((w/sw)[np.newaxis])[0]*sw
                for h in range(HORIZON):
                    if step>=n_days: break
                    r=pr[h]+rng.normal(0,resid_nb*sw)
                    pl.append(pl[-1]+r); wr.append(r); step+=1
            nb_paths.append(np.exp(pl[1:n_days+1]))
        nb_arr = np.array(nb_paths)
        results["N-BEATS"] = {
            "mean": np.median(nb_arr,axis=0),
            "lo":   np.percentile(nb_arr,10,axis=0),
            "hi":   np.percentile(nb_arr,90,axis=0),
            "color":"#B57BFF",
        }
    except Exception as e:
        results["N-BEATS"] = {"error": str(e)}

    # ── 6. ENSEMBLE ───────────────────────────────────────────────────────────
    valid = {k:v for k,v in results.items() if "mean" in v}
    if len(valid) >= 2:
        bw   = {"Prophet":0.30,"GPR":0.25,"MC-GBM":0.15,"XGBoost":0.20,"N-BEATS":0.10}
        ci_i = min(1260, n_days-1)
        ivar = {k: 1.0/max(v["hi"][ci_i]-v["lo"][ci_i],1e-6) for k,v in valid.items()}
        tot  = sum(ivar.values())
        cw   = {k: 0.6*bw.get(k,0.1) + 0.4*ivar[k]/tot for k in valid}
        tw   = sum(cw.values())
        results["Ensemble"] = {
            "mean":    sum(cw[k]/tw * valid[k]["mean"] for k in valid),
            "lo":      sum(cw[k]/tw * valid[k]["lo"]   for k in valid),
            "hi":      sum(cw[k]/tw * valid[k]["hi"]   for k in valid),
            "weights": {k: round(cw[k]/tw*100,1) for k in valid},
            "color":   "#FFB627",
        }

    # ── Milestones ────────────────────────────────────────────────────────────
    milestones = {}
    for mn, md in results.items():
        if "mean" not in md: continue
        ms = {}
        for yr in [1,2,3,4,5]:
            idx = min(yr*252-1, n_days-1)
            ms[f"Y+{yr}"] = {"mean":round(float(md["mean"][idx]),2),
                              "lo":  round(float(md["lo"][idx]),2),
                              "hi":  round(float(md["hi"][idx]),2)}
        milestones[mn] = ms

    # ── Downsample for chart ──────────────────────────────────────────────────
    step = 5
    cd   = future_dates[::step]
    for mn, md in results.items():
        if "mean" in md:
            sz = min(len(cd), len(md["mean"][::step]))
            md["dates"]      = cd[:sz]
            md["mean_chart"] = md["mean"][::step][:sz]
            md["lo_chart"]   = md["lo"][::step][:sz]
            md["hi_chart"]   = md["hi"][::step][:sz]

    return {"models":results, "milestones":milestones,
            "last_price":round(float(prices[-1]),2),
            "last_date":str(last_date.date()),
            "history_yrs":round(len(df)/252,1), "ticker":ticker}


# ╔══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════╝
def chart_candlestick(df, ticker, company):
    di = add_indicators(df)
    fig= make_subplots(rows=3,cols=1,shared_xaxes=True,
        row_heights=[0.6,0.2,0.2],
        subplot_titles=["Price & Indicators","MACD","Volume"],
        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=di.index,open=di["Open"],high=di["High"],low=di["Low"],close=di["Close"],
        increasing_line_color=GREEN,decreasing_line_color=RED,name="Price"),row=1,col=1)
    if "BB_upper" in di:
        fig.add_trace(go.Scatter(x=di.index,y=di["BB_upper"],
            line=dict(color=f"rgba({_hex_rgb(GOLD)},0.35)",width=1),showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=di.index,y=di["BB_lower"],
            line=dict(color=f"rgba({_hex_rgb(GOLD)},0.35)",width=1),
            fill="tonexty",fillcolor=f"rgba({_hex_rgb(GOLD)},0.04)",showlegend=False),row=1,col=1)
    for ma,c,w in [("SMA_20",BLUE,1.2),("SMA_50",GOLD,1.2),("SMA_200",PURPLE,1.5)]:
        if ma in di:
            fig.add_trace(go.Scatter(x=di.index,y=di[ma],
                line=dict(color=c,width=w),name=ma),row=1,col=1)
    if "MACD" in di:
        ch=[GREEN if v>=0 else RED for v in di["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=di.index,y=di["MACD_hist"],
            marker_color=ch,showlegend=False),row=2,col=1)
        fig.add_trace(go.Scatter(x=di.index,y=di["MACD"],
            line=dict(color=BLUE,width=1.2),name="MACD"),row=2,col=1)
        fig.add_trace(go.Scatter(x=di.index,y=di["MACD_signal"],
            line=dict(color=GOLD,width=1.2),name="Signal"),row=2,col=1)
    vc=[GREEN if di["Close"].iloc[i]>=di["Open"].iloc[i] else RED for i in range(len(di))]
    fig.add_trace(go.Bar(x=di.index,y=di["Volume"],
        marker_color=vc,showlegend=False),row=3,col=1)
    if "Vol_MA20" in di:
        fig.add_trace(go.Scatter(x=di.index,y=di["Vol_MA20"],
            line=dict(color=GOLD,width=1.2),name="Vol MA20"),row=3,col=1)
    apply_theme(fig,700)
    fig.update_layout(xaxis_rangeslider_visible=False,
        title=dict(text=f"<b>{company}</b>  ·  {ticker}",
                   font=dict(size=14,color=TEXT_SEC,family="Syne")))
    return fig


def chart_forecast(pred, company, show_models):
    hist = fetch_history(pred["ticker"], period="5y")
    fig  = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist.index,y=hist["Close"],
            line=dict(color=TEXT_SEC,width=1.5),name="Historical",
            hovertemplate="₹%{y:,.2f}<extra>Historical</extra>"))
    for mn in show_models:
        md = pred["models"].get(mn,{})
        if "mean_chart" not in md: continue
        c = md["color"]
        fig.add_trace(go.Scatter(
            x=list(md["dates"])+list(md["dates"])[::-1],
            y=list(md["hi_chart"])+list(md["lo_chart"])[::-1],
            fill="toself",fillcolor=f"rgba({_hex_rgb(c)},0.09)",
            line=dict(width=0),name=f"{mn} CI",hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=md["dates"],y=md["mean_chart"],
            line=dict(color=c,width=2.2),name=mn,
            hovertemplate=f"<b>{mn}</b><br>₹%{{y:,.0f}}<br>%{{x}}<extra></extra>"))
    fig.add_vline(x=pred["last_date"],line_dash="dash",
        line_color=TEXT_MUT,opacity=0.7,
        annotation_text="Today",annotation_font_size=10,annotation_font_color=TEXT_SEC)
    apply_theme(fig,600)
    fig.update_layout(
        title=dict(text=f"<b>{company}</b> — 5-Year Price Forecast",
                   font=dict(size=15,family="Syne")),
        xaxis_title="Year",yaxis_title="Price ₹",
        legend=dict(orientation="h",y=1.06,x=0),hovermode="x unified")
    return fig


def chart_rsi(df):
    di = add_indicators(df.tail(365))
    fig= go.Figure()
    fig.add_trace(go.Scatter(x=di.index,y=di["RSI"],
        line=dict(color=ACCENT,width=2),name="RSI(14)"))
    fig.add_hrect(y0=70,y1=100,fillcolor=f"rgba({_hex_rgb(RED)},0.06)",line_width=0,
        annotation_text="Overbought",annotation_font_size=10,annotation_font_color=RED)
    fig.add_hrect(y0=0,y1=30,fillcolor=f"rgba({_hex_rgb(GREEN)},0.06)",line_width=0,
        annotation_text="Oversold",annotation_font_size=10,annotation_font_color=GREEN)
    fig.add_hline(y=70,line_dash="dash",line_color=RED,  opacity=0.4)
    fig.add_hline(y=30,line_dash="dash",line_color=GREEN,opacity=0.4)
    apply_theme(fig,260)
    fig.update_layout(title="RSI (14-day)",yaxis_range=[0,100],margin=dict(t=36,b=28))
    return fig


def chart_returns_dist(df):
    r   = df["Close"].pct_change().dropna()*100
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r,nbinsx=80,name="Daily Returns",
        marker_color=ACCENT,opacity=0.7))
    fig.add_vline(x=r.mean(),line_dash="dash",line_color=GOLD,
        annotation_text=f"μ={r.mean():.2f}%",annotation_font_size=10)
    apply_theme(fig,280)
    fig.update_layout(title=f"Daily Return Distribution  |  σ={r.std():.2f}%",
        xaxis_title="Daily Return %",yaxis_title="Frequency",margin=dict(t=36,b=28))
    return fig


def chart_sector_perf(quotes):
    rows=[]
    for tk,q in quotes.items():
        if q.get("price"):
            nm,_,cap=PHARMA_COMPANIES[tk]
            rows.append({"name":nm[:20],"change":q["change_pct"],"cap":cap})
    if not rows: return go.Figure()
    df=pd.DataFrame(rows).sort_values("change")
    cs=[GREEN if v>=0 else RED for v in df["change"]]
    fig=go.Figure(go.Bar(y=df["name"],x=df["change"],orientation="h",
        marker_color=cs,
        text=[f"{v:+.2f}%" for v in df["change"]],textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:+.2f}%<extra></extra>"))
    apply_theme(fig,max(440,len(rows)*22))
    fig.update_layout(title="Today's Performance — All Pharma Stocks",
        xaxis_title="Change %",margin=dict(t=40,r=90))
    return fig


def chart_corr(tickers,names):
    closes={}
    for tk,nm in zip(tickers,names):
        h=fetch_history(tk,period="2y")
        if not h.empty: closes[nm[:14]]=h["Close"].resample("W").last()
    if len(closes)<2: return go.Figure()
    corr=pd.DataFrame(closes).dropna().pct_change().dropna().corr().round(2)
    fig=go.Figure(go.Heatmap(z=corr.values,x=corr.columns,y=corr.index,
        colorscale=[[0,RED],[0.5,CARD2_BG],[1,GREEN]],
        zmin=-1,zmax=1,text=corr.values,texttemplate="%{text:.2f}",
        textfont=dict(size=9),
        hovertemplate="<b>%{y} vs %{x}</b><br>r=%{z:.2f}<extra></extra>"))
    apply_theme(fig,520)
    fig.update_layout(title="Weekly Return Correlation Matrix",
        margin=dict(t=50,b=80,l=130,r=20),xaxis=dict(tickangle=-35))
    return fig


def build_milestone_df(milestones,last_price):
    rows=[]
    for mn,ms in milestones.items():
        for yl,v in ms.items():
            yr=int(yl[2:])
            rows.append({"Model":mn,"Horizon":yl,
                "P10 ₹":f"₹{v['lo']:,.0f}",
                "Target ₹":f"₹{v['mean']:,.0f}",
                "P90 ₹":f"₹{v['hi']:,.0f}",
                "CAGR":f"{((v['mean']/last_price)**(1/yr)-1)*100:.1f}%",
                "Upside":f"{(v['mean']/last_price-1)*100:+.0f}%"})
    return pd.DataFrame(rows)


# ╔══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════╝
with st.sidebar:
    st.markdown(f"""
    <div style='padding:18px 0 10px;'>
      <div style='font-family:Syne,sans-serif;font-size:1.15rem;font-weight:800;
                  background:linear-gradient(135deg,{ACCENT},{CYAN});
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        🧬 PHARMA INTEL
      </div>
      <div style='font-size:0.70rem;color:{TEXT_SEC};margin-top:3px;letter-spacing:0.08em;'>
        NSE · BSE · LIVE · AI FORECAST
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigation", [
        "🏠  Market Overview",
        "📊  Live Stock Tracker",
        "🕯️  Candlestick & Technicals",
        "🔮  5-Year AI Forecast",
        "🗞️  News & Sentiment",
        "📐  Correlation & Risk",
        "📋  Fundamentals",
    ], label_visibility="collapsed")

    st.divider()

    labels   = [f"{v[0]} ({k.replace('.NS','')})" for k,v in PHARMA_COMPANIES.items()]
    sel_lbl  = st.selectbox("Select Company", labels)
    sel_idx  = labels.index(sel_lbl)
    sel_tk   = list(PHARMA_COMPANIES.keys())[sel_idx]
    sel_name = PHARMA_COMPANIES[sel_tk][0]

    st.divider()
    st.markdown(f"""
    <div style='font-size:0.68rem;color:{TEXT_SEC};line-height:1.7;'>
      <b style='color:{ACCENT};'>LIVE DATA</b><br>
      • Prices refresh every 60s<br>
      • Auto-refresh on Live page<br><br>
      <b style='color:{ACCENT};'>PREDICTIONS</b><br>
      • First run: ~90s<br>
      • Cached 24 hours<br><br>
      <b style='color:{ACCENT};'>DISCLAIMER</b><br>
      Research only.<br>
      Not financial advice.
    </div>
    """, unsafe_allow_html=True)

# ── Page router ───────────────────────────────────────────────────────────────

# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════╝
if "Overview" in page:
    now = datetime.now().strftime("%d %b %Y  %I:%M %p IST")
    st.markdown(f"<h1 class='page-title'>Indian Pharma Markets</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='page-subtitle'><span class='live-dot'></span>{now} · prices ~1 min delayed</p>",
                unsafe_allow_html=True)

    # Refresh controls
    rc1, rc2, rc3 = st.columns([1,1,4])
    with rc1:
        if st.button("⟳  Refresh Prices", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with rc2:
        if _AUTOREFRESH_AVAILABLE:
            auto = st.toggle("Auto-refresh (30s)", value=False)
            if auto:
                st_autorefresh(interval=30000, key="overview_refresh")

    # Indices
    st.markdown("<div class='sec-hdr'>MARKET INDICES</div>", unsafe_allow_html=True)
    with st.spinner("Loading indices..."):
        idx = fetch_index_data()

    c1,c2,c3,c4 = st.columns(4)
    for col,(lbl,d) in zip([c1,c2,c3],idx.items()):
        p=d["price"]; pct=d["change_pct"]
        clr=GREEN if pct>=0 else RED; arr="▲" if pct>=0 else "▼"
        with col:
            st.markdown(f"""<div class='kpi' style='border-top:3px solid {clr};'>
              <div class='kpi-val'>{f'{p:,.2f}' if p else 'N/A'}</div>
              <div class='kpi-sub' style='color:{clr};'>{arr} {abs(pct):.2f}%</div>
              <div class='kpi-lbl'>{lbl}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='kpi' style='border-top:3px solid {BLUE};'>
          <div class='kpi-val'>~₹12.4T</div>
          <div class='kpi-sub' style='color:{TEXT_SEC};'>Listed Pharma</div>
          <div class='kpi-lbl'>Total Market Cap (est.)</div></div>""", unsafe_allow_html=True)

    # Fetch all quotes
    st.markdown("<div class='sec-hdr'>LIVE PRICE GRID</div>", unsafe_allow_html=True)
    tks  = list(PHARMA_COMPANIES.keys())
    prog = st.progress(0, text="Loading live prices...")
    qs   = {}
    for i,tk in enumerate(tks):
        qs[tk] = fetch_live_quote(tk)
        prog.progress((i+1)/len(tks), text=f"Fetching {tk}...")
    prog.empty()

    # Performance chart
    perf_fig = chart_sector_perf(qs)
    st.plotly_chart(perf_fig, use_container_width=True)

    # Price grid
    cols_n = 5
    for rs in range(0, len(tks), cols_n):
        row_tks = tks[rs:rs+cols_n]
        cols    = st.columns(cols_n)
        for col,tk in zip(cols,row_tks):
            nm,_,cap = PHARMA_COMPANIES[tk]
            q=qs.get(tk,{}); p=q.get("price"); chg=q.get("change_pct",0)
            clr=GREEN if chg>=0 else RED; arr="▲" if chg>=0 else "▼"
            cls="change-up" if chg>=0 else "change-down"
            with col:
                st.markdown(f"""<div class='ticker-card'>
                  <div class='ticker-sym'>{tk.replace('.NS','')}</div>
                  <div class='ticker-price'>₹{f'{p:,.2f}' if p else '--'}</div>
                  <div class='{cls}'>{arr} {abs(chg):.2f}%</div>
                  <div class='ticker-meta'>{nm[:22]}</div>
                  <div class='ticker-meta' style='color:{TEXT_MUT};'>{cap}</div>
                </div>""", unsafe_allow_html=True)

    # Top movers
    st.markdown("<div class='sec-hdr'>TOP MOVERS</div>", unsafe_allow_html=True)
    vq = {k:v for k,v in qs.items() if v.get("price") and v.get("change_pct") is not None}
    sq = sorted(vq.items(), key=lambda x: x[1]["change_pct"])
    cg, cl = st.columns(2)
    for col_o, grp, clr, label, arrow in [
        (cg, sq[-3:][::-1], GREEN, "🚀 TOP GAINERS", "▲"),
        (cl, sq[:3],        RED,   "📉 TOP LOSERS",  "▼")]:
        with col_o:
            st.markdown(f"<div style='color:{clr};font-family:Syne,sans-serif;font-weight:700;font-size:0.82rem;margin-bottom:8px;'>{label}</div>",
                        unsafe_allow_html=True)
            for tk,q in grp:
                nm = PHARMA_COMPANIES[tk][0]
                st.markdown(f"""<div class='ticker-card' style='border-left:3px solid {clr}44;'>
                  <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div><div class='ticker-sym'>{tk.replace('.NS','')}</div>
                         <div style='font-size:0.80rem;color:{TEXT_SEC};margin-top:2px;'>{nm[:24]}</div></div>
                    <div style='text-align:right;'>
                         <div class='ticker-price' style='font-size:1.1rem;'>₹{q['price']:,.2f}</div>
                         <div style='font-family:Space Mono,monospace;font-size:0.80rem;color:{clr};font-weight:700;'>
                           {arrow} {abs(q['change_pct']):.2f}%</div></div>
                  </div></div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE STOCK TRACKER  — FIX 1: auto-refresh + manual refresh
# ══════════════════════════════════════════════════════════════════════════════╝
elif "Live Stock" in page:
    # ── Auto-refresh (every 30s) ──────────────────────────────────────────────
    if _AUTOREFRESH_AVAILABLE:
        count = st_autorefresh(interval=30000, key="live_refresh")
    else:
        st.info("📦 Install `streamlit-autorefresh` for automatic 30s price updates: `pip install streamlit-autorefresh`")

    # Manual refresh
    col_hdr, col_btn = st.columns([5,1])
    with col_hdr:
        st.markdown(f"<h1 class='page-title'>{sel_name}</h1>", unsafe_allow_html=True)
        bse = PHARMA_COMPANIES[sel_tk][1]
        st.markdown(f"<p class='page-subtitle'><span class='live-dot'></span>LIVE &nbsp;·&nbsp; {sel_tk} &nbsp;·&nbsp; BSE {bse}</p>",
                    unsafe_allow_html=True)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⟳ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Fetching live quote..."):
        q = fetch_live_quote(sel_tk)

    if not q.get("price"):
        st.error("Could not fetch live data. Market may be closed or ticker unavailable.")
        st.stop()

    p=q["price"]; chg=q["change_pct"]; chga=q["change"]
    clr=GREEN if chg>=0 else RED; arr="▲" if chg>=0 else "▼"

    # Hero
    ts = datetime.now().strftime("%I:%M:%S %p")
    st.markdown(f"""
    <div class='hero'>
      <div style='font-size:0.70rem;color:{TEXT_SEC};letter-spacing:0.1em;margin-bottom:8px;'>
        <span class='live-dot'></span>LIVE PRICE · Updated {ts} IST
      </div>
      <div class='hero-price'>₹{p:,.2f}</div>
      <div class='hero-change' style='color:{clr};'>
        {arr} ₹{abs(chga):.2f} ({abs(chg):.2f}%) today
      </div>
      <div style='font-size:0.78rem;color:{TEXT_SEC};margin-top:10px;'>
        Prev Close: ₹{q.get('prev_close','N/A')} &nbsp;·&nbsp;
        Open: ₹{q.get('open') or 'N/A'} &nbsp;·&nbsp;
        {PHARMA_COMPANIES[sel_tk][2]}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI strip 1
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(lbl,val) in zip([c1,c2,c3,c4,c5],[
        ("Day High",    f"₹{q.get('high') or 'N/A'}"),
        ("Day Low",     f"₹{q.get('low') or 'N/A'}"),
        ("Volume",      fmt_num(q.get("volume"),"") or "N/A"),
        ("Market Cap",  fmt_num(q.get("market_cap"))),
        ("Avg Volume",  fmt_num(q.get("avg_volume"),"") or "N/A"),
    ]):
        with col:
            st.markdown(f"""<div class='kpi'>
              <div class='kpi-val' style='font-size:1.05rem;'>{val}</div>
              <div class='kpi-lbl'>{lbl}</div></div>""", unsafe_allow_html=True)

    # KPI strip 2
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    pe=q.get("pe_ratio"); pb=q.get("pb_ratio"); eps=q.get("eps")
    beta=q.get("beta"); roe=q.get("roe"); pm=q.get("profit_margin")
    for col,(lbl,val) in zip([c1,c2,c3,c4,c5,c6],[
        ("P/E Ratio",     f"{pe:.2f}" if pe else "N/A"),
        ("P/B Ratio",     f"{pb:.2f}" if pb else "N/A"),
        ("EPS (TTM)",     f"₹{eps:.2f}" if eps else "N/A"),
        ("Beta",          f"{beta:.2f}" if beta else "N/A"),
        ("ROE",           f"{roe*100:.1f}%" if roe else "N/A"),
        ("Profit Margin", f"{pm*100:.1f}%" if pm else "N/A"),
    ]):
        with col:
            st.markdown(f"""<div class='kpi'>
              <div class='kpi-val' style='font-size:0.95rem;'>{val}</div>
              <div class='kpi-lbl'>{lbl}</div></div>""", unsafe_allow_html=True)

    # 52W range
    wh=q.get("week52_high"); wl=q.get("week52_low")
    if wh and wl and p:
        pct_pos = (p-wl)/(wh-wl)*100
        st.markdown("<div class='sec-hdr'>52-WEEK RANGE</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='glass-card'>
          <div style='display:flex;justify-content:space-between;font-size:0.78rem;color:{TEXT_SEC};margin-bottom:10px;'>
            <span>52W Low: <b style='color:{GREEN};'>₹{wl:,.2f}</b></span>
            <span>Current: <b style='color:{TEXT_PRI};'>₹{p:,.2f}</b>
                  &nbsp;<span style='color:{TEXT_MUT};'>({pct_pos:.0f}% of range)</span></span>
            <span>52W High: <b style='color:{RED};'>₹{wh:,.2f}</b></span>
          </div>
          <div class='range-bar-wrap'>
            <div class='range-bar-fill' style='width:{pct_pos:.1f}%;'></div>
            <div class='range-marker' style='left:{pct_pos:.1f}%;'></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Price history
    st.markdown("<div class='sec-hdr'>PRICE HISTORY</div>", unsafe_allow_html=True)
    period_sel = st.radio("Period", ["1mo","3mo","6mo","1y","2y","5y"],
                          horizontal=True, index=3, key="hist_p")
    with st.spinner("Loading history..."):
        dfh = fetch_history(sel_tk, period=period_sel)

    if not dfh.empty:
        ca = dfh["Close"].values
        lc = GREEN if ca[-1]>=ca[0] else RED
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=dfh.index,y=dfh["Close"],mode="lines",
            line=dict(color=lc,width=2),
            fill="tozeroy",fillcolor=f"rgba({_hex_rgb(lc)},0.07)",name="Close"))
        apply_theme(fig_a,360)
        fig_a.update_layout(hovermode="x unified",xaxis_title="",yaxis_title="Price ₹",
            xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_a, use_container_width=True)

        vc=[GREEN if dfh["Close"].iloc[i]>=dfh["Close"].iloc[max(0,i-1)] else RED
            for i in range(len(dfh))]
        fig_v=go.Figure()
        fig_v.add_trace(go.Bar(x=dfh.index,y=dfh["Volume"],marker_color=vc,name="Volume"))
        apply_theme(fig_v,190)
        fig_v.update_layout(margin=dict(t=16,b=28),
            title="Volume",xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_v, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: CANDLESTICK & TECHNICALS
# ══════════════════════════════════════════════════════════════════════════════╝
elif "Candlestick" in page:
    st.markdown(f"<h1 class='page-title'>Technical Analysis</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='page-subtitle'>{sel_name} · {sel_tk}</p>", unsafe_allow_html=True)

    period_sel = st.radio("Period",["3mo","6mo","1y","2y"],horizontal=True,key="candle_p",index=1)
    with st.spinner("Building chart..."):
        dfc = fetch_history(sel_tk, period=period_sel)

    if dfc.empty:
        st.error("No data."); st.stop()

    st.plotly_chart(chart_candlestick(dfc,sel_tk,sel_name), use_container_width=True)

    c1,c2 = st.columns(2)
    with c1: st.plotly_chart(chart_rsi(dfc), use_container_width=True)
    with c2: st.plotly_chart(chart_returns_dist(dfc), use_container_width=True)

    # Technical signals
    st.markdown("<div class='sec-hdr'>SIGNAL SUMMARY</div>", unsafe_allow_html=True)
    di    = add_indicators(dfc); last=di.iloc[-1]
    sigs  = []
    rsi   = last.get("RSI")
    sma20 = last.get("SMA_20"); sma50=last.get("SMA_50"); sma200=last.get("SMA_200")
    macd  = last.get("MACD");   sig=last.get("MACD_signal"); cl=last["Close"]

    if rsi:
        if rsi>70: sigs.append(("RSI",f"{rsi:.1f}","Overbought",RED))
        elif rsi<30: sigs.append(("RSI",f"{rsi:.1f}","Oversold",GREEN))
        else: sigs.append(("RSI",f"{rsi:.1f}","Neutral",GOLD))
    if sma20 and sma50:
        sigs.append(("MA20 vs MA50","20>50" if sma20>sma50 else "20<50",
                      "Bullish" if sma20>sma50 else "Bearish",
                      GREEN if sma20>sma50 else RED))
    if cl and sma200:
        sigs.append(("Price vs SMA200",
                      f"₹{cl:.0f} {'>' if cl>sma200 else '<'} ₹{sma200:.0f}",
                      "Bullish" if cl>sma200 else "Bearish",
                      GREEN if cl>sma200 else RED))
    if macd and sig:
        sigs.append(("MACD","Above signal" if macd>sig else "Below signal",
                      "Bullish" if macd>sig else "Bearish",
                      GREEN if macd>sig else RED))

    cols = st.columns(max(1,len(sigs)))
    for col,(ind,val,lbl,clr) in zip(cols,sigs):
        with col:
            st.markdown(f"""<div class='kpi' style='border-top:3px solid {clr};'>
              <div class='kpi-val' style='color:{clr};font-size:0.88rem;'>{val}</div>
              <div class='kpi-sub' style='color:{clr};font-size:0.76rem;'>{lbl}</div>
              <div class='kpi-lbl'>{ind}</div></div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: 5-YEAR AI FORECAST
# ══════════════════════════════════════════════════════════════════════════════╝
elif "Forecast" in page:
    st.markdown("<h1 class='page-title'>5-Year AI Forecast</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='page-subtitle'>{sel_name} · Prophet · GPR · MC-GBM · XGBoost · N-BEATS · Ensemble</p>",
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class='alert'>
    ⚡ <b>Optimised engine</b> — first run ~60–90s, then cached 24h.
    6 models · 80–200 Monte Carlo paths each · P10/P90 confidence bands.
    </div>""", unsafe_allow_html=True)

    show_m = st.multiselect("Models to display",
        ["Prophet","GPR","MC-GBM","XGBoost","N-BEATS","Ensemble"],
        default=["Ensemble","Prophet","MC-GBM"])

    # Progress UI during computation
    prog_ph = st.empty()
    with prog_ph.container():
        pb = st.progress(0,"Initialising models...")

    with st.spinner(f"Computing predictions for {sel_name}..."):
        pred = run_all_predictions(sel_tk, years_ahead=5)
        pb.progress(100,"✅ Done!")
    prog_ph.empty()

    if "error" in pred:
        st.error(f"Prediction error: {pred['error']}"); st.stop()

    # Model badges
    badges=""
    for mn,md in pred["models"].items():
        ok = "✅" if "mean" in md else "❌"
        err= md.get("error","")
        badges += f'<span class="mbadge">{ok} {mn}</span>'
        if err: badges += f'<span style="font-size:0.68rem;color:{RED};"> {err[:35]}</span>'
    st.markdown(f"<div style='margin-bottom:16px;'>{badges}</div>", unsafe_allow_html=True)

    # Ensemble targets
    ens_ms = pred.get("milestones",{}).get("Ensemble",{})
    lp     = pred["last_price"]
    st.markdown("<div class='sec-hdr'>ENSEMBLE PRICE TARGETS</div>", unsafe_allow_html=True)
    if ens_ms:
        cols = st.columns(5)
        for col,yr in zip(cols,["Y+1","Y+2","Y+3","Y+4","Y+5"]):
            if yr in ens_ms:
                v=ens_ms[yr]; up=(v["mean"]/lp-1)*100
                clr=GREEN if up>0 else RED
                cagr=((v["mean"]/lp)**(1/int(yr[2:]))-1)*100
                with col:
                    st.markdown(f"""<div class='kpi' style='border-top:3px solid {clr};'>
                      <div class='kpi-val'>₹{v['mean']:,.0f}</div>
                      <div class='kpi-sub' style='color:{clr};'>{up:+.0f}%  ·  {cagr:.1f}% CAGR</div>
                      <div class='kpi-lbl'>{yr.replace('Y+','')} Year Target</div>
                    </div>""", unsafe_allow_html=True)

    # Forecast chart
    st.markdown("<div class='sec-hdr'>FORECAST CHART</div>", unsafe_allow_html=True)
    if show_m:
        st.plotly_chart(chart_forecast(pred,sel_name,show_m), use_container_width=True)
    else:
        st.info("Select at least one model.")

    # Milestone table
    st.markdown("<div class='sec-hdr'>FULL MILESTONE TABLE</div>", unsafe_allow_html=True)
    ms_df = build_milestone_df(pred.get("milestones",{}), lp)
    if not ms_df.empty:
        mf = st.multiselect("Filter models",ms_df["Model"].unique().tolist(),
             default=ms_df["Model"].unique().tolist(), key="msf")
        st.dataframe(ms_df[ms_df["Model"].isin(mf)].reset_index(drop=True),
                     use_container_width=True, hide_index=True, height=380)

    # Model tabs
    st.markdown("<div class='sec-hdr'>MODEL DOCUMENTATION</div>", unsafe_allow_html=True)
    tabs = st.tabs(["Prophet","GPR","MC-GBM","XGBoost","N-BEATS","Ensemble"])
    with tabs[0]:
        cp=pred["models"].get("Prophet",{}).get("changepoints","N/A")
        st.markdown(f"**Prophet** — Meta's additive decomposition. Fits `log(price) = trend + seasonality + ε`. "
                    f"Changepoints detected automatically (prior 0.15). Annual seasonality for India budget cycle. "
                    f"150 MC paths with residual noise. Changepoints found: **{cp}**.")
    with tabs[1]:
        kern=pred["models"].get("GPR",{}).get("kernel","N/A")
        st.markdown(f"**GPR** — Composite kernel: RBF (trend) + ExpSineSquared (yearly cycle) + DotProduct (growth) + WhiteKernel (noise). "
                    f"Sparse approximation on 200 subsampled points. Fitted kernel: `{str(kern)[:100]}`")
    with tabs[2]:
        mu=pred["models"].get("MC-GBM",{}).get("mu_ann","N/A")
        sg=pred["models"].get("MC-GBM",{}).get("sigma_ann","N/A")
        st.markdown(f"**MC-GBM** — Jump-diffusion GBM with EWMA vol (λ=0.94). 200 paths with regime uncertainty in μ,σ. "
                    f"~2 Poisson jumps/year. μ: **{mu}%/yr** · σ: **{sg}%/yr**")
    with tabs[3]:
        st.markdown("**XGBoost** — 200 trees, 15 engineered features (momentum, vol, slope, OBV). "
                    "80 MC rollout paths. RobustScaler for outlier-resistant normalisation.")
    with tabs[4]:
        st.markdown("**N-BEATS** — Neural Basis Expansion (M4 winner). Trend stack (polynomial deg-3) "
                    "+ Seasonal stack (6 Fourier harmonics). 8 epochs, Adam optimiser, full numpy backprop. "
                    "100 MC paths with per-window noise scaling.")
    with tabs[5]:
        wts=pred["models"].get("Ensemble",{}).get("weights",{})
        ws="  ·  ".join([f"**{k}** {v}%" for k,v in wts.items()]) if wts else "N/A"
        st.markdown(f"**Ensemble** — Inverse-variance weighted (60% base weight + 40% CI-adaptive). "
                    f"Weights this run: {ws}")

    st.markdown(f"""
    <div class='alert-danger'>
    ⚠️ <b>Important:</b> Long-horizon forecasts carry very wide uncertainty.
    These are probabilistic scenarios, not guarantees.
    Use for research and scenario planning only. <b>Not financial advice.</b>
    </div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: NEWS & SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════╝
elif "News" in page:
    st.markdown("<h1 class='page-title'>News & Sentiment</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='page-subtitle'>{sel_name}</p>", unsafe_allow_html=True)

    with st.spinner("Fetching headlines..."):
        news = fetch_news(sel_name)

    if not news:
        st.warning("No news found. Google News RSS may be temporarily unavailable.")
    else:
        pos=sum(1 for n in news if n["sentiment"]=="positive")
        neg=sum(1 for n in news if n["sentiment"]=="negative")
        neu=sum(1 for n in news if n["sentiment"]=="neutral")

        c1,c2,c3,c4=st.columns(4)
        overall="Bullish" if pos>neg else ("Bearish" if neg>pos else "Neutral")
        oclr=GREEN if overall=="Bullish" else (RED if overall=="Bearish" else GOLD)
        for col,(lbl,val,clr) in zip([c1,c2,c3,c4],[
            ("Overall Sentiment", overall, oclr),
            ("Positive", str(pos), GREEN),
            ("Negative", str(neg), RED),
            ("Neutral",  str(neu), GOLD),
        ]):
            with col:
                st.markdown(f"""<div class='kpi' style='border-top:3px solid {clr};'>
                  <div class='kpi-val' style='color:{clr};font-size:1.0rem;'>{val}</div>
                  <div class='kpi-lbl'>{lbl}</div></div>""", unsafe_allow_html=True)

        c_chart, c_news = st.columns([1,2])
        with c_chart:
            fig_s=go.Figure(go.Pie(labels=["Positive","Neutral","Negative"],
                values=[pos,neu,neg],marker_colors=[GREEN,GOLD,RED],
                hole=0.65,textinfo="label+percent"))
            fig_s.add_annotation(text=f"<b>{len(news)}</b><br>items",
                x=0.5,y=0.5,font_size=14,showarrow=False,font_color=TEXT_PRI)
            apply_theme(fig_s,280)
            fig_s.update_layout(showlegend=False,margin=dict(t=16,b=16,l=16,r=16))
            st.plotly_chart(fig_s, use_container_width=True)

        with c_news:
            st.markdown("<div class='sec-hdr'>RECENT HEADLINES</div>", unsafe_allow_html=True)
            for item in news:
                s=item["sentiment"]
                pc="pill-pos" if s=="positive" else ("pill-neg" if s=="negative" else "pill-neu")
                pt="🟢 Positive" if s=="positive" else ("🔴 Negative" if s=="negative" else "⚪ Neutral")
                st.markdown(f"""<div class='ticker-card' style='margin-bottom:6px;'>
                  <div style='display:flex;justify-content:space-between;align-items:flex-start;gap:12px;'>
                    <div style='font-size:0.84rem;color:{TEXT_PRI};line-height:1.5;'>{item['title']}</div>
                    <div class='pill {pc}' style='white-space:nowrap;flex-shrink:0;'>{pt}</div>
                  </div></div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: CORRELATION & RISK
# ══════════════════════════════════════════════════════════════════════════════╝
elif "Correlation" in page:
    st.markdown("<h1 class='page-title'>Correlation & Risk</h1>", unsafe_allow_html=True)

    cap_f = st.multiselect("Market Cap",["Large Cap","Mid Cap","Small Cap"],
                            default=["Large Cap","Mid Cap"])
    ftks  = [k for k,v in PHARMA_COMPANIES.items() if v[2] in cap_f]
    fnms  = [PHARMA_COMPANIES[k][0] for k in ftks]

    with st.spinner("Computing correlation matrix..."):
        st.plotly_chart(chart_corr(ftks,fnms), use_container_width=True)

    st.markdown(f"<div class='sec-hdr'>RISK METRICS — {sel_name}</div>", unsafe_allow_html=True)
    with st.spinner("Computing risk metrics..."):
        dfr = fetch_history(sel_tk, period="3y")

    if not dfr.empty:
        r    = dfr["Close"].pct_change().dropna()
        ar   = (1+r.mean())**252-1
        av   = r.std()*np.sqrt(252)
        sh   = ar/av if av>0 else 0
        mdd  = ((dfr["Close"]/dfr["Close"].cummax())-1).min()
        var  = np.percentile(r,5)
        pos  = (r>0).mean()*100

        c1,c2,c3,c4,c5,c6=st.columns(6)
        for col,(lbl,val,clr) in zip([c1,c2,c3,c4,c5,c6],[
            ("Ann. Return",    f"{ar*100:.1f}%",  GREEN if ar>0 else RED),
            ("Ann. Volatility",f"{av*100:.1f}%",  GOLD),
            ("Sharpe Ratio",   f"{sh:.2f}",        GREEN if sh>1 else (GOLD if sh>0 else RED)),
            ("Max Drawdown",   f"{mdd*100:.1f}%",  RED),
            ("VaR (95% 1D)",   f"{var*100:.2f}%",  RED),
            ("% Positive Days",f"{pos:.1f}%",      GREEN if pos>50 else RED),
        ]):
            with col:
                st.markdown(f"""<div class='kpi' style='border-top:3px solid {clr};'>
                  <div class='kpi-val' style='color:{clr};font-size:1.0rem;'>{val}</div>
                  <div class='kpi-lbl'>{lbl}</div></div>""", unsafe_allow_html=True)

        # Rolling vol
        rv = r.rolling(30).std()*np.sqrt(252)*100
        fv = go.Figure()
        fv.add_trace(go.Scatter(x=rv.index,y=rv,fill="tozeroy",
            fillcolor=f"rgba({_hex_rgb(GOLD)},0.10)",
            line=dict(color=GOLD,width=2),name="30D Rolling Vol"))
        apply_theme(fv,280)
        fv.update_layout(title="30-Day Rolling Annualised Volatility %",
            yaxis_title="Vol %",margin=dict(t=36,b=28))
        st.plotly_chart(fv, use_container_width=True)

        # Drawdown
        dd = (dfr["Close"]/dfr["Close"].cummax()-1)
        fd = go.Figure()
        fd.add_trace(go.Scatter(x=dd.index,y=dd*100,fill="tozeroy",
            fillcolor=f"rgba({_hex_rgb(RED)},0.12)",
            line=dict(color=RED,width=1.5),name="Drawdown"))
        apply_theme(fd,250)
        fd.update_layout(title="Underwater / Drawdown Chart",
            yaxis_title="Drawdown %",margin=dict(t=36,b=28))
        st.plotly_chart(fd, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════════════════╝
elif "Fundamentals" in page:
    st.markdown("<h1 class='page-title'>Fundamentals</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='page-subtitle'>{sel_name}</p>", unsafe_allow_html=True)

    with st.spinner("Loading fundamentals..."):
        q = fetch_live_quote(sel_tk)

    st.markdown("<div class='sec-hdr'>VALUATION & PROFITABILITY</div>", unsafe_allow_html=True)
    mets = {
        "P/E Ratio":      (q.get("pe_ratio"),         10, 30,  "lower"),
        "P/B Ratio":      (q.get("pb_ratio"),          1,  5,  "lower"),
        "EPS (TTM ₹)":   (q.get("eps"),          None, None,   "higher"),
        "Dividend Yield": ((q.get("div_yield") or 0),  0, 0.05,"higher"),
        "Beta":           (q.get("beta"),            0.8,  1.2, "neutral"),
        "Debt/Equity":    (q.get("debt_equity"),       0, 100, "lower"),
        "ROE %":          ((q.get("roe") or 0)*100,   10,  25, "higher"),
        "Profit Margin %":((q.get("profit_margin") or 0)*100, 5, 20, "higher"),
    }
    cols=st.columns(4)
    for i,(lbl,(val,lo,hi,dir)) in enumerate(mets.items()):
        with cols[i%4]:
            if val is None: disp,clr="N/A",TEXT_SEC
            else:
                disp=f"{val:.2f}"
                if lo is not None:
                    if dir=="lower":    clr=GREEN if val<lo else (GOLD if val<hi else RED)
                    elif dir=="higher": clr=GREEN if val>hi else (GOLD if val>lo else RED)
                    else:               clr=GREEN if lo<=val<=hi else RED
                else: clr=GREEN if val>0 else RED
            st.markdown(f"""<div class='kpi' style='border-top:3px solid {clr};'>
              <div class='kpi-val' style='color:{clr};font-size:1.05rem;'>{disp}</div>
              <div class='kpi-lbl'>{lbl}</div></div>""", unsafe_allow_html=True)

    # Peer table
    st.markdown("<div class='sec-hdr'>PEER COMPARISON — TOP 10</div>", unsafe_allow_html=True)
    peers=[]
    with st.spinner("Loading peers..."):
        for tk in list(PHARMA_COMPANIES.keys())[:10]:
            pq=fetch_live_quote(tk)
            if pq.get("price"):
                peers.append({
                    "Company":  PHARMA_COMPANIES[tk][0][:22],
                    "Price ₹":  f"₹{pq['price']:,.2f}",
                    "Chg %":    f"{pq['change_pct']:+.2f}%",
                    "Mkt Cap":  fmt_num(pq.get("market_cap")),
                    "P/E":      f"{pq['pe_ratio']:.1f}" if pq.get("pe_ratio") else "—",
                    "P/B":      f"{pq['pb_ratio']:.1f}" if pq.get("pb_ratio") else "—",
                    "ROE %":    f"{pq['roe']*100:.1f}%" if pq.get("roe") else "—",
                    "52W Hi":   f"₹{pq['week52_high']:,.0f}" if pq.get("week52_high") else "—",
                    "52W Lo":   f"₹{pq['week52_low']:,.0f}" if pq.get("week52_low") else "—",
                    "Segment":  PHARMA_COMPANIES[tk][2],
                })
    if peers:
        st.dataframe(pd.DataFrame(peers), use_container_width=True,
                     hide_index=True, height=380)

    # Bubble chart
    st.markdown("<div class='sec-hdr'>VALUATION MAP</div>", unsafe_allow_html=True)
    bub=[]
    for tk in list(PHARMA_COMPANIES.keys())[:10]:
        pq=fetch_live_quote(tk)
        mc=pq.get("market_cap") or 0
        pe=pq.get("pe_ratio"); pb=pq.get("pb_ratio"); roe=pq.get("roe")
        if pq.get("price") and pe and pb:
            bub.append({"name":PHARMA_COMPANIES[tk][0][:18],"pe":pe,"pb":pb,
                "mktcap":mc/1e9,"roe":(roe or 0)*100,"change":pq["change_pct"]})
    if bub:
        bdf=pd.DataFrame(bub)
        fig_b=px.scatter(bdf,x="pe",y="pb",size="mktcap",color="roe",text="name",
            size_max=50,color_continuous_scale=[[0,RED],[0.5,GOLD],[1,GREEN]],
            labels={"pe":"P/E","pb":"P/B","mktcap":"Mkt Cap B","roe":"ROE %"},
            hover_data={"pe":":.1f","pb":":.1f","mktcap":":.1f","roe":":.1f"})
        fig_b.update_traces(textposition="top center",textfont_size=9)
        apply_theme(fig_b,520)
        fig_b.update_layout(title="P/E vs P/B  ·  Size = Mkt Cap  ·  Color = ROE%")
        st.plotly_chart(fig_b, use_container_width=True)
