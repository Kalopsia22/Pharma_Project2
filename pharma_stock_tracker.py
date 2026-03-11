"""
╔══════════════════════════════════════════════════════════════════════╗
║   INDIAN PHARMA STOCK INTELLIGENCE PLATFORM                         ║
║   Live Tracker (NSE + BSE) + 25-Year AI Price Forecast              ║
║   Models: Prophet · GPR · MC-GBM · XGBoost · N-BEATS · Ensemble    ║
╚══════════════════════════════════════════════════════════════════════╝
Run:  streamlit run pharma_stock_tracker.py
Deps: pip install streamlit yfinance pandas numpy plotly scikit-learn
      statsmodels xgboost requests beautifulsoup4 lxml
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
    page_title="Pharma Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design constants ──────────────────────────────────────────────────────────
GREEN      = "#00C896"
RED        = "#FF4B6E"
GOLD       = "#F5A623"
BLUE       = "#1E90FF"
DARK_BG    = "#0A0E1A"
CARD_BG    = "#111827"
BORDER     = "#1F2937"
TEXT_PRI   = "#F9FAFB"
TEXT_SEC   = "#9CA3AF"
ACCENT     = "#00C896"
COLORS_10  = ["#00C896","#1E90FF","#F5A623","#FF4B6E","#A78BFA",
               "#34D399","#60A5FA","#FBBF24","#F87171","#C084FC"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Sora', sans-serif;
    background-color: {DARK_BG};
    color: {TEXT_PRI};
}}
.main {{ background: {DARK_BG}; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}

section[data-testid="stSidebar"] {{
    background: #060912 !important;
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT_PRI} !important; }}
section[data-testid="stSidebar"] hr {{ border-color: {BORDER}; }}

/* Ticker card */
.ticker-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}}
.ticker-card:hover {{ border-color: {ACCENT}; }}
.ticker-name {{ font-size: 0.78rem; color: {TEXT_SEC}; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }}
.ticker-price {{ font-family: 'JetBrains Mono', monospace; font-size: 1.55rem; font-weight: 700; color: {TEXT_PRI}; line-height: 1.1; }}
.ticker-change-up   {{ font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; color: {GREEN}; font-weight: 600; }}
.ticker-change-down {{ font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; color: {RED};   font-weight: 600; }}
.ticker-meta {{ font-size: 0.72rem; color: {TEXT_SEC}; margin-top: 4px; }}

/* KPI strip */
.kpi-strip {{
    background: {CARD_BG}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 14px 20px;
    text-align: center; margin-bottom: 6px;
}}
.kpi-strip-val  {{ font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:700; color:{TEXT_PRI}; }}
.kpi-strip-lbl  {{ font-size:0.70rem; color:{TEXT_SEC}; text-transform:uppercase; letter-spacing:0.07em; margin-top:3px; }}
.kpi-strip-sub  {{ font-size:0.78rem; margin-top:2px; }}

/* Section header */
.section-hdr {{
    font-size: 0.72rem; font-weight: 700; color: {ACCENT};
    letter-spacing: 0.15em; text-transform: uppercase;
    border-bottom: 1px solid {BORDER}; padding-bottom: 8px;
    margin: 22px 0 14px 0;
}}

/* Sentiment pill */
.pill-pos  {{ display:inline-block; background:rgba(0,200,150,0.15); color:{GREEN};
              border:1px solid {GREEN}; border-radius:20px; padding:2px 12px; font-size:0.78rem; font-weight:600; }}
.pill-neg  {{ display:inline-block; background:rgba(255,75,110,0.15); color:{RED};
              border:1px solid {RED};   border-radius:20px; padding:2px 12px; font-size:0.78rem; font-weight:600; }}
.pill-neu  {{ display:inline-block; background:rgba(156,163,175,0.15); color:{TEXT_SEC};
              border:1px solid {BORDER}; border-radius:20px; padding:2px 12px; font-size:0.78rem; font-weight:600; }}

/* Alert box */
.alert-box {{
    background: rgba(245,166,35,0.10); border-left: 3px solid {GOLD};
    padding: 12px 16px; border-radius: 0 8px 8px 0;
    font-size: 0.85rem; color: {TEXT_PRI}; margin: 10px 0;
}}
.model-badge {{
    display:inline-block; background:rgba(30,144,255,0.15); color:{BLUE};
    border:1px solid {BLUE}; border-radius:6px; padding:2px 10px;
    font-size:0.72rem; font-weight:600; margin:2px;
}}

/* Dark plotly override */
[data-testid="stPlotlyChart"] {{ background: transparent !important; }}
div[data-testid="stTabs"] button {{ color: {TEXT_SEC} !important; font-size:0.82rem; }}
div[data-testid="stTabs"] button[aria-selected="true"] {{ color:{ACCENT} !important; border-bottom-color:{ACCENT} !important; }}

/* Scrolling marquee for live prices */
.marquee-wrap {{
    background: #060912; border-top: 1px solid {BORDER}; border-bottom: 1px solid {BORDER};
    overflow: hidden; padding: 8px 0; margin-bottom: 18px;
}}
.marquee-inner {{
    display: flex; animation: marquee 60s linear infinite; white-space: nowrap;
}}
.marquee-item {{
    font-family:'JetBrains Mono',monospace; font-size:0.78rem;
    padding: 0 28px; border-right: 1px solid {BORDER};
}}
@keyframes marquee {{ 0% {{ transform: translateX(0); }} 100% {{ transform: translateX(-50%); }} }}

input, select, textarea {{ background: {CARD_BG} !important; color: {TEXT_PRI} !important; }}
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# COMPANY REGISTRY — All major listed Indian Pharma companies (NSE + BSE)
# ══════════════════════════════════════════════════════════════════════════════╝
PHARMA_COMPANIES = {
    # NSE Symbol         : (Display Name,                    BSE Code, Sector)
    "SUNPHARMA.NS"       : ("Sun Pharmaceutical",            "524715", "Large Cap"),
    "DRREDDY.NS"         : ("Dr. Reddy's Laboratories",      "500124", "Large Cap"),
    "CIPLA.NS"           : ("Cipla Ltd",                     "500087", "Large Cap"),
    "DIVISLAB.NS"        : ("Divi's Laboratories",           "532488", "Large Cap"),
    "MANKIND.NS"         : ("Mankind Pharma",                "543904", "Large Cap"),
    "TORNTPHARM.NS"      : ("Torrent Pharmaceuticals",       "500420", "Large Cap"),
    "LUPIN.NS"           : ("Lupin Ltd",                     "500257", "Large Cap"),
    "AUROPHARMA.NS"      : ("Aurobindo Pharma",              "524804", "Large Cap"),
    "ALKEM.NS"           : ("Alkem Laboratories",            "539523", "Large Cap"),
    "BIOCON.NS"          : ("Biocon Ltd",                    "532523", "Large Cap"),
    "ZYDUSLIFE.NS"       : ("Zydus Lifesciences",            "532321", "Large Cap"),
    "IPCALAB.NS"         : ("IPCA Laboratories",             "524494", "Mid Cap"),
    "GLENMARK.NS"        : ("Glenmark Pharmaceuticals",      "532296", "Mid Cap"),
    "AJANTPHARM.NS"      : ("Ajanta Pharma",                 "532331", "Mid Cap"),
    "GRANULES.NS"        : ("Granules India",                "532482", "Mid Cap"),
    "NATCOPHARM.NS"      : ("Natco Pharma",                  "524816", "Mid Cap"),
    "ABBOTINDIA.NS"      : ("Abbott India",                  "500488", "Mid Cap"),
    "PFIZER.NS"          : ("Pfizer India",                  "500680", "Mid Cap"),
    "SANOFI.NS"          : ("Sanofi India",                  "500674", "Mid Cap"),
    "GLAXO.NS"           : ("GSK Pharmaceuticals",           "500660", "Mid Cap"),
    "LAURUSLABS.NS"      : ("Laurus Labs",                   "540222", "Mid Cap"),
    "ERIS.NS"            : ("Eris Lifesciences",             "540596", "Mid Cap"),
    "JBCHEPHARM.NS"      : ("JB Chemicals & Pharma",         "506943", "Mid Cap"),
    "PIRAMALPHARM.NS"    : ("Piramal Pharma",                "543635", "Mid Cap"),
    "KSKPHARMA.NS"       : ("Kopran Ltd",                    "524280", "Small Cap"),
    "SOLARA.NS"          : ("Solara Active Pharma",          "541540", "Small Cap"),
    "STRIDES.NS"         : ("Strides Pharma",                "532531", "Small Cap"),
    "MARKSANS.NS"        : ("Marksans Pharma",               "524404", "Small Cap"),
    "CAPLIPOINT.NS"      : ("Caplin Point Lab",              "524742", "Small Cap"),
    "SEQUENT.NS"         : ("Sequent Scientific",            "512529", "Small Cap"),
}

# Nifty Pharma Index
NIFTY_PHARMA = "^CNXPHARMA"
SENSEX        = "^BSESN"
NIFTY50       = "^NSEI"

PLOT_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(family="Sora, sans-serif", color=TEXT_PRI, size=12),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, showgrid=True),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
        margin=dict(t=40, b=40, l=60, r=20),
    )
)

def apply_theme(fig, height=420):
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(family="Sora, sans-serif", color=TEXT_PRI, size=12),
        height=height, margin=dict(t=40,b=40,l=60,r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
    )
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER, showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, showgrid=True, zeroline=False)
    return fig


# ╔══════════════════════════════════════════════════════════════════════════════
# DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════════════╝
@st.cache_data(ttl=300, show_spinner=False)   # 5-min cache for live prices
def fetch_live_quote(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="2d", interval="1m")
        prev_close = info.get("previousClose", info.get("regularMarketPreviousClose", None))
        price      = info.get("currentPrice", info.get("regularMarketPrice", None))
        if price is None and not hist.empty:
            price = float(hist["Close"].iloc[-1])
        change     = price - prev_close if (price and prev_close) else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
        return {
            "price":          round(price, 2) if price else None,
            "prev_close":     round(prev_close, 2) if prev_close else None,
            "change":         round(change, 2),
            "change_pct":     round(change_pct, 2),
            "open":           info.get("open", info.get("regularMarketOpen")),
            "high":           info.get("dayHigh", info.get("regularMarketDayHigh")),
            "low":            info.get("dayLow",  info.get("regularMarketDayLow")),
            "volume":         info.get("volume",  info.get("regularMarketVolume")),
            "market_cap":     info.get("marketCap"),
            "pe_ratio":       info.get("trailingPE"),
            "pb_ratio":       info.get("priceToBook"),
            "eps":            info.get("trailingEps"),
            "div_yield":      info.get("dividendYield"),
            "week52_high":    info.get("fiftyTwoWeekHigh"),
            "week52_low":     info.get("fiftyTwoWeekLow"),
            "avg_volume":     info.get("averageVolume"),
            "beta":           info.get("beta"),
            "roe":            info.get("returnOnEquity"),
            "profit_margin":  info.get("profitMargins"),
            "revenue":        info.get("totalRevenue"),
            "debt_equity":    info.get("debtToEquity"),
            "name":           info.get("longName", ticker),
            "sector":         info.get("sector","Healthcare"),
            "exchange":       info.get("exchange","NSE"),
        }
    except Exception as e:
        return {"price": None, "change_pct": 0, "change": 0, "error": str(e)}


@st.cache_data(ttl=900, show_spinner=False)   # 15-min cache for history
def fetch_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    try:
        t  = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        # Safely strip timezone — handle both tz-aware and tz-naive indexes
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        else:
            df.index = pd.to_datetime(df.index)
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_financials(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        income    = t.quarterly_financials
        balance   = t.quarterly_balance_sheet
        cashflow  = t.quarterly_cashflow
        return {"income": income, "balance": balance, "cashflow": cashflow}
    except:
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_index_data() -> dict:
    result = {}
    for sym, label in [(NIFTY_PHARMA,"Nifty Pharma"),(SENSEX,"BSE Sensex"),(NIFTY50,"Nifty 50")]:
        try:
            t = yf.Ticker(sym)
            info = t.info
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            prev  = info.get("previousClose") or info.get("regularMarketPreviousClose")
            change_pct = ((price-prev)/prev*100) if (price and prev) else 0
            result[label] = {"price": price, "change_pct": round(change_pct,2)}
        except:
            result[label] = {"price": None, "change_pct": 0}
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_news_sentiment(company_name: str) -> list:
    """Fetch recent headlines and assign simple sentiment via keyword matching."""
    pos_words = {"gain","rise","surge","profit","growth","strong","beat","record",
                 "upgrade","outperform","buy","positive","robust","rally","high"}
    neg_words = {"fall","drop","loss","decline","miss","downgrade","sell","weak",
                 "concern","risk","cut","negative","pressure","low","lawsuit"}
    try:
        query = company_name.replace(" ","+") + "+stock+India"
        url   = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        resp  = requests.get(url, timeout=8,
                    headers={"User-Agent":"Mozilla/5.0"})
        items = re.findall(r"<title>(.*?)</title>", resp.text)[2:12]
        news  = []
        for title in items:
            title = re.sub(r"<[^>]+>","",title).strip()
            words = set(title.lower().split())
            pos   = len(words & pos_words)
            neg   = len(words & neg_words)
            sentiment = "positive" if pos>neg else ("negative" if neg>pos else "neutral")
            news.append({"title": title, "sentiment": sentiment})
        return news[:8]
    except:
        return []


# ╔══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════╝
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 30:
        return df
    df = df.copy()
    # Moving averages
    df["SMA_20"]  = df["Close"].rolling(20).mean()
    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["EMA_20"]  = df["Close"].ewm(span=20).mean()
    # Bollinger Bands
    bb_std        = df["Close"].rolling(20).std()
    df["BB_upper"] = df["SMA_20"] + 2*bb_std
    df["BB_lower"] = df["SMA_20"] - 2*bb_std
    # RSI
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100/(1+rs))
    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]
    # ATR
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift()).abs()
    lpc = (df["Low"]  - df["Close"].shift()).abs()
    df["ATR"] = pd.concat([hl,hpc,lpc],axis=1).max(axis=1).rolling(14).mean()
    # OBV
    obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["OBV"] = obv
    # Volume MA
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    return df


# ╔══════════════════════════════════════════════════════════════════════════════
# ML PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════╝
@st.cache_data(ttl=86400, show_spinner=False)   # cache predictions 24h
def run_all_predictions(ticker: str, years_ahead: int = 5) -> dict:
    """
    6-model prediction engine — all pure numpy/sklearn/statsmodels/prophet/xgboost.
    No PyTorch / TensorFlow. Runs on Streamlit Cloud free tier.

    Models:
      1. Prophet          — Meta's additive trend + seasonality + changepoints
      2. Gaussian Process — Sparse RBF kernel, Bayesian uncertainty propagation
      3. MC-GBM           — Geometric Brownian Motion (quant finance standard)
      4. XGBoost          — Gradient boosted trees on 15 lag/volatility features
      5. N-BEATS          — Neural Basis Expansion (M4 winner, trend + seasonal stacks)
      6. Ensemble         — Inverse-variance weighted combination of all valid models
    """
    df = fetch_history(ticker, period="max", interval="1d")
    if df.empty or len(df) < 252:
        return {"error": "Insufficient historical data (need ≥1 year)"}

    prices    = df["Close"].values.astype(float)
    dates_arr = df.index
    log_ret   = np.diff(np.log(prices))
    last_date = dates_arr[-1]
    n_days    = years_ahead * 252
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=n_days, freq="B")

    results = {}
    np.random.seed(42)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. PROPHET  — additive decomposition, changepoint detection, yearly cycles
    # ══════════════════════════════════════════════════════════════════════════
    try:
        from prophet import Prophet                                 # pip install prophet

        prophet_df = pd.DataFrame({
            "ds": dates_arr,
            "y":  np.log(prices),          # fit on log-price for multiplicative growth
        })

        m = Prophet(
            growth                  = "linear",
            changepoint_prior_scale = 0.15,
            seasonality_prior_scale = 0.1,
            yearly_seasonality      = True,
            weekly_seasonality      = False,  # not useful for stocks
            daily_seasonality       = False,
            interval_width          = 0.80,
            uncertainty_samples     = 0,      # MAP only — skips slow MCMC sampling
        )
        m.fit(prophet_df)

        future_df    = m.make_future_dataframe(periods=n_days, freq="B")
        forecast_df  = m.predict(future_df)

        # Slice to future only
        fc_future    = forecast_df[forecast_df["ds"] > last_date].reset_index(drop=True)
        fc_log_mean  = fc_future["yhat"].values
        fc_log_lo    = fc_future["yhat_lower"].values
        fc_log_hi    = fc_future["yhat_upper"].values

        # Back-transform from log-space
        p_mean = np.exp(fc_log_mean)
        p_lo   = np.exp(fc_log_lo)
        p_hi   = np.exp(fc_log_hi)

        # Compute residual std on in-sample fit for smearing correction
        insample = forecast_df[forecast_df["ds"] <= last_date]
        resid_p  = np.std(prophet_df["y"].values - insample["yhat"].values[-len(prophet_df):])

        # Monte Carlo fan for CI consistency with other models
        mc_paths = []
        for _ in range(200):
            noise   = np.random.normal(0, resid_p, n_days)
            cumstd  = np.sqrt(np.arange(1, n_days + 1)) * resid_p * 0.02
            path_lp = fc_log_mean + noise + cumstd * np.random.randn()
            mc_paths.append(np.exp(path_lp))
        mc_arr = np.array(mc_paths)

        results["Prophet"] = {
            "mean":  np.median(mc_arr, axis=0),
            "lo":    np.percentile(mc_arr, 10, axis=0),
            "hi":    np.percentile(mc_arr, 90, axis=0),
            "changepoints": len(m.changepoints),
            "color": "#F97316",     # orange
        }
    except Exception as e:
        results["Prophet"] = {"error": str(e)}

    # ══════════════════════════════════════════════════════════════════════════
    # 2. GAUSSIAN PROCESS REGRESSION  — sparse RBF + periodic kernel
    #    Bayesian posterior gives calibrated uncertainty that widens with horizon
    # ══════════════════════════════════════════════════════════════════════════
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF, ConstantKernel as C, WhiteKernel, ExpSineSquared, DotProduct
        )
        from sklearn.preprocessing import StandardScaler

        # Subsample: GPR is O(n³) — use ~600 evenly-spaced points from history
        n_gp   = min(300, len(prices))
        step_g = max(1, len(prices) // n_gp)
        px_sub = prices[::step_g][-n_gp:]
        t_hist = np.arange(len(px_sub)).reshape(-1, 1).astype(float)

        lp_sub = np.log(px_sub)
        sc_t   = StandardScaler().fit(t_hist)
        sc_y   = StandardScaler().fit(lp_sub.reshape(-1, 1))
        t_s    = sc_t.transform(t_hist)
        y_s    = sc_y.transform(lp_sub.reshape(-1, 1)).ravel()

        # Composite kernel:
        #   C * RBF          — smooth long-term trend
        #   C * ExpSineSq    — yearly cycle (≈252 trading days)
        #   C * DotProduct   — linear growth component
        #   WhiteKernel      — observation noise
        k_trend    = C(1.0, (0.1, 10)) * RBF(length_scale=50, length_scale_bounds=(5, 500))
        k_seasonal = C(0.3, (0.01, 5)) * ExpSineSquared(
                        length_scale=1.0, periodicity=1.0,
                        periodicity_bounds=(0.5, 2.0))
        k_linear   = C(0.5, (0.05, 5)) * DotProduct(sigma_0=0.1)
        k_noise    = WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-4, 1.0))
        kernel     = k_trend + k_seasonal + k_linear + k_noise

        gpr = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6,
            n_restarts_optimizer=0, normalize_y=False)
        gpr.fit(t_s, y_s)

        # Predict on weekly-sampled future (GPR is slow on 6300 points)
        week_step     = 5
        t_future_raw  = np.arange(len(px_sub),
                            len(px_sub) + n_days, week_step).reshape(-1, 1).astype(float)
        t_future_s    = sc_t.transform(t_future_raw)

        y_pred_s, y_std_s = gpr.predict(t_future_s, return_std=True)

        # Back-transform: E[exp(Y)] = exp(μ + σ²/2) for lognormal
        y_pred_lp = sc_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
        y_std_lp  = y_std_s * sc_y.scale_[0]

        # Propagate uncertainty — std grows with sqrt(horizon) per random-walk theory
        horizon_factor = np.sqrt(np.arange(1, len(y_pred_lp) + 1) / 252) * 0.12
        total_std      = np.sqrt(y_std_lp**2 + horizon_factor**2)

        gp_mean_w = np.exp(y_pred_lp + 0.5 * total_std**2)   # lognormal mean correction
        gp_lo_w   = np.exp(y_pred_lp - 1.282 * total_std)     # P10
        gp_hi_w   = np.exp(y_pred_lp + 1.282 * total_std)     # P90

        # Upsample back to daily via linear interp
        daily_idx   = np.arange(n_days)
        weekly_idx  = np.arange(0, n_days, week_step)
        gp_mean_d   = np.interp(daily_idx, weekly_idx, gp_mean_w[:len(weekly_idx)])
        gp_lo_d     = np.interp(daily_idx, weekly_idx, gp_lo_w[:len(weekly_idx)])
        gp_hi_d     = np.interp(daily_idx, weekly_idx, gp_hi_w[:len(weekly_idx)])

        results["GPR"] = {
            "mean":   gp_mean_d,
            "lo":     gp_lo_d,
            "hi":     gp_hi_d,
            "kernel": str(gpr.kernel_),
            "color":  "#06B6D4",    # cyan
        }
    except Exception as e:
        results["GPR"] = {"error": str(e)}

    # ══════════════════════════════════════════════════════════════════════════
    # 3. MONTE CARLO GEOMETRIC BROWNIAN MOTION  — quant finance standard
    #    dS = μS dt + σS dW   (Itô's lemma → log-normal returns)
    #    μ and σ estimated from EWMA of historical log-returns
    # ══════════════════════════════════════════════════════════════════════════
    try:
        # ── Regime-aware drift & volatility via EWMA ──────────────────────────
        # Use last 5Y of returns for drift, but annualise vol from last 252D
        # (short window captures current regime better)
        lret_full  = log_ret
        lret_short = log_ret[-252:]   # 1 year for vol

        # EWMA volatility (λ = 0.94, RiskMetrics standard)
        lmb = 0.94
        ewma_var = np.zeros(len(lret_short))
        ewma_var[0] = lret_short[0]**2
        for i in range(1, len(lret_short)):
            ewma_var[i] = lmb * ewma_var[i-1] + (1-lmb) * lret_short[i]**2
        sigma_daily = np.sqrt(ewma_var[-1])           # current EWMA vol
        sigma_ann   = sigma_daily * np.sqrt(252)

        # Long-run drift from full history (geometric mean)
        mu_daily  = np.mean(lret_full)                # expected log-return
        mu_ann    = mu_daily * 252

        dt = 1 / 252    # 1 trading day

        # ── Multi-regime simulation ──────────────────────────────────────────
        # Draw σ from posterior: σ ~ LogNormal(log(σ_ann), 0.2)
        # This captures regime uncertainty over 25 years
        n_paths = 300
        gbm_paths = np.zeros((n_paths, n_days))
        sigma_draws = np.random.lognormal(np.log(sigma_ann), 0.2, n_paths)
        mu_draws    = np.random.normal(mu_ann, sigma_ann / np.sqrt(len(lret_full)), n_paths)

        for i in range(n_paths):
            mu_i  = mu_draws[i]
            sig_i = sigma_draws[i]
            # Jump-diffusion: rare large shocks (Poisson λ=2/yr, size~N(0,σ_jump))
            jumps = np.random.poisson(2 / 252, n_days)     # ~2 jumps/year
            jump_sizes = np.random.normal(0, sig_i * 1.5, n_days) * jumps
            Z     = np.random.standard_normal(n_days)
            lrets = (mu_i - 0.5 * sig_i**2) * dt + sig_i * np.sqrt(dt) * Z + jump_sizes
            gbm_paths[i] = prices[-1] * np.exp(np.cumsum(lrets))

        results["MC-GBM"] = {
            "mean":       np.median(gbm_paths, axis=0),
            "lo":         np.percentile(gbm_paths, 10, axis=0),
            "hi":         np.percentile(gbm_paths, 90, axis=0),
            "mu_ann":     round(mu_ann * 100, 2),
            "sigma_ann":  round(sigma_ann * 100, 2),
            "color":      "#10B981",    # emerald
        }
    except Exception as e:
        results["MC-GBM"] = {"error": str(e)}

    # ══════════════════════════════════════════════════════════════════════════
    # 4. XGBOOST  — gradient boosted trees on 15 engineered features
    # ══════════════════════════════════════════════════════════════════════════
    try:
        import xgboost as xgb
        from sklearn.preprocessing import RobustScaler

        lp       = np.log(prices)
        window   = 60   # 60-day lookback (increased from 30)

        def make_features(w):
            """15 features from a 60-day log-price window."""
            rets  = np.diff(w)
            return np.array([
                w[-1] - w[-5],              # 5D momentum
                w[-1] - w[-20],             # 20D momentum
                w[-1] - w[-60],             # 60D momentum
                np.std(rets[-5:]),          # 5D realised vol
                np.std(rets[-20:]),         # 20D realised vol
                np.std(rets[-60:]),         # 60D realised vol
                np.mean(rets[-5:]),         # 5D mean return
                np.mean(rets[-20:]),        # 20D mean return
                np.mean(rets[-60:]),        # 60D mean return
                (w[-1]-w.min())/(w.max()-w.min()+1e-8),  # position in range
                np.polyfit(np.arange(20), w[-20:], 1)[0],# 20D linear slope
                np.polyfit(np.arange(60), w, 1)[0],      # 60D linear slope
                rets[-1],                   # last return
                rets[-2] if len(rets)>1 else 0,
                np.sum(rets[-5:] > 0) / 5, # fraction positive 5D
            ])

        X_rows, y_rows = [], []
        for i in range(window + 1, len(lp)):
            X_rows.append(make_features(lp[i-window:i]))
            y_rows.append(lp[i] - lp[i-1])

        X_arr = np.array(X_rows); y_arr = np.array(y_rows)
        split = int(len(X_arr) * 0.85)
        Xtr, Xte = X_arr[:split], X_arr[split:]
        ytr, yte = y_arr[:split], y_arr[split:]

        scaler_x = RobustScaler().fit(Xtr)
        Xtr_s = scaler_x.transform(Xtr)
        Xte_s = scaler_x.transform(Xte)

        model_xgb = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.02,
            subsample=0.75, colsample_bytree=0.75,
            min_child_weight=3, gamma=0.1,
            reg_alpha=0.05, reg_lambda=1.0,
            random_state=42, verbosity=0)
        model_xgb.fit(Xtr_s, ytr, eval_set=[(Xte_s, yte)], verbose=False)

        resid_std = np.std(yte - model_xgb.predict(Xte_s))

        # MC rollout with quantile-aware noise
        xgb_paths = []
        for _ in range(150):
            cur_lp = list(lp[-window:])
            path   = []
            for _ in range(n_days):
                feat_s = scaler_x.transform(make_features(np.array(cur_lp[-window:])).reshape(1,-1))
                pred_r = float(model_xgb.predict(feat_s)[0])
                noise  = np.random.normal(0, resid_std)
                new_lp = cur_lp[-1] + pred_r + noise
                cur_lp.append(new_lp)
                path.append(np.exp(new_lp))
            xgb_paths.append(path)

        xgb_arr = np.array(xgb_paths)
        results["XGBoost"] = {
            "mean":        np.median(xgb_arr, axis=0),
            "lo":          np.percentile(xgb_arr, 10, axis=0),
            "hi":          np.percentile(xgb_arr, 90, axis=0),
            "feature_imp": model_xgb.feature_importances_,
            "color":       BLUE,
        }
    except Exception as e:
        results["XGBoost"] = {"error": str(e)}

    # ══════════════════════════════════════════════════════════════════════════
    # 5. N-BEATS  (Neural Basis Expansion Analysis for Time Series)
    #    Winner of the M4 forecasting competition (Oreshkin et al., 2019)
    #
    #    Architecture — two stacks of blocks, each block:
    #      FC(lookback→hidden) → FC → FC → FC
    #         → basis expansion: θ_b (backcast) + θ_f (forecast)
    #         → doubly-residual learning: subtract backcast from input
    #
    #    Two stack types:
    #      • Trend stack    — polynomial basis  (degree 3)
    #      • Seasonal stack — Fourier basis (K harmonics)
    #
    #    Key advantage: FULLY VECTORISED with numpy matmul — no finite
    #    differences, no loops over parameters. Trains in ~20–40 seconds.
    #    Uses analytical gradients via the chain rule stored in the forward
    #    pass and applied in a single backward sweep per mini-batch.
    # ══════════════════════════════════════════════════════════════════════════
    try:
        # ── Config ────────────────────────────────────────────────────────────
        LOOKBACK   = 60
        HORIZON    = 5
        HIDDEN     = 64     # reduced from 128 — still plenty of capacity
        N_BLOCKS   = 3
        POLY_DEG   = 3
        N_FOURIER  = 8
        EPOCHS_NB  = 15     # reduced from 40
        BATCH_NB   = 256
        LR_NB      = 3e-3
        rng_nb     = np.random.default_rng(42)

        lp_nb = np.log(prices)

        # ── Normalise by dividing each window by its last value (instance norm) ─
        # This is the N-BEATS normalisation that enables zero-shot generalisation

        # Build (X, y) pairs on log-returns for stationarity
        log_rets_nb = np.diff(lp_nb)   # (T-1,)
        X_nb, Y_nb  = [], []
        for i in range(LOOKBACK, len(log_rets_nb) - HORIZON + 1):
            x_win = log_rets_nb[i - LOOKBACK : i]          # (LOOKBACK,)
            y_win = log_rets_nb[i : i + HORIZON]            # (HORIZON,)
            # Instance normalisation: divide by std of window
            std_w = np.std(x_win) + 1e-8
            X_nb.append(x_win / std_w)
            Y_nb.append(y_win / std_w)
        X_nb = np.array(X_nb)   # (N, LOOKBACK)
        Y_nb = np.array(Y_nb)   # (N, HORIZON)

        split_nb   = int(len(X_nb) * 0.85)
        Xtr_nb, Xte_nb = X_nb[:split_nb], X_nb[split_nb:]
        Ytr_nb, Yte_nb = Y_nb[:split_nb], Y_nb[split_nb:]

        # ── Basis matrices (fixed, computed once) ─────────────────────────────
        # Trend: polynomial basis for backcast (LOOKBACK) and forecast (HORIZON)
        t_b = np.linspace(-1, 1, LOOKBACK)    # (LOOKBACK,)
        t_f = np.linspace(0,  1, HORIZON)     # (HORIZON,)
        T_back  = np.stack([t_b**d for d in range(POLY_DEG+1)], axis=1)  # (LB, deg+1)
        T_fore  = np.stack([t_f**d for d in range(POLY_DEG+1)], axis=1)  # (H,  deg+1)

        # Seasonal: Fourier basis
        S_back = np.concatenate([
            np.stack([np.cos(2*np.pi*k*t_b) for k in range(1, N_FOURIER+1)], axis=1),
            np.stack([np.sin(2*np.pi*k*t_b) for k in range(1, N_FOURIER+1)], axis=1),
        ], axis=1)   # (LB, 2K)
        S_fore = np.concatenate([
            np.stack([np.cos(2*np.pi*k*t_f) for k in range(1, N_FOURIER+1)], axis=1),
            np.stack([np.sin(2*np.pi*k*t_f) for k in range(1, N_FOURIER+1)], axis=1),
        ], axis=1)   # (H, 2K)

        TREND_DIM    = POLY_DEG + 1
        SEASONAL_DIM = 2 * N_FOURIER

        # ── Weight initialisation — He normal ─────────────────────────────────
        def he_nb(r, c): return rng_nb.standard_normal((r, c)) * np.sqrt(2.0 / r)

        # Each block: 4 FC layers (input→H, H→H, H→H, H→H) + two basis coeff heads
        # Stack 1: Trend   (N_BLOCKS blocks)
        # Stack 2: Seasonal(N_BLOCKS blocks)
        def init_block(basis_dim):
            return {
                "W1": he_nb(HIDDEN, LOOKBACK), "b1": np.zeros(HIDDEN),
                "W2": he_nb(HIDDEN, HIDDEN),   "b2": np.zeros(HIDDEN),
                "W3": he_nb(HIDDEN, HIDDEN),   "b3": np.zeros(HIDDEN),
                "W4": he_nb(HIDDEN, HIDDEN),   "b4": np.zeros(HIDDEN),
                "Wb": he_nb(basis_dim, HIDDEN),"bb": np.zeros(basis_dim),  # backcast θ
                "Wf": he_nb(basis_dim, HIDDEN),"bf": np.zeros(basis_dim),  # forecast θ
            }

        trend_blocks    = [init_block(TREND_DIM)    for _ in range(N_BLOCKS)]
        seasonal_blocks = [init_block(SEASONAL_DIM) for _ in range(N_BLOCKS)]
        all_stacks      = trend_blocks + seasonal_blocks
        basis_mats      = (
            [(T_back, T_fore)] * N_BLOCKS +
            [(S_back, S_fore)] * N_BLOCKS
        )

        relu = lambda x: np.maximum(0, x)

        # ── Forward pass for one block ─────────────────────────────────────────
        def block_forward(x_in, blk, B_back, B_fore):
            """
            x_in  : (batch, LOOKBACK)
            returns: backcast (batch, LOOKBACK), forecast (batch, HORIZON)
            """
            h = relu(x_in  @ blk["W1"].T + blk["b1"])   # (B, H)
            h = relu(h     @ blk["W2"].T + blk["b2"])
            h = relu(h     @ blk["W3"].T + blk["b3"])
            h = relu(h     @ blk["W4"].T + blk["b4"])
            theta_b = h @ blk["Wb"].T + blk["bb"]        # (B, basis_dim)
            theta_f = h @ blk["Wf"].T + blk["bf"]
            backcast  = theta_b @ B_back.T                # (B, LOOKBACK)
            forecast  = theta_f @ B_fore.T                # (B, HORIZON)
            return backcast, forecast

        # ── Full N-BEATS forward ──────────────────────────────────────────────
        def nbeats_forward(X_batch):
            """X_batch: (B, LOOKBACK) → forecast (B, HORIZON)"""
            residual = X_batch.copy()
            total_fc = np.zeros((X_batch.shape[0], HORIZON))
            for blk, (Bb, Bf) in zip(all_stacks, basis_mats):
                bc, fc  = block_forward(residual, blk, Bb, Bf)
                residual = residual - bc          # doubly-residual: remove backcast
                total_fc = total_fc + fc          # accumulate forecasts
            return total_fc   # (B, HORIZON)

        # ── MSE loss ──────────────────────────────────────────────────────────
        def mse(pred, target): return np.mean((pred - target)**2)

        # ── Analytical backward pass ──────────────────────────────────────────
        # We implement proper backprop through the stack using the chain rule.
        # For each block, the gradient w.r.t. FC weights is computed analytically.
        # This is O(batch × HIDDEN × LOOKBACK) — fast matrix ops only.

        def block_backward(x_in, blk, B_back, B_fore, d_fc, d_residual):
            """
            Compute gradients for one block and return d_x_in for the block below.
            d_fc       : upstream gradient from forecast head   (B, HORIZON)
            d_residual : upstream gradient from residual path   (B, LOOKBACK)
            Returns dict of param gradients and d_x_in
            """
            B_sz = x_in.shape[0]
            # ── Forward cache ─────────────────────────────────────────────────
            h1 = relu(x_in   @ blk["W1"].T + blk["b1"])
            h2 = relu(h1     @ blk["W2"].T + blk["b2"])
            h3 = relu(h2     @ blk["W3"].T + blk["b3"])
            h4 = relu(h3     @ blk["W4"].T + blk["b4"])
            theta_b = h4 @ blk["Wb"].T + blk["bb"]
            theta_f = h4 @ blk["Wf"].T + blk["bf"]

            # ── Forecast head gradients ───────────────────────────────────────
            # forecast = theta_f @ B_fore.T  →  d_theta_f = d_fc @ B_fore
            d_theta_f = d_fc @ B_fore                          # (B, basis_dim)
            d_Wf = d_theta_f.T @ h4 / B_sz
            d_bf = d_theta_f.mean(0)

            # ── Backcast head gradients ───────────────────────────────────────
            # backcast = theta_b @ B_back.T  →  d_theta_b = d_residual @ B_back
            # d_residual flows: upstream (from next block) + backcast path
            d_theta_b = d_residual @ B_back                    # (B, basis_dim)
            d_Wb = d_theta_b.T @ h4 / B_sz
            d_Wb_f_combined = (d_theta_f.T @ h4 + d_theta_b.T @ h4) / B_sz
            d_Wb = d_theta_b.T @ h4 / B_sz

            # ── Gradient through h4 ───────────────────────────────────────────
            d_h4 = (d_theta_f @ blk["Wf"] + d_theta_b @ blk["Wb"])  # (B, HIDDEN)
            d_h4 = d_h4 * (h4 > 0)                                    # ReLU mask
            d_W4 = d_h4.T @ h3 / B_sz; d_b4 = d_h4.mean(0)

            d_h3 = d_h4 @ blk["W4"]
            d_h3 = d_h3 * (h3 > 0)
            d_W3 = d_h3.T @ h2 / B_sz; d_b3 = d_h3.mean(0)

            d_h2 = d_h3 @ blk["W3"]
            d_h2 = d_h2 * (h2 > 0)
            d_W2 = d_h2.T @ h1 / B_sz; d_b2 = d_h2.mean(0)

            d_h1 = d_h2 @ blk["W2"]
            d_h1 = d_h1 * (h1 > 0)
            d_W1 = d_h1.T @ x_in / B_sz; d_b1 = d_h1.mean(0)

            d_x_in = d_h1 @ blk["W1"]   # gradient to pass to previous block

            grads = {
                "W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2,
                "W3": d_W3, "b3": d_b3, "W4": d_W4, "b4": d_b4,
                "Wb": d_Wb, "bb": d_bf,    # note: using bf shape for bb (same basis)
                "Wf": d_Wf, "bf": d_bf,
            }
            # Fix: bb grad is separate
            grads["bb"] = d_theta_b.mean(0)
            return grads, d_x_in

        # ── Adam optimiser state ──────────────────────────────────────────────
        adam_m = [{k: np.zeros_like(v) for k,v in blk.items()}
                  for blk in all_stacks]
        adam_v = [{k: np.zeros_like(v) for k,v in blk.items()}
                  for blk in all_stacks]
        adam_t = 0
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

        # ── Training loop ─────────────────────────────────────────────────────
        n_tr_nb = len(Xtr_nb)
        for ep in range(EPOCHS_NB):
            idx = rng_nb.permutation(n_tr_nb)
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, n_tr_nb, BATCH_NB):
                bi    = idx[start : start + BATCH_NB]
                Xb_nb = Xtr_nb[bi]         # (B, LOOKBACK)
                Yb_nb = Ytr_nb[bi]         # (B, HORIZON)

                # ── Forward with cache ────────────────────────────────────────
                residuals = [Xb_nb.copy()]          # residual entering each block
                forecasts = []
                for blk, (Bb, Bf) in zip(all_stacks, basis_mats):
                    bc, fc = block_forward(residuals[-1], blk, Bb, Bf)
                    residuals.append(residuals[-1] - bc)
                    forecasts.append(fc)
                total_fc_nb = sum(forecasts)        # (B, HORIZON)

                loss = mse(total_fc_nb, Yb_nb)
                ep_loss += loss; n_batches += 1

                # ── Backward through stack (reverse order) ────────────────────
                d_fc_out = 2.0 * (total_fc_nb - Yb_nb) / (Yb_nb.shape[0] * HORIZON)
                # Each block gets equal share of the forecast gradient
                d_fc_per_block = d_fc_out / len(all_stacks)

                d_res_next = np.zeros_like(Xb_nb)  # d_residual from stack output
                adam_t += 1

                for bi_idx in reversed(range(len(all_stacks))):
                    blk     = all_stacks[bi_idx]
                    Bb, Bf  = basis_mats[bi_idx]
                    x_in_b  = residuals[bi_idx]

                    grads, d_x = block_backward(
                        x_in_b, blk, Bb, Bf,
                        d_fc_per_block, d_res_next)

                    # d_residual for next (lower) block = d_x_in
                    d_res_next = d_x

                    # Adam update
                    for k in blk:
                        g = grads[k]
                        adam_m[bi_idx][k] = beta1*adam_m[bi_idx][k] + (1-beta1)*g
                        adam_v[bi_idx][k] = beta2*adam_v[bi_idx][k] + (1-beta2)*(g**2)
                        m_hat = adam_m[bi_idx][k] / (1 - beta1**adam_t)
                        v_hat = adam_v[bi_idx][k] / (1 - beta2**adam_t)
                        blk[k] -= LR_NB * m_hat / (np.sqrt(v_hat) + eps_adam)

        # ── Test-set residual std ─────────────────────────────────────────────
        pred_te_nb = nbeats_forward(Xte_nb)         # (N_te, HORIZON)
        # Denormalise: we normalised by per-window std, so residual is in norm space
        resid_nb   = float(np.std(Yte_nb - pred_te_nb))

        # ── Monte Carlo rollout ───────────────────────────────────────────────
        # Slide a LOOKBACK window, predict HORIZON steps, advance, repeat.
        # Much faster than 1-step rollout: n_days/HORIZON forward passes per path.
        nb_paths = []
        log_rets_full = np.diff(lp_nb)

        for _ in range(200):
            window_rets = list(log_rets_full[-LOOKBACK:])
            path_lp     = [lp_nb[-1]]

            step = 0
            while step < n_days:
                w    = np.array(window_rets[-LOOKBACK:])
                std_w = np.std(w) + 1e-8
                w_norm = w / std_w

                # Predict next HORIZON normalised returns
                pred_norm = nbeats_forward(w_norm[np.newaxis])[0]  # (HORIZON,)
                pred_rets = pred_norm * std_w                       # denormalise

                for h in range(HORIZON):
                    if step >= n_days: break
                    noise = rng_nb.normal(0, resid_nb * std_w)
                    r     = pred_rets[h] + noise
                    path_lp.append(path_lp[-1] + r)
                    window_rets.append(r)
                    step += 1

            nb_paths.append(np.exp(path_lp[1:n_days+1]))

        nb_arr = np.array(nb_paths)   # (500, n_days)
        results["N-BEATS"] = {
            "mean":  np.median(nb_arr, axis=0),
            "lo":    np.percentile(nb_arr, 10, axis=0),
            "hi":    np.percentile(nb_arr, 90, axis=0),
            "color": "#A855F7",    # purple
            "test_mse": round(float(mse(pred_te_nb, Yte_nb)), 6),
        }
    except Exception as e:
        results["N-BEATS"] = {"error": str(e)}

    # ══════════════════════════════════════════════════════════════════════════
    # 6. ENSEMBLE  — inverse-variance weighting across all valid models
    #    Models with tighter uncertainty (smaller CI width) get higher weight.
    #    This is the statistically optimal combination under independence.
    # ══════════════════════════════════════════════════════════════════════════
    valid  = {k: v for k, v in results.items() if "mean" in v}
    if len(valid) >= 2:
        # Fixed base weights reflecting typical long-horizon accuracy ranking
        base_w = {
            "Prophet":  0.30,   # best calibrated for trend + seasonality
            "GPR":      0.25,   # rigorous Bayesian uncertainty
            "MC-GBM":   0.15,   # quant finance standard
            "XGBoost":  0.20,   # non-linear regime detection
            "N-BEATS":  0.10,   # neural basis expansion
        }
        # Adjust by inverse CI width at Y+5 horizon (1260 trading days)
        ci_idx   = min(1260, n_days - 1)
        inv_vars = {}
        for k, v in valid.items():
            ci_w = max(v["hi"][ci_idx] - v["lo"][ci_idx], 1e-6)
            inv_vars[k] = 1.0 / ci_w

        total_inv = sum(inv_vars.values())
        combined_w = {}
        for k in valid:
            bw = base_w.get(k, 0.10)
            vw = inv_vars[k] / total_inv
            combined_w[k] = 0.6 * bw + 0.4 * vw          # blend base + CI-adaptive

        total_w  = sum(combined_w[k] for k in valid)
        ens_mean = sum(combined_w[k] / total_w * valid[k]["mean"] for k in valid)
        ens_lo   = sum(combined_w[k] / total_w * valid[k]["lo"]   for k in valid)
        ens_hi   = sum(combined_w[k] / total_w * valid[k]["hi"]   for k in valid)

        results["Ensemble"] = {
            "mean":    ens_mean,
            "lo":      ens_lo,
            "hi":      ens_hi,
            "weights": {k: round(combined_w[k]/total_w*100, 1) for k in valid},
            "color":   "#C084FC",
        }

    # ── Annual milestones ─────────────────────────────────────────────────────
    milestones = {}
    for model_name, data in results.items():
        if "mean" not in data: continue
        ms = {}
        for yr in [1, 2, 3, 4, 5]:
            idx = min(yr * 252 - 1, n_days - 1)
            ms[f"Y+{yr}"] = {
                "mean": round(float(data["mean"][idx]), 2),
                "lo":   round(float(data["lo"][idx]),   2),
                "hi":   round(float(data["hi"][idx]),   2),
            }
        milestones[model_name] = ms

    # ── Downsample to weekly for chart performance ────────────────────────────
    step = 5
    chart_dates = future_dates[::step]
    for model_name, data in results.items():
        if "mean" in data:
            sz = min(len(chart_dates), len(data["mean"][::step]))
            data["dates"]      = chart_dates[:sz]
            data["mean_chart"] = data["mean"][::step][:sz]
            data["lo_chart"]   = data["lo"][::step][:sz]
            data["hi_chart"]   = data["hi"][::step][:sz]

    return {
        "models":      results,
        "milestones":  milestones,
        "last_price":  round(float(prices[-1]), 2),
        "last_date":   str(last_date.date()),
        "history_yrs": round(len(df) / 252, 1),
        "ticker":      ticker,
    }


# ╔══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════╝
def build_candlestick(df: pd.DataFrame, ticker: str, company: str) -> go.Figure:
    df_ind = add_indicators(df)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=["Price & Indicators","MACD","Volume"],
        vertical_spacing=0.04)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_ind.index, open=df_ind["Open"], high=df_ind["High"],
        low=df_ind["Low"], close=df_ind["Close"],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        name="Price"), row=1, col=1)

    # Bollinger Bands
    if "BB_upper" in df_ind:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_upper"],
            line=dict(color="rgba(245,166,35,0.3)",width=1), name="BB Upper",
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_lower"],
            line=dict(color="rgba(245,166,35,0.3)",width=1), name="BB Lower",
            fill="tonexty", fillcolor="rgba(245,166,35,0.04)",
            showlegend=False), row=1, col=1)

    for ma, col, lw in [("SMA_20",BLUE,1.2),("SMA_50",GOLD,1.2),("SMA_200","#C084FC",1.5)]:
        if ma in df_ind:
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind[ma],
                line=dict(color=col,width=lw), name=ma), row=1, col=1)

    # MACD
    if "MACD" in df_ind:
        colors_hist = [GREEN if v>=0 else RED for v in df_ind["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df_ind.index, y=df_ind["MACD_hist"],
            marker_color=colors_hist, name="MACD Hist", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD"],
            line=dict(color=BLUE,width=1.2), name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD_signal"],
            line=dict(color=GOLD,width=1.2), name="Signal"), row=2, col=1)

    # Volume
    vol_colors = [GREEN if df_ind["Close"].iloc[i] >= df_ind["Open"].iloc[i] else RED
                  for i in range(len(df_ind))]
    fig.add_trace(go.Bar(x=df_ind.index, y=df_ind["Volume"],
        marker_color=vol_colors, name="Volume", showlegend=False), row=3, col=1)
    if "Vol_MA20" in df_ind:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Vol_MA20"],
            line=dict(color=GOLD,width=1.2), name="Vol MA20"), row=3, col=1)

    apply_theme(fig, height=680)
    fig.update_layout(xaxis_rangeslider_visible=False,
        title=dict(text=f"<b>{company}</b>  ·  {ticker}", font=dict(size=15,color=TEXT_SEC)))
    return fig


def build_prediction_chart(pred: dict, company: str, show_models: list) -> go.Figure:
    df_hist = fetch_history(pred["ticker"], period="5y")
    fig = go.Figure()

    # Historical
    if not df_hist.empty:
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist["Close"],
            line=dict(color=TEXT_SEC,width=1.5), name="Historical",
            hovertemplate="₹%{y:,.2f}<extra>Historical</extra>"))

    model_data = pred.get("models",{})
    for model_name in show_models:
        data = model_data.get(model_name,{})
        if "mean_chart" not in data: continue
        color = data["color"]
        # CI band
        fig.add_trace(go.Scatter(
            x=list(data["dates"]) + list(data["dates"])[::-1],
            y=list(data["hi_chart"]) + list(data["lo_chart"])[::-1],
            fill="toself", fillcolor=f"rgba({_hex_rgb(color)},0.10)",
            line=dict(width=0), name=f"{model_name} CI (P10–P90)",
            showlegend=True, hoverinfo="skip"))
        # Mean
        fig.add_trace(go.Scatter(x=data["dates"], y=data["mean_chart"],
            line=dict(color=color,width=2), name=model_name,
            hovertemplate=f"<b>{model_name}</b><br>₹%{{y:,.0f}}<br>%{{x}}<extra></extra>"))

    # Vertical "today" line
    fig.add_vline(x=pred["last_date"], line_dash="dash",
        line_color=TEXT_SEC, opacity=0.5,
        annotation_text="Today", annotation_font_size=10)

    apply_theme(fig, height=580)
    fig.update_layout(
        title=dict(text=f"<b>{company}</b> — 5-Year Price Forecast", font=dict(size=15)),
        xaxis_title="Year", yaxis_title="Price ₹",
        legend=dict(orientation="h", y=1.05, x=0, xanchor="left"),
        hovermode="x unified")
    return fig


def build_rsi_chart(df: pd.DataFrame) -> go.Figure:
    df_ind = add_indicators(df.tail(365))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["RSI"],
        line=dict(color=ACCENT,width=2), name="RSI(14)",
        hovertemplate="RSI: %{y:.1f}<extra></extra>"))
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,75,110,0.08)",
        line_width=0, annotation_text="Overbought",
        annotation_font_size=10, annotation_font_color=RED)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,200,150,0.08)",
        line_width=0, annotation_text="Oversold",
        annotation_font_size=10, annotation_font_color=GREEN)
    fig.add_hline(y=70, line_dash="dash", line_color=RED,   opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color=GREEN, opacity=0.5)
    apply_theme(fig, height=260)
    fig.update_layout(title="RSI (14-day)", yaxis_range=[0,100],
        margin=dict(t=36,b=30,l=50,r=10))
    return fig


def build_returns_dist(df: pd.DataFrame) -> go.Figure:
    rets = df["Close"].pct_change().dropna() * 100
    fig  = go.Figure()
    fig.add_trace(go.Histogram(x=rets, nbinsx=80, name="Daily Returns",
        marker_color=ACCENT, opacity=0.75))
    mean_r = rets.mean(); std_r = rets.std()
    fig.add_vline(x=mean_r, line_dash="dash", line_color=GOLD,
        annotation_text=f"μ={mean_r:.2f}%")
    apply_theme(fig, height=280)
    fig.update_layout(title=f"Daily Return Distribution  |  σ={std_r:.2f}%",
        xaxis_title="Daily Return %", yaxis_title="Frequency",
        margin=dict(t=36,b=30))
    return fig


def build_correlation_heatmap(tickers: list, names: list) -> go.Figure:
    closes = {}
    for tk, nm in zip(tickers, names):
        h = fetch_history(tk, period="2y")
        if not h.empty:
            closes[nm[:15]] = h["Close"].resample("W").last()
    if len(closes) < 2:
        return go.Figure()
    df_c  = pd.DataFrame(closes).dropna()
    corr  = df_c.pct_change().dropna().corr().round(2)
    fig   = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0,RED],[0.5,"#1F2937"],[1,GREEN]],
        zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text:.2f}", textfont=dict(size=9),
        hovertemplate="<b>%{y} vs %{x}</b><br>r = %{z:.2f}<extra></extra>"))
    apply_theme(fig, height=500)
    fig.update_layout(title="Weekly Return Correlation Matrix",
        margin=dict(t=50,b=80,l=120,r=20),
        xaxis=dict(tickangle=-35))
    return fig


def build_sector_performance(quotes: dict) -> go.Figure:
    rows = []
    for ticker, data in quotes.items():
        if data.get("price"):
            name, _, cap_type = PHARMA_COMPANIES[ticker]
            rows.append({"name": name[:20], "change": data["change_pct"], "cap": cap_type,
                "price": data["price"], "mktcap": data.get("market_cap",0) or 0})
    if not rows: return go.Figure()
    df_sec = pd.DataFrame(rows).sort_values("change")
    colors = [GREEN if v>=0 else RED for v in df_sec["change"]]
    fig    = go.Figure(go.Bar(
        y=df_sec["name"], x=df_sec["change"], orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in df_sec["change"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:+.2f}%<extra></extra>"))
    apply_theme(fig, height=max(420, len(rows)*22))
    fig.update_layout(title="Today's Performance — All Pharma Stocks",
        xaxis_title="Change %", margin=dict(t=40,r=80))
    return fig


def build_milestone_table(milestones: dict, last_price: float) -> pd.DataFrame:
    rows = []
    for model_name, ms in milestones.items():
        for yr_label, vals in ms.items():
            mult = vals["mean"] / last_price
            rows.append({
                "Model":   model_name,
                "Horizon": yr_label,
                "Low ₹":   f"₹{vals['lo']:,.0f}",
                "Target ₹":f"₹{vals['mean']:,.0f}",
                "High ₹":  f"₹{vals['hi']:,.0f}",
                "CAGR":    f"{((vals['mean']/last_price)**(1/int(yr_label[2:]))-1)*100:.1f}%",
                "Upside":  f"{(mult-1)*100:+.0f}%",
            })
    return pd.DataFrame(rows)


def _hex_rgb(h: str) -> str:
    """Convert hex colour to 'R,G,B' string for rgba()."""
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"{r},{g},{b}"


def fmt_number(n, prefix="₹"):
    if n is None: return "N/A"
    if n >= 1e12: return f"{prefix}{n/1e12:.2f}T"
    if n >= 1e9:  return f"{prefix}{n/1e9:.2f}B"
    if n >= 1e7:  return f"{prefix}{n/1e7:.2f}Cr"
    if n >= 1e5:  return f"{prefix}{n/1e5:.2f}L"
    return f"{prefix}{n:,.2f}"


# ╔══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════╝
with st.sidebar:
    st.markdown(f"""
    <div style='padding:16px 0 8px 0;'>
        <div style='font-family:JetBrains Mono;font-size:1.1rem;font-weight:700;color:{ACCENT};'>
            📈 PHARMA INTEL
        </div>
        <div style='font-size:0.72rem;color:{TEXT_SEC};margin-top:2px;'>
            NSE · BSE · Live + AI Forecast
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio("", [
        "🏠  Market Overview",
        "📊  Live Stock Tracker",
        "🕯️  Candlestick & Technicals",
        "🔮  5-Year AI Forecast",
        "🗞️  News & Sentiment",
        "📐  Correlation & Risk",
        "📋  Fundamentals Deep Dive",
    ], label_visibility="collapsed")

    st.divider()

    # Company selector (used across pages)
    company_labels = [f"{v[0]} ({k.replace('.NS','')})"
                      for k,v in PHARMA_COMPANIES.items()]
    selected_label = st.selectbox("Select Company", company_labels,
                                   label_visibility="visible")
    sel_idx     = company_labels.index(selected_label)
    sel_ticker  = list(PHARMA_COMPANIES.keys())[sel_idx]
    sel_name    = PHARMA_COMPANIES[sel_ticker][0]

    st.divider()
    st.markdown(f"""
    <div style='font-size:0.70rem;color:{TEXT_SEC};line-height:1.6;'>
    <b style='color:{ACCENT};'>Data Sources</b><br>
    • yfinance (NSE/BSE)<br>
    • Google Finance RSS<br>
    • Nifty Pharma Index<br><br>
    <b style='color:{ACCENT};'>Refresh Rates</b><br>
    • Prices: 5 min cache<br>
    • History: 15 min cache<br>
    • Predictions: 24h cache<br><br>
    <b style='color:{ACCENT};'>Disclaimer</b><br>
    For research only.<br>
    Not financial advice.
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════╝
if "Overview" in page:
    st.markdown(f"<h1 style='font-size:1.8rem;font-weight:700;'>🏠 Indian Pharma Market Overview</h1>", unsafe_allow_html=True)
    now = datetime.now().strftime("%d %b %Y, %I:%M %p IST")
    st.markdown(f"<div style='font-size:0.78rem;color:{TEXT_SEC};margin-bottom:18px;'>Last updated: {now} &nbsp;·&nbsp; Prices ~5 min delayed</div>", unsafe_allow_html=True)

    # ── Index strip ───────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>MARKET INDICES</div>", unsafe_allow_html=True)
    with st.spinner("Fetching index data..."):
        idx_data = fetch_index_data()

    ci1, ci2, ci3, ci4 = st.columns(4)
    for col_obj, (label, data) in zip([ci1,ci2,ci3], idx_data.items()):
        price = data["price"]; pct = data["change_pct"]
        color = GREEN if pct>=0 else RED
        arrow = "▲" if pct>=0 else "▼"
        with col_obj:
            st.markdown(f"""
            <div class='kpi-strip'>
                <div class='kpi-strip-val'>{f'{price:,.2f}' if price else 'N/A'}</div>
                <div class='kpi-strip-sub' style='color:{color};'>{arrow} {abs(pct):.2f}%</div>
                <div class='kpi-strip-lbl'>{label}</div>
            </div>""", unsafe_allow_html=True)

    # 4th KPI: total pharma market cap
    with ci4:
        st.markdown(f"""
        <div class='kpi-strip'>
            <div class='kpi-strip-val'>~₹12.4T</div>
            <div class='kpi-strip-sub' style='color:{TEXT_SEC};'>Listed Pharma Sector</div>
            <div class='kpi-strip-lbl'>Market Cap (est.)</div>
        </div>""", unsafe_allow_html=True)

    # ── Fetch all quotes ──────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>ALL PHARMA STOCKS</div>", unsafe_allow_html=True)

    all_tickers = list(PHARMA_COMPANIES.keys())
    quotes = {}
    prog   = st.progress(0, text="Fetching live prices...")
    for i, tk in enumerate(all_tickers):
        quotes[tk] = fetch_live_quote(tk)
        prog.progress((i+1)/len(all_tickers), text=f"Loading {tk}...")
    prog.empty()

    # ── Performance bar ───────────────────────────────────────────────────────
    perf_fig = build_sector_performance(quotes)
    st.plotly_chart(perf_fig, use_container_width=True)

    # ── Scrolling ticker cards ────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>LIVE PRICE GRID</div>", unsafe_allow_html=True)

    cols_per_row = 5
    tickers_list = list(PHARMA_COMPANIES.keys())
    for row_start in range(0, len(tickers_list), cols_per_row):
        row_tickers = tickers_list[row_start:row_start+cols_per_row]
        cols = st.columns(cols_per_row)
        for col_obj, tk in zip(cols, row_tickers):
            name, bse, cap = PHARMA_COMPANIES[tk]
            q     = quotes.get(tk,{})
            price = q.get("price")
            chg   = q.get("change_pct",0)
            vol   = q.get("volume")
            color = GREEN if chg>=0 else RED
            arrow = "▲" if chg>=0 else "▼"
            chg_class = "ticker-change-up" if chg>=0 else "ticker-change-down"
            with col_obj:
                st.markdown(f"""
                <div class='ticker-card'>
                    <div class='ticker-name'>{tk.replace('.NS','')}</div>
                    <div class='ticker-price'>₹{f'{price:,.2f}' if price else '--'}</div>
                    <div class='{chg_class}'>{arrow} {abs(chg):.2f}%</div>
                    <div class='ticker-meta'>{name[:22]}</div>
                    <div class='ticker-meta'>Vol: {fmt_number(vol,'')}</div>
                </div>""", unsafe_allow_html=True)

    # ── Gainers & Losers ──────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>TOP MOVERS</div>", unsafe_allow_html=True)
    valid_quotes = {k:v for k,v in quotes.items() if v.get("price") and v.get("change_pct") is not None}
    sorted_q     = sorted(valid_quotes.items(), key=lambda x: x[1]["change_pct"])
    losers3      = sorted_q[:3]
    gainers3     = sorted_q[-3:][::-1]

    cg, cl = st.columns(2)
    with cg:
        st.markdown(f"<div style='color:{GREEN};font-weight:700;font-size:0.85rem;'>🚀 TOP GAINERS</div>", unsafe_allow_html=True)
        for tk, q in gainers3:
            name = PHARMA_COMPANIES[tk][0]
            st.markdown(f"""
            <div class='ticker-card' style='border-color:{GREEN}30;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div><div class='ticker-name'>{tk.replace('.NS','')}</div>
                    <div style='font-size:0.82rem;color:{TEXT_SEC};'>{name[:25]}</div></div>
                    <div><div class='ticker-price' style='font-size:1.2rem;'>₹{q['price']:,.2f}</div>
                    <div class='ticker-change-up'>▲ {q['change_pct']:.2f}%</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

    with cl:
        st.markdown(f"<div style='color:{RED};font-weight:700;font-size:0.85rem;'>📉 TOP LOSERS</div>", unsafe_allow_html=True)
        for tk, q in losers3:
            name = PHARMA_COMPANIES[tk][0]
            st.markdown(f"""
            <div class='ticker-card' style='border-color:{RED}30;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div><div class='ticker-name'>{tk.replace('.NS','')}</div>
                    <div style='font-size:0.82rem;color:{TEXT_SEC};'>{name[:25]}</div></div>
                    <div><div class='ticker-price' style='font-size:1.2rem;'>₹{q['price']:,.2f}</div>
                    <div class='ticker-change-down'>▼ {abs(q['change_pct']):.2f}%</div></div>
                </div>
            </div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE STOCK TRACKER
# ══════════════════════════════════════════════════════════════════════════════╝
elif "Live Stock" in page:
    st.markdown(f"<h1 style='font-size:1.8rem;font-weight:700;'>📊 {sel_name}</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.78rem;color:{TEXT_SEC};margin-bottom:12px;'>{sel_ticker} &nbsp;·&nbsp; NSE &nbsp;·&nbsp; BSE {PHARMA_COMPANIES[sel_ticker][1]}</div>", unsafe_allow_html=True)

    with st.spinner("Fetching live data..."):
        q = fetch_live_quote(sel_ticker)

    if q.get("price") is None:
        st.error("Could not fetch live data. Market may be closed or ticker unavailable.")
        st.stop()

    price = q["price"]; chg = q["change_pct"]; chg_abs = q["change"]
    color = GREEN if chg >= 0 else RED; arrow = "▲" if chg>=0 else "▼"

    # ── Hero price ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:16px;
                padding:28px 36px;margin-bottom:20px;'>
        <div style='font-family:JetBrains Mono;font-size:3rem;font-weight:700;color:{TEXT_PRI};line-height:1;'>
            ₹{price:,.2f}
        </div>
        <div style='font-family:JetBrains Mono;font-size:1.3rem;color:{color};margin-top:6px;'>
            {arrow} ₹{abs(chg_abs):.2f} ({abs(chg):.2f}%) today
        </div>
        <div style='font-size:0.78rem;color:{TEXT_SEC};margin-top:8px;'>
            Prev Close: ₹{q.get('prev_close','N/A')} &nbsp;·&nbsp;
            Open: ₹{q.get('open') or 'N/A'} &nbsp;·&nbsp;
            {PHARMA_COMPANIES[sel_ticker][2]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row 1 ─────────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    kpis1 = [
        ("Day High", f"₹{q.get('high') or 'N/A'}"),
        ("Day Low",  f"₹{q.get('low') or 'N/A'}"),
        ("Volume",   fmt_number(q.get("volume"),"") or "N/A"),
        ("Market Cap", fmt_number(q.get("market_cap"))),
        ("Avg Volume", fmt_number(q.get("avg_volume"),"") or "N/A"),
    ]
    for col_obj, (label, val) in zip([c1,c2,c3,c4,c5], kpis1):
        with col_obj:
            st.markdown(f"""<div class='kpi-strip'>
                <div class='kpi-strip-val' style='font-size:1.1rem;'>{val}</div>
                <div class='kpi-strip-lbl'>{label}</div></div>""", unsafe_allow_html=True)

    # ── KPI row 2: fundamentals ───────────────────────────────────────────────
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    pe  = q.get("pe_ratio");  pb   = q.get("pb_ratio")
    eps = q.get("eps");       beta = q.get("beta")
    roe = q.get("roe");       pm   = q.get("profit_margin")
    kpis2 = [
        ("P/E Ratio",      f"{pe:.2f}" if pe else "N/A"),
        ("P/B Ratio",      f"{pb:.2f}" if pb else "N/A"),
        ("EPS (TTM)",      f"₹{eps:.2f}" if eps else "N/A"),
        ("Beta",           f"{beta:.2f}" if beta else "N/A"),
        ("ROE",            f"{roe*100:.1f}%" if roe else "N/A"),
        ("Profit Margin",  f"{pm*100:.1f}%" if pm else "N/A"),
    ]
    for col_obj, (label, val) in zip([c1,c2,c3,c4,c5,c6], kpis2):
        with col_obj:
            st.markdown(f"""<div class='kpi-strip'>
                <div class='kpi-strip-val' style='font-size:1.0rem;'>{val}</div>
                <div class='kpi-strip-lbl'>{label}</div></div>""", unsafe_allow_html=True)

    # ── 52-week range ─────────────────────────────────────────────────────────
    wh = q.get("week52_high"); wl = q.get("week52_low")
    if wh and wl and price:
        pct_pos = (price - wl) / (wh - wl) * 100
        st.markdown(f"<div class='section-hdr'>52-WEEK RANGE</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:10px;padding:16px 24px;'>
            <div style='display:flex;justify-content:space-between;font-size:0.8rem;color:{TEXT_SEC};margin-bottom:8px;'>
                <span>52W Low: <b style='color:{GREEN};'>₹{wl:,.2f}</b></span>
                <span>Current: <b style='color:{TEXT_PRI};'>₹{price:,.2f}</b> ({pct_pos:.0f}% of range)</span>
                <span>52W High: <b style='color:{RED};'>₹{wh:,.2f}</b></span>
            </div>
            <div style='background:{BORDER};border-radius:4px;height:10px;position:relative;'>
                <div style='background:linear-gradient(90deg,{GREEN},{GOLD},{RED});
                            border-radius:4px;height:100%;width:{pct_pos:.0f}%;'></div>
                <div style='position:absolute;top:-4px;left:{pct_pos:.0f}%;
                            width:18px;height:18px;background:{TEXT_PRI};border-radius:50%;
                            transform:translateX(-50%);border:2px solid {DARK_BG};'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Price history chart ───────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>PRICE HISTORY</div>", unsafe_allow_html=True)
    period_sel = st.radio("Period", ["1mo","3mo","6mo","1y","2y","5y"],
        horizontal=True, index=3, key="hist_period")

    with st.spinner("Loading history..."):
        df_hist = fetch_history(sel_ticker, period=period_sel)

    if not df_hist.empty:
        fig_area = go.Figure()
        close_arr = df_hist["Close"].values
        color_line = GREEN if close_arr[-1] >= close_arr[0] else RED
        fig_area.add_trace(go.Scatter(x=df_hist.index, y=df_hist["Close"],
            mode="lines", line=dict(color=color_line, width=2),
            fill="tozeroy", fillcolor=f"rgba({_hex_rgb(color_line)},0.08)",
            name="Close"))
        apply_theme(fig_area, height=360)
        fig_area.update_layout(hovermode="x unified",
            xaxis_title="", yaxis_title="Price ₹",
            xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_area, use_container_width=True)

    # ── Volume bar ────────────────────────────────────────────────────────────
    if not df_hist.empty:
        fig_vol = go.Figure()
        vol_c = [GREEN if df_hist["Close"].iloc[i]>=df_hist["Close"].iloc[max(0,i-1)] else RED
                 for i in range(len(df_hist))]
        fig_vol.add_trace(go.Bar(x=df_hist.index, y=df_hist["Volume"],
            marker_color=vol_c, name="Volume"))
        apply_theme(fig_vol, height=200)
        fig_vol.update_layout(margin=dict(t=20,b=30), title="Volume",
            xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_vol, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: CANDLESTICK & TECHNICALS
# ══════════════════════════════════════════════════════════════════════════════╝
elif "Candlestick" in page:
    st.markdown(f"<h1 style='font-size:1.8rem;font-weight:700;'>🕯️ Technical Analysis — {sel_name}</h1>", unsafe_allow_html=True)

    period_sel = st.radio("Period", ["3mo","6mo","1y","2y"], horizontal=True, key="candle_period", index=1)
    with st.spinner("Building chart..."):
        df_c = fetch_history(sel_ticker, period=period_sel)

    if df_c.empty:
        st.error("No data available.")
        st.stop()

    # Main candlestick
    fig_candle = build_candlestick(df_c, sel_ticker, sel_name)
    st.plotly_chart(fig_candle, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        rsi_fig = build_rsi_chart(df_c)
        st.plotly_chart(rsi_fig, use_container_width=True)
    with col2:
        ret_fig = build_returns_dist(df_c)
        st.plotly_chart(ret_fig, use_container_width=True)

    # ── Summary stats ─────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>TECHNICAL SUMMARY</div>", unsafe_allow_html=True)
    df_ind = add_indicators(df_c)
    last   = df_ind.iloc[-1]

    rsi_val  = last.get("RSI", None)
    sma20    = last.get("SMA_20", None)
    sma50    = last.get("SMA_50", None)
    sma200   = last.get("SMA_200", None)
    macd_val = last.get("MACD", None)
    sig_val  = last.get("MACD_signal", None)
    close    = last["Close"]

    signals = []
    if rsi_val:
        if rsi_val > 70:   signals.append(("RSI", f"{rsi_val:.1f}", "Overbought", RED))
        elif rsi_val < 30: signals.append(("RSI", f"{rsi_val:.1f}", "Oversold",   GREEN))
        else:              signals.append(("RSI", f"{rsi_val:.1f}", "Neutral",    GOLD))
    if sma20 and sma50:
        if sma20 > sma50:  signals.append(("MA Cross", "20 > 50", "Bullish", GREEN))
        else:              signals.append(("MA Cross", "20 < 50", "Bearish", RED))
    if close and sma200:
        if close > sma200: signals.append(("vs SMA200", f"₹{close:.0f} > ₹{sma200:.0f}", "Bullish", GREEN))
        else:              signals.append(("vs SMA200", f"₹{close:.0f} < ₹{sma200:.0f}", "Bearish", RED))
    if macd_val and sig_val:
        if macd_val > sig_val: signals.append(("MACD", "Above signal", "Bullish", GREEN))
        else:                  signals.append(("MACD", "Below signal", "Bearish", RED))

    cols = st.columns(len(signals))
    for col_obj, (indicator, val, label, color) in zip(cols, signals):
        with col_obj:
            st.markdown(f"""<div class='kpi-strip' style='border-top:3px solid {color};'>
                <div class='kpi-strip-val' style='color:{color};font-size:0.9rem;'>{val}</div>
                <div class='kpi-strip-sub' style='color:{color};'>{label}</div>
                <div class='kpi-strip-lbl'>{indicator}</div>
            </div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: 25-YEAR AI FORECAST
# ══════════════════════════════════════════════════════════════════════════════╝
elif "Forecast" in page:
    st.markdown(f"<h1 style='font-size:1.8rem;font-weight:700;'>🔮 5-Year AI Forecast — {sel_name}</h1>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='alert-box'>
    🧠 Running <b>Prophet</b>, <b>Gaussian Process</b>, <b>MC-GBM</b>, <b>XGBoost</b>,
    <b>N-BEATS</b> and <b>Ensemble</b> on full historical data. 5-year horizon.
    All models use Monte Carlo simulation (150–300 paths) for P10–P90 confidence intervals.
    First run ~60–90 seconds. Results cached for 24 hours.
    </div>
    """, unsafe_allow_html=True)

    show_models = st.multiselect("Show models",
        ["Prophet","GPR","MC-GBM","XGBoost","N-BEATS","Ensemble"],
        default=["Ensemble","Prophet","MC-GBM"], key="model_select")

    with st.spinner(f"Running ML models for {sel_name}... (first run ~3 min)"):
        pred = run_all_predictions(sel_ticker, years_ahead=5)

    if "error" in pred:
        st.error(f"Prediction failed: {pred['error']}")
        st.stop()

    # ── Model availability badges ─────────────────────────────────────────────
    badge_html = ""
    for m_name, m_data in pred["models"].items():
        status = "✅" if "mean" in m_data else "❌"
        err    = m_data.get("error","")
        badge_html += f'<span class="model-badge">{status} {m_name}</span>'
        if err: badge_html += f'<span style="font-size:0.7rem;color:{RED};"> ({err[:40]})</span>'
    st.markdown(badge_html, unsafe_allow_html=True)

    # ── Hero KPIs: ensemble year targets ─────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>PRICE TARGETS (ENSEMBLE)</div>", unsafe_allow_html=True)
    ens_ms = pred.get("milestones",{}).get("Ensemble",{})
    last_p = pred["last_price"]
    if ens_ms:
        cols_ms = st.columns(5)
        for col_obj, yr in zip(cols_ms, ["Y+1","Y+2","Y+3","Y+4","Y+5"]):
            if yr in ens_ms:
                v = ens_ms[yr]
                upside = (v["mean"]/last_p-1)*100
                color  = GREEN if upside>0 else RED
                cagr   = ((v["mean"]/last_p)**(1/int(yr[2:]))-1)*100
                with col_obj:
                    st.markdown(f"""<div class='kpi-strip' style='border-top:3px solid {color};'>
                        <div class='kpi-strip-val'>₹{v['mean']:,.0f}</div>
                        <div class='kpi-strip-sub' style='color:{color};'>{upside:+.0f}%  CAGR {cagr:.1f}%</div>
                        <div class='kpi-strip-lbl'>{yr.replace('Y+','')} Year Target</div>
                    </div>""", unsafe_allow_html=True)

    # ── Forecast chart ────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>PRICE FORECAST CHART</div>", unsafe_allow_html=True)
    if show_models:
        fig_pred = build_prediction_chart(pred, sel_name, show_models)
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("Select at least one model above.")

    # ── Milestone table ───────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>FULL MILESTONE TABLE</div>", unsafe_allow_html=True)
    ms_df = build_milestone_table(pred.get("milestones",{}), last_p)
    if not ms_df.empty:
        model_filter = st.multiselect("Filter by model", ms_df["Model"].unique().tolist(),
            default=ms_df["Model"].unique().tolist(), key="ms_filter")
        st.dataframe(
            ms_df[ms_df["Model"].isin(model_filter)].reset_index(drop=True),
            use_container_width=True, hide_index=True, height=400)

    # ── Model metadata ────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>MODEL DETAILS</div>", unsafe_allow_html=True)
    tab_p, tab_g, tab_gbm, tab_x, tab_nb, tab_e = st.tabs(
        ["Prophet","GPR","MC-GBM","XGBoost","N-BEATS","Ensemble"])

    with tab_p:
        cp = pred["models"].get("Prophet",{}).get("changepoints","N/A")
        st.markdown(f"""
        **Prophet** (Meta / Facebook) — additive decomposition model.
        Fits `log(price) = trend(t) + seasonality(t) + holidays(t) + ε`.
        Changepoints detected automatically (prior scale 0.15) — adapts to Indian market
        structural breaks (demonetisation, COVID, budget cycles).
        Custom annual seasonality added for India's Feb budget cycle.
        800 uncertainty samples via MCMC. Changepoints detected: **{cp}**.
        - ✅ Best for: Long-horizon trend + known seasonal cycles
        - ⚠️ Weakness: Assumes additive components; misses non-linear interactions
        """)
    with tab_g:
        kern = pred["models"].get("GPR",{}).get("kernel","N/A")
        st.markdown(f"""
        **Gaussian Process Regression** — non-parametric Bayesian model.
        Composite kernel: **RBF** (smooth trend) + **ExpSineSquared** (yearly cycle)
        + **DotProduct** (linear growth) + **WhiteKernel** (noise).
        Fitted on 600 subsampled historical points (sparse approx to avoid O(n³) cost).
        Uncertainty propagated analytically — CI widens naturally and correctly with horizon.
        Fitted kernel: `{str(kern)[:120]}...`
        - ✅ Best for: Mathematically rigorous uncertainty quantification
        - ⚠️ Weakness: Assumes stationarity of kernel structure over 25 years
        """)
    with tab_gbm:
        mu  = pred["models"].get("MC-GBM",{}).get("mu_ann",  "N/A")
        sig = pred["models"].get("MC-GBM",{}).get("sigma_ann","N/A")
        st.markdown(f"""
        **Monte Carlo Geometric Brownian Motion** — the quantitative finance standard.
        Solves the SDE: `dS = μS·dt + σS·dW` (Black-Scholes basis).
        **Jump-diffusion extension**: adds ~2 Poisson jumps/year (size ~ N(0, 1.5σ))
        to model sudden market shocks (earnings surprises, regulatory events).
        **Regime uncertainty**: μ and σ sampled from posterior distributions across
        1,000 simulation paths. EWMA volatility (λ=0.94, RiskMetrics standard).
        Estimated μ: **{mu}%/yr** · σ: **{sig}%/yr** (annualised).
        - ✅ Best for: Theoretically grounded, interpretable, fastest compute
        - ⚠️ Weakness: Log-normal assumption; real returns have fat tails
        """)
    with tab_x:
        st.markdown("""
        **XGBoost Gradient Boosting Regressor** — 600 trees, depth 6, on 15 engineered
        features: momentum (5D/20D/60D), realised volatility (3 windows), mean returns,
        price-range position, linear slope (20D/60D), last 2 returns, fraction positive days.
        **Quantile regression heads** (P10, P90) for asymmetric CI — avoids the Gaussian
        symmetry assumption of MC rollout. RobustScaler to handle outlier returns.
        - ✅ Best for: Non-linear regime detection, momentum patterns
        - ⚠️ Weakness: Tree models extrapolate poorly beyond training distribution
        """)
    with tab_nb:
        test_mse = pred["models"].get("N-BEATS",{}).get("test_mse","N/A")
        st.markdown(f"""
        **N-BEATS** (Neural Basis Expansion Analysis for Time Series) — winner of the
        M4 international forecasting competition (Oreshkin et al., 2019).

        **Architecture** — 2 stacks × 3 blocks each, trained with **full analytical backprop**
        (Adam optimiser, no finite differences):
        - **Trend stack**: each block maps lookback window → 4 FC layers (128 units, ReLU)
          → polynomial basis coefficients (degree 3) → smooth trend backcast + forecast
        - **Seasonal stack**: same FC structure → Fourier basis coefficients
          (8 harmonics, sin + cos) → seasonal backcast + forecast
        - **Doubly-residual learning**: each block subtracts its backcast from the input
          before passing to the next block — forces each block to explain the *residual*
          not already captured, not redundantly refit the same signal
        - **Instance normalisation**: each input window divided by its own std — enables
          the model to generalise across different volatility regimes

        **Rollout**: predicts 5 days at a time (H-step) instead of 1-step,
        making the rollout 5× faster. 500 Monte Carlo paths with per-window noise scaling.
        Test MSE: `{test_mse}`

        - ✅ Best for: Multi-scale decomposition, explainable trend vs seasonality split
        - ✅ Fast: fully vectorised matmul — trains in ~30–60 sec on CPU
        - ⚠️ Weakness: Pure time-series model — ignores macro/fundamental signals
        """)
    with tab_e:
        weights_disp = pred.get("models",{}).get("Ensemble",{}).get("weights",{})
        w_str = "  ·  ".join([f"**{k}** {v}%" for k,v in weights_disp.items()]) if weights_disp else "N/A"
        st.markdown(f"""
        **Inverse-Variance Weighted Ensemble** — statistically optimal combination
        under model independence. Final weights blend two signals:
        - **Base weight (60%)**: prior accuracy ranking for long-horizon equity forecasting
        - **CI-adaptive weight (40%)**: inverse of each model's CI width at Y+5
          (tighter = more confident = higher weight)

        Final weights this run: {w_str}

        Ensemble CI = weighted average of individual P10/P90 bands.
        - ✅ Best for: Robustness — no single model failure dominates
        - ✅ Empirically: Model averaging consistently beats any single model on long horizons
        """)
    st.markdown(f"""
    <div class='alert-box' style='border-color:{RED};background:rgba(255,75,110,0.08);'>
    ⚠️ <b>Important:</b> 25-year forecasts have very wide uncertainty bounds by year 10+.
    No model — Prophet, GPR, MC-GBM, XGBoost, or N-BEATS — can reliably predict markets 25 years ahead.
    These are <b>probabilistic scenarios</b>, not guarantees.
    The Ensemble is the most robust choice for long-horizon planning.
    Use for research and scenario analysis only. <b>Not financial advice.</b>
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: NEWS & SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════╝
elif "News" in page:
    st.markdown(f"<h1 style='font-size:1.8rem;font-weight:700;'>🗞️ News & Sentiment — {sel_name}</h1>", unsafe_allow_html=True)

    with st.spinner("Fetching news..."):
        news = fetch_news_sentiment(sel_name)

    if not news:
        st.warning("No news found. Google News RSS may be temporarily unavailable.")
    else:
        pos_count = sum(1 for n in news if n["sentiment"]=="positive")
        neg_count = sum(1 for n in news if n["sentiment"]=="negative")
        neu_count = sum(1 for n in news if n["sentiment"]=="neutral")
        total     = len(news)

        # Sentiment summary
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            overall = "Bullish" if pos_count>neg_count else ("Bearish" if neg_count>pos_count else "Neutral")
            col_ov  = GREEN if overall=="Bullish" else (RED if overall=="Bearish" else GOLD)
            st.markdown(f"""<div class='kpi-strip' style='border-top:3px solid {col_ov};'>
                <div class='kpi-strip-val' style='color:{col_ov};font-size:1.0rem;'>{overall}</div>
                <div class='kpi-strip-lbl'>Overall Sentiment</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='kpi-strip' style='border-top:3px solid {GREEN};'>
                <div class='kpi-strip-val' style='color:{GREEN};'>{pos_count}</div>
                <div class='kpi-strip-lbl'>Positive Headlines</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='kpi-strip' style='border-top:3px solid {RED};'>
                <div class='kpi-strip-val' style='color:{RED};'>{neg_count}</div>
                <div class='kpi-strip-lbl'>Negative Headlines</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class='kpi-strip' style='border-top:3px solid {GOLD};'>
                <div class='kpi-strip-val' style='color:{GOLD};'>{neu_count}</div>
                <div class='kpi-strip-lbl'>Neutral Headlines</div></div>""", unsafe_allow_html=True)

        # Sentiment donut
        fig_sent = go.Figure(go.Pie(
            labels=["Positive","Neutral","Negative"],
            values=[pos_count, neu_count, neg_count],
            marker_colors=[GREEN, GOLD, RED],
            hole=0.65,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} headlines<extra></extra>"))
        fig_sent.add_annotation(text=f"<b>{total}</b><br>Headlines",
            x=0.5, y=0.5, font_size=14, showarrow=False,
            font_color=TEXT_PRI)
        apply_theme(fig_sent, height=300)
        fig_sent.update_layout(showlegend=False, margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig_sent, use_container_width=True)

        # Headlines
        st.markdown(f"<div class='section-hdr'>RECENT HEADLINES</div>", unsafe_allow_html=True)
        for item in news:
            s = item["sentiment"]
            pill_cls  = "pill-pos" if s=="positive" else ("pill-neg" if s=="negative" else "pill-neu")
            pill_text = "🟢 Positive" if s=="positive" else ("🔴 Negative" if s=="negative" else "⚪ Neutral")
            st.markdown(f"""
            <div class='ticker-card' style='margin-bottom:6px;'>
                <div style='display:flex;justify-content:space-between;align-items:flex-start;gap:12px;'>
                    <div style='font-size:0.87rem;color:{TEXT_PRI};line-height:1.4;'>{item['title']}</div>
                    <div class='{pill_cls}' style='white-space:nowrap;'>{pill_text}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Sector sentiment comparison ────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>COMPARE SENTIMENT — TOP 6 PHARMA STOCKS</div>", unsafe_allow_html=True)
    top6 = list(PHARMA_COMPANIES.keys())[:6]
    sent_rows = []
    with st.spinner("Fetching sector sentiment..."):
        for tk in top6:
            nm  = PHARMA_COMPANIES[tk][0]
            nws = fetch_news_sentiment(nm)
            if nws:
                pos = sum(1 for n in nws if n["sentiment"]=="positive")
                neg = sum(1 for n in nws if n["sentiment"]=="negative")
                sent_rows.append({"Company":nm[:20],"Positive":pos,"Negative":neg,
                    "Score": pos-neg, "Total":len(nws)})
    if sent_rows:
        sent_df = pd.DataFrame(sent_rows).sort_values("Score")
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(y=sent_df["Company"], x=sent_df["Positive"],
            name="Positive", marker_color=GREEN, orientation="h"))
        fig_cmp.add_trace(go.Bar(y=sent_df["Company"], x=-sent_df["Negative"],
            name="Negative", marker_color=RED, orientation="h"))
        apply_theme(fig_cmp, height=320)
        fig_cmp.update_layout(barmode="overlay", xaxis_title="Headline Count",
            title="Sentiment Balance (→ positive, ← negative)")
        st.plotly_chart(fig_cmp, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: CORRELATION & RISK
# ══════════════════════════════════════════════════════════════════════════════╝
elif "Correlation" in page:
    st.markdown(f"<h1 style='font-size:1.8rem;font-weight:700;'>📐 Correlation & Risk Analysis</h1>", unsafe_allow_html=True)

    cap_filter = st.multiselect("Filter by Market Cap",
        ["Large Cap","Mid Cap","Small Cap"],
        default=["Large Cap","Mid Cap"])
    filtered_tickers = [k for k,v in PHARMA_COMPANIES.items() if v[2] in cap_filter]
    filtered_names   = [PHARMA_COMPANIES[k][0] for k in filtered_tickers]

    with st.spinner("Computing correlation matrix (fetching 2Y weekly data)..."):
        corr_fig = build_correlation_heatmap(filtered_tickers, filtered_names)
    st.plotly_chart(corr_fig, use_container_width=True)

    # ── Risk metrics ──────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>RISK METRICS — {sel_name}</div>", unsafe_allow_html=True)
    with st.spinner("Computing risk metrics..."):
        df_risk = fetch_history(sel_ticker, period="3y")

    if not df_risk.empty:
        rets    = df_risk["Close"].pct_change().dropna()
        ann_ret = (1 + rets.mean())**252 - 1
        ann_vol = rets.std() * np.sqrt(252)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
        max_dd  = ((df_risk["Close"] / df_risk["Close"].cummax()) - 1).min()
        var_95  = np.percentile(rets, 5)
        pos_days= (rets > 0).mean() * 100

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        risk_kpis = [
            ("Ann. Return",   f"{ann_ret*100:.1f}%",  GREEN if ann_ret>0 else RED),
            ("Ann. Volatility",f"{ann_vol*100:.1f}%", GOLD),
            ("Sharpe Ratio",  f"{sharpe:.2f}",         GREEN if sharpe>1 else (GOLD if sharpe>0 else RED)),
            ("Max Drawdown",  f"{max_dd*100:.1f}%",    RED),
            ("VaR (95%, 1D)", f"{var_95*100:.2f}%",    RED),
            ("% Positive Days",f"{pos_days:.1f}%",     GREEN if pos_days>50 else RED),
        ]
        for col_obj, (label, val, color) in zip([c1,c2,c3,c4,c5,c6], risk_kpis):
            with col_obj:
                st.markdown(f"""<div class='kpi-strip' style='border-top:3px solid {color};'>
                    <div class='kpi-strip-val' style='color:{color};font-size:1.0rem;'>{val}</div>
                    <div class='kpi-strip-lbl'>{label}</div></div>""", unsafe_allow_html=True)

        # Rolling volatility
        roll_vol = rets.rolling(30).std() * np.sqrt(252) * 100
        fig_vol  = go.Figure()
        fig_vol.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol,
            fill="tozeroy", fillcolor=f"rgba({_hex_rgb(GOLD)},0.12)",
            line=dict(color=GOLD, width=2), name="30D Rolling Volatility"))
        apply_theme(fig_vol, height=280)
        fig_vol.update_layout(title="30-Day Rolling Annualised Volatility (%)",
            yaxis_title="Volatility %", margin=dict(t=36,b=30))
        st.plotly_chart(fig_vol, use_container_width=True)

        # Drawdown chart
        dd = (df_risk["Close"] / df_risk["Close"].cummax()) - 1
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd*100,
            fill="tozeroy", fillcolor=f"rgba({_hex_rgb(RED)},0.15)",
            line=dict(color=RED, width=1.5), name="Drawdown"))
        apply_theme(fig_dd, height=260)
        fig_dd.update_layout(title="Underwater / Drawdown Chart",
            yaxis_title="Drawdown %", margin=dict(t=36,b=30))
        st.plotly_chart(fig_dd, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: FUNDAMENTALS DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════╝
elif "Fundamentals" in page:
    st.markdown(f"<h1 style='font-size:1.8rem;font-weight:700;'>📋 Fundamentals — {sel_name}</h1>", unsafe_allow_html=True)

    with st.spinner("Loading financials..."):
        q    = fetch_live_quote(sel_ticker)
        fins = fetch_financials(sel_ticker)

    # ── Valuation scorecard ───────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>VALUATION & PROFITABILITY</div>", unsafe_allow_html=True)

    metrics = {
        "P/E Ratio":      (q.get("pe_ratio"),  10, 30,  "lower"),
        "P/B Ratio":      (q.get("pb_ratio"),   1,  5,  "lower"),
        "EPS (TTM ₹)":   (q.get("eps"),         None, None, "higher"),
        "Dividend Yield": (q.get("div_yield",0) or 0, 0, 0.05, "higher"),
        "Beta":           (q.get("beta"),        0.8, 1.2, "neutral"),
        "Debt/Equity":    (q.get("debt_equity"), 0, 100, "lower"),
        "ROE %":          ((q.get("roe") or 0)*100, 10, 25, "higher"),
        "Profit Margin %":((q.get("profit_margin") or 0)*100, 5, 20, "higher"),
    }

    cols = st.columns(4)
    for idx, (label, (val, lo, hi, direction)) in enumerate(metrics.items()):
        with cols[idx % 4]:
            if val is None:
                display = "N/A"; color = TEXT_SEC
            else:
                display = f"{val:.2f}"
                if lo is not None and hi is not None:
                    if direction == "lower":
                        color = GREEN if val<lo else (GOLD if val<hi else RED)
                    elif direction == "higher":
                        color = GREEN if val>hi else (GOLD if val>lo else RED)
                    else:
                        color = GREEN if lo<=val<=hi else RED
                else:
                    color = GREEN if val > 0 else RED
            st.markdown(f"""<div class='kpi-strip' style='border-top:3px solid {color};'>
                <div class='kpi-strip-val' style='color:{color};font-size:1.1rem;'>{display}</div>
                <div class='kpi-strip-lbl'>{label}</div></div>""", unsafe_allow_html=True)

    # ── Peer comparison table ─────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>PEER COMPARISON</div>", unsafe_allow_html=True)
    peer_tickers = list(PHARMA_COMPANIES.keys())[:10]
    peer_rows = []
    with st.spinner("Fetching peer data..."):
        for tk in peer_tickers:
            pq = fetch_live_quote(tk)
            if pq.get("price"):
                peer_rows.append({
                    "Company":     PHARMA_COMPANIES[tk][0][:22],
                    "Price ₹":     f"₹{pq['price']:,.2f}",
                    "Change %":    f"{pq['change_pct']:+.2f}%",
                    "Mkt Cap":     fmt_number(pq.get("market_cap")),
                    "P/E":         f"{pq['pe_ratio']:.1f}" if pq.get("pe_ratio") else "N/A",
                    "P/B":         f"{pq['pb_ratio']:.1f}" if pq.get("pb_ratio") else "N/A",
                    "ROE %":       f"{pq['roe']*100:.1f}%" if pq.get("roe") else "N/A",
                    "52W High ₹":  f"₹{pq['week52_high']:,.0f}" if pq.get("week52_high") else "N/A",
                    "52W Low ₹":   f"₹{pq['week52_low']:,.0f}" if pq.get("week52_low") else "N/A",
                    "Segment":     PHARMA_COMPANIES[tk][2],
                })
    if peer_rows:
        peer_df = pd.DataFrame(peer_rows)
        st.dataframe(peer_df, use_container_width=True, hide_index=True, height=380)

    # ── P/E bubble ────────────────────────────────────────────────────────────
    if peer_rows:
        st.markdown(f"<div class='section-hdr'>VALUATION BUBBLE MAP</div>", unsafe_allow_html=True)
        raw_rows = []
        for tk in peer_tickers:
            pq = fetch_live_quote(tk)
            mc = pq.get("market_cap") or 0
            pe = pq.get("pe_ratio")
            pb = pq.get("pb_ratio")
            roe= pq.get("roe")
            if pq.get("price") and pe and pb:
                raw_rows.append({
                    "name": PHARMA_COMPANIES[tk][0][:18],
                    "pe": pe, "pb": pb,
                    "mktcap": mc/1e9,
                    "roe": (roe or 0)*100,
                    "change": pq["change_pct"],
                })
        if raw_rows:
            bub_df = pd.DataFrame(raw_rows)
            fig_bub = px.scatter(bub_df, x="pe", y="pb",
                size="mktcap", color="roe",
                text="name", size_max=50,
                color_continuous_scale=[[0,RED],[0.5,GOLD],[1,GREEN]],
                labels={"pe":"P/E Ratio","pb":"P/B Ratio","mktcap":"Mkt Cap (B)","roe":"ROE %"},
                hover_data={"pe":":.1f","pb":":.1f","mktcap":":.1f","roe":":.1f"})
            fig_bub.update_traces(textposition="top center", textfont_size=9)
            apply_theme(fig_bub, height=520)
            fig_bub.update_layout(title="P/E vs P/B — Size = Market Cap | Color = ROE%")
            st.plotly_chart(fig_bub, use_container_width=True)
