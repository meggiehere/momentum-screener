import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time

# =====================================================================
# NSE Momentum-Fundamental Alpha Screener
# 4-Layer Engine: Trend Filter -> Momentum Rank -> Risk/Sizing -> Fundamentals
# =====================================================================

st.set_page_config(page_title="NSE Momentum Alpha Screener", page_icon="📈", layout="wide")
st.title("📈 NSE Momentum-Fundamental Alpha Screener")
st.caption("Individual-stock momentum screener with entry/exit price guidance for the Indian equity market.")

# ---------------------------------------------------------------------
# Sidebar — Controls
# ---------------------------------------------------------------------
st.sidebar.header("1. Portfolio & Risk")
account_capital = st.sidebar.number_input("Total Capital (₹)", min_value=1000.0, value=100000.0, step=5000.0)
max_risk_pct = st.sidebar.number_input("Risk Per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
port_size = st.sidebar.number_input("Target Portfolio Slots", min_value=1, max_value=50, value=10, step=1)
sell_buffer_rank = st.sidebar.number_input("Sell Buffer Rank (hysteresis)", min_value=1, max_value=200, value=port_size + 15, step=1)

st.sidebar.header("2. Entry Parameters")
ema_period = st.sidebar.slider("Entry EMA Window (days)", 10, 50, 20)
max_ext_pct = st.sidebar.slider("Max Buy Extension above EMA (%)", 1.0, 10.0, 5.0, step=0.5)
atr_mult = st.sidebar.slider("ATR Stop Multiplier", 1.0, 5.0, 3.0, step=0.1)

st.sidebar.header("3. Fundamentals (Lazy Fetch)")
top_n_fundamentals = st.sidebar.number_input("Fetch fundamentals for Top N ranked stocks only", min_value=5, max_value=100, value=30, step=5)
st.sidebar.caption("Yahoo Finance rate-limits bulk fundamental calls, so EPS trend and intrinsic value are only fetched for the top-ranked stocks after momentum scoring — not the whole universe.")

# ---------------------------------------------------------------------
# Ticker input
# ---------------------------------------------------------------------
st.subheader("Universe")
raw_input = st.text_area(
    "Paste NSE/BSE tickers (comma or newline separated). Example: RELIANCE, TCS, INFY, HDFCBANK",
    height=100,
)


def parse_tickers(text):
    out = []
    for t in text.replace("\n", ",").split(","):
        t = t.strip().upper()
        if not t:
            continue
        if not (t.endswith(".NS") or t.endswith(".BO")):
            t = f"{t}.NS"
        out.append(t)
    return sorted(set(out))


def clean_array(values, default=0.0):
    arr = np.array(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, default)
    return arr


tickers = parse_tickers(raw_input)
st.write(f"**{len(tickers)} unique tickers loaded**")

# ---------------------------------------------------------------------
# Run engine
# ---------------------------------------------------------------------
if st.button("🚀 Run Screener") and tickers:
    progress = st.progress(0)
    status = st.empty()

    try:
        # -------------------- Data fetch --------------------
        status.text("Downloading 2 years of daily price history...")
        raw = yf.download(tickers, period="2y", progress=False)
        progress.progress(10)

        if len(tickers) == 1:
            close = raw[["Close"]].copy(); close.columns = [tickers[0]]
            high = raw[["High"]].copy(); high.columns = [tickers[0]]
            low = raw[["Low"]].copy(); low.columns = [tickers[0]]
        else:
            close = raw["Close"]
            high = raw["High"]
            low = raw["Low"]

        # -------------------- Layer 1: Trend Filter --------------------
        status.text("Layer 1: Applying 100-day SMA trend filter...")
        min_days = 273 + 10  # 12M lookback (252+21 skip) plus buffer
        valid_cols = close.count()[close.count() >= min_days].index
        close = close[valid_cols].ffill(limit=5)
        high = high[valid_cols].ffill(limit=5)
        low = low[valid_cols].ffill(limit=5)

        if close.empty:
            st.warning("No tickers have enough history (need ~14 months of trading days).")
            st.stop()

        sma100 = close.rolling(100).mean().iloc[-1]
        price_now = close.iloc[-1]
        trend_pass = price_now[price_now > sma100].index.tolist()

        if not trend_pass:
            st.warning("No stocks passed the 100-day SMA trend filter today. Bearish breadth — consider staying in cash.")
            st.stop()

        close, high, low = close[trend_pass], high[trend_pass], low[trend_pass]
        price_now = price_now[trend_pass]
        progress.progress(30)

        # -------------------- Layer 2: Momentum Ranking --------------------
        status.text("Layer 2: Scoring blended momentum, near-high, and quality...")
        P_skip = close.iloc[-22]
        P_12m = close.iloc[-273]
        P_6m = close.iloc[-147]
        P_3m = close.iloc[-84]

        R_blend = 0.50 * (P_skip / P_12m - 1) + 0.30 * (P_skip / P_6m - 1) + 0.20 * (P_skip / P_3m - 1)

        daily_ret = close.pct_change().clip(upper=0)
        DD_126 = np.sqrt((daily_ret ** 2).rolling(126).sum().iloc[-1] / 126)
        score_raw = R_blend / (DD_126 + 0.002)

        near_high = price_now / close.rolling(252).max().iloc[-1]

        t_seq = pd.Series(np.arange(len(close)), index=close.index)
        roll_r = close.rolling(126).corr(t_seq)
        quality = ((roll_r ** 2) * np.sign(roll_r)).iloc[-1]

        def pct_rank(s):
            s = s.reindex(trend_pass)
            valid = s.dropna()
            if len(valid) <= 1:
                return pd.Series(100.0, index=s.index)
            ranks = (valid.rank(method="min") - 1) / (len(valid) - 1) * 100.0
            return ranks.reindex(s.index)

        score_final = (
            pct_rank(score_raw).fillna(0)
            + pct_rank(near_high).fillna(0)
            + pct_rank(quality).fillna(0)
        ) / 3.0

        progress.progress(50)

        # -------------------- Layer 3: Risk & Sizing --------------------
        status.text("Layer 3: Computing extension, ATR stop-loss, and position sizing...")
        ema_n = close.ewm(span=ema_period, adjust=False).mean().iloc[-1]
        max_allowed_price = ema_n * (1 + max_ext_pct / 100.0)
        is_extended = price_now > max_allowed_price
        buy_zone_tag = pd.Series(np.where(is_extended, "🟡 Wait Dip", "🟢 IN ZONE"), index=price_now.index)

        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        atr14 = tr.rolling(14).mean().iloc[-1]
        atr_stop = price_now - atr14 * atr_mult

        cash_per_slot = account_capital / port_size
        max_shares_cash = np.floor(clean_array(cash_per_slot / price_now))

        risk_per_share = (price_now - atr_stop)
        risk_per_share = risk_per_share.where(risk_per_share > 0.01, 0.01)
        total_risk_rupees = account_capital * (max_risk_pct / 100.0)
        max_shares_risk = np.floor(clean_array(total_risk_rupees / risk_per_share))

        shares_to_buy = np.minimum(max_shares_cash, max_shares_risk)
        shares_to_buy = np.nan_to_num(shares_to_buy, nan=0.0, posinf=0.0, neginf=0.0)
        shares_to_buy = np.clip(shares_to_buy, 0, None).astype(int)
        shares_to_buy = pd.Series(shares_to_buy, index=price_now.index)

        progress.progress(65)

        df = pd.DataFrame({
            "Ticker": score_final.index,
            "Price (₹)": price_now.reindex(score_final.index).values,
            "Score": score_final.values,
            "Buy Zone": buy_zone_tag.reindex(score_final.index).values,
            f"{ema_period}-EMA (₹)": ema_n.reindex(score_final.index).values,
            "ATR Stop (₹)": atr_stop.reindex(score_final.index).values,
            "Shares to Buy": shares_to_buy.reindex(score_final.index).values,
        }).sort_values("Score", ascending=False).reset_index(drop=True)
        df.index += 1
        df.index.name = "Rank"

        # -------------------- Layer 4: Lazy Fundamentals (Top N only) --------------------
        progress.progress(75)
        status.text(f"Layer 4: Fetching fundamentals for the top {int(top_n_fundamentals)} ranked stocks...")

        top_tickers = df.head(int(top_n_fundamentals))["Ticker"].tolist()
        eps_trend_map, iv_map = {}, {}

        for i, t in enumerate(top_tickers):
            eps_tag, iv = "⚪ N/A", 0.0
            try:
                tk = yf.Ticker(t)
                qf = tk.quarterly_financials
                if qf is not None and not qf.empty and "Net Income" in qf.index:
                    ni = qf.loc["Net Income"].dropna()
                    if len(ni) >= 2:
                        eps_tag = "🟢 EPS Up" if ni.iloc[0] > ni.iloc[1] else "🔴 EPS Down"
                info = tk.info
                eps = info.get("trailingEps", 0) or 0
                bv = info.get("bookValue", 0) or 0
                if eps > 0 and bv > 0:
                    iv = float(np.sqrt(22.5 * eps * bv))
            except Exception:
                pass
            eps_trend_map[t] = eps_tag
            iv_map[t] = iv
            time.sleep(0.1)
            progress.progress(75 + int(20 * (i + 1) / max(len(top_tickers), 1)))

        df["EPS Trend"] = df["Ticker"].map(eps_trend_map).fillna("⚪ Not Ranked")
        df["Intrinsic Value (₹)"] = df["Ticker"].map(iv_map).fillna(0.0)

        val_tags = []
        for p, iv in zip(df["Price (₹)"], df["Intrinsic Value (₹)"]):
            if iv <= 0:
                val_tags.append("⚪ N/A")
            elif p < iv:
                val_tags.append("🟢 Discount")
            else:
                val_tags.append("🔴 Premium")
        df["Valuation"] = val_tags

        # -------------------- Cleanup --------------------
        numeric_cols = ["Price (₹)", "Score", f"{ema_period}-EMA (₹)", "ATR Stop (₹)", "Intrinsic Value (₹)"]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], 0).fillna(0).round(2)

        df = df[[
            "Ticker", "Price (₹)", "Shares to Buy", "Score", "Buy Zone",
            "Intrinsic Value (₹)", "Valuation", "EPS Trend",
            f"{ema_period}-EMA (₹)", "ATR Stop (₹)",
        ]]

        progress.progress(100)
        status.success(f"Done — {len(df)} stocks passed the trend filter and were ranked.")

        st.subheader("Leaderboard")
        st.dataframe(df, use_container_width=True, height=600)
        st.download_button("📥 Download CSV", df.to_csv(), file_name="momentum_leaderboard.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("How to read Buy / Sell price")
        st.markdown(f"""
- **Buy price**: Only buy when a stock is tagged 🟢 **IN ZONE** — price is within {max_ext_pct}% of its {ema_period}-day EMA. A 🟡 **Wait Dip** tag means the stock is extended above its short-term trend; chasing it here erodes the momentum edge, so wait for a pullback toward the EMA.
- **Stop-loss (hard sell)**: The **ATR Stop** column is your initial hard stop, set at {atr_mult}× the 14-day Average True Range below price. A daily close below this level is an exit signal, independent of the ranking.
- **Rotation-based sell (rank-based)**: This is a *relative-strength* ranking system, so a holding should also be sold/replaced once it drifts out of your buffer zone — i.e. its rank falls below ~{int(sell_buffer_rank)} — even if the ATR stop hasn't been hit yet. That rotation from weak names into strong ones is where most of the momentum alpha comes from, not the stop-loss alone.
- **Position size** already reflects both your risk budget ({max_risk_pct}% of capital per trade, capped by the ATR-based stop distance) and your cash budget (capital ÷ {int(port_size)} slots) — the smaller of the two is used.
""")

    except Exception as e:
        status.empty()
        st.error(f"Error running screener: {e}")
