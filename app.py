import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import json
import os

# -------------------------------------------------------------------
# File Storage Setup (For Permanent Memory)
# -------------------------------------------------------------------
SAVE_FILE = "saved_universes.json"

def load_saved_lists():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r") as f:
            return json.load(f)
    return {"nse_500": "", "custom": "", "oneoff": ""}

def save_lists_to_file(nse_500, custom, oneoff):
    data = {"nse_500": nse_500, "custom": custom, "oneoff": oneoff}
    with open(SAVE_FILE, "w") as f:
        json.dump(data, f)

def process_raw_tickers(text_input):
    parsed_tickers = []
    raw_tickers = text_input.replace('\n', ',').split(',')
    for t in raw_tickers:
        t = t.strip().upper()
        if not t: continue
        if not (t.endswith('.NS') or t.endswith('.BO')):
            t = f"{t}.BO" if t.isdigit() else f"{t}.NS"
        parsed_tickers.append(t)
    return list(set(parsed_tickers))

# -------------------------------------------------------------------
# Market Regime Filter (Layer 1)
# -------------------------------------------------------------------
@st.cache_data(ttl=3600) 
def get_market_regime():
    try:
        nifty = yf.Ticker("^NSEI").history(period="2y")
        if nifty.empty: return None, 0, 0, 0, 0, 0
            
        close_series = nifty['Close']
        sma5 = close_series.rolling(window=5).mean()
        sma200 = close_series.rolling(window=200).mean()
        upper_band = sma200 * 1.015
        lower_band = sma200 * 0.985
        
        conditions = [(sma5 > upper_band), (sma5 < lower_band)]
        choices = [1, -1]
        
        raw_regime = np.select(conditions, choices, default=np.nan)
        regime_series = pd.Series(raw_regime, index=close_series.index).ffill().dropna()
        
        is_bull = True if (not regime_series.empty and regime_series.iloc[-1] == 1) else False
        return is_bull, float(close_series.iloc[-1]), float(sma5.iloc[-1]), float(sma200.iloc[-1]), float(upper_band.iloc[-1]), float(lower_band.iloc[-1])
    except Exception:
        return None, 0, 0, 0, 0, 0

# -------------------------------------------------------------------
# Helper: Bulk Fundamental Fetching (Layer 2)
# -------------------------------------------------------------------
def get_bulk_fundamentals(tickers):
    fundamental_results = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            # Fetch quarterly net income for QoQ check
            q_fin = tk.quarterly_financials
            if q_fin is not None and not q_fin.empty and 'Net Income' in q_fin.index:
                net_inc = q_fin.loc['Net Income'].dropna()
                if len(net_inc) >= 2:
                    is_accel = net_inc.iloc[0] > net_inc.iloc[1]
                    tag = "🟢 EPS Accelerating" if is_accel else "🔴 EPS Declining"
                else:
                    tag = "⚪ Limited Qtr Data"
            else:
                tag = "⚪ Data Unavailable"
        except Exception:
            tag = "⚪ Fetch Error"
        fundamental_results[t] = tag
    return fundamental_results

# -------------------------------------------------------------------
# 1. Page Configuration
# -------------------------------------------------------------------
st.set_page_config(page_title="MK Quant Master Engine", page_icon="📈", layout="wide")

st.title("🚀 MK_QUANT_ALPHA: Master Unified Dashboard")

# -------------------------------------------------------------------
# 2. Sidebar Settings
# -------------------------------------------------------------------
st.sidebar.title("⚙️ Engine Controls")

st.sidebar.header("0. Universe Input")
universe_choice = st.sidebar.selectbox("Target Universe:", ["NSE Top 500", "Custom Universe (Screener)", "One-Off List / CSV"])

st.sidebar.header("1. Portfolio Execution & Risk")
account_capital = st.sidebar.number_input("Total Capital (₹)", min_value=1000.0, value=10000.0, step=5000.0)
max_risk_pct = st.sidebar.number_input("Max Risk Per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
port_size = st.sidebar.number_input("Target Hold Slots", min_value=1, max_value=50, value=5, step=1)
sell_rank = st.sidebar.number_input("Hysteresis Sell Rank", min_value=1, max_value=100, value=25, step=1)

st.sidebar.header("2. Strategy Weights (%)")
w1 = st.sidebar.number_input("Momentum (W1)", value=80, step=5)
w2 = st.sidebar.number_input("NearHigh (W2)", value=20, step=5)
w3 = st.sidebar.number_input("Quality R² (W3)", value=0, step=5)

st.sidebar.header("3. Extension & Risk Parameters")
ema_period = st.sidebar.slider("Entry EMA Window", 10, 50, 20)
max_ext_pct = st.sidebar.slider("Max Buy Extension (%)", 1.0, 10.0, 5.0, step=0.5)
atr_mult = st.sidebar.slider("ATR Stop Multiplier", 1.0, 5.0, 3.0, step=0.1)
check_fundamentals = st.sidebar.checkbox("Enable Layer 2 Fundamentals (QoQ Net Income)", value=True)

# -------------------------------------------------------------------
# 3. Macro Regime Header (Layer 1)
# -------------------------------------------------------------------
saved_data = load_saved_lists()
if 'memory_nse500' not in st.session_state: st.session_state['memory_nse500'] = saved_data.get('nse_500', "")
if 'memory_custom' not in st.session_state: st.session_state['memory_custom'] = saved_data.get('custom', "")
if 'memory_oneoff' not in st.session_state: st.session_state['memory_oneoff'] = saved_data.get('oneoff', "")

is_bull, nse_close, sma5, sma200, upper_band, lower_band = get_market_regime()

if is_bull:
    st.success(f"🟢 **MACRO REGIME: BULLISH** | Nifty 50 5-DMA ({sma5:.0f}) > Upper Hysteresis Band ({upper_band:.0f}). Systemic Liquidity Expanding.")
else:
    st.error(f"🔴 **MACRO REGIME: BEARISH** | Nifty 50 5-DMA ({sma5:.0f}) < Lower Hysteresis Band ({lower_band:.0f}). Capital Protection Active.")

st.markdown("---")

# Universe Text Areas
tickers = []
if universe_choice == "NSE Top 500":
    new_text = st.text_area("Paste NSE Top 500 Tickers:", value=st.session_state['memory_nse500'], height=100)
    if new_text != st.session_state['memory_nse500']:
        st.session_state['memory_nse500'] = new_text
        save_lists_to_file(st.session_state['memory_nse500'], st.session_state['memory_custom'], st.session_state['memory_oneoff'])
    tickers = process_raw_tickers(st.session_state['memory_nse500'])

elif universe_choice == "Custom Universe (Screener)":
    new_text = st.text_area("Paste Custom Tickers:", value=st.session_state['memory_custom'], height=100)
    if new_text != st.session_state['memory_custom']:
        st.session_state['memory_custom'] = new_text
        save_lists_to_file(st.session_state['memory_nse500'], st.session_state['memory_custom'], st.session_state['memory_oneoff'])
    tickers = process_raw_tickers(st.session_state['memory_custom'])

elif universe_choice == "One-Off List / CSV":
    new_text = st.text_area("Paste Tickers:", value=st.session_state['memory_oneoff'], height=100)
    if new_text != st.session_state['memory_oneoff']:
        st.session_state['memory_oneoff'] = new_text
        save_lists_to_file(st.session_state['memory_nse500'], st.session_state['memory_custom'], st.session_state['memory_oneoff'])
    tickers = process_raw_tickers(st.session_state['memory_oneoff'])

st.write(f"**Total Unique Tickers Loaded:** {len(tickers)}")

# -------------------------------------------------------------------
# 4. Engine Processing Pipeline
# -------------------------------------------------------------------
if st.button("🚀 Run Master Engine") and len(tickers) > 0:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Fetching market price data...")
        raw_data = yf.download(tickers, period="2y", progress=False)
        
        if len(tickers) == 1:
            prices_df = raw_data[['Close']].copy(); prices_df.columns = [tickers[0]]
            high_df = raw_data[['High']].copy(); high_df.columns = [tickers[0]]
            low_df = raw_data[['Low']].copy(); low_df.columns = [tickers[0]]
        else:
            prices_df = raw_data['Close']
            high_df = raw_data['High']
            low_df = raw_data['Low']

        progress_bar.progress(20)
        status_text.text("Applying Layer 3 Dual-Momentum Filter (Price > 100-SMA)...")

        # History and Dual-Momentum Filter
        min_days = 252 + 21
        valid_history = prices_df.count()[prices_df.count() >= min_days].index
        prices_df = prices_df[valid_history].ffill(limit=5)
        high_df = high_df[valid_history].ffill(limit=5)
        low_df = low_df[valid_history].ffill(limit=5)

        sma100 = prices_df.rolling(100).mean().iloc[-1]
        current_price = prices_df.iloc[-1]
        
        valid_mom = current_price[current_price > sma100].index.tolist()
        if not valid_mom:
            st.warning("No stocks passed the 100-SMA trend filter.")
            st.stop()

        prices_df, high_df, low_df = prices_df[valid_mom], high_df[valid_mom], low_df[valid_mom]
        current_price = current_price[valid_mom]

        progress_bar.progress(40)
        status_text.text("Calculating 20-EMA Extension & ATR Stop-Loss...")

        # Extension Check (Layer 4)
        ema_20 = prices_df.ewm(span=ema_period, adjust=False).mean().iloc[-1]
        max_allowed_price = ema_20 * (1 + (max_ext_pct / 100.0))
        is_extended = current_price > max_allowed_price

        # ATR Calculation
        prev_close = prices_df.shift(1)
        tr = np.maximum(high_df - low_df, np.maximum((high_df - prev_close).abs(), (low_df - prev_close).abs()))
        atr_14 = tr.rolling(14).mean().iloc[-1]
        atr_stop = current_price - (atr_14 * atr_mult)

        progress_bar.progress(60)
        status_text.text("Calculating Multi-Timeframe Blended Momentum & R² Quality...")

        # Momentum & Quality Math
        P_skip, P_12M, P_6M, P_3M = prices_df.iloc[-22], prices_df.iloc[-273], prices_df.iloc[-147], prices_df.iloc[-84]
        R_blend = (0.50 * ((P_skip / P_12M) - 1)) + (0.30 * ((P_skip / P_6M) - 1)) + (0.20 * ((P_skip / P_3M) - 1))

        daily_ret = prices_df.pct_change(1).clip(upper=0)
        DD_126 = np.sqrt((daily_ret ** 2).rolling(126).sum().iloc[-1] / 126)
        Score_raw = R_blend / (DD_126 + 0.002)

        NearHigh = current_price / prices_df.rolling(252).max().iloc[-1]

        time_seq = pd.Series(np.arange(len(prices_df)), index=prices_df.index)
        rolling_r = prices_df.rolling(126).corr(time_seq)
        Quality = ((rolling_r ** 2) * np.sign(rolling_r)).iloc[-1]

        # Percentile Normalization
        def percentile(s):
            return (s.rank(method='min') - 1) / (len(s) - 1) * 100.0 if len(s) > 1 else pd.Series(100.0, index=s.index)

        w_tot = w1 + w2 + w3 if (w1 + w2 + w3) > 0 else 1
        Score_Final = ((w1/w_tot)*percentile(Score_raw)) + ((w2/w_tot)*percentile(NearHigh)) + ((w3/w_tot)*percentile(Quality))

        progress_bar.progress(80)
        
        # Fundamental Check (Layer 2)
        fund_tags = {}
        if check_fundamentals:
            status_text.text("Fetching Layer 2 Fundamental Data (QoQ Net Income)...")
            fund_tags = get_bulk_fundamentals(valid_mom)
        else:
            fund_tags = {t: "⚪ Filter Disabled" for t in valid_mom}

        # Sizing & Execution Math
        cash_budget_per_stock = account_capital / port_size
        max_shares_cash = np.floor(cash_budget_per_stock / current_price)

        risk_per_share = (current_price - atr_stop).replace(0, 0.01)
        total_risk = account_capital * (max_risk_pct / 100.0)
        max_shares_risk = np.floor(total_risk / risk_per_share)

        final_shares = np.minimum(max_shares_risk, max_shares_cash).clip(lower=0).astype(int)

        # Build Master Output
        df = pd.DataFrame({
            'Ticker': Score_Final.index,
            'Price (₹)': current_price.values,
            'Shares to Buy': final_shares.values,
            'Score_Final': Score_Final.values,
            'Entry Status': np.where(is_extended, "🟡 EXTENDED (Wait Dip)", "🟢 IN BUY ZONE"),
            'Fundamental Health': [fund_tags.get(t, "⚪ N/A") for t in Score_Final.index],
            'Blended Return (%)': R_blend.values * 100,
            'R² Quality': Quality.values,
            '20-EMA (₹)': ema_20.values,
            'ATR Stop-Loss (₹)': atr_stop.values
        }).sort_values(by='Score_Final', ascending=False).reset_index(drop=True)

        df.index += 1
        df.index.name = 'Rank'

        # Unified Action Tags
        actions = []
        for rank, row in df.iterrows():
            if rank <= port_size:
                if not is_bull:
                    actions.append('🔴 HOLD CASH (Bear Regime)')
                elif "Declining" in row['Fundamental Health']:
                    actions.append('🟡 TOXIC BREAKOUT (Half Size)')
                elif "EXTENDED" in row['Entry Status']:
                    actions.append('🟡 HOLD (Wait for Dip)')
                else:
                    actions.append('🟢 BUY / STRONG HOLD')
            elif rank <= sell_rank:
                actions.append('🟡 HOLD (Buffer Zone)')
            else:
                actions.append('🔴 SELL / REPLACE')

        df.insert(0, 'Action Tag', actions)

        # Rounding
        for col in ['Price (₹)', 'Score_Final', 'Blended Return (%)', 'R² Quality', '20-EMA (₹)', 'ATR Stop-Loss (₹)']:
            df[col] = df[col].round(2)

        progress_bar.progress(100)
        status_text.success(f"✅ Analysis Complete! {len(df)} stocks evaluated across all 4 Layers.")

        # Display Master Dashboard
        st.subheader("🏆 Unified Master Leaderboard")
        st.dataframe(df, use_container_width=True, height=600)

        st.download_button("📥 Download Master Analysis CSV", data=df.to_csv(), file_name="mk_master_leaderboard.csv", mime="text/csv")

    except Exception as e:
        status_text.empty()
        st.error(f"Error executing engine: {str(e)}")
