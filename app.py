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
    """Loads the saved lists from a local JSON file."""
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r") as f:
            return json.load(f)
    return {"nse_500": "", "custom": "", "oneoff": ""}

def save_lists_to_file(nse_500, custom, oneoff):
    """Saves the current lists to a local JSON file."""
    data = {
        "nse_500": nse_500,
        "custom": custom,
        "oneoff": oneoff
    }
    with open(SAVE_FILE, "w") as f:
        json.dump(data, f)

# -------------------------------------------------------------------
# Helper Function: Smart Ticker Logic
# -------------------------------------------------------------------
def process_raw_tickers(text_input):
    parsed_tickers = []
    raw_tickers = text_input.replace('\n', ',').split(',')
    
    for t in raw_tickers:
        t = t.strip().upper()
        if not t: continue
            
        if not (t.endswith('.NS') or t.endswith('.BO')):
            t = f"{t}.BO" if t.isdigit() else f"{t}.NS"
        parsed_tickers.append(t)
        
    return parsed_tickers

# -------------------------------------------------------------------
# Helper Function: Market Regime Filter (Cached for speed)
# -------------------------------------------------------------------
@st.cache_data(ttl=3600) # Cache clears every 1 hour
def get_market_regime():
    try:
        # yf.Ticker().history() guarantees a clean structure without MultiIndex issues
        nifty = yf.Ticker("^NSEI").history(period="1y")
        
        # If Yahoo returns empty data, raise an error to bypass the cache
        if nifty.empty:
            raise ValueError("Empty data from Yahoo Finance")
            
        close_series = nifty['Close']
        
        # Calculate moving averages
        current_close = float(close_series.dropna().iloc[-1])
        sma50 = float(close_series.rolling(50).mean().dropna().iloc[-1])
        sma200 = float(close_series.rolling(200).mean().dropna().iloc[-1])
        
        is_bull = (current_close > sma50) and (current_close > sma200)
        return is_bull, current_close, sma50, sma200
        
    except Exception as e:
        # Clear the cache so it retries instantly next time instead of waiting an hour
        st.cache_data.clear() 
        return None, 0, 0, 0

# -------------------------------------------------------------------
# 1. Page Configuration & Setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="MK Momentum Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stDataFrame { width: 100%; }
    .stButton>button { width: 100%; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Persistent Memory Initialization
# -------------------------------------------------------------------
saved_data = load_saved_lists()

if 'memory_nse500' not in st.session_state:
    st.session_state['memory_nse500'] = saved_data.get('nse_500', "")
if 'memory_custom' not in st.session_state:
    st.session_state['memory_custom'] = saved_data.get('custom', "")
if 'memory_oneoff' not in st.session_state:
    st.session_state['memory_oneoff'] = saved_data.get('oneoff', "")

# -------------------------------------------------------------------
# 2. Sidebar: Strategy Settings & Parameters
# -------------------------------------------------------------------
st.sidebar.title("⚙️ Strategy Settings")

st.sidebar.header("0. Select Target Universe")
universe_choice = st.sidebar.selectbox(
    "Choose which universe to rank:", 
    ["NSE Top 500", "Custom Universe (Screener)", "One-Off List / CSV"]
)
st.sidebar.markdown("---")

st.sidebar.header("1. Component Weights (%)")
w1 = st.sidebar.number_input("Momentum (W1)", min_value=0, max_value=100, value=80, step=5)
w2 = st.sidebar.number_input("NearHigh (W2)", min_value=0, max_value=100, value=20, step=5)
w3 = st.sidebar.number_input("Quality (W3)", min_value=0, max_value=100, value=0, step=5)

w_total = w1 + w2 + w3
if w_total != 100:
    st.sidebar.warning(f"⚠️ Weights sum to {w_total}, not 100.")

st.sidebar.header("2. Timeframe Blend (Sum=1.0)")
w_12m = st.sidebar.number_input("12M Return Weight", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
w_6m = st.sidebar.number_input("6M Return Weight", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
w_3m = st.sidebar.number_input("3M Return Weight", min_value=0.0, max_value=1.0, value=0.20, step=0.05)

st.sidebar.header("3. Lookback & Windows")
skip_days = st.sidebar.slider("Skip Days (Momentum lag)", 5, 60, 21)
dd_lookback = st.sidebar.slider("Downside Lookback (Days)", 30, 252, 126)
nh_lookback = st.sidebar.slider("NearHigh Lookback (Days)", 63, 504, 252)

st.sidebar.header("4. Portfolio Risk Controls")
atr_mult = st.sidebar.slider("ATR Stop Multiplier", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
tox_cutoff = st.sidebar.slider("Toxicity Cutoff (%)", 0, 50, 10)

# -------------------------------------------------------------------
# 3. Main UI: Data Input & Market Regime
# -------------------------------------------------------------------
st.title("🚀 Quant Equity Screener: MK_MOMENTUM")

# Market Regime UI Banner
is_bull, nse_close, nse_50, nse_200 = get_market_regime()
if is_bull is True:
    st.success(f"🟢 **Bull Regime: Deploy Capital** | Nifty 50 ({nse_close:.0f}) is trading above its 50-DMA ({nse_50:.0f}) and 200-DMA ({nse_200:.0f}).")
elif is_bull is False:
    st.error(f"🔴 **Bear Regime: Hold Cash** | Nifty 50 ({nse_close:.0f}) has lost primary trend support (50-DMA: {nse_50:.0f}, 200-DMA: {nse_200:.0f}).")
else:
    st.warning("⚠️ Market Regime data temporarily unavailable.")

st.markdown("---")
tickers = []

if universe_choice == "NSE Top 500":
    st.subheader("📥 NSE Top 500 Universe")
    st.info("Your list is automatically saved to the server. It will remain here even if you close the app.")
    
    new_text = st.text_area("Paste NSE Top 500 Tickers:", value=st.session_state['memory_nse500'], height=150)
    if new_text != st.session_state['memory_nse500']:
        st.session_state['memory_nse500'] = new_text
        save_lists_to_file(st.session_state['memory_nse500'], st.session_state['memory_custom'], st.session_state['memory_oneoff'])
    
    if st.session_state['memory_nse500']:
        tickers = process_raw_tickers(st.session_state['memory_nse500'])

elif universe_choice == "Custom Universe (Screener)":
    st.subheader("📥 Custom Universe (Small/Micro Caps)")
    st.info("Your list is automatically saved to the server. It will remain here even if you close the app.")
    
    new_text = st.text_area("Paste Custom Tickers:", value=st.session_state['memory_custom'], height=150)
    if new_text != st.session_state['memory_custom']:
        st.session_state['memory_custom'] = new_text
        save_lists_to_file(st.session_state['memory_nse500'], st.session_state['memory_custom'], st.session_state['memory_oneoff'])
    
    if st.session_state['memory_custom']:
        tickers = process_raw_tickers(st.session_state['memory_custom'])

elif universe_choice == "One-Off List / CSV":
    st.subheader("📥 One-Off Manual Input or CSV")
    col1, col2 = st.columns(2)

    with col1:
        new_text = st.text_area("Paste Tickers:", value=st.session_state['memory_oneoff'], height=150)
        if new_text != st.session_state['memory_oneoff']:
            st.session_state['memory_oneoff'] = new_text
            save_lists_to_file(st.session_state['memory_nse500'], st.session_state['memory_custom'], st.session_state['memory_oneoff'])

    with col2:
        uploaded_file = st.file_uploader("Upload CSV (Must contain a 'Ticker' column)", type=['csv'])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            ticker_col = next((col for col in df_upload.columns if 'ticker' in col.lower()), None)
            if ticker_col:
                tickers.extend(df_upload[ticker_col].dropna().astype(str).tolist())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    if st.session_state['memory_oneoff']:
        tickers.extend(process_raw_tickers(st.session_state['memory_oneoff']))

# Deduplicate tickers
tickers = list(set(tickers))
st.write(f"**Total unique tickers ready for processing:** {len(tickers)}")

# -------------------------------------------------------------------
# 4. Engine Execution
# -------------------------------------------------------------------
if st.button("🚀 Run Momentum Engine") and len(tickers) > 0:
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Fetching market data (High, Low, Close) from Yahoo Finance...")
    
    try:
        raw_data = yf.download(tickers, period="2y", progress=False)
        
        # Safely parse multi-ticker vs single-ticker structures
        if len(tickers) == 1:
            prices_df = raw_data[['Close']].copy()
            prices_df.columns = [tickers[0]]
            high_df = raw_data[['High']].copy()
            high_df.columns = [tickers[0]]
            low_df = raw_data[['Low']].copy()
            low_df.columns = [tickers[0]]
        else:
            prices_df = raw_data['Close']
            high_df = raw_data['High']
            low_df = raw_data['Low']
                
        if prices_df.empty:
            st.error("Could not find price data.")
            st.stop()
            
        progress_bar.progress(20)
        status_text.text("Data fetched. Applying minimum data filters...")

        min_required_days = 252 + skip_days
        valid_counts = prices_df.count()
        valid_tickers = valid_counts[valid_counts >= min_required_days].index.tolist()
        
        if not valid_tickers:
            st.error("No tickers have enough history.")
            st.stop()
            
        # Filter and forward-fill missing data across all price frames
        prices_df = prices_df[valid_tickers].ffill(limit=5)
        high_df = high_df[valid_tickers].ffill(limit=5)
        low_df = low_df[valid_tickers].ffill(limit=5)
        
        progress_bar.progress(40)
        status_text.text("Calculating 14-Day Average True Range (ATR)...")

        # Calculate True Range (Vectorized)
        prev_close = prices_df.shift(1)
        tr1 = high_df - low_df
        tr2 = (high_df - prev_close).abs()
        tr3 = (low_df - prev_close).abs()
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        # 14-Day Simple Moving Average of True Range
        atr_14 = true_range.rolling(window=14).mean().iloc[-1]

        progress_bar.progress(55)
        status_text.text("Calculating Multi-Timeframe Blended Returns & Risk...")

        current_price = prices_df.iloc[-1]
        P_skip = prices_df.iloc[-(1 + skip_days)]
        P_12M = prices_df.iloc[-(252 + skip_days + 1)]
        P_6M = prices_df.iloc[-(126 + skip_days + 1)]
        P_3M = prices_df.iloc[-(63 + skip_days + 1)]

        R_12M = (P_skip / P_12M) - 1
        R_6M  = (P_skip / P_6M) - 1
        R_3M  = (P_skip / P_3M) - 1
        R_blend = (w_12m * R_12M) + (w_6m * R_6M) + (w_3m * R_3M)

        daily_returns = prices_df.pct_change(1)
        negative_returns = daily_returns.clip(upper=0)
        DD_126 = np.sqrt((negative_returns ** 2).rolling(dd_lookback).sum().iloc[-1] / dd_lookback)

        progress_bar.progress(70)
        status_text.text("Evaluating Proximity & Quality...")

        Score_raw = R_blend / (DD_126 + 0.002)
        High_52W = prices_df.rolling(nh_lookback).max().iloc[-1]
        NearHigh = current_price / High_52W
        Quality = (daily_returns > 0).rolling(dd_lookback).sum().iloc[-1] / dd_lookback

        if tox_cutoff > 0:
            dd_threshold = np.percentile(DD_126.dropna(), 100 - tox_cutoff)
            valid_mask = DD_126 <= dd_threshold
            
            Score_raw = Score_raw[valid_mask]
            NearHigh = NearHigh[valid_mask]
            Quality = Quality[valid_mask]
            current_price = current_price[valid_mask]
            R_blend = R_blend[valid_mask]
            DD_126 = DD_126[valid_mask]
            atr_14 = atr_14[valid_mask]
            
        progress_bar.progress(85)
        status_text.text("Normalizing and generating final ranks...")

        def calc_percentile(series):
            s_clean = series.dropna()
            if len(s_clean) <= 1:
                return pd.Series(100.0, index=s_clean.index)
            return (s_clean.rank(method='min') - 1) / (len(s_clean) - 1) * 100.0

        Percentile_ScoreRaw = calc_percentile(Score_raw)
        Percentile_NearHigh = calc_percentile(NearHigh)
        Percentile_Quality  = calc_percentile(Quality)

        w_tot = w_total if w_total > 0 else 1
        Score_Final = ((w1/w_tot) * Percentile_ScoreRaw) + ((w2/w_tot) * Percentile_NearHigh) + ((w3/w_tot) * Percentile_Quality)

        # Calculate Dynamic ATR Stop Loss
        atr_stop_loss = current_price - (atr_14 * atr_mult)

        results_df = pd.DataFrame({
            'Ticker': Score_Final.index,
            'Price (₹)': current_price.values,
            'Score_Final': Score_Final.values,
            'Blended Return (%)': R_blend.values * 100,
            'Downside Dev (%)': DD_126.values * 100,
            'NearHigh Ratio': NearHigh.values,
            'ATR Stop-Loss (₹)': atr_stop_loss.values
        }).sort_values(by='Score_Final', ascending=False).reset_index(drop=True)
        
        results_df.index = results_df.index + 1
        results_df.index.name = 'Rank'
        
        for col in ['Price (₹)', 'Score_Final', 'Blended Return (%)', 'Downside Dev (%)', 'ATR Stop-Loss (₹)']:
            results_df[col] = results_df[col].round(2)
        results_df['NearHigh Ratio'] = results_df['NearHigh Ratio'].round(3)

        progress_bar.progress(100)
        status_text.success(f"✅ Calculation complete! {len(results_df)} stocks qualified.")

        st.subheader("🏆 Leaderboard")
        st.dataframe(
            results_df.style.background_gradient(cmap='RdYlGn', subset=['Score_Final', 'Blended Return (%)'])
                          .background_gradient(cmap='RdYlGn_r', subset=['Downside Dev (%)']),
            use_container_width=True, height=600
        )

        st.download_button("📥 Download Rankings CSV", data=results_df.to_csv(), file_name="mk_momentum_rankings.csv", mime="text/csv")

    except Exception as e:
        status_text.empty()
        st.error(f"An error occurred: {str(e)}")
