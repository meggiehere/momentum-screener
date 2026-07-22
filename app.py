import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# -------------------------------------------------------------------
# 1. Page Configuration & Setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="MK Momentum Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mobile responsiveness and cleaner tables
st.markdown("""
    <style>
    .stDataFrame { width: 100%; }
    .stButton>button { width: 100%; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. Sidebar: Strategy Settings & Parameters
# -------------------------------------------------------------------
st.sidebar.title("⚙️ Strategy Settings")

st.sidebar.header("1. Component Weights (%)")
w1 = st.sidebar.number_input("Momentum (W1)", min_value=0, max_value=100, value=80, step=5)
w2 = st.sidebar.number_input("NearHigh (W2)", min_value=0, max_value=100, value=20, step=5)
w3 = st.sidebar.number_input("Quality (W3)", min_value=0, max_value=100, value=0, step=5)

w_total = w1 + w2 + w3
if w_total != 100:
    st.sidebar.warning(f"⚠️ Weights sum to {w_total}, not 100. They will be normalized automatically.")

st.sidebar.header("2. Timeframe Blend (Sum=1.0)")
w_12m = st.sidebar.number_input("12M Return Weight", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
w_6m = st.sidebar.number_input("6M Return Weight", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
w_3m = st.sidebar.number_input("3M Return Weight", min_value=0.0, max_value=1.0, value=0.20, step=0.05)

st.sidebar.header("3. Lookback & Windows")
skip_days = st.sidebar.slider("Skip Days (Momentum lag)", 5, 60, 21)
dd_lookback = st.sidebar.slider("Downside Lookback (Days)", 30, 252, 126)
nh_lookback = st.sidebar.slider("NearHigh Lookback (Days)", 63, 504, 252)

st.sidebar.header("4. Portfolio Risk Controls")
trail_sl_pct = st.sidebar.slider("Trailing Stop Loss (%)", 5, 30, 15)
tox_cutoff = st.sidebar.slider("Toxicity Cutoff (%)", 0, 50, 10, help="Excludes top X% most volatile downside stocks")

# -------------------------------------------------------------------
# 3. Main UI: Data Input
# -------------------------------------------------------------------
st.title("🚀 Quant Equity Screener: MK_MOMENTUM")
st.markdown("Rank Indian Equities (NSE/BSE) using a factor-momentum and downside-risk algorithmic engine.")

st.subheader("📥 Input Tickers")
col1, col2 = st.columns(2)

with col1:
    ticker_text = st.text_area(
        "Paste Tickers (Comma or newline separated)",
        "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, 544026",
        height=150
    )

with col2:
    uploaded_file = st.file_uploader("Or Upload CSV (Must contain a 'Ticker' column)", type=['csv'])

# Process Tickers
tickers = []

# Process CSV if uploaded
if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)
        ticker_col = next((col for col in df_upload.columns if 'ticker' in col.lower()), None)
        if ticker_col:
            tickers.extend(df_upload[ticker_col].dropna().astype(str).tolist())
        else:
            st.error("CSV must contain a column named 'Ticker'")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# Process Text Area using Smart Logic
if ticker_text:
    raw_tickers = ticker_text.replace('\n', ',').split(',')
    
    for t in raw_tickers:
        t = t.strip().upper()
        
        # Skip empty strings
        if not t:
            continue
            
        # Check if the user already included a suffix
        if not (t.endswith('.NS') or t.endswith('.BO')):
            # If the ticker is strictly digits, it's a BSE stock
            if t.isdigit():
                t = f"{t}.BO"
            # Otherwise, default to NSE
            else:
                t = f"{t}.NS"
                
        tickers.append(t)

# Deduplicate tickers
tickers = list(set(tickers))

st.write(f"**Total unique tickers loaded:** {len(tickers)}")

# -------------------------------------------------------------------
# 4. Engine Execution
# -------------------------------------------------------------------
if st.button("🚀 Run Momentum Engine") and len(tickers) > 0:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Fetching market data from Yahoo Finance...")
    
    try:
        # Fetch data: 2 years history to ensure enough data for 252-day lookbacks
        raw_data = yf.download(tickers, period="2y", progress=False)
        
        # Handle yfinance single vs multi-ticker output structure gracefully
        if len(tickers) == 1:
            if 'Close' in raw_data.columns:
                prices_df = raw_data[['Close']].rename(columns={'Close': tickers[0]})
            else:
                prices_df = pd.DataFrame()
        else:
            # yfinance returns MultiIndex columns for multiple tickers
            if 'Close' in raw_data.columns:
                prices_df = raw_data['Close']
            elif isinstance(raw_data.columns, pd.MultiIndex) and 'Close' in raw_data.columns.levels[0]:
                prices_df = raw_data['Close']
            else:
                prices_df = raw_data
                
        if prices_df.empty:
            st.error("Could not find price data for the provided tickers.")
            st.stop()
            
        progress_bar.progress(30)
        status_text.text("Data fetched. Applying minimum data filters...")

        # Minimum Data Filter: (252 + SKIP) trading days
        min_required_days = 252 + skip_days
        valid_counts = prices_df.count()
        valid_tickers = valid_counts[valid_counts >= min_required_days].index.tolist()
        
        if not valid_tickers:
            st.error(f"No tickers have the minimum required history of {min_required_days} trading days.")
            st.stop()
            
        prices_df = prices_df[valid_tickers]
        # Forward fill up to 5 days to handle minor data gaps
        prices_df = prices_df.ffill(limit=5)
        
        progress_bar.progress(50)
        status_text.text("Calculating Multi-Timeframe Blended Returns & Risk...")

        # Core Mathematical Logic Vectorization
        current_price = prices_df.iloc[-1]
        P_skip = prices_df.iloc[-(1 + skip_days)]
        P_12M = prices_df.iloc[-(252 + skip_days + 1)]
        P_6M = prices_df.iloc[-(126 + skip_days + 1)]
        P_3M = prices_df.iloc[-(63 + skip_days + 1)]

        # A. Multi-Timeframe Blended Return
        R_12M = (P_skip / P_12M) - 1
        R_6M  = (P_skip / P_6M) - 1
        R_3M  = (P_skip / P_3M) - 1
        
        R_blend = (w_12m * R_12M) + (w_6m * R_6M) + (w_3m * R_3M)

        # B. Downside Deviation
        daily_returns = prices_df.pct_change(1)
        negative_returns = daily_returns.clip(upper=0)
        DD_126 = np.sqrt((negative_returns ** 2).rolling(dd_lookback).sum().iloc[-1] / dd_lookback)

        progress_bar.progress(70)
        status_text.text("Evaluating 52-Week High Proximity & Quality...")

        # C. Raw Risk-Adjusted Score
        Score_raw = R_blend / (DD_126 + 0.002)

        # D. 52-Week High Proximity & Quality
        High_52W = prices_df.rolling(nh_lookback).max().iloc[-1]
        NearHigh = current_price / High_52W
        Quality = (daily_returns > 0).rolling(dd_lookback).sum().iloc[-1] / dd_lookback

        # Apply Toxicity Cutoff
        if tox_cutoff > 0:
            dd_threshold = np.percentile(DD_126.dropna(), 100 - tox_cutoff)
            valid_mask = DD_126 <= dd_threshold
            
            Score_raw = Score_raw[valid_mask]
            NearHigh = NearHigh[valid_mask]
            Quality = Quality[valid_mask]
            current_price = current_price[valid_mask]
            R_blend = R_blend[valid_mask]
            DD_126 = DD_126[valid_mask]
            
        progress_bar.progress(85)
        status_text.text("Normalizing and generating final ranks...")

        # F. Percentile Normalization (Strictly Less Than Rank)
        def calc_percentile(series):
            s_clean = series.dropna()
            N = len(s_clean)
            if N <= 1:
                return pd.Series(100.0, index=s_clean.index)
            return (s_clean.rank(method='min') - 1) / (N - 1) * 100.0

        Percentile_ScoreRaw = calc_percentile(Score_raw)
        Percentile_NearHigh = calc_percentile(NearHigh)
        Percentile_Quality  = calc_percentile(Quality)

        # G. Final Composite Score
        # Avoid division by zero if all weights are set to 0
        w_tot = w_total if w_total > 0 else 1
        norm_w1, norm_w2, norm_w3 = (w1/w_tot), (w2/w_tot), (w3/w_tot)

        Score_Final = (norm_w1 * Percentile_ScoreRaw) + (norm_w2 * Percentile_NearHigh) + (norm_w3 * Percentile_Quality)

        # Assemble Output DataFrame
        results_df = pd.DataFrame({
            'Ticker': Score_Final.index,
            'Price (₹)': current_price.values,
            'Score_Final': Score_Final.values,
            'Blended Return (%)': R_blend.values * 100,
            'Downside Dev (%)': DD_126.values * 100,
            'NearHigh Ratio': NearHigh.values,
            'Stop-Loss (₹)': current_price.values * (1 - (trail_sl_pct / 100))
        })

        # Sort and Rank
        results_df = results_df.sort_values(by='Score_Final', ascending=False).reset_index(drop=True)
        results_df.index = results_df.index + 1
        results_df.index.name = 'Rank'
        
        # Format table numbers
        results_df['Price (₹)'] = results_df['Price (₹)'].round(2)
        results_df['Score_Final'] = results_df['Score_Final'].round(2)
        results_df['Blended Return (%)'] = results_df['Blended Return (%)'].round(2)
        results_df['Downside Dev (%)'] = results_df['Downside Dev (%)'].round(2)
        results_df['NearHigh Ratio'] = results_df['NearHigh Ratio'].round(3)
        results_df['Stop-Loss (₹)'] = results_df['Stop-Loss (₹)'].round(2)

        progress_bar.progress(100)
        status_text.success(f"✅ Calculation complete! {len(results_df)} stocks qualified.")

        # -------------------------------------------------------------------
        # 5. Dashboard Output
        # -------------------------------------------------------------------
        st.subheader("🏆 Leaderboard")
        st.dataframe(
            results_df.style.background_gradient(cmap='RdYlGn', subset=['Score_Final', 'Blended Return (%)'])
                          .background_gradient(cmap='RdYlGn_r', subset=['Downside Dev (%)']),
            use_container_width=True,
            height=600
        )

        # Export Functionality
        csv = results_df.to_csv()
        st.download_button(
            label="📥 Download Rankings CSV",
            data=csv,
            file_name="mk_momentum_rankings.csv",
            mime="text/csv",
        )

    except Exception as e:
        status_text.empty()
        st.error(f"An error occurred during execution: {str(e)}")
        st.exception(e)
