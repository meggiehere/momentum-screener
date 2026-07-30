import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time

st.set_page_config(page_title="Clean Momentum Engine", page_icon="⚡", layout="wide")
st.title("⚡ The Clean Momentum Engine")
st.markdown("Focusing on Price Action, Trend Quality, and Basic Valuation.")

# -------------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------------
st.sidebar.header("1. Execution Settings")
account_capital = st.sidebar.number_input("Total Capital (₹)", value=10000.0, step=1000.0)
max_risk_pct = st.sidebar.number_input("Risk Per Trade (%)", value=1.0, step=0.1)
port_size = st.sidebar.number_input("Target Slots", value=3, step=1)
ema_period = st.sidebar.slider("Buy Zone EMA Window", 10, 50, 20)
max_ext_pct = st.sidebar.slider("Max Buy Extension (%)", 1.0, 10.0, 5.0, step=0.5)

st.sidebar.header("2. Strategy Weights")
w1 = st.sidebar.number_input("Momentum (W1)", value=80, step=5)
w2 = st.sidebar.number_input("NearHigh (W2)", value=20, step=5)
w3 = st.sidebar.number_input("Quality R² (W3)", value=0, step=5)

# -------------------------------------------------------------------
# Helper: Fetch Quick Fundamentals (Only for Top Stocks)
# -------------------------------------------------------------------
def get_quick_fundamentals(tickers):
    results = {}
    for t in tickers:
        tag, iv = "⚪ N/A", 0.0
        try:
            tk = yf.Ticker(t)
            
            # EPS QoQ Check
            q_fin = tk.quarterly_financials
            if q_fin is not None and not q_fin.empty and 'Net Income' in q_fin.index:
                net_inc = q_fin.loc['Net Income'].dropna()
                if len(net_inc) >= 2:
                    tag = "🟢 EPS Up" if net_inc.iloc[0] > net_inc.iloc[1] else "🔴 EPS Down"
            
            # Intrinsic Value (Graham Proxy)
            info = tk.info
            eps = info.get('trailingEps', 0)
            bv = info.get('bookValue', 0)
            if eps and bv and eps > 0 and bv > 0:
                iv = np.sqrt(22.5 * eps * bv)
                
        except Exception:
            pass
            
        results[t] = {"qoq": tag, "iv": iv}
        time.sleep(0.1) # Tiny pause to prevent rate limits
    return results

# -------------------------------------------------------------------
# Engine Core
# -------------------------------------------------------------------
tickers_input = st.text_area("Paste Tickers (Comma or newline separated):", height=100)

if st.button("🚀 Run Clean Engine") and tickers_input:
    # 1. Clean Tickers
    raw_tickers = tickers_input.replace('\n', ',').split(',')
    valid_tickers = []
    for t in raw_tickers:
        t = t.strip().upper()
        if t: valid_tickers.append(f"{t}.BO" if t.isdigit() else (t if t.endswith('.NS') or t.endswith('.BO') else f"{t}.NS"))
    valid_tickers = list(set(valid_tickers))
    
    st.write(f"Processing **{len(valid_tickers)}** tickers...")
    progress = st.progress(0)
    
    # 2. Download Price Data
    data = yf.download(valid_tickers, period="2y", progress=False)
    if len(valid_tickers) == 1:
        prices, highs, lows = data[['Close']], data[['High']], data[['Low']]
        prices.columns, highs.columns, lows.columns = valid_tickers, valid_tickers, valid_tickers
    else:
        prices, highs, lows = data['Close'], data['High'], data['Low']
        
    prices = prices.ffill(limit=5).dropna(axis=1, thresh=250)
    highs, lows = highs[prices.columns], lows[prices.columns]
    
    progress.progress(30)
    
    # 3. Apply 100-SMA Filter
    sma100 = prices.rolling(100).mean().iloc[-1]
    current_price = prices.iloc[-1]
    passed_sma = current_price[current_price > sma100].index.tolist()
    
    if not passed_sma:
        st.warning("No stocks passed the 100-SMA filter.")
        st.stop()
        
    prices, highs, lows = prices[passed_sma], highs[passed_sma], lows[passed_sma]
    current_price = current_price[passed_sma]
    
    progress.progress(50)
    
    # 4. Momentum & Quality Math
    P_skip, P_12M, P_6M, P_3M = prices.iloc[-22], prices.iloc[-273], prices.iloc[-147], prices.iloc[-84]
    R_blend = (0.50 * ((P_skip / P_12M) - 1)) + (0.30 * ((P_skip / P_6M) - 1)) + (0.20 * ((P_skip / P_3M) - 1))
    
    daily_ret = prices.pct_change(1).clip(upper=0)
    DD_126 = np.sqrt((daily_ret ** 2).rolling(126).sum().iloc[-1] / 126)
    Score_raw = R_blend / (DD_126 + 0.002)
    NearHigh = current_price / prices.rolling(252).max().iloc[-1]
    
    time_seq = pd.Series(np.arange(len(prices)), index=prices.index)
    rolling_r = prices.rolling(126).corr(time_seq)
    Quality = ((rolling_r ** 2) * np.sign(rolling_r)).iloc[-1]
    
    def percentile(s):
        return (s.rank(method='min') - 1) / (len(s) - 1) * 100.0 if len(s) > 1 else pd.Series(100.0, index=s.index)
        
    w_tot = w1 + w2 + w3 if (w1 + w2 + w3) > 0 else 1
    Score_Final = ((w1/w_tot)*percentile(Score_raw)) + ((w2/w_tot)*percentile(NearHigh)) + ((w3/w_tot)*percentile(Quality))
    Score_Final = Score_Final.sort_values(ascending=False)
    
    progress.progress(70)
    
    # 5. Risk & Extension Math (Only for sorted stocks)
    ordered_tickers = Score_Final.index
    top_prices = current_price[ordered_tickers]
    
    ema_20 = prices[ordered_tickers].ewm(span=ema_period, adjust=False).mean().iloc[-1]
    is_extended = top_prices > (ema_20 * (1 + (max_ext_pct / 100.0)))
    
    tr = np.maximum(highs - lows, np.maximum((highs - prices.shift(1)).abs(), (lows - prices.shift(1)).abs()))
    atr_stop = top_prices - (tr[ordered_tickers].rolling(14).mean().iloc[-1] * 3.0)
    
    cash_budget = account_capital / port_size
    shares_cash = np.floor(cash_budget / top_prices)
    shares_risk = np.floor((account_capital * (max_risk_pct / 100.0)) / (top_prices - atr_stop).replace(0, 0.01))
    final_shares = np.minimum(shares_risk, shares_cash).clip(lower=0).astype(int)
    
    # 6. Fast Fetch Fundamentals ONLY for Top 30
    top_30 = ordered_tickers[:30].tolist()
    st.text("Quick-fetching Fundamentals for Top 30 stocks...")
    fund_data = get_quick_fundamentals(top_30)
    
    qoq_tags = [fund_data.get(t, {}).get("qoq", "⚪ Skipped") for t in ordered_tickers]
    iv_vals = [fund_data.get(t, {}).get("iv", 0.0) for t in ordered_tickers]
    
    progress.progress(100)
    
    # 7. Build Clean DataFrame
    df = pd.DataFrame({
        'Ticker': ordered_tickers,
        'Price (₹)': top_prices.values,
        'Shares': final_shares.values,
        'Score': Score_Final.values,
        'Buy Zone?': np.where(is_extended, "🟡 Wait Dip", "🟢 IN ZONE"),
        'Intrinsic Val (₹)': iv_vals,
        'EPS Trend': qoq_tags,
        '20-EMA (₹)': ema_20.values,
        'ATR Stop (₹)': atr_stop.values
    }).reset_index(drop=True)
    
    df.index += 1
    
    # Add simple premium/discount visual tag
    df['Valuation'] = np.where(df['Intrinsic Val (₹)'] == 0, "N/A", 
                      np.where(df['Price (₹)'] < df['Intrinsic Val (₹)'], "🟢 Discount", "🔴 Premium"))
    cols = df.columns.tolist()
    cols.insert(6, cols.pop(cols.index('Valuation')))
    df = df[cols]
    
    for col in ['Price (₹)', 'Score', 'Intrinsic Val (₹)', '20-EMA (₹)', 'ATR Stop (₹)']:
        df[col] = df[col].round(2)
        
    st.success("✅ Clean Engine complete.")
    st.dataframe(df, use_container_width=True)
