import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from scipy.signal import find_peaks

# --- Page Setup ---
st.set_page_config(
    page_title="Delta Exchange Historical & Trendline Viewer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Delta Exchange Historical Chart & Extrapolated Swing Channel")

BASE_URL = "https://api.india.delta.exchange"  # Change to "https://api.delta.exchange" for global API

# --- Sidebar Inputs ---
st.sidebar.header("Data Parameters")

symbol = st.sidebar.text_input("Symbol", value="BTCUSD").upper()

# Updated Resolution Options (1M now handled via daily resampling)
resolutions = {
    "1min": "1 Minute",
    "5m": "5 Minutes",
    "15m": "15 Minutes",
    "1h": "1 Hour",
    "4h": "4 Hours",
    "1d": "1 Day",
    "1M": "1 Month"
}
selected_res = st.sidebar.selectbox("Resolution", options=list(resolutions.keys()), index=6)

# Date Picker Range
default_start = datetime.now() - timedelta(days=90)
default_end = datetime.now()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(default_start, default_end),
    max_value=datetime.now()
)

st.sidebar.header("Chart Settings")
show_sma = st.sidebar.checkbox("Add 20-period Moving Average (SMA)", value=True)
show_trendlines = st.sidebar.checkbox("Enable Extrapolated Channel Lines", value=True)
extrapolate_backward = st.sidebar.checkbox("Extrapolate Backwards (To Start of Chart)", value=True)
future_steps = st.sidebar.slider("Future Extrapolation Steps", min_value=1, max_value=24, value=3)
swing_distance = st.sidebar.slider("Swing Sensitivity (Min Candle Gap)", min_value=1, max_value=10, value=1)


# --- Function: Extrapolate Swing Highs and Swing Lows (Forward & Backward) ---
def extrapolate_swing_channel(
        fig: go.Figure,
        df: pd.DataFrame,
        extend_steps: int = 3,
        distance: int = 1,
        extrapolate_bwd: bool = True
) -> go.Figure:
    if len(df) < 2:
        return fig

    df_sorted = df.copy().reset_index(drop=True)

    # 1. Identify Swing High & Low indices
    high_peaks_idx, _ = find_peaks(df_sorted['high'].values, distance=distance)
    low_peaks_idx, _ = find_peaks(-df_sorted['low'].values, distance=distance)

    # Helper function to reliably select 2 distinct points (with fallback if find_peaks returns < 2 points)
    def get_points(peak_indices, col_name, find_max=True):
        if len(peak_indices) >= 2:
            subset = df_sorted.iloc[peak_indices]
            top2 = subset.nlargest(2, col_name) if find_max else subset.nsmallest(2, col_name)
            return top2.sort_values('time')

        # Fallback: Select the 2 extreme points separated in time
        sorted_df = df_sorted.sort_values(col_name, ascending=not find_max)
        p1 = sorted_df.iloc[0]

        candidates = df_sorted[abs(df_sorted.index - p1.name) >= max(1, distance)]
        if candidates.empty:
            candidates = df_sorted[df_sorted.index != p1.name]

        p2 = candidates.sort_values(col_name, ascending=not find_max).iloc[0]
        return pd.DataFrame([p1, p2]).sort_values('time')

    swing_highs = get_points(high_peaks_idx, 'high', find_max=True)
    swing_lows = get_points(low_peaks_idx, 'low', find_max=False)

    p1_high, p2_high = swing_highs.iloc[0], swing_highs.iloc[1]
    p1_low, p2_low = swing_lows.iloc[0], swing_lows.iloc[1]

    # Calculate time horizons for extrapolation
    time_diffs = df_sorted['time'].diff().dropna()
    step_td = time_diffs.median() if not time_diffs.empty else pd.Timedelta(days=30)

    start_time = df_sorted['time'].iloc[0]
    last_time = df_sorted['time'].iloc[-1]
    future_end_time = last_time + (step_td * extend_steps)

    # Helper to calculate start & end points for linear slope (y = mx + c)
    def calculate_projection(p1, p2, col_name):
        dx_sec = (p2['time'] - p1['time']).total_seconds()
        slope = (p2[col_name] - p1[col_name]) / dx_sec if dx_sec != 0 else 0.0

        # Determine start point (Backwards Extrapolation)
        if extrapolate_bwd:
            x_start = start_time
            dt_back_sec = (start_time - p1['time']).total_seconds()
            y_start = p1[col_name] + (slope * dt_back_sec)
        else:
            x_start = p1['time']
            y_start = p1[col_name]

        # Determine end point (Forward Extrapolation)
        x_end = future_end_time
        dt_fwd_sec = (future_end_time - p1['time']).total_seconds()
        y_end = p1[col_name] + (slope * dt_fwd_sec)

        return [x_start, x_end], [y_start, y_end]

    # --- Draw Resistance Line (Highs) ---
    x_res, y_res = calculate_projection(p1_high, p2_high, 'high')
    fig.add_trace(go.Scatter(
        x=x_res, y=y_res, mode='lines', name='Extrapolated Resistance',
        line=dict(color='#ff4b4b', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=[p1_high['time'], p2_high['time']], y=[p1_high['high'], p2_high['high']],
        mode='markers', name='Swing High Anchors',
        marker=dict(color='#ff4b4b', size=9, symbol='triangle-down')
    ))

    # --- Draw Support Line (Lows) ---
    x_sup, y_sup = calculate_projection(p1_low, p2_low, 'low')
    fig.add_trace(go.Scatter(
        x=x_sup, y=y_sup, mode='lines', name='Extrapolated Support',
        line=dict(color='#00d47b', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=[p1_low['time'], p2_low['time']], y=[p1_low['low'], p2_low['low']],
        mode='markers', name='Swing Low Anchors',
        marker=dict(color='#00d47b', size=9, symbol='triangle-up')
    ))

    return fig


# --- Function to Paginate & Fetch Historical Data ---
def fetch_all_historical_candles(symbol_name, resolution, start_dt, end_dt):
    start_ts = int(datetime.combine(start_dt, datetime.min.time()).timestamp())
    end_ts = int(datetime.combine(end_dt, datetime.max.time()).timestamp())

    # Map internal key back to valid Delta API resolutions
    if resolution == "1min":
        api_resolution = "1m"
    elif resolution == "1M":
        api_resolution = "1d"  # Fetch daily data and resample locally to monthly
    else:
        api_resolution = resolution

    all_candles = []
    current_end = end_ts

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_span = end_ts - start_ts if end_ts > start_ts else 1

    while current_end > start_ts:
        params = {
            'symbol': symbol_name,
            'resolution': api_resolution,
            'start': str(start_ts),
            'end': str(current_end)
        }

        try:
            res = requests.get(f"{BASE_URL}/v2/history/candles", params=params)
            if res.status_code != 200:
                st.error(f"API Error {res.status_code}: {res.text}")
                break

            data = res.json().get('result', [])
            if not data:
                break

            all_candles.extend(data)

            oldest_time = min(item['time'] for item in data)
            if oldest_time <= start_ts:
                break

            current_end = oldest_time - 1

            fetched_span = end_ts - current_end
            progress = min(1.0, max(0.0, fetched_span / total_span))
            progress_bar.progress(progress)
            status_text.text(f"Fetched {len(all_candles):,} raw candles...")

            time.sleep(0.05)

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            break

    progress_bar.empty()
    status_text.empty()

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df = df.drop_duplicates(subset=['time'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('time').reset_index(drop=True)

    # --- Resample Daily to Monthly if 1M was selected ---
    if resolution == "1M":
        df = df.set_index('time')
        df_monthly = df.resample('ME').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        return df_monthly

    return df


# --- Main Application Logic ---
if len(date_range) == 2:
    start_date, end_date = date_range

    if st.sidebar.button("Fetch & Render Chart", type="primary"):
        st.session_state['df'] = fetch_all_historical_candles(symbol, selected_res, start_date, end_date)

if 'df' in st.session_state and not st.session_state['df'].empty:
    df = st.session_state['df']

    st.success(f"Successfully loaded **{len(df):,}** candles for **{symbol}**!")

    # --- Build Plotly Figure ---
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="OHLC",
        increasing_line_color='#0ecb81',
        decreasing_line_color='#f6465d'
    ))

    # Optional SMA
    if show_sma and len(df) >= 20:
        df['SMA20'] = df['close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['SMA20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='#f39c12', width=1.5)
        ))

    # Optional Extrapolated Swing Channel
    if show_trendlines:
        fig = extrapolate_swing_channel(
            fig,
            df,
            extend_steps=future_steps,
            distance=swing_distance,
            extrapolate_bwd=extrapolate_backward
        )

    fig.update_layout(
        title=f"{symbol} ({resolutions[selected_res]}) Historical Chart & Extrapolated Channel",
        yaxis_title="Price (USD)",
        xaxis_title="Time",
        template="plotly_dark",
        height=650,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics & Actions
    latest = df.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Latest Open", f"${latest['open']:,.2f}")
    c2.metric("Period High", f"${df['high'].max():,.2f}")
    c3.metric("Period Low", f"${df['low'].min():,.2f}")
    c4.metric("Latest Close", f"${latest['close']:,.2f}")
    c5.metric("Total Volume", f"{df['volume'].sum():,.0f}")

    st.markdown("### Historical Data Actions")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv_data,
        file_name=f"{symbol}_{selected_res}_candles.csv",
        mime="text/csv"
    )

    with st.expander("View Raw Candle Table"):
        st.dataframe(df[['time', 'open', 'high', 'low', 'close', 'volume']].sort_values('time', ascending=False),
                     use_container_width=True)

else:
    st.info("Configure parameters in the sidebar and click **Fetch & Render Chart** to load.")