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
    page_title="Delta Exchange Multi-Resolution Historical Viewer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Delta Exchange Historical Chart & Extrapolated Swing Channel")

BASE_URL = "https://api.india.delta.exchange"  # Change to "https://api.delta.exchange" for global API

# --- Sidebar Inputs ---
st.sidebar.header("Data Parameters")

symbol = st.sidebar.text_input("Symbol", value="BTCUSD").upper()

# Date Picker Range
default_start = datetime.now() - timedelta(days=180)
default_end = datetime.now()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(default_start, default_end),
    max_value=datetime.now()
)

st.sidebar.header("Chart & Layer Settings")
display_layer = st.sidebar.radio(
    "Primary Candlestick Series",
    options=["Daily Candles (Overlaid on Monthly Trend)", "Monthly Aggregated Candles"],
    index=0
)

show_daily_sma = st.sidebar.checkbox("Add Daily 20-period Moving Average (SMA)", value=True)
show_trendlines = st.sidebar.checkbox("Enable Extrapolated Channel Lines", value=True)
extrapolate_backward = st.sidebar.checkbox("Extrapolate Channel Backwards", value=True)
enable_canvas_drawing = st.sidebar.checkbox("Enable Native UI Drawing Tools", value=True)
future_steps = st.sidebar.slider("Future Extrapolation Steps", min_value=1, max_value=24, value=3)
swing_distance = st.sidebar.slider("Swing Sensitivity (Min Candle Gap)", min_value=1, max_value=10, value=1)


# --- DEDICATED FUNCTION: Enable Native Plotly Interactive Drawing Tools & Eraser ---
def enable_plotly_drawing_tools(fig: go.Figure, default_color: str = "#00ffff") -> tuple:
    fig.update_layout(
        dragmode="drawline",
        newshape=dict(
            line=dict(color=default_color, width=2, dash='solid'),
            fillcolor="rgba(0, 255, 255, 0.1)"
        )
    )

    plotly_config = {
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawrect',
            'eraseshape'
        ],
        'displaylogo': False,
        'scrollZoom': True
    }

    return fig, plotly_config


# --- Function: Extrapolate Swing Highs and Swing Lows ---
def extrapolate_swing_channel(
        fig: go.Figure,
        df: pd.DataFrame,
        extend_steps: int = 3,
        distance: int = 1,
        extrapolate_bwd: bool = True
):
    if len(df) < 2:
        return fig, 0.0, 0.0

    df_sorted = df.copy().reset_index(drop=True)

    high_peaks_idx, _ = find_peaks(df_sorted['high'].values, distance=distance)
    low_peaks_idx, _ = find_peaks(-df_sorted['low'].values, distance=distance)

    def get_points(peak_indices, col_name, find_max=True):
        if len(peak_indices) >= 2:
            subset = df_sorted.iloc[peak_indices]
            top2 = subset.nlargest(2, col_name) if find_max else subset.nsmallest(2, col_name)
            return top2.sort_values('time')

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

    time_diffs = df_sorted['time'].diff().dropna()
    step_td = time_diffs.median() if not time_diffs.empty else pd.Timedelta(days=30)

    start_time = df_sorted['time'].iloc[0]
    last_time = df_sorted['time'].iloc[-1]
    future_end_time = last_time + (step_td * extend_steps)

    def calculate_projection(p1, p2, col_name):
        dx_sec = (p2['time'] - p1['time']).total_seconds()
        slope = (p2[col_name] - p1[col_name]) / dx_sec if dx_sec != 0 else 0.0

        if extrapolate_bwd:
            x_start = start_time
            dt_back_sec = (start_time - p1['time']).total_seconds()
            y_start = p1[col_name] + (slope * dt_back_sec)
        else:
            x_start = p1['time']
            y_start = p1[col_name]

        x_end = future_end_time
        dt_fwd_sec = (future_end_time - p1['time']).total_seconds()
        y_end = p1[col_name] + (slope * dt_fwd_sec)

        return [x_start, x_end], [y_start, y_end], slope

    # --- Resistance Line ---
    x_res, y_res, res_slope = calculate_projection(p1_high, p2_high, 'high')
    fig.add_trace(go.Scatter(
        x=x_res, y=y_res, mode='lines', name='Extrapolated Resistance',
        line=dict(color='#ff4b4b', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=[p1_high['time'], p2_high['time']], y=[p1_high['high'], p2_high['high']],
        mode='markers', name='Swing High Anchors',
        marker=dict(color='#ff4b4b', size=9, symbol='triangle-down')
    ))

    # --- Support Line ---
    x_sup, y_sup, sup_slope = calculate_projection(p1_low, p2_low, 'low')
    fig.add_trace(go.Scatter(
        x=x_sup, y=y_sup, mode='lines', name='Extrapolated Support',
        line=dict(color='#00d47b', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=[p1_low['time'], p2_low['time']], y=[p1_low['low'], p2_low['low']],
        mode='markers', name='Swing Low Anchors',
        marker=dict(color='#00d47b', size=9, symbol='triangle-up')
    ))

    return fig, res_slope, sup_slope


# --- Function: Standard Monthly Parallel Lines ---
def draw_monthly_parallel_lines(
        fig: go.Figure,
        df: pd.DataFrame,
        monthly_line_configs: list,
        res_slope: float,
        sup_slope: float,
        extend_steps: int = 3
) -> go.Figure:
    if len(df) == 0 or not monthly_line_configs:
        return fig

    time_diffs = df['time'].diff().dropna()
    step_td = time_diffs.median() if not time_diffs.empty else pd.Timedelta(days=30)

    chart_start_time = df['time'].iloc[0]
    chart_end_time = df['time'].iloc[-1] + (step_td * extend_steps)

    for cfg in monthly_line_configs:
        if not cfg.get('enabled', True):
            continue

        selected_year_month = cfg['year_month']
        draw_mode = cfg['draw_mode']
        single_color = cfg['color']

        if not draw_mode or draw_mode == "-- Select Level --":
            continue

        month_df = df[df['time'].dt.strftime('%Y-%m') == selected_year_month]
        if month_df.empty:
            continue

        def add_parallel_trace(price_val, anchor_time, slope_to_use, label_title, marker_symbol):
            dt_start_sec = (chart_start_time - anchor_time).total_seconds()
            dt_end_sec = (chart_end_time - anchor_time).total_seconds()

            y_start = price_val + (slope_to_use * dt_start_sec)
            y_end = price_val + (slope_to_use * dt_end_sec)

            fig.add_trace(go.Scatter(
                x=[chart_start_time, chart_end_time],
                y=[y_start, y_end],
                mode='lines',
                name=label_title,
                line=dict(color=single_color, width=2, dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=[anchor_time],
                y=[price_val],
                mode='markers',
                name=f"{label_title} Anchor",
                marker=dict(color=single_color, size=10, symbol=marker_symbol)
            ))

        if draw_mode in ["Resistance Only (High w/ Resistance Slope)", "Both Parallel (High & Low Same Slope)"]:
            target_res = month_df.loc[month_df['high'].idxmax()]
            add_parallel_trace(target_res['high'], target_res['time'], res_slope,
                               f"{selected_year_month} Parallel Resistance", 'triangle-down')

        if draw_mode in ["Support Only (Low w/ Support Slope)", "Both Parallel (High & Low Same Slope)"]:
            target_sup = month_df.loc[month_df['low'].idxmin()]
            add_parallel_trace(target_sup['low'], target_sup['time'], sup_slope,
                               f"{selected_year_month} Parallel Support", 'triangle-up')

    return fig


# --- Function: Dedicated Cross-Parallel Monthly Lines ---
def draw_monthly_cross_parallel_lines(
        fig: go.Figure,
        df: pd.DataFrame,
        cross_configs: list,
        res_slope: float,
        sup_slope: float,
        extend_steps: int = 3
) -> go.Figure:
    if len(df) == 0 or not cross_configs:
        return fig

    time_diffs = df['time'].diff().dropna()
    step_td = time_diffs.median() if not time_diffs.empty else pd.Timedelta(days=30)

    chart_start_time = df['time'].iloc[0]
    chart_end_time = df['time'].iloc[-1] + (step_td * extend_steps)

    for cfg in cross_configs:
        if not cfg.get('enabled', True):
            continue

        selected_year_month = cfg['year_month']
        draw_mode = cfg['draw_mode']
        single_color = cfg['color']

        if not draw_mode or draw_mode == "-- Select Level --":
            continue

        month_df = df[df['time'].dt.strftime('%Y-%m') == selected_year_month]
        if month_df.empty:
            continue

        def add_parallel_trace(price_val, anchor_time, slope_to_use, label_title, marker_symbol):
            dt_start_sec = (chart_start_time - anchor_time).total_seconds()
            dt_end_sec = (chart_end_time - anchor_time).total_seconds()

            y_start = price_val + (slope_to_use * dt_start_sec)
            y_end = price_val + (slope_to_use * dt_end_sec)

            fig.add_trace(go.Scatter(
                x=[chart_start_time, chart_end_time],
                y=[y_start, y_end],
                mode='lines',
                name=label_title,
                line=dict(color=single_color, width=2, dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=[anchor_time],
                y=[price_val],
                mode='markers',
                name=f"{label_title} Anchor",
                marker=dict(color=single_color, size=10, symbol=marker_symbol)
            ))

        if draw_mode in ["High Parallel to Support Slope", "Cross-Parallel Both"]:
            target_res = month_df.loc[month_df['high'].idxmax()]
            add_parallel_trace(target_res['high'], target_res['time'], sup_slope,
                               f"{selected_year_month} High (w/ Support Slope)", 'triangle-down')

        if draw_mode in ["Low Parallel to Resistance Slope", "Cross-Parallel Both"]:
            target_sup = month_df.loc[month_df['low'].idxmin()]
            add_parallel_trace(target_sup['low'], target_sup['time'], res_slope,
                               f"{selected_year_month} Low (w/ Resistance Slope)", 'triangle-up')

    return fig


# --- Function: Dedicated Monthly Horizontal Lines ---
def draw_monthly_horizontal_lines(
        fig: go.Figure,
        df: pd.DataFrame,
        horizontal_configs: list,
        extend_steps: int = 3
) -> go.Figure:
    if len(df) == 0 or not horizontal_configs:
        return fig

    time_diffs = df['time'].diff().dropna()
    step_td = time_diffs.median() if not time_diffs.empty else pd.Timedelta(days=30)

    chart_start_time = df['time'].iloc[0]
    chart_end_time = df['time'].iloc[-1] + (step_td * extend_steps)

    for cfg in horizontal_configs:
        if not cfg.get('enabled', True):
            continue

        selected_year_month = cfg['year_month']
        level_mode = cfg['mode']
        color = cfg['color']

        if not level_mode or level_mode == "-- Select Level --":
            continue

        month_df = df[df['time'].dt.strftime('%Y-%m') == selected_year_month]
        if month_df.empty:
            continue

        if level_mode in ["Month High Horizontal", "Both Horizontal (High & Low)"]:
            target_high = month_df.loc[month_df['high'].idxmax()]
            high_val = target_high['high']
            anchor_time = target_high['time']

            fig.add_trace(go.Scatter(
                x=[chart_start_time, chart_end_time],
                y=[high_val, high_val],
                mode='lines',
                name=f"{selected_year_month} Horizontal High (${high_val:,.2f})",
                line=dict(color=color, width=1.5, dash='dashdot')
            ))
            fig.add_trace(go.Scatter(
                x=[anchor_time],
                y=[high_val],
                mode='markers',
                name=f"{selected_year_month} High Anchor",
                marker=dict(color=color, size=9, symbol='circle')
            ))

        if level_mode in ["Month Low Horizontal", "Both Horizontal (High & Low)"]:
            target_low = month_df.loc[month_df['low'].idxmin()]
            low_val = target_low['low']
            anchor_time = target_low['time']

            fig.add_trace(go.Scatter(
                x=[chart_start_time, chart_end_time],
                y=[low_val, low_val],
                mode='lines',
                name=f"{selected_year_month} Horizontal Low (${low_val:,.2f})",
                line=dict(color=color, width=1.5, dash='dashdot')
            ))
            fig.add_trace(go.Scatter(
                x=[anchor_time],
                y=[low_val],
                mode='markers',
                name=f"{selected_year_month} Low Anchor",
                marker=dict(color=color, size=9, symbol='circle')
            ))

    return fig


# --- Function to Paginate & Fetch Historical Data ---
def fetch_all_historical_candles(symbol_name, resolution, start_dt, end_dt):
    start_ts = int(datetime.combine(start_dt, datetime.min.time()).timestamp())
    end_ts = int(datetime.combine(end_dt, datetime.max.time()).timestamp())

    api_resolution = "1d" if resolution == "1M" else resolution

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
            status_text.text(f"Fetching {symbol_name} historical candles...")

            time.sleep(0.05)

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            break

    progress_bar.empty()
    status_text.empty()

    if not all_candles:
        return pd.DataFrame(), pd.DataFrame()

    df_daily = pd.DataFrame(all_candles)
    df_daily = df_daily.drop_duplicates(subset=['time'])
    df_daily['time'] = pd.to_datetime(df_daily['time'], unit='s')
    df_daily = df_daily.sort_values('time').reset_index(drop=True)

    # Monthly Aggregated Resample
    df_monthly = df_daily.set_index('time').resample('ME').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()

    return df_daily, df_monthly


# --- Main Application Logic ---
if len(date_range) == 2:
    start_date, end_date = date_range

    if st.sidebar.button("Fetch & Render Chart", type="primary"):
        df_daily, df_monthly = fetch_all_historical_candles(symbol, "1d", start_date, end_date)
        st.session_state['df_daily'] = df_daily
        st.session_state['df_monthly'] = df_monthly

if 'df_daily' in st.session_state and not st.session_state['df_daily'].empty:
    df_daily = st.session_state['df_daily']
    df_monthly = st.session_state['df_monthly']

    st.success(f"Loaded **{len(df_daily):,}** daily candles across **{len(df_monthly)}** months for **{symbol}**!")

    available_months = df_monthly['time'].dt.strftime('%Y-%m').unique().tolist()

    # --- SIDEBAR CONTROL 1: Standard Monthly Parallel Lines ---
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Monthly Parallel Lines")
    num_monthly_lines = st.sidebar.number_input("Number of Standard Selections", min_value=0, max_value=5, value=1,
                                                key="sb_num_par")

    monthly_line_configs = []
    for i in range(int(num_monthly_lines)):
        with st.sidebar.expander(f"Standard Selection {i + 1}", expanded=True):
            is_enabled = st.checkbox("Enable / Render", value=True, key=f"enable_par_{i}")
            sel_month = st.selectbox("Select Month", options=available_months, key=f"month_par_sel_{i}")

            draw_mode = st.selectbox(
                "Lines to Render",
                options=[
                    "-- Select Level --",
                    "Both Parallel (High & Low Same Slope)",
                    "Support Only (Low w/ Support Slope)",
                    "Resistance Only (High w/ Resistance Slope)"
                ],
                index=0,
                key=f"draw_par_mode_{i}"
            )
            selected_color = st.color_picker("Line Color", value="#00ffff" if i % 2 == 0 else "#ff00ff",
                                             key=f"line_par_col_{i}")

            monthly_line_configs.append({
                'enabled': is_enabled,
                'year_month': sel_month,
                'draw_mode': draw_mode,
                'color': selected_color
            })

    # --- SIDEBAR CONTROL 2: Monthly Cross-Parallel Lines ---
    st.sidebar.markdown("---")
    st.sidebar.header("🔀 Monthly Cross-Parallel Lines")
    num_cross_lines = st.sidebar.number_input("Number of Cross-Parallel Selections", min_value=0, max_value=5, value=0,
                                              key="sb_num_cross")

    cross_line_configs = []
    for i in range(int(num_cross_lines)):
        with st.sidebar.expander(f"Cross-Parallel Selection {i + 1}", expanded=True):
            is_enabled = st.checkbox("Enable / Render", value=True, key=f"enable_cross_{i}")
            sel_month = st.selectbox("Select Month", options=available_months, key=f"month_cross_sel_{i}")

            draw_mode = st.selectbox(
                "Cross-Parallel Option",
                options=[
                    "-- Select Level --",
                    "High Parallel to Support Slope",
                    "Low Parallel to Resistance Slope",
                    "Cross-Parallel Both"
                ],
                index=0,
                key=f"draw_cross_mode_{i}"
            )
            selected_color = st.color_picker("Line Color", value="#a29bfe" if i % 2 == 0 else "#fd79a8",
                                             key=f"line_cross_col_{i}")

            cross_line_configs.append({
                'enabled': is_enabled,
                'year_month': sel_month,
                'draw_mode': draw_mode,
                'color': selected_color
            })

    # --- SIDEBAR CONTROL 3: Dedicated Monthly Horizontal Lines ---
    st.sidebar.markdown("---")
    st.sidebar.header("➖ Monthly Horizontal Lines")
    num_hz_lines = st.sidebar.number_input("Number of Horizontal Selections", min_value=0, max_value=5, value=0,
                                           key="sb_num_hz")

    horizontal_line_configs = []
    for i in range(int(num_hz_lines)):
        with st.sidebar.expander(f"Horizontal Selection {i + 1}", expanded=True):
            is_enabled = st.checkbox("Enable / Render", value=True, key=f"enable_hz_{i}")
            sel_month = st.selectbox("Select Month", options=available_months, key=f"month_hz_sel_{i}")

            hz_mode = st.selectbox(
                "Level to Render",
                options=[
                    "-- Select Level --",
                    "Both Horizontal (High & Low)",
                    "Month High Horizontal",
                    "Month Low Horizontal"
                ],
                index=0,
                key=f"mode_hz_{i}"
            )
            selected_color = st.color_picker("Line Color", value="#e67e22" if i % 2 == 0 else "#f1c40f",
                                             key=f"line_hz_col_{i}")

            horizontal_line_configs.append({
                'enabled': is_enabled,
                'year_month': sel_month,
                'mode': hz_mode,
                'color': selected_color
            })

    # --- Build Plotly Figure ---
    fig = go.Figure()

    # Dynamic Layer Selection
    if display_layer == "Daily Candles (Overlaid on Monthly Trend)":
        active_df = df_daily
        candle_name = "Daily OHLC"
    else:
        active_df = df_monthly
        candle_name = "Monthly OHLC"

    # Candlestick Series
    fig.add_trace(go.Candlestick(
        x=active_df['time'],
        open=active_df['open'],
        high=active_df['high'],
        low=active_df['low'],
        close=active_df['close'],
        name=candle_name,
        increasing_line_color='#0ecb81',
        decreasing_line_color='#f6465d'
    ))

    # Daily SMA
    if show_daily_sma and len(df_daily) >= 20:
        df_daily['SMA20'] = df_daily['close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=df_daily['time'],
            y=df_daily['SMA20'],
            mode='lines',
            name='Daily SMA 20',
            line=dict(color='#f39c12', width=1.5)
        ))

    # Extrapolated Swing Channel calculated on Monthly Data
    res_slope, sup_slope = 0.0, 0.0
    if show_trendlines:
        fig, res_slope, sup_slope = extrapolate_swing_channel(
            fig,
            df_monthly,
            extend_steps=future_steps,
            distance=swing_distance,
            extrapolate_bwd=extrapolate_backward
        )

    # Render Standard Parallel Lines using Monthly Anchors
    if monthly_line_configs:
        fig = draw_monthly_parallel_lines(
            fig,
            df_monthly,
            monthly_line_configs,
            res_slope=res_slope,
            sup_slope=sup_slope,
            extend_steps=future_steps
        )

    # Render Cross-Parallel Lines using Monthly Anchors
    if cross_line_configs:
        fig = draw_monthly_cross_parallel_lines(
            fig,
            df_monthly,
            cross_line_configs,
            res_slope=res_slope,
            sup_slope=sup_slope,
            extend_steps=future_steps
        )

    # Render Dedicated Horizontal Lines using Monthly Anchors
    if horizontal_line_configs:
        fig = draw_monthly_horizontal_lines(
            fig,
            df_monthly,
            horizontal_line_configs,
            extend_steps=future_steps
        )

    # Layout Customization
    fig.update_layout(
        title=f"{symbol} Multi-Resolution Interactive Chart (Daily Candles with Monthly Levels)",
        yaxis_title="Price (USD)",
        xaxis_title="Timeline",
        template="plotly_dark",
        height=680,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117"
    )

    fig.update_xaxes(
        dtick="M1",
        tickformat="%b %Y",
        ticklabelmode="period",
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.1)",
        gridwidth=1,
        ticks="outside",
        ticklen=6,
        tickcolor="#888888"
    )

    # Apply Canvas Drawing and Eraser
    plotly_config = {}
    if enable_canvas_drawing:
        fig, plotly_config = enable_plotly_drawing_tools(fig, default_color="#00ffff")

    st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    # Summary Metrics
    latest_daily = df_daily.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Latest Daily Open", f"${latest_daily['open']:,.2f}")
    c2.metric("Period High", f"${df_daily['high'].max():,.2f}")
    c3.metric("Period Low", f"${df_daily['low'].min():,.2f}")
    c4.metric("Latest Daily Close", f"${latest_daily['close']:,.2f}")
    c5.metric("Total Daily Volume", f"{df_daily['volume'].sum():,.0f}")

    st.markdown("### Data Downloads")
    csv_data = df_daily.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Daily Data as CSV",
        data=csv_data,
        file_name=f"{symbol}_daily_candles.csv",
        mime="text/csv"
    )

    with st.expander("View Raw Daily Candle Table"):
        st.dataframe(df_daily[['time', 'open', 'high', 'low', 'close', 'volume']].sort_values('time', ascending=False),
                     use_container_width=True)

else:
    st.info("Configure date parameters in the sidebar and click **Fetch & Render Chart** to load.")