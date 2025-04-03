import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Load the data
with open('nifty50_analysis_results.json') as f:
    data = json.load(f)

# Convert to DataFrame for easier manipulation
df = pd.DataFrame.from_dict(data, orient='index')

# Sidebar filters
st.sidebar.title("Filters")
selected_stocks = st.sidebar.multiselect(
    "Select Stocks", 
    df.index.tolist(), 
    default=df.index.tolist()[:5]
)

# Main dashboard
st.title("Nifty50 Stock Analysis Dashboard")
st.write("Comprehensive analysis of Nifty50 stocks including technical indicators, fundamental metrics, and earnings predictions")

# Overview section
st.header("Stock Overview")
overview_cols = ['status', 'technical.last_close', 'technical.trend', 'earnings.next_earnings_prediction']
overview_df = df[df.index.isin(selected_stocks)][overview_cols]
overview_df.columns = ['Status', 'Last Close', 'Trend', 'Next Earnings']
st.dataframe(overview_df.style.format({
    'Last Close': '{:.2f}'
}), height=(len(selected_stocks) * 35 + 38))

# Technical Analysis Section
st.header("Technical Analysis")

# Technical metrics comparison
tech_cols = [
    'technical.rsi', 'technical.ma_20', 'technical.ma_50', 'technical.ma_200',
    'technical.macd_line', 'technical.macd_signal', 'technical.bb_upper', 
    'technical.bb_lower', 'technical.last_close'
]
tech_df = df[df.index.isin(selected_stocks)][tech_cols]
tech_df.columns = [
    'RSI', 'MA(20)', 'MA(50)', 'MA(200)', 'MACD Line', 'MACD Signal', 
    'BB Upper', 'BB Lower', 'Last Close'
]

# Normalize for comparison
tech_normalized = tech_df.copy()
for col in tech_normalized.columns:
    if col != 'RSI':  # RSI is already normalized
        tech_normalized[col] = (tech_normalized[col] - tech_normalized[col].min()) / (tech_normalized[col].max() - tech_normalized[col].min())

fig = px.bar(
    tech_normalized.T, 
    barmode='group',
    title="Normalized Technical Indicators Comparison"
)
st.plotly_chart(fig, use_container_width=True)

# RSI Analysis
st.subheader("RSI Analysis")
rsi_fig = go.Figure()
for stock in selected_stocks:
    rsi_fig.add_trace(go.Bar(
        x=[stock],
        y=[df.loc[stock, 'technical.rsi']],
        name=stock,
        text=f"{df.loc[stock, 'technical.rsi']:.2f}",
        textposition='auto'
    ))
rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
rsi_fig.update_layout(title="RSI Values", yaxis_title="RSI")
st.plotly_chart(rsi_fig, use_container_width=True)

# Moving Averages
st.subheader("Moving Averages vs Last Close")
ma_fig = go.Figure()
for stock in selected_stocks:
    ma_fig.add_trace(go.Bar(
        x=['MA(20)', 'MA(50)', 'MA(200)', 'Last Close'],
        y=[
            df.loc[stock, 'technical.ma_20'],
            df.loc[stock, 'technical.ma_50'],
            df.loc[stock, 'technical.ma_200'],
            df.loc[stock, 'technical.last_close']
        ],
        name=stock
    ))
ma_fig.update_layout(barmode='group', title="Moving Averages Comparison")
st.plotly_chart(ma_fig, use_container_width=True)

# Fundamental Analysis Section
st.header("Fundamental Analysis")

# Revenue and Profit Trends
st.subheader("Revenue and Profit Trends")
fundamental_metrics = st.selectbox(
    "Select Metric", 
    ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']
)

fund_fig = go.Figure()
for stock in selected_stocks:
    if isinstance(df.loc[stock, 'fundamental.historical'], dict):
        historical_data = df.loc[stock, 'fundamental.historical'].get(fundamental_metrics, {})
        if historical_data:
            dates = list(historical_data.keys())
            values = list(historical_data.values())
            fund_fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name=stock
            ))
fund_fig.update_layout(title=f"{fundamental_metrics} Trend")
st.plotly_chart(fund_fig, use_container_width=True)

# EPS Analysis
st.subheader("EPS Analysis")
eps_fig = go.Figure()
for stock in selected_stocks:
    if isinstance(df.loc[stock, 'fundamental.historical'], dict):
        eps_data = df.loc[stock, 'fundamental.historical'].get('Basic EPS', {})
        if eps_data:
            dates = list(eps_data.keys())
            values = list(eps_data.values())
            eps_fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name=stock
            ))
eps_fig.update_layout(title="Basic EPS Trend")
st.plotly_chart(eps_fig, use_container_width=True)

# Earnings Section
st.header("Earnings Analysis")

# Earnings Date Prediction
st.subheader("Next Earnings Date Prediction")
earnings_df = df[df.index.isin(selected_stocks)][['earnings.next_earnings_prediction', 'earnings.prediction_metadata.confidence_score']]
earnings_df.columns = ['Next Earnings Date', 'Confidence Score']
st.dataframe(earnings_df.style.format({
    'Confidence Score': '{:.0f}%'
}))

# Earnings Interval Analysis
st.subheader("Earnings Interval Analysis")
interval_fig = go.Figure()
for stock in selected_stocks:
    if isinstance(df.loc[stock, 'earnings.prediction_metadata'], dict):
        intervals = df.loc[stock, 'earnings.prediction_metadata'].get('day_intervals', [])
        if intervals:
            interval_fig.add_trace(go.Box(
                y=intervals,
                name=stock,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
interval_fig.update_layout(title="Distribution of Days Between Earnings Reports")
st.plotly_chart(interval_fig, use_container_width=True)

# Error Analysis
st.header("Error Analysis")
error_stocks = df[df['status'] == 'failed'].index.tolist()
if error_stocks:
    st.warning(f"The following stocks encountered errors during analysis: {', '.join(error_stocks)}")
    selected_error = st.selectbox("Select stock to view error details", error_stocks)
    st.code(df.loc[selected_error, 'error'], language='python')
else:
    st.success("No errors encountered in the analysis")

# Stock Details Section
st.header("Detailed Stock Analysis")
selected_detail = st.selectbox("Select stock for detailed view", selected_stocks)

st.subheader(f"Details for {selected_detail}")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Technical Indicators**")
    tech_data = df.loc[selected_detail, 'technical']
    tech_df = pd.DataFrame.from_dict(tech_data, orient='index', columns=['Value'])
    st.dataframe(tech_df.style.format({'Value': '{:.2f}'}))

with col2:
    st.markdown("**Earnings Information**")
    earnings_data = df.loc[selected_detail, 'earnings']
    if isinstance(earnings_data['prediction_metadata'], dict):
        pred_meta = earnings_data['prediction_metadata']
        st.metric("Next Earnings Prediction", earnings_data['next_earnings_prediction'])
        st.metric("Confidence Score", f"{pred_meta['confidence_score']}%")
        st.metric("Prediction Method", pred_meta['prediction_method'])
        st.write(f"Consistent Quarterly: {'Yes' if pred_meta['consistent_quarterly'] else 'No'}")

# Historical Earnings Dates
if isinstance(earnings_data['historical_dates'], list):
    st.subheader("Historical Earnings Dates")
    dates = pd.to_datetime(earnings_data['historical_dates'])
    dates_df = pd.DataFrame({'Date': dates.sort_values(ascending=False)})
    st.dataframe(dates_df)

# Fundamental Predictions
st.subheader("Fundamental Predictions")
if isinstance(df.loc[selected_detail, 'fundamental.predictions'], dict):
    preds = df.loc[selected_detail, 'fundamental.predictions']
    preds_df = pd.DataFrame.from_dict(preds, orient='index', columns=['Predicted Value'])
    st.dataframe(preds_df.style.format({'Predicted Value': '{:.2f}'}))
