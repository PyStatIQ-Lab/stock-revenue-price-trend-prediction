import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Load the data
with open('nifty50_analysis_results.json') as f:
    data = json.load(f)

# Convert to DataFrame and flatten the nested structure
df = pd.json_normalize(data).set_index('symbol')

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
overview_df = df.loc[selected_stocks, overview_cols]
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
tech_df = df.loc[selected_stocks, tech_cols]
tech_df.columns = [
    'RSI', 'MA(20)', 'MA(50)', 'MA(200)', 'MACD Line', 'MACD Signal', 
    'BB Upper', 'BB Lower', 'Last Close'
]

# Normalize for comparison (except RSI which is already normalized)
tech_normalized = tech_df.copy()
for col in tech_normalized.columns:
    if col != 'RSI' and tech_normalized[col].max() != tech_normalized[col].min():
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
    rsi_value = df.loc[stock, 'technical.rsi']
    if pd.notna(rsi_value):
        rsi_fig.add_trace(go.Bar(
            x=[stock],
            y=[rsi_value],
            name=stock,
            text=f"{rsi_value:.2f}",
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
    ma_20 = df.loc[stock, 'technical.ma_20']
    ma_50 = df.loc[stock, 'technical.ma_50']
    ma_200 = df.loc[stock, 'technical.ma_200']
    last_close = df.loc[stock, 'technical.last_close']
    
    if all(pd.notna(val) for val in [ma_20, ma_50, ma_200, last_close]):
        ma_fig.add_trace(go.Bar(
            x=['MA(20)', 'MA(50)', 'MA(200)', 'Last Close'],
            y=[ma_20, ma_50, ma_200, last_close],
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
    historical_data = df.loc[stock, 'fundamental.historical']
    if isinstance(historical_data, dict) and fundamental_metrics in historical_data:
        metric_data = historical_data[fundamental_metrics]
        if isinstance(metric_data, dict):
            dates = list(metric_data.keys())
            values = list(metric_data.values())
            if all(isinstance(v, (int, float)) for v in values):
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
    historical_data = df.loc[stock, 'fundamental.historical']
    if isinstance(historical_data, dict) and 'Basic EPS' in historical_data:
        eps_data = historical_data['Basic EPS']
        if isinstance(eps_data, dict):
            dates = list(eps_data.keys())
            values = list(eps_data.values())
            if all(isinstance(v, (int, float)) for v in values):
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
if 'earnings.next_earnings_prediction' in df.columns and 'earnings.prediction_metadata.confidence_score' in df.columns:
    earnings_df = df.loc[selected_stocks, ['earnings.next_earnings_prediction', 'earnings.prediction_metadata.confidence_score']]
    earnings_df.columns = ['Next Earnings Date', 'Confidence Score']
    st.dataframe(earnings_df.style.format({
        'Confidence Score': '{:.0f}%'
    }))

# Error Analysis
st.header("Error Analysis")
if 'status' in df.columns:
    error_stocks = df[df['status'] == 'failed'].index.tolist()
    if error_stocks:
        st.warning(f"The following stocks encountered errors during analysis: {', '.join(error_stocks)}")
        selected_error = st.selectbox("Select stock to view error details", error_stocks)
        if 'error' in df.columns:
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
    if 'technical' in df.columns:
        tech_data = df.loc[selected_detail, 'technical']
        if isinstance(tech_data, dict):
            tech_df = pd.DataFrame.from_dict(tech_data, orient='index', columns=['Value'])
            st.dataframe(tech_df.style.format({'Value': '{:.2f}'}))

with col2:
    st.markdown("**Earnings Information**")
    if 'earnings' in df.columns:
        earnings_data = df.loc[selected_detail, 'earnings']
        if isinstance(earnings_data, dict):
            if 'next_earnings_prediction' in earnings_data:
                st.metric("Next Earnings Prediction", earnings_data['next_earnings_prediction'])
            if 'prediction_metadata' in earnings_data and isinstance(earnings_data['prediction_metadata'], dict):
                pred_meta = earnings_data['prediction_metadata']
                if 'confidence_score' in pred_meta:
                    st.metric("Confidence Score", f"{pred_meta['confidence_score']}%")
                if 'prediction_method' in pred_meta:
                    st.metric("Prediction Method", pred_meta['prediction_method'])
                if 'consistent_quarterly' in pred_meta:
                    st.write(f"Consistent Quarterly: {'Yes' if pred_meta['consistent_quarterly'] else 'No'}")

# Historical Earnings Dates
if 'earnings' in df.columns:
    earnings_data = df.loc[selected_detail, 'earnings']
    if isinstance(earnings_data, dict) and 'historical_dates' in earnings_data and isinstance(earnings_data['historical_dates'], list):
        st.subheader("Historical Earnings Dates")
        dates = pd.to_datetime(earnings_data['historical_dates'])
        dates_df = pd.DataFrame({'Date': dates.sort_values(ascending=False)})
        st.dataframe(dates_df)

# Fundamental Predictions
st.subheader("Fundamental Predictions")
if 'fundamental.predictions' in df.columns:
    preds = df.loc[selected_detail, 'fundamental.predictions']
    if isinstance(preds, dict):
        preds_df = pd.DataFrame.from_dict(preds, orient='index', columns=['Predicted Value'])
        st.dataframe(preds_df.style.format({'Predicted Value': '{:.2f}'}))
