import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Load the data
with open('nifty50_analysis_results.json') as f:
    data = json.load(f)

# Convert to DataFrame - properly handle nested structure
records = []
for symbol, stock_data in data.items():
    record = {'symbol': symbol}
    # Flatten the nested structure
    for category, values in stock_data.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        record[f"{category}.{key}.{subkey}"] = subvalue
                else:
                    record[f"{category}.{key}"] = value
        else:
            record[category] = values
    records.append(record)

df = pd.DataFrame(records).set_index('symbol')

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
available_overview_cols = [col for col in overview_cols if col in df.columns]
overview_df = df.loc[selected_stocks, available_overview_cols]
overview_df.columns = ['Status', 'Last Close', 'Trend', 'Next Earnings'][:len(available_overview_cols)]
st.dataframe(overview_df.style.format({
    'Last Close': '{:.2f}'
} if 'Last Close' in overview_df.columns else {}), 
height=(len(selected_stocks) * 35 + 38))

# Technical Analysis Section (same as before)
# ... [previous technical analysis code] ...

# Fundamental Analysis Section - Enhanced
st.header("Fundamental Analysis")

# Historical Fundamentals
st.subheader("Historical Fundamentals")
fundamental_metrics = st.selectbox(
    "Select Fundamental Metric", 
    ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'Basic EPS']
)

fund_fig = go.Figure()
for stock in selected_stocks:
    if 'fundamental.historical' in df.columns:
        historical_data = df.loc[stock, 'fundamental.historical']
        if isinstance(historical_data, dict) and fundamental_metrics in historical_data:
            metric_data = historical_data[fundamental_metrics]
            if isinstance(metric_data, dict):
                dates = sorted(metric_data.keys())
                values = [metric_data[date] for date in dates]
                if all(isinstance(v, (int, float)) for v in values):
                    fund_fig.add_trace(go.Scatter(
                        x=dates,
                        y=values,
                        mode='lines+markers',
                        name=stock
                    ))
fund_fig.update_layout(
    title=f"{fundamental_metrics} Trend",
    xaxis_title="Date",
    yaxis_title=fundamental_metrics
)
st.plotly_chart(fund_fig, use_container_width=True)

# Financial Ratios Comparison
st.subheader("Financial Ratios Comparison")
if 'fundamental.historical' in df.columns:
    ratio_options = ['Gross Profit Margin', 'Operating Margin', 'Net Profit Margin']
    selected_ratio = st.selectbox("Select Ratio", ratio_options)
    
    ratio_fig = go.Figure()
    for stock in selected_stocks:
        historical_data = df.loc[stock, 'fundamental.historical']
        if isinstance(historical_data, dict):
            # Calculate ratios for the most recent period
            latest_date = None
            ratios = []
            
            if 'Total Revenue' in historical_data and isinstance(historical_data['Total Revenue'], dict):
                revenue_dates = sorted(historical_data['Total Revenue'].keys())
                if revenue_dates:
                    latest_date = revenue_dates[-1]
                    try:
                        revenue = historical_data['Total Revenue'][latest_date]
                        if selected_ratio == 'Gross Profit Margin' and 'Gross Profit' in historical_data:
                            gross_profit = historical_data['Gross Profit'].get(latest_date, 0)
                            ratio = (gross_profit / revenue) * 100 if revenue != 0 else 0
                            ratios.append(ratio)
                        elif selected_ratio == 'Operating Margin' and 'Operating Income' in historical_data:
                            op_income = historical_data['Operating Income'].get(latest_date, 0)
                            ratio = (op_income / revenue) * 100 if revenue != 0 else 0
                            ratios.append(ratio)
                        elif selected_ratio == 'Net Profit Margin' and 'Net Income' in historical_data:
                            net_income = historical_data['Net Income'].get(latest_date, 0)
                            ratio = (net_income / revenue) * 100 if revenue != 0 else 0
                            ratios.append(ratio)
                    except (TypeError, ZeroDivisionError):
                        pass
            
            if ratios:
                ratio_fig.add_trace(go.Bar(
                    x=[stock],
                    y=ratios,
                    name=stock,
                    text=[f"{r:.1f}%" for r in ratios],
                    textposition='auto'
                ))
    
    ratio_fig.update_layout(
        title=f"{selected_ratio} Comparison (Latest Available)",
        yaxis_title=f"{selected_ratio} (%)"
    )
    st.plotly_chart(ratio_fig, use_container_width=True)

# Predictions Section
st.header("Financial Predictions")

if 'fundamental.predictions' in df.columns:
    prediction_metrics = st.selectbox(
        "Select Prediction Metric",
        ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']
    )
    
    pred_fig = go.Figure()
    
    # Historical vs Predicted Comparison
    for stock in selected_stocks:
        predictions = df.loc[stock, 'fundamental.predictions']
        historical = df.loc[stock, 'fundamental.historical']
        
        if isinstance(predictions, dict) and prediction_metrics in predictions:
            # Get historical data
            hist_values = []
            hist_dates = []
            if isinstance(historical, dict) and prediction_metrics in historical:
                metric_data = historical[prediction_metrics]
                if isinstance(metric_data, dict):
                    hist_dates = sorted(metric_data.keys())
                    hist_values = [metric_data[date] for date in hist_dates]
            
            # Add historical trace
            if hist_values:
                pred_fig.add_trace(go.Scatter(
                    x=hist_dates,
                    y=hist_values,
                    mode='lines+markers',
                    name=f"{stock} Historical",
                    line=dict(color='blue')
                ))
            
            # Add predicted value
            pred_value = predictions[prediction_metrics]
            if pred_value is not None and isinstance(pred_value, (int, float)):
                if hist_dates:
                    last_date = pd.to_datetime(hist_dates[-1])
                    pred_date = last_date + pd.DateOffset(months=3)  # Assuming next quarter
                else:
                    pred_date = datetime.now().strftime('%Y-%m-%d')
                
                pred_fig.add_trace(go.Scatter(
                    x=[pred_date],
                    y=[pred_value],
                    mode='markers',
                    name=f"{stock} Predicted",
                    marker=dict(color='red', size=10)
                ))
    
    pred_fig.update_layout(
        title=f"{prediction_metrics} - Historical vs Predicted",
        xaxis_title="Date",
        yaxis_title=prediction_metrics
    )
    st.plotly_chart(pred_fig, use_container_width=True)
    
    # Predictions Table
    st.subheader("Detailed Predictions")
    pred_cols = [f'fundamental.predictions.{m}' for m in ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']]
    available_pred_cols = [col for col in pred_cols if col in df.columns]
    
    if available_pred_cols:
        pred_df = df.loc[selected_stocks, available_pred_cols]
        pred_df.columns = [col.replace('fundamental.predictions.', '') for col in available_pred_cols]
        st.dataframe(pred_df.style.format('{:.2f}'))
else:
    st.warning("No prediction data available")

# Stock Details Section - Enhanced with Fundamentals and Predictions
st.header("Detailed Stock Analysis")
selected_detail = st.selectbox("Select stock for detailed view", selected_stocks)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Fundamental Metrics")
    if 'fundamental.historical' in df.columns:
        historical_data = df.loc[selected_detail, 'fundamental.historical']
        if isinstance(historical_data, dict):
            # Show latest available fundamentals
            latest_data = {}
            for metric in ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']:
                if metric in historical_data and isinstance(historical_data[metric], dict):
                    dates = sorted(historical_data[metric].keys())
                    if dates:
                        latest_date = dates[-1]
                        latest_data[metric] = historical_data[metric][latest_date]
            
            if latest_data:
                latest_df = pd.DataFrame.from_dict(latest_data, orient='index', columns=['Value'])
                st.dataframe(latest_df.style.format({'Value': '{:,.2f}'}))

with col2:
    st.subheader("Predicted Values")
    if 'fundamental.predictions' in df.columns:
        predictions = df.loc[selected_detail, 'fundamental.predictions']
        if isinstance(predictions, dict):
            pred_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['Predicted Value'])
            st.dataframe(pred_df.style.format({'Predicted Value': '{:,.2f}'}))
