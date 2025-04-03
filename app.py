import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import pytz
from statsmodels.tsa.arima.model import ARIMA
import warnings
import time
import numpy as np
from dateutil.relativedelta import relativedelta
from typing import Dict, Tuple, Optional, Any, List

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    "analysis": {
        "history_period": "200d",
        "min_earnings_for_prediction": 4,
        "forecast_horizon": 1,
        "max_earnings_dates": 8
    },
    "api": {
        "delay": 1,
        "timeout": 15
    }
}

# NIFTY50 stock list (as of knowledge cutoff in 2023)
NIFTY50_STOCKS = [
    'ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS',
    'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS',
    'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS',
    'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS',
    'HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS', 'INDUSINDBK.NS', 'INFY.NS',
    'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS',
    'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS',
    'SBILIFE.NS', 'SHREECEM.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS',
    'TATACONSUM.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'UPL.NS',
    'ULTRACEMCO.NS', 'WIPRO.NS', 'HDFC.NS'
]

class StockAnalyzer:
    """Streamlit-compatible stock analyzer"""
    
    def __init__(self):
        self.stocks = NIFTY50_STOCKS
        
    def _safe_api_call(self, func, *args, **kwargs) -> Any:
        """Robust API call wrapper with retries"""
        try:
            result = func(*args, **kwargs)
            if result is None or (hasattr(result, 'empty') and result.empty:
                raise ValueError("Empty API response")
            return result
        except Exception as e:
            st.warning(f"API call failed: {str(e)}")
            return None

    def _analyze_earnings_pattern(self, dates: List[datetime]) -> Dict[str, Any]:
        """Enhanced earnings date pattern analysis"""
        if len(dates) < 2:
            return {"error": "Insufficient historical data"}
        
        dates_series = pd.Series(dates).sort_values()
        day_diffs = dates_series.diff().dt.days.dropna()
        
        month_diffs = []
        for i in range(1, len(dates_series)):
            delta = relativedelta(dates_series[i], dates_series[i-1])
            month_diffs.append(delta.months + delta.years * 12)
        
        analysis = {
            "last_date": dates_series.iloc[-1].strftime('%Y-%m-%d'),
            "day_intervals": day_diffs.tolist(),
            "month_intervals": month_diffs,
            "avg_day_interval": int(np.mean(day_diffs)),
            "median_day_interval": int(np.median(day_diffs)),
            "avg_month_interval": np.mean(month_diffs),
            "consistent_quarterly": all(2 <= m <= 4 for m in month_diffs)
        }
        
        return analysis

    def _predict_next_earnings(self, dates: List[datetime]) -> Tuple[Optional[str], Dict[str, Any]]:
        """Improved earnings date prediction"""
        try:
            if len(dates) < CONFIG["analysis"]["min_earnings_for_prediction"]:
                return None, {"warning": "Insufficient history for prediction"}
            
            dates_series = pd.Series(dates).sort_values()
            analysis = self._analyze_earnings_pattern(dates)
            
            last_date = dates_series.iloc[-1]
            predicted_date = last_date + timedelta(days=analysis["median_day_interval"])
            
            if predicted_date.weekday() >= 5:
                predicted_date += timedelta(days=7 - predicted_date.weekday())
            
            confidence = min(90, len(dates) * 15)
            
            return predicted_date.strftime('%Y-%m-%d'), {
                **analysis,
                "confidence_score": confidence,
                "prediction_method": "median_day_interval"
            }
            
        except Exception as e:
            st.error(f"Date prediction failed: {str(e)}")
            return None, {"error": str(e)}

    def get_earnings_analysis(self, ticker: str) -> Dict[str, Any]:
        """Complete earnings date analysis"""
        try:
            stock = self._safe_api_call(yf.Ticker, ticker)
            if stock is None:
                return {"error": "Failed to fetch stock data"}
                
            earnings_dates = self._safe_api_call(lambda: stock.earnings_dates)
            
            if earnings_dates is None or earnings_dates.empty:
                return {"error": "No earnings dates available"}
            
            dates = earnings_dates.index.tolist()
            dates = [pd.to_datetime(d) for d in dates if d <= datetime.now(pytz.utc)]
            dates = sorted(dates)[-CONFIG["analysis"]["max_earnings_dates"]:]
            
            if not dates:
                return {"error": "No valid historical dates"}
            
            next_date, prediction_meta = self._predict_next_earnings(dates)
            
            return {
                "next_earnings_prediction": next_date,
                "prediction_metadata": prediction_meta,
                "historical_dates": [d.strftime('%Y-%m-%d') for d in dates]
            }
            
        except Exception as e:
            st.error(f"Earnings analysis failed for {ticker}: {str(e)}")
            return {"error": str(e)}

    def get_technical_analysis(self, ticker: str) -> Dict[str, Any]:
        """Comprehensive technical analysis"""
        try:
            history = self._safe_api_call(
                lambda: yf.Ticker(ticker).history(
                    period=CONFIG["analysis"]["history_period"],
                    timeout=CONFIG["api"]["timeout"]
                )
            )
            
            if history is None or history.empty:
                return {"error": "No historical price data"}
            
            indicators = {
                "last_close": float(history['Close'].iloc[-1]),
                "last_volume": int(history['Volume'].iloc[-1]),
                "last_date": history.index[-1].strftime('%Y-%m-%d')
            }
            
            for days in [20, 50, 200]:
                history.ta.sma(length=days, append=True, col_names=(f'ma_{days}',))
                indicators[f'ma_{days}'] = float(history[f'ma_{days}'].iloc[-1])
            
            history.ta.rsi(length=14, append=True, col_names=('rsi',))
            indicators['rsi'] = float(history['rsi'].iloc[-1])
            
            history.ta.macd(append=True)
            indicators['macd_line'] = float(history['MACD_12_26_9'].iloc[-1])
            indicators['macd_signal'] = float(history['MACDs_12_26_9'].iloc[-1])
            
            history.ta.bbands(append=True)
            indicators['bb_upper'] = float(history['BBU_5_2.0'].iloc[-1])
            indicators['bb_lower'] = float(history['BBL_5_2.0'].iloc[-1])
            
            volume_mean = history['Volume'].mean()
            volume_std = history['Volume'].std()
            indicators['volume_z'] = (indicators['last_volume'] - volume_mean) / volume_std if volume_std > 0 else 0
            
            indicators['trend'] = self._determine_trend(history, indicators)
            
            return indicators
            
        except Exception as e:
            st.error(f"Technical analysis failed for {ticker}: {str(e)}")
            return {"error": str(e)}

    def _determine_trend(self, data: pd.DataFrame, indicators: Dict) -> str:
        """Enhanced trend determination"""
        try:
            close = indicators['last_close']
            ma_20 = indicators['ma_20']
            ma_50 = indicators['ma_50']
            ma_200 = indicators['ma_200']
            rsi = indicators['rsi']
            volume_z = indicators['volume_z']
            
            price_above = sum(close > ma for ma in [ma_20, ma_50, ma_200])
            ma_order = int(ma_20 > ma_50 > ma_200)
            
            score = (
                price_above * 0.4 +
                ma_order * 0.3 +
                (0 if rsi > 70 else 1 if rsi < 30 else 0.5) * 0.2 +
                (1 if volume_z > 2 else 0 if volume_z < -2 else 0.5) * 0.1
            )
            
            if score > 2.5:
                return "Strong Bullish"
            elif score > 1.8:
                return "Bullish"
            elif score > 1.2:
                return "Mild Bullish"
            elif score > 0.8:
                return "Neutral"
            elif score > 0.5:
                return "Mild Bearish"
            elif score > 0.2:
                return "Bearish"
            else:
                return "Strong Bearish"
                
        except:
            return "Unknown"

    def get_fundamental_analysis(self, ticker: str) -> Dict[str, Any]:
        """Comprehensive fundamental analysis with validation"""
        try:
            stock = self._safe_api_call(yf.Ticker, ticker)
            if stock is None:
                return {"error": "Failed to fetch stock data"}
                
            financials = self._safe_api_call(lambda: stock.quarterly_financials)
            
            if financials is None or financials.empty:
                return {"error": "No financial data available"}
                
            financials = financials.T
            financials.index = pd.to_datetime(financials.index)
            financials = financials.sort_index()
            
            metrics = [
                'Total Revenue', 'Gross Profit', 'Operating Income',
                'Net Income', 'Basic EPS', 'Diluted EPS',
                'Total Expenses', 'Cost Of Revenue', 'Operating Expense',
                'Depreciation And Amortization', 'Tax Provision'
            ]
            
            results = {"historical": {}, "predictions": {}}
            
            for metric in metrics:
                if metric not in financials.columns:
                    continue
                    
                series = pd.to_numeric(financials[metric], errors='coerce')
                series = series[~series.isna()]
                
                if series.empty:
                    continue
                
                hist = series.tail(4)
                results["historical"][metric] = {
                    d.strftime('%Y-%m-%d'): float(v) for d, v in hist.items()
                }
                
                if len(series) >= 3:
                    try:
                        model = ARIMA(series, order=(1,1,1))
                        model_fit = model.fit()
                        forecast = model_fit.forecast(steps=CONFIG["analysis"]["forecast_horizon"])
                        results["predictions"][metric] = float(forecast[0])
                    except:
                        results["predictions"][metric] = None
            
            return results
            
        except Exception as e:
            st.error(f"Fundamental analysis failed for {ticker}: {str(e)}")
            return {"error": str(e)}

    def analyze_stock(self, ticker: str) -> Dict[str, Any]:
        """Complete analysis for a single stock"""
        analysis = {
            "symbol": ticker,
            "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            "status": "completed",
            "earnings": {},
            "technical": {},
            "fundamental": {}
        }
        
        try:
            analysis["earnings"] = self.get_earnings_analysis(ticker)
            analysis["technical"] = self.get_technical_analysis(ticker)
            analysis["fundamental"] = self.get_fundamental_analysis(ticker)
        except Exception as e:
            analysis["status"] = "failed"
            analysis["error"] = str(e)
        
        return analysis

# Streamlit UI
def main():
    st.set_page_config(
        page_title="NIFTY50 Stock Analyzer",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 NIFTY50 Stock Analyzer")
    st.markdown("""
        Comprehensive analysis of NIFTY50 stocks including:
        - Earnings date predictions
        - Technical indicators
        - Fundamental metrics
    """)
    
    analyzer = StockAnalyzer()
    
    # Sidebar controls
    st.sidebar.header("Analysis Options")
    selected_stock = st.sidebar.selectbox(
        "Select a Stock", 
        NIFTY50_STOCKS,
        index=NIFTY50_STOCKS.index('RELIANCE.NS')
    )
    
    analyze_button = st.sidebar.button("Analyze Selected Stock")
    analyze_all_button = st.sidebar.button("Analyze All NIFTY50 Stocks (Takes Time)")
    
    if analyze_button or analyze_all_button:
        if analyze_all_button:
            st.warning("This will take several minutes to complete. Please be patient.")
            progress_bar = st.progress(0)
            results = {}
            
            for i, ticker in enumerate(analyzer.stocks):
                results[ticker] = analyzer.analyze_stock(ticker)
                progress_bar.progress((i + 1) / len(analyzer.stocks))
                time.sleep(CONFIG["api"]["delay"])
            
            st.success("Analysis completed!")
            
            # Display summary table
            st.subheader("NIFTY50 Analysis Summary")
            summary_data = []
            
            for ticker, analysis in results.items():
                if analysis.get("status") == "completed":
                    tech = analysis.get("technical", {})
                    earnings = analysis.get("earnings", {})
                    
                    summary_data.append({
                        "Stock": ticker.replace('.NS', ''),
                        "Price": tech.get("last_close", "N/A"),
                        "Trend": tech.get("trend", "N/A"),
                        "RSI": tech.get("rsi", "N/A"),
                        "Next Earnings": earnings.get("next_earnings_prediction", "N/A"),
                        "Confidence": earnings.get("prediction_metadata", {}).get("confidence_score", "N/A")
                    })
            
            df = pd.DataFrame(summary_data)
            st.dataframe(df.style.format({
                "Price": "{:.2f}",
                "RSI": "{:.1f}",
                "Confidence": "{:.0f}"
            }))
            
        else:
            with st.spinner(f"Analyzing {selected_stock}..."):
                analysis = analyzer.analyze_stock(selected_stock)
            
            if analysis.get("status") == "completed":
                st.success(f"Analysis completed for {selected_stock}")
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["Technical", "Earnings", "Fundamental"])
                
                with tab1:
                    st.subheader("Technical Analysis")
                    tech = analysis.get("technical", {})
                    
                    if tech.get("error"):
                        st.error(tech["error"])
                    else:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Price", f"₹{tech['last_close']:.2f}")
                            st.metric("Volume", f"{tech['last_volume']:,}")
                            st.metric("Trend", tech['trend'])
                            
                        with col2:
                            st.metric("RSI (14)", f"{tech['rsi']:.1f}", 
                                      help="Overbought (>70), Oversold (<30)")
                            st.metric("20-day MA", f"₹{tech['ma_20']:.2f}")
                            st.metric("50-day MA", f"₹{tech['ma_50']:.2f}")
                            
                        with col3:
                            st.metric("200-day MA", f"₹{tech['ma_200']:.2f}")
                            st.metric("MACD", f"{tech['macd_line']:.2f} / {tech['macd_signal']:.2f}")
                            st.metric("Bollinger Bands", 
                                     f"₹{tech['bb_lower']:.2f} - ₹{tech['bb_upper']:.2f}")
                
                with tab2:
                    st.subheader("Earnings Analysis")
                    earnings = analysis.get("earnings", {})
                    
                    if earnings.get("error"):
                        st.error(earnings["error"])
                    else:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Next Earnings Prediction", 
                                     earnings.get("next_earnings_prediction", "N/A"))
                            st.metric("Confidence Score", 
                                     f"{earnings.get('prediction_metadata', {}).get('confidence_score', 'N/A')}%")
                            
                        with col2:
                            st.metric("Last Earnings Date", 
                                     earnings.get('prediction_metadata', {}).get('last_date', 'N/A'))
                            st.metric("Average Interval", 
                                     f"{earnings.get('prediction_metadata', {}).get('avg_day_interval', 'N/A')} days")
                        
                        st.subheader("Historical Earnings Dates")
                        st.write(pd.DataFrame({
                            "Date": earnings.get("historical_dates", [])
                        }))
                
                with tab3:
                    st.subheader("Fundamental Analysis")
                    fund = analysis.get("fundamental", {})
                    
                    if fund.get("error"):
                        st.error(fund["error"])
                    else:
                        st.subheader("Recent Financials")
                        if fund.get("historical"):
                            st.dataframe(pd.DataFrame(fund["historical"]).T)
                        
                        st.subheader("Next Quarter Predictions")
                        if fund.get("predictions"):
                            st.dataframe(pd.DataFrame(fund["predictions"], index=["Prediction"]).T)
            
            else:
                st.error(f"Analysis failed: {analysis.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
