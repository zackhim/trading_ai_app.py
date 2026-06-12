import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

st.set_page_config(page_title="Trading AI App", layout="wide")
st.title("Trading AI App")
st.caption("Educational prototype for trade analysis and signal generation. Not financial advice.")

symbol = st.text_input("Ticker symbol", value="AAPL")
period = st.selectbox("History", ["6mo", "1y", "2y", "5y"], index=1)

@st.cache_data
def load_data(sym, per):
    df = yf.download(sym, period=per, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df.reset_index()
    return df

def features(df):
    d = df.copy()
    d["ret1"] = d["Close"].pct_change()
    d["ma5"] = d["Close"].rolling(5).mean()
    d["ma20"] = d["Close"].rolling(20).mean()
    d["vol20"] = d["Close"].rolling(20).std()
    d["rsi"] = 100 - (100 / (1 + (d["Close"].diff().clip(lower=0).rolling(14).mean() / (-d["Close"].diff().clip(upper=0).rolling(14).mean()).abs())))
    d["target_up"] = (d["Close"].shift(-1) > d["Close"]).astype(int)
    d = d.dropna().reset_index(drop=True)
    return d

df = load_data(symbol, period)
if df.empty:
    st.error("No data found for that ticker.")
    st.stop()

feat = features(df)
cols = ["ret1", "ma5", "ma20", "vol20", "rsi"]
X = feat[cols]
y = feat["target_up"]

if len(feat) < 80:
    st.warning("Not enough data for a reliable model.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)

last = feat.iloc[-1:][cols]
prob_up = float(clf.predict_proba(last)[0, 1])
recommendation = "BUY" if prob_up >= 0.55 else "SELL" if prob_up <= 0.45 else "HOLD"

st.metric("Model accuracy", f"{acc:.2%}")
st.metric("Up probability", f"{prob_up:.2%}")
st.metric("Signal", recommendation)

future_days = st.slider("Forecast days", 5, 30, 7)
idx = np.arange(len(feat)).reshape(-1, 1)
lin = LinearRegression().fit(idx, feat["Close"])
future_idx = np.arange(len(feat), len(feat) + future_days).reshape(-1, 1)
forecast = lin.predict(future_idx)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close"))
fig.add_trace(go.Scatter(x=feat["Date"], y=feat["ma5"], name="MA5"))
fig.add_trace(go.Scatter(x=feat["Date"], y=feat["ma20"], name="MA20"))
future_dates = pd.date_range(feat["Date"].iloc[-1], periods=future_days + 1, freq="B")[1:]
fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", line=dict(dash="dash")))
fig.update_layout(height=600, xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Recent data")
st.dataframe(feat[["Date", "Close", "ma5", "ma20", "rsi", "target_up"]].tail(15), use_container_width=True)

st.subheader("How to use")
st.write("1. Enter a ticker like AAPL, TSLA, or MSFT.")
st.write("2. Review the BUY/SELL/HOLD signal and probability.")
st.write("3. Use the chart and recent indicators to inspect the trend.")
