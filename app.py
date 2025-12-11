import io
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Prophet optional
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


st.set_page_config(page_title="Telecom AI Forecast", layout="wide", page_icon="📈")

st.markdown("""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
    <div style="font-size:35px;">📈</div>
    <div>
        <div style="font-size:25px; font-weight:600;">Telecom AI Forecast App</div>
        <div style="font-size:13px; color:#888;">
            Live data • AI & Prophet models • Buy/Sell Signals • Excel/PDF Export
        </div>
    </div>
</div>
<hr/>
""", unsafe_allow_html=True)


TICKER_MAP = {
    "Bell (BCE.TO)": "BCE.TO",
    "Telus (T.TO)": "T.TO",
    "Rogers (RCI-B.TO)": "RCI-B.TO",
}

@st.cache_data(show_spinner=False)
def fetch_price_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        return df

    # Handle different possible price column names from Yahoo Finance
    possible_cols = ["Adj Close", "Adj_Close", "Close", "close"]

    price_col = None
    for col in possible_cols:
        if col in df.columns:
            price_col = col
            break

    if price_col is None:
        raise KeyError("Price column not found in Yahoo Finance data")

    df = df[[price_col]].rename(columns={price_col: "price"})
    df.index = pd.to_datetime(df.index)

    return df.reset_index().rename(columns={"Date": "date"})




def add_features(df):
    df = df.copy()
    df["return"] = df["price"].pct_change()
    df["ma_short"] = df["price"].rolling(20).mean()
    df["ma_long"] = df["price"].rolling(50).mean()
    df["vol_20"] = df["return"].rolling(20).std()
    df["day"] = (df["date"] - df["date"].min()).dt.days
    return df


def train_rf(df):
    df2 = add_features(df)
    df2 = df2.dropna()
    X = df2[["day", "ma_short", "ma_long", "vol_20"]]
    y = df2["price"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model


def forecast_rf(df, model, horizon):
    df2 = add_features(df)
    last_date = df2["date"].max()
    future_dates = [last_date + dt.timedelta(days=i) for i in range(1, horizon + 1)]
    future = pd.DataFrame({"date": future_dates})
    future["day"] = (future["date"] - df2["date"].min()).dt.days

    last_ma_short = df2["ma_short"].iloc[-1]
    last_ma_long = df2["ma_long"].iloc[-1]
    last_vol = df2["vol_20"].iloc[-1]

    future["ma_short"] = last_ma_short
    future["ma_long"] = last_ma_long
    future["vol_20"] = last_vol

    future["rf"] = model.predict(future[["day", "ma_short", "ma_long", "vol_20"]])
    return future


def forecast_prophet(df, horizon):
    if not PROPHET_AVAILABLE:
        return None
    df_p = df.rename(columns={"date": "ds", "price": "y"})
    m = Prophet(daily_seasonality=True)
    m.fit(df_p)
    future = m.make_future_dataframe(periods=horizon)
    fc = m.predict(future)
    return fc.rename(columns={"ds": "date", "yhat": "prophet"})[["date", "prophet"]]


def hybrid_model(df, rf_fc, p_fc):
    df["hybrid"] = df["price"]
    if rf_fc is not None and p_fc is not None:
        return df, rf_fc.merge(p_fc, on="date", how="outer")
    elif rf_fc is not None:
        return df, rf_fc.rename(columns={"rf": "hybrid"})
    else:
        return df, p_fc.rename(columns={"prophet": "hybrid"})


def buy_sell(df):
    df = df.copy()
    df["ma20"] = df["price"].rolling(20).mean()
    df["ma50"] = df["price"].rolling(50).mean()
    df["signal"] = np.where(df["ma20"] > df["ma50"], "BUY", "SELL")
    return df


def generate_pdf(ticker_name, hist, hybrid):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, 750, f"{ticker_name} – AI Forecast Report")

    last_price = float(hist["price"].iloc[-1])
    c.setFont("Helvetica", 12)
    c.drawString(40, 720, f"Last price: {last_price:.2f}")

    c.drawString(40, 700, "Forecast data:")
    y = 680
    for _, r in hybrid.head(10).iterrows():
        c.drawString(40, y, str(r.to_dict()))
        y -= 15
        if y < 40:
            c.showPage()
            y = 750

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect("Companies", list(TICKER_MAP.keys()), ["Bell (BCE.TO)"])
    years = st.slider("Years of history", 1, 10, 3)
    horizon = st.slider("Forecast days", 30, 365, 180)


start = dt.date.today() - dt.timedelta(days=365 * years)
end = dt.date.today()

tab1, tab2, tab3 = st.tabs(["📊 Forecasts", "📈 Signals", "📄 PDF"])


for t_name in tickers:
    t = TICKER_MAP[t_name]
    df = fetch_price_data(t, start, end)
    if df.empty:
        st.error(f"No data for {t_name}")
        continue

    with tab1:
        st.subheader(f"{t_name} – Price")
        st.plotly_chart(px.line(df, x="date", y="price"), use_container_width=True)

        model = train_rf(df)
        rf_fc = forecast_rf(df, model, horizon)
        p_fc = forecast_prophet(df, horizon)
        hist, hybrid = hybrid_model(df, rf_fc, p_fc)

        st.subheader("Hybrid Forecast")
        st.plotly_chart(
            px.line(hybrid, x="date", y=hybrid.columns[1:], title="Forecast"),
            use_container_width=True
        )

    with tab2:
        sig = buy_sell(df)
        st.subheader(f"{t_name} – Trading Signals")
        st.plotly_chart(px.line(sig, x="date", y=["price", "ma20", "ma50"]), use_container_width=True)
        st.metric("Latest Signal", sig["signal"].iloc[-1])

    with tab3:
        if st.button(f"Download PDF for {t_name}"):
            pdf = generate_pdf(t_name, df, hybrid)
            st.download_button("Download PDF", pdf, file_name="forecast.pdf", mime="application/pdf")

