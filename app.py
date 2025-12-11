import io
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Prophet is optional – app should still run without it
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# -----------------------------------------------------------------------------
# Streamlit page config & header
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Telecom AI Forecast",
    layout="wide",
    page_icon="📈",
)

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Ticker map
# -----------------------------------------------------------------------------
TICKER_MAP = {
    "Bell (BCE.TO)": "BCE.TO",
    "Telus (T.TO)": "T.TO",
    "Rogers (RCI-B.TO)": "RCI-B.TO",
}

# -----------------------------------------------------------------------------
# Data download + cleaning
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_price_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download daily prices from Yahoo Finance and return df[date, price]."""
    df = yf.download(ticker, start=start, end=end)

    # If nothing came back, just return empty frame
    if df.empty:
        return pd.DataFrame(columns=["date", "price"])

    # Try multiple possible price columns from Yahoo
    possible_cols = ["Adj Close", "Adj_Close", "Close", "close"]
    price_col = None
    for col in possible_cols:
        if col in df.columns:
            price_col = col
            break

    # Still nothing? Return empty safely
    if price_col is None:
        return pd.DataFrame(columns=["date", "price"])

    # Keep only the price column
    df = df[[price_col]].rename(columns={price_col: "price"})
    df.index = pd.to_datetime(df.index)

    # Ensure there is always a 'date' column
    df = df.reset_index()
    df.columns = ["date", "price"]

    # Make sure types are clean
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["price"])

    return df


# -----------------------------------------------------------------------------
# Feature engineering & ML models
# -----------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical features for the Random Forest model."""
    df = df.copy()
    df["return"] = df["price"].pct_change()
    df["ma_short"] = df["price"].rolling(20).mean()
    df["ma_long"] = df["price"].rolling(50).mean()
    df["vol_20"] = df["return"].rolling(20).std()
    df["day"] = (df["date"] - df["date"].min()).dt.days
    return df


def train_rf(df: pd.DataFrame) -> RandomForestRegressor | None:
    """Train a Random Forest on historical prices."""
    df2 = add_features(df)
    df2 = df2.dropna()

    if len(df2) < 40:
        # not enough data to train a model
        return None

    X = df2[["day", "ma_short", "ma_long", "vol_20"]]
    y = df2["price"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def forecast_rf(df: pd.DataFrame, model: RandomForestRegressor, horizon: int) -> pd.DataFrame:
    """Use trained RF model to forecast 'horizon' future days."""
    df2 = add_features(df)
    df2 = df2.dropna()

    last_date = df2["date"].max()
    future_dates = [last_date + dt.timedelta(days=i) for i in range(1, horizon + 1)]
    future = pd.DataFrame({"date": future_dates})

    # same feature structure
    future["day"] = (future["date"] - df2["date"].min()).dt.days
    future["ma_short"] = df2["ma_short"].iloc[-1]
    future["ma_long"] = df2["ma_long"].iloc[-1]
    future["vol_20"] = df2["vol_20"].iloc[-1]

    future["rf"] = model.predict(future[["day", "ma_short", "ma_long", "vol_20"]])
    return future[["date", "rf"]]


def forecast_prophet(df: pd.DataFrame, horizon: int) -> pd.DataFrame | None:
    """Forecast with Prophet if available."""
    if not PROPHET_AVAILABLE:
        return None

    df_p = df.rename(columns={"date": "ds", "price": "y"}).copy()
    df_p["y"] = pd.to_numeric(df_p["y"], errors="coerce")
    df_p = df_p.dropna(subset=["y"])

    if df_p.empty:
        return None

    m = Prophet(daily_seasonality=True)
    m.fit(df_p)

    future = m.make_future_dataframe(periods=horizon)
    fc = m.predict(future)

    out = fc[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "prophet"})
    out["date"] = pd.to_datetime(out["date"])
    return out


def hybrid_model(
    hist: pd.DataFrame,
    rf_fc: pd.DataFrame | None,
    p_fc: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine RF and Prophet forecasts into a 'hybrid' forecast dataframe.
    Returns (historical_df, forecast_df).
    """
    hist = hist.copy()

    # build forecast frame
    if rf_fc is None and p_fc is None:
        # no models → just return history
        hist["hybrid"] = hist["price"]
        return hist, hist[["date", "hybrid"]]

    if rf_fc is not None and p_fc is not None:
        fc = rf_fc.merge(p_fc, on="date", how="outer").sort_values("date")
        # simple average hybrid
        fc["hybrid"] = fc[["rf", "prophet"]].mean(axis=1)
    elif rf_fc is not None:
        fc = rf_fc.copy()
        fc["hybrid"] = fc["rf"]
    else:
        fc = p_fc.copy()
        fc["hybrid"] = fc["prophet"]

    return hist, fc[["date", "hybrid"]]


# -----------------------------------------------------------------------------
# Trading signals & PDF report
# -----------------------------------------------------------------------------
def buy_sell(df: pd.DataFrame) -> pd.DataFrame:
    """Simple MA crossover signals."""
    df = df.copy()
    df["ma20"] = df["price"].rolling(20).mean()
    df["ma50"] = df["price"].rolling(50).mean()
    df["signal"] = np.where(df["ma20"] > df["ma50"], "BUY", "SELL")
    return df


def generate_pdf(ticker_name: str, hist: pd.DataFrame, forecast: pd.DataFrame) -> bytes:
    """Create a simple PDF summary for a ticker."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, 750, f"{ticker_name} – AI Forecast Report")

    if not hist.empty:
        last_price = float(hist["price"].iloc[-1])
        c.setFont("Helvetica", 12)
        c.drawString(40, 720, f"Last price: {last_price:.2f}")

    c.drawString(40, 700, "Sample forecast rows:")
    y = 680
    for _, row in forecast.head(10).iterrows():
        c.drawString(40, y, f"{row['date'].date()}: {row['hybrid']:.2f}")
        y -= 15
        if y < 40:
            c.showPage()
            y = 750

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    selected_names = st.multiselect(
        "Companies",
        list(TICKER_MAP.keys()),
        default=["Bell (BCE.TO)"],
    )
    years = st.slider("Years of history", 1, 10, 3)
    horizon = st.slider("Forecast days", 30, 365, 180)

start = dt.date.today() - dt.timedelta(days=365 * years)
end = dt.date.today()

tab_forecast, tab_signals, tab_pdf = st.tabs(["📊 Forecasts", "📈 Signals", "📄 PDF"])

# -----------------------------------------------------------------------------
# Main loop over selected tickers
# -----------------------------------------------------------------------------
for t_name in selected_names:
    ticker = TICKER_MAP[t_name]
    df = fetch_price_data(ticker, start, end)

    if df.empty:
        st.warning(f"No data available for {t_name}.")
        continue

    # ----------------- PRICE + FORECAST TAB -----------------
    with tab_forecast:
        st.subheader(f"{t_name} – Price")

        fig_price = px.line(
            df,
            x="date",
            y="price",
            title=f"{t_name} – Price",
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # Train RF and forecast
        rf_model = train_rf(df)
        rf_fc = forecast_rf(df, rf_model, horizon) if rf_model is not None else None
        p_fc = forecast_prophet(df, horizon)

        hist, hybrid_fc = hybrid_model(df, rf_fc, p_fc)

        st.subheader(f"{t_name} – Hybrid Forecast")

        fig_hybrid = px.line(
            hybrid_fc,
            x="date",
            y="hybrid",
            title=f"{t_name} – Hybrid Forecast (AI + Prophet)",
        )
        st.plotly_chart(fig_hybrid, use_container_width=True)

    # ----------------- SIGNALS TAB -----------------
    with tab_signals:
        st.subheader(f"{t_name} – Trading Signals")

        sig_df = buy_sell(df)
        sig_df = sig_df.dropna(subset=["ma20", "ma50"])

        if sig_df.empty:
            st.info("Not enough data to compute moving-average signals yet.")
        else:
            fig_sig = px.line(
                sig_df,
                x="date",
                y=["price", "ma20", "ma50"],
                title=f"{t_name} – Price & Moving Averages",
            )
            st.plotly_chart(fig_sig, use_container_width=True)
            st.metric("Latest Signal", sig_df["signal"].iloc[-1])

    # ----------------- PDF TAB -----------------
    with tab_pdf:
        st.subheader(f"{t_name} – PDF Report")
        if st.button(f"Generate PDF for {t_name}"):
            pdf_bytes = generate_pdf(t_name, df, hybrid_fc)
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"{ticker}_forecast.pdf",
                mime="application/pdf",
            )
