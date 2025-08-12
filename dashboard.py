import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import date, timedelta


# ! === DATA LOADER ===
@st.cache_data()
def load_data(ticker, start_date, end_date):
    data = yf.download(tickers=ticker, start=start_date, end=end_date, auto_adjust=True)
    data.sort_index()
    data.columns = data.columns.droplevel(1)
    return data


# ! ========== PERIOD DETERMINATION FOR PREDICTIONS ==========
def determine_period(range_value, n_preds, periods):
    return n_preds * periods.get(range_value, 30)


# ! ========== CALCULATE METRICS ==========
def calculate_metrics(data):
    last_price = data["Close"].iloc[-1]
    previous_price = data["Close"].iloc[-2]
    diff = last_price - previous_price
    change_rate = diff / previous_price * 100
    high, low, volume = (
        data["High"].max(),
        data["Low"].min(),
        data["Volume"].sum(),
    )
    return last_price, diff, change_rate, high, low, volume


# ! ========== CONSTANTS ==========
TICKERS = (
    "AAPL",
    "AMZN",
    "GOOG",
    "MSFT",
    "GME",
    "TSLA",
    "META",
    "GS",
    "DJIA",
    "SPX",
    "COMP",
)
START_DATE = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
RANGE_OPTIONS = {
    "Last 7 Days": "7d",
    "Last 4 Weeks": "4w",
    "Last 6 Months": "6m",
    "Last 5 Years": "5y",
    "All": "max",
}
BASE_PERIODS = {"7d": 1, "4w": 7, "6m": 30, "5y": 365, "max": 365}
PRED_UNITS = {
    "7d": "Days",
    "4w": "Weeks",
    "6m": "Months",
    "5y": "Years",
    "max": "Years",
}
GRAPH_HEIGHT = 600

# ! ========== CONFIG ==========
st.set_page_config(
    layout="wide",
    page_title="StockLab",
    page_icon="chart_with_upwards_trend",
)
st.title("📈 StockLab")
st.markdown(
    """
    Use the sidebar to select stock tickers, specify time frames, and customise prediction horizons. The dashboard provides visual insights through candlestick and line charts, forecast trends, and comparison between actual and predicted prices.
    """
)
# ! ========== SIDEBAR ==========

st.sidebar.title("Parameters")
# * Ticker
selected_ticker = st.sidebar.selectbox("Ticker", TICKERS, key="ticker")
# * Chart Type
selected_chart_type = st.sidebar.selectbox(
    "Chart Type", ["Candlestick", "Line"], key="chart"
)
# * Show Range
selected_range = st.sidebar.selectbox(
    "Show Range", list(RANGE_OPTIONS.keys()), key="range"
)
range_value = RANGE_OPTIONS[selected_range]
# * Number of Predictions
selected_n_predictions = st.sidebar.number_input(
    f"Number of Predictions ({PRED_UNITS[range_value]})",
    step=1,
    min_value=1,
    max_value=10,
    key="n_pred",
)

cap = st.sidebar.caption("Loading data... Please wait...")
data = load_data(selected_ticker, START_DATE, TODAY)
cap.caption(f"Data loaded: {pd.Timestamp.now().strftime('%d-%m-%Y %H:%M:%S')}")


period = determine_period(range_value, selected_n_predictions, BASE_PERIODS)


# ! ========== SET X AXIS RANGE FOR MAIN CHART ==========
def get_xaxis_range(today, range_selection, min_index):
    end_date = pd.to_datetime(today)
    if range_selection == "7d":
        start_date = end_date - timedelta(days=7)
    elif range_selection == "4w":
        start_date = end_date - timedelta(weeks=4)
    elif range_selection == "6m":
        start_date = end_date - timedelta(days=180)
    elif range_selection == "5y":
        start_date = end_date - timedelta(days=5 * 365)
    elif range_selection == "max":
        start_date = min_index
    else:
        start_date = end_date - timedelta(days=30)
    return [start_date, end_date]


# ! ========== SET Y AXIS RANGE FOR MAIN CHART ==========
def get_yaxis_range(data, xrange):
    start_date, end_date = pd.to_datetime(xrange[0]), pd.to_datetime(xrange[1])
    filtered_data = data.loc[(data.index >= start_date) & (data.index <= end_date)]
    if filtered_data.empty:
        return None
    y_min = filtered_data["Low"].min()
    y_max = filtered_data["High"].max()
    y_padding = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 1
    return [y_min - y_padding, y_max + y_padding]


# ! ========== SET AXIS RANGE FOR PREDICTED GRAPH ==========
def get_forecast_xaxis_range(
    today, range_selection, n_predictions, fcast_min, fcast_max
):
    today_date = pd.to_datetime(today)
    if range_selection == "7d":
        start_date = today_date - timedelta(days=7)
        end_date = today_date + timedelta(days=n_predictions)
    elif range_selection == "4w":
        start_date = today_date - timedelta(days=7)
        end_date = today_date + timedelta(weeks=n_predictions)
    elif range_selection == "6m":
        start_date = today_date - timedelta(days=180)
        end_date = today_date + timedelta(days=n_predictions * 30)
    elif range_selection == "5y":
        start_date = today_date - timedelta(days=5 * 365)
        end_date = today_date + timedelta(days=n_predictions * 365)
    elif range_selection == "max":
        start_date, end_date = fcast_min, fcast_max
    else:
        start_date = today_date - timedelta(days=30)
        end_date = today_date + timedelta(days=n_predictions * 30)
    return [start_date, end_date]


# ! ========== MAIN CHART ==========
xrange = get_xaxis_range(TODAY, range_value, data.index.min())
yrange = get_yaxis_range(data, xrange)
last_price, diff, change_rate, high, low, volume = calculate_metrics(data)
st.metric(
    label=f"{selected_ticker}",
    value=f"{last_price:.2f} USD",
    delta=f"{diff:.2f} ({change_rate:.2f}%)",
)
col_1, col_2, col_3 = st.columns(3)
col_1.metric("High", f"{high:.2f} USD")
col_2.metric("Low", f"{low:.2f} USD")
col_3.metric("Volume", f"{volume:,}")
if selected_chart_type == "Candlestick":
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
            )
        ]
    )
else:
    fig = px.line(data, x=data.index, y="Close", render_mode="webg1")

fig.update_layout(
    title="Main Chart",
    xaxis_rangeslider_visible=False,
    xaxis_range=xrange,
    yaxis_range=yrange,
    height=GRAPH_HEIGHT,
    xaxis_title="Date",
    yaxis_title="Stock Price in USD ($)",
)
st.plotly_chart(fig, use_container_width=True)

# ! ========== CURRENT DATA ==========
st.subheader("Current data")
current_data = data.copy()
current_data.index = current_data.index.strftime("%Y-%m-%d")
st.write(current_data.tail())

# ! ========== FORECASTER (FB PROPHET) ==========
data_reset = data.reset_index()
date_col = data_reset.columns[0]
df_train = data_reset.rename(columns={date_col: "ds", "Close": "y"})[["ds", "y"]]
model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)
forecast_table = forecast[forecast["ds"] >= pd.Timestamp(date.today())][
    ["ds", "yhat", "trend", "yhat_lower", "yhat_upper"]
]
forecast_table["ds"] = forecast_table["ds"].dt.date
forecast_table.set_index("ds", inplace=True)
st.subheader(f"Predicted data ({selected_n_predictions} {PRED_UNITS[range_value]})")
st.write(forecast_table)

# ! ========== PREDICTIONS ==========
forecast_xrange = get_forecast_xaxis_range(
    TODAY,
    range_value,
    selected_n_predictions,
    forecast["ds"].min(),
    forecast["ds"].max(),
)
fig_1 = plot_plotly(model, forecast, trend=True)
fig_1.add_vline(
    x=TODAY,
    line_width=2,
    line_dash="dash",
    line_color="green",
)
fig_1.update_layout(
    title="Predicted Price and Trend",
    xaxis_range=forecast_xrange,
    height=GRAPH_HEIGHT,
    xaxis_title="Date",
    yaxis_title="Stock Price in USD ($)",
)
st.plotly_chart(fig_1)

# ! ========== FORECAST COMPONENTS ==========
st.subheader("Forecast Components")
fig_2 = model.plot_components(forecast)
st.pyplot(fig_2)


# ! ========== ACTUAL VS PREDICTED VALUES UNTIL TODAY ==========
df_actual = df_train.set_index("ds")
df_forecast = forecast.set_index("ds")[["yhat"]]
compare_df = df_actual.join(df_forecast, how="left")

fig_compare = go.Figure()

# Actual Close Prices
fig_compare.add_trace(
    go.Scatter(
        x=compare_df.index,
        y=compare_df["y"],
        mode="lines",
        name="Actual Values (Close)",
    )
)

# Forecasted Prices (yhat)
fig_compare.add_trace(
    go.Scatter(
        x=compare_df.index,
        y=compare_df["yhat"],
        mode="lines",
        name="Predictions (yhat)",
    )
)

fig_compare.update_layout(
    title="Actual vs Forecasted Close Prices",
    xaxis_title="Date",
    yaxis_title="Stock Price in USD ($)",
    height=GRAPH_HEIGHT,
)
st.plotly_chart(fig_compare)
