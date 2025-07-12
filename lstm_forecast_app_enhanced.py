import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

st.title("📈 LSTM Financial Forecasting App")
uploaded_file = st.file_uploader("Upload your Excel/CSV file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    else:
        df = pd.read_excel(uploaded_file, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    st.write("Raw Data", df.tail())

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

    seq_length = 60

    # Check if data is long enough
    if len(scaled_data) <= seq_length:
        st.error(f"Your dataset is too short. At least {seq_length + 1} rows are required, but you only provided {len(scaled_data)}.")
        st.stop()

    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32)

    last_sequence = scaled_data[-seq_length:]
    forecast = []
    for _ in range(30):
        input_seq = last_sequence[-seq_length:]
        pred = model.predict(input_seq.reshape(1, seq_length, 1), verbose=0)
        forecast.append(pred[0, 0])
        last_sequence = np.append(last_sequence, pred, axis=0)

    forecast_prices = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    future_dates = pd.date_range(df["Date"].max() + pd.Timedelta(days=1), periods=30)
    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast_Close": forecast_prices.flatten()})

    st.subheader("📊 Forecast for Next 30 Days")
    st.line_chart(forecast_df.set_index("Date"))
    st.dataframe(forecast_df)