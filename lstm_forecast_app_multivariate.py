import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

st.title("📊 Multivariate LSTM Financial Forecasting App")

uploaded_file = st.file_uploader("Upload your Excel/CSV file with columns: Date, Open, High, Low, Close, Volume", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    else:
        df = pd.read_excel(uploaded_file, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    st.write("Raw Data", df.tail())

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        st.error(f"File must contain the columns: {required_columns}")
        st.stop()

    # Select features and scale them
    features = required_columns
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    # Sequence length: 20% of data
    seq_length = max(1, int(len(scaled_data) * 0.2))

    if len(scaled_data) <= seq_length:
        st.error(f"Not enough rows to generate sequences. Minimum needed: {seq_length + 1}, provided: {len(scaled_data)}.")
        st.stop()

    # Build sequences
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])
        y.append(scaled_data[i, features.index("Close")])  # Target = Close
    X, y = np.array(X), np.array(y)

    # Define model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(64))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=5, batch_size=32)

    # Forecast
    last_seq = scaled_data[-seq_length:]
    forecast = []
    for _ in range(30):
        input_seq = last_seq[-seq_length:]
        input_seq = input_seq.reshape(1, seq_length, len(features))
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        new_row = last_seq[-1].copy()
        new_row[features.index("Close")] = pred_scaled
        forecast.append(pred_scaled)
        last_seq = np.append(last_seq, [new_row], axis=0)

    forecast_prices = MinMaxScaler().fit(df[["Close"]]).inverse_transform(np.array(forecast).reshape(-1, 1))
    forecast_dates = pd.date_range(start=df["Date"].max() + pd.Timedelta(days=1), periods=30)
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast_Close": forecast_prices.flatten()})

    st.subheader("📈 Forecast for Next 30 Days")
    st.line_chart(forecast_df.set_index("Date"))
    st.dataframe(forecast_df)