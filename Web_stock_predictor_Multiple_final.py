import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error("File not found. Please provide a valid file path.")
        return None

def load_nn_model(model_path):
    try:
        return load_model(model_path)
    except Exception as e:
        st.error("Error loading the model:", e)
        return None

def plot_graph(figsize, data_dict, extra_data=None):
    fig = plt.figure(figsize=figsize)
    for label, values in data_dict.items():
        plt.plot(values, label=label)
    if extra_data:
        plt.plot(extra_data["data"], label=extra_data["label"], linestyle='--')
    plt.legend()
    return fig

def predict_next_close(model, scaler, last_100_days):
    last_100_days = last_100_days.reshape(-1, 1)
    scaled_last_100_days = scaler.transform(last_100_days)
    prediction = model.predict(np.array([scaled_last_100_days]))
    return scaler.inverse_transform(prediction)[0][0]

def main():
    st.title("Stock Price Predictor App")
    model_name = st.selectbox("Select Model", [
        "ADANIPORTS", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE",
        "BHARTIARTL", "BRITANNIA", "CIPLA", "DIVISLAB", "DRREDDY", "EICHERMOT",
        "GRASIM", "HCLTECH", "HDFC", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
        "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC",
        "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NTPC", "POWERGRID",
        "SBILIFE", "SHREECEM", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TITAN",
        "UPL", "WIPRO"
    ])
    model_path = f"C:/Users/mohammad/OneDrive/Desktop/CI & AT/Nifty50/Latest_stock_price_model_{model_name}.keras"
    model = load_nn_model(model_path)
    if model is None:
        st.error("Model loading failed.")
        return

    df_path = f'C:/Users/mohammad/OneDrive/Desktop/CI & AT/Nifty50/output_{model_name.lower()}.csv'
    df = load_data(df_path)
    if df is None:
        return

    st.subheader("Stock Data")
    st.write(df)

    splitting_len = int(len(df) * 0.7)
    x_test = pd.DataFrame(df.Close[splitting_len:])

    ma_days = [100, 200, 250]
    for days in ma_days:
        df[f'MA_for_{days}_days'] = df.Close.rolling(days).mean()
        st.subheader(f'Original Close Price and MA for {days} days')
        st.pyplot(plot_graph((15, 6), {"Close": df['Close'], f"MA_for_{days}_days": df[f'MA_for_{days}_days']}))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test[['Close']])

    x_data = []
    y_data = []

    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i - 100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    predictions = model.predict(x_data)

    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    ploting_data = pd.DataFrame(
        {'Original Test Data': inv_y_test.reshape(-1), 'Predictions': inv_pre.reshape(-1)},
        index=df.index[splitting_len + 100:]
    )

    st.subheader("Original values vs Predicted values")
    st.write(ploting_data)

    st.subheader('Original Close Price vs Predicted Close price')
    fig = plt.figure(figsize=(15, 6))
    plt.plot(pd.concat([df.Close[:splitting_len + 100], ploting_data], axis=0))
    plt.legend(["Data - not used", "Original Test data", "Predicted Test data"])
    st.pyplot(fig)

    last_100_days = x_test[-100:].values
    tomorrow_close = predict_next_close(model, scaler, last_100_days)
    st.subheader("Predicted Close Price for Tomorrow")
    st.write(f"The predicted close price for tomorrow is: {tomorrow_close:.2f}")

if __name__ == "__main__":
    main()
