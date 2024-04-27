# Nifty50 Stock Price Predictor

## Overview
This project predicts the closing prices of stocks in the Nifty50 index using LSTM (Long Short-Term Memory) neural networks.

## Data Collection and Preprocessing
- Historical stock data for various Nifty50 stocks is collected from CSV files.
- Data preprocessing includes:
  - Converting timestamps to datetime format
  - Setting date as index
  - Calculating moving averages (MA) for 100 and 250 days
  - Calculating percentage change in closing prices
  - Dropping NaN values

## Model Training
- LSTM models are trained for each stock in the Nifty50 index.
- For each stock:
  - Features are scaled using MinMaxScaler.
  - Data is prepared in sequences of 100 days windowed format for LSTM.
  - The LSTM model architecture consists of two LSTM layers (128 and 64 units), followed by two Dense layers (25 and 1 unit).
  - Model is compiled using Adam optimizer and Mean Squared Error (MSE) loss.
  - Model is trained with a batch size of 1 and for 2 epochs.

## Model Saving
- Trained models are saved with the respective stock tickers for future use.

## Stock Price Predictor App
- A Streamlit web application is developed to predict tomorrow's closing price for a selected stock from Nifty50.
- Users can choose a stock ticker from the dropdown menu.
- The app loads the corresponding trained LSTM model and historical stock data.
- It displays the original and predicted closing prices, along with moving averages, using matplotlib.
- Users can see the predicted close price for the next day.

## Files
1. `nifty50_stock_price_prediction.py`: Script for training LSTM models for Nifty50 stocks.
2. `nifty50_stock_price_predictor_app.py`: Streamlit web app for predicting stock prices.

## Usage
1. Clone the repository.
2. Run `nifty50_stock_price_prediction.py` to train LSTM models.
3. Run `nifty50_stock_price_predictor_app.py` to launch the web app.
4. Select a stock from the dropdown menu and view the predicted closing price for tomorrow.

## Contributing
- Fork the repository.
- Make changes and create a new branch.
- Commit your changes and push to the branch.
- Create a pull request.

## License
- This project is licensed under the MIT License.

## Acknowledgements
- Data sourced from Nifty50 stock CSV files.
- Built with Python, Streamlit, Keras, Pandas, NumPy, and Matplotlib.
