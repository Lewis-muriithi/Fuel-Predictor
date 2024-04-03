import pickle
import datetime
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Add an image. 
st.image('/content/fuel-price.jpg')

def load_model(model_path):
    return pickle.load(open(model_path, 'rb'))

def fuel_prediction(start_date, pred_period, exchange_rate_model, town_model):
    try:
        # Forecast exchange rate
        exchange_rate_pred = exchange_rate_model.get_forecast(steps=pred_period)
        exchange_rate_forecasted = exchange_rate_pred.predicted_mean

        # Forecast town fuel prices
        town_pred = town_model.get_forecast(steps=pred_period)
        town_forecasted = town_pred.predicted_mean

        # Divide town fuel prices by exchange rate to get final predictions
        final_forecasted = town_forecasted / exchange_rate_forecasted

        return final_forecasted, exchange_rate_forecasted
    except Exception as e:
        st.error(f"Error occurred: {e}")

# Title
st.title("Fuel Price Predictor")

# Load exchange rate model
exchange_rate_model = load_model('/content/exchange_rate_model.pkl')

# Load all models for Nairobi
nrb_super_model = load_model('/content/nrb_super_model.pkl')
nrb_diesel_model = load_model('/content/nrb_diesel_model.pkl')
nrb_kerosene_model = load_model('/content/nrb_kerosene_model.pkl')

# Load all models for Mombasa
mbs_super_model = load_model('/content/mbs_super_model.pkl')
mbs_diesel_model = load_model('/content/mbs_diesel_model.pkl')
mbs_kerosene_model = load_model('/content/mbs_kerosene_model.pkl')

# Load all models for Kisumu
ksm_super_model = load_model('/content/ksm_super_model.pkl')
ksm_diesel_model = load_model('/content/ksm_diesel_model.pkl')
ksm_kerosene_model = load_model('/content/ksm_kerosene_model.pkl')

# Getting input data from the user
start_date = st.date_input('Start date', datetime.date(2023, 12, 15))
pred_period = st.number_input('Prediction period', step=1, value=None)

def main():
  # Creating buttons for prediction
  if st.button('Fuel predictions'):
    # Perform prediction for Nairobi
    st.subheader("Nairobi Predictions:")
    nrb_super_pred, exchange_rate_pred = fuel_prediction(start_date, pred_period, exchange_rate_model, nrb_super_model)
    nrb_diesel_pred, _ = fuel_prediction(start_date, pred_period, exchange_rate_model, nrb_diesel_model)
    nrb_kerosene_pred, _ = fuel_prediction(start_date, pred_period, exchange_rate_model, nrb_kerosene_model)

    # Create DataFrame for predictions
    index = pd.date_range(start=start_date, periods=pred_period, freq='MS') + pd.Timedelta(days=14)
    nairobi_predictions = pd.DataFrame({
         'Date': index,
         'Exchange_Rate': exchange_rate_pred,
         'Nairobi Super Prediction': nrb_super_pred,
         'Nairobi Diesel Prediction': nrb_diesel_pred,
         'Nairobi Kerosene Prediction': nrb_kerosene_pred
      })
    nairobi_predictions.set_index('Date', inplace=True)
    st.write(nairobi_predictions)

    # Plot fuel prices for Nairobi
    #st.write("Nairobi Fuel Price Predictions")
    plt.figure(figsize=(6, 4))
    plt.plot(nairobi_predictions.index, nairobi_predictions['Nairobi Super Prediction'], label='Nairobi Super')
    plt.plot(nairobi_predictions.index, nairobi_predictions['Nairobi Diesel Prediction'], label='Nairobi Diesel')
    plt.plot(nairobi_predictions.index, nairobi_predictions['Nairobi Kerosene Prediction'], label='Nairobi Kerosene')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Nairobi Fuel Price Predictions')
    plt.legend()
    st.pyplot()

    # Predictions for Mombasa
    st.subheader("Mombasa Predictions")
    mombasa_super_pred, _ = fuel_prediction(start_date, pred_period, exchange_rate_model, mbs_super_model)
    mombasa_diesel_pred, _ = fuel_prediction(start_date, pred_period, exchange_rate_model, mbs_diesel_model)
    mombasa_kerosene_pred, _ = fuel_prediction(start_date, pred_period, exchange_rate_model, mbs_kerosene_model)

    mombasa_predictions = pd.DataFrame({
          'Date': index,
          'Exchange_Rate': exchange_rate_pred,
          'Mombasa Super Prediction': mombasa_super_pred,
          'Mombasa Diesel Prediction': mombasa_diesel_pred,
          'Mombasa Kerosene Prediction': mombasa_kerosene_pred
      })
    mombasa_predictions.set_index('Date', inplace=True)
    st.write(mombasa_predictions)

    # Plot fuel prices for Mombasa
    #st.write("Mombasa Fuel Price Predictions")

    width = st.sidebar.slider("plot width", 1, 10, 6)
    height = st.sidebar.slider("plot height", 1, 10, 4)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(mombasa_predictions.index, mombasa_predictions['Mombasa Super Prediction'], label='Mombasa Super')
    ax.plot(mombasa_predictions.index, mombasa_predictions['Mombasa Diesel Prediction'], label='Mombasa Diesel')
    ax.plot(mombasa_predictions.index, mombasa_predictions['Mombasa Kerosene Prediction'], label='Mombasa Kerosene')
    ax.legend()

    st.pyplot(fig)

    # plt.figure(figsize=(6, 4))
    # plt.plot(mombasa_predictions.index, mombasa_predictions['Mombasa Super Prediction'], label='Mombasa Super')
    # plt.plot(mombasa_predictions.index, mombasa_predictions['Mombasa Diesel Prediction'], label='Mombasa Diesel')
    # plt.plot(mombasa_predictions.index, mombasa_predictions['Mombasa Kerosene Prediction'], label='Mombasa Kerosene')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.title('Mombasa Fuel Price Predictions')
    # plt.legend()
    # st.pyplot()

    # Predictions for Kisumu
    st.subheader("Kisumu Predictions")
    kisumu_super_pred, _ = fuel_prediction(start_date, pred_period, exchange_rate_model, ksm_super_model)
    kisumu_diesel_pred, _ = fuel_prediction(start_date, pred_period, exchange_rate_model, ksm_diesel_model)
    kisumu_kerosene_pred, _ = fuel_prediction(start_date, pred_period, exchange_rate_model, ksm_kerosene_model)

    kisumu_predictions = pd.DataFrame({
         'Date': index,
         'Exchange_Rate': exchange_rate_pred,
         'Kisumu Super Prediction': kisumu_super_pred,
         'Kisumu Diesel Prediction': kisumu_diesel_pred,
         'Kisumu Kerosene Prediction': kisumu_kerosene_pred
      })
    kisumu_predictions.set_index('Date', inplace=True)
    st.write(kisumu_predictions)

    # Plot fuel prices for NaKisumuirobi
    #st.write("Kisumu Fuel Price Predictions")
    plt.figure(figsize=(6, 4))
    plt.plot(kisumu_predictions.index, kisumu_predictions['Kisumu Super Prediction'], label='Kisumu Super')
    plt.plot(kisumu_predictions.index, kisumu_predictions['Kisumu Diesel Prediction'], label='Kisumu Diesel')
    plt.plot(kisumu_predictions.index, kisumu_predictions['Kisumu Kerosene Prediction'], label='Kisumu Kerosene')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Kisumu Fuel Price Predictions')
    plt.legend()
    st.pyplot()


if __name__ == '__main__':
    main()