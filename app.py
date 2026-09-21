import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('logit.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Input fields for features
# Based on the 'x' DataFrame columns: Traffic_Congestion, Delivery_Distance, Vehicle_Condition, Driver_Experience, Customer_Location_Category, Weather_Conditions, Road_Infrastructure_Quality, Number_of_Vehicles_in_Fleet, Fuel_Price, Warehouse_Processing_Time

traffic_congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
delivery_distance = st.number_input('Delivery Distance (km)', min_value=0.1, max_value=100.0, value=25.0, step=0.1)
vehicle_condition = st.slider('Vehicle Condition (1-5)', 1, 5, 3)
driver_experience = st.number_input('Driver Experience (years)', min_value=0, max_value=50, value=5, step=1)
customer_location_category = st.slider('Customer Location Category (1-3)', 1, 3, 2)
weather_conditions = st.slider('Weather Conditions (1-5)', 1, 5, 3)
road_infrastructure_quality = st.slider('Road Infrastructure Quality (1-5)', 1, 5, 3)
number_of_vehicles_in_fleet = st.number_input('Number of Vehicles in Fleet', min_value=1, max_value=1000, value=50, step=1)
fuel_price = st.number_input('Fuel Price ($/gallon)', min_value=1.0, max_value=10.0, value=3.0, step=0.1)
warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, max_value=300, value=60, step=1)


if st.button('Predict Delivery Delay'):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([[traffic_congestion,
                                delivery_distance,
                                vehicle_condition,
                                driver_experience,
                                customer_location_category,
                                weather_conditions,
                                road_infrastructure_quality,
                                number_of_vehicles_in_fleet,
                                fuel_price,
                                warehouse_processing_time]],
                              columns=['Traffic_Congestion',
                                       'Delivery_Distance',
                                       'Vehicle_Condition',
                                       'Driver_Experience',
                                       'Customer_Location_Category',
                                       'Weather_Conditions',
                                       'Road_Infrastructure_Quality',
                                       'Number_of_Vehicles_in_Fleet',
                                       'Fuel_Price',
                                       'Warehouse_Processing_Time'])
    
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader('Prediction Result:')
    if prediction == 1:
        st.error(f"**Delay Predicted!** (Probability: {prediction_proba[1]:.2f})")
    else:
        st.success(f"**No Delay Predicted!** (Probability: {prediction_proba[0]:.2f})")

# To run this app:
# 1. Save the code above as `app.py` in your Colab environment.
# 2. Run the following commands in a new cell:
#    !pip install streamlit
#    !streamlit run app.py & npx localtunnel --port 8501
