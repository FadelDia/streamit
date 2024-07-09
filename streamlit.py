! pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeRegressor
from snapml import DecisionTreeRegressor as SnapDecisionTreeRegressor
import time
import gc

# Title of the Streamlit app
st.title("Taxi Tip Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the data
    raw_data = pd.read_csv(uploaded_file)

    # Preprocess the data
    raw_data = raw_data[raw_data['tip_amount'] > 0]
    raw_data = raw_data[(raw_data['tip_amount'] <= raw_data['fare_amount'])]
    raw_data = raw_data[((raw_data['fare_amount'] >=2) & (raw_data['fare_amount'] < 200))]
    clean_data = raw_data.drop(['total_amount'], axis=1)
    del raw_data
    gc.collect()

    clean_data['tpep_dropoff_datetime'] = pd.to_datetime(clean_data['tpep_dropoff_datetime'])
    clean_data['tpep_pickup_datetime'] = pd.to_datetime(clean_data['tpep_pickup_datetime'])
    clean_data['pickup_hour'] = clean_data['tpep_pickup_datetime'].dt.hour
    clean_data['dropoff_hour'] = clean_data['tpep_dropoff_datetime'].dt.hour
    clean_data['pickup_day'] = clean_data['tpep_pickup_datetime'].dt.weekday
    clean_data['dropoff_day'] = clean_data['tpep_dropoff_datetime'].dt.weekday
    clean_data['trip_time'] = (clean_data['tpep_dropoff_datetime'] - clean_data['tpep_pickup_datetime']).dt.total_seconds()

    # You might want to add a slider here to control the number of rows used
    first_n_rows = st.slider("Number of rows to use", 1000, len(clean_data), 200000)
    clean_data = clean_data.head(first_n_rows)

    clean_data = clean_data.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)
    get_dummy_col = ["VendorID","RatecodeID","store_and_fwd_flag","PULocationID", "DOLocationID","payment_type", "pickup_hour", "dropoff_hour", "pickup_day", "dropoff_day"]
    proc_data = pd.get_dummies(clean_data, columns = get_dummy_col)
    del clean_data
    gc.collect()

    y = proc_data[['tip_amount']].values.astype('float32')
    proc_data = proc_data.drop(['tip_amount'], axis=1)
    X = proc_data.values
    X = normalize(X, axis=1, norm='l1', copy=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model selection
    model_choice = st.selectbox("Choose a model:", ["Scikit-Learn Decision Tree", "Snap ML Decision Tree"])

    if model_choice == "Scikit-Learn Decision Tree":
        # Train Scikit-Learn model
        sklearn_dt = DecisionTreeRegressor(max_depth=8, random_state=35)
        t0 = time.time()
        sklearn_dt.fit(X_train, y_train)
        sklearn_time = time.time()-t0
        st.write("[Scikit-Learn] Training time (s):  {0:.5f}".format(sklearn_time))

        # Make predictions
        y_pred_sklearn = sklearn_dt.predict(X_test)
        st.write("Scikit-Learn Predictions:", y_pred_sklearn)

    elif model_choice == "Snap ML Decision Tree":
        # Train Snap ML model
        snapml_dt = SnapDecisionTreeRegressor
