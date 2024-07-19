import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming 'df_half' is available in your Streamlit environment
# If not, you'll need to load it here (e.g., from a CSV)
# df_half = pd.read_csv(...) 

# --- Model Training (This part can be hidden in a separate script if needed) ---
# Separate features and target variable
X = df_half.drop('CHURN', axis=1)
y = df_half['CHURN']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Remove 'user_id' column
X_train = X_train.drop('user_id', axis=1)
X_test = X_test.drop('user_id', axis=1)

# Apply one-hot encoding to 'TOP_PACK'
X_train = pd.get_dummies(X_train, columns=['TOP_PACK'])
X_test = pd.get_dummies(X_test, columns=['TOP_PACK'])

# Ensure train and test sets have the same columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Initialize and train the classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# --- Streamlit App ---
st.title('Churn Prediction App')

# Get feature names from the trained model
feature_names = clf.feature_names_in_

# Create input fields dynamically
input_data = {}
for feature in feature_names:
    # Assuming all features are numerical
    input_data[feature] = st.number_input(feature)

# Create a validation button
if st.button('Predict'):
    # Create a DataFrame from the input values
    input_df = pd.DataFrame([input_data])

    # Make prediction using the trained model
    prediction = clf.predict(input_df)

    # Display the prediction
    st.write('Prediction:', prediction[0])
