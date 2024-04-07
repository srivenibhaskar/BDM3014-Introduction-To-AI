import streamlit as st
import joblib

# Load scaler values
scaler = joblib.load("scaler_val.pkl")
mean_amount = scaler.center_[0]
std_amount = scaler.scale_[0]

# User interface
st.title("Credit Card Fraud Detection")

# Input fields
amount = st.number_input("Enter transaction amount:", min_value=0.01, step=0.01)
model_option = st.selectbox("Select model:", ("Logistic Regression", "Naive Bayes", "Decision Tree", "Random Forest"))

# Load saved test set predictions
test_predictions = joblib.load("test_predictions.pkl")

if st.button("Get Prediction"):
    # Scale the input amount
    scaled_amount = (amount - mean_amount) / std_amount
    
    # Map model option to index in the list of models
    model_index_map = {"Logistic Regression": 0, "Naive Bayes": 1, "Decision Tree": 2, "Random Forest": 3}
    
    # Check if the selected model option is valid
    if model_option in model_index_map:
        # Extract the prediction based on the selected model option
        model_index = model_index_map[model_option]
        selected_prediction = test_predictions[model_index]
        
        # Display the prediction
        st.write("Predicted Class:", selected_prediction)
    else:
        st.error("Invalid model option selected.")
