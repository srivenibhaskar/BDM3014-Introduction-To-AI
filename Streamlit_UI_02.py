import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

scaler = joblib.load("scaler_val.pkl")
mean_amount = scaler.center_[0]
std_amount = scaler.scale_[0]

test_predictions = joblib.load("test_predictions.pkl")

vars = joblib.load("my_variables.pkl")
true_labels = vars['y']

model_descriptions = {
    "Logistic Regression": "Logistic Regression is a linear classification algorithm that predicts the probability of a binary outcome.",
    "Naive Bayes": "Naive Bayes is a probabilistic classifier based on Bayes' theorem with strong (naive) independence assumptions.",
    "Decision Tree": "Decision Tree builds a tree-like structure where each internal node represents a feature, each branch represents a decision, and each leaf node represents an outcome.",
    "Random Forest": "Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes as the prediction.",
    "K-Nearest Neighbors (KNN)": "K-Nearest Neighbors (KNN) is a non-parametric and instance-based learning algorithm that classifies objects based on their closest training examples in the feature space.",
    "XGBoost": "XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.",
}

def display_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    st.subheader("Model Performance Metrics:")
    st.write("Accuracy:", accuracy)
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1 Score:", f1)

def display_feature_importance(feature_importance):
    st.subheader("Feature Importance:")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=X.columns)
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    st.pyplot(plt)

st.title("Credit Card Fraud Detection")

amount = st.number_input("Enter transaction amount:", min_value=0.01, step=0.01)
model_option = st.selectbox("Select model:", ("Logistic Regression", "Naive Bayes", "Decision Tree", "Random Forest", "K-Nearest Neighbors (KNN)", "XGBoost"))

if model_option in model_descriptions:
    st.subheader("Model Description:")
    st.write(model_descriptions[model_option])

if st.button("Get Prediction"):
    scaled_amount = (amount - mean_amount) / std_amount
    
    model_index_map = {
        "Logistic Regression": 0,
        "Naive Bayes": 1,
        "Decision Tree": 2,
        "Random Forest": 3,
        "K-Nearest Neighbors (KNN)": 4,
        "XGBoost": 5
    }
    
    if model_option in model_index_map:
        model_index = model_index_map[model_option]
        selected_prediction = [test_predictions[model_index]] * len(true_labels)
        
        prediction_text = "Fraudulent" if selected_prediction[0] == 1 else "Not Fraudulent"
        st.write("Predicted Class:", prediction_text)
        
        display_metrics(true_labels, selected_prediction)
        
        if model_option in ["Decision Tree", "Random Forest", "XGBoost"]:
            feature_importance = joblib.load(f"{model_option.lower()}_feature_importance.pkl")
            display_feature_importance(feature_importance)
            
    else:
        st.error("Invalid model option selected.")
