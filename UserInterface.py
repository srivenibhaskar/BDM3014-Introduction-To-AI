import streamlit as st
import joblib
import numpy as np
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

scaler = joblib.load("scaler_val.pkl")
mean_amount = scaler.center_[0]
std_amount = scaler.scale_[0]

def scale_amount(amount):
    scaled_amount = (amount - mean_amount) / std_amount
    return scaled_amount

def predict_fraud(v_features, amount):
    model = joblib.load('test_predictions.pkl')
    scaled_amount = scale_amount(amount)
    all_features = list(map(float, v_features.split())) + [scaled_amount]
    prediction = model.predict([all_features])[0]
    return prediction, model

def main():
    st.title('Credit Card Fraud Detection')

    v_features = st.sidebar.text_input('Enter values for features V1-V28', '')
    amount = st.sidebar.number_input('Enter Amount', min_value=0.0)

    if st.sidebar.button('Predict'):
        prediction, model = predict_fraud(v_features, amount)

        vars = joblib.load('my_variables.pkl')
        X_train = vars['x_train_adasyn']

        all_features_array = np.array(list(map(float, v_features.split())) + [scale_amount(amount)]).reshape(1, -1)

        feature_names = X_train.columns.tolist()
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=feature_names, class_names=['Non-Fraud', 'Fraud'], discretize_continuous=True)
        explanation = explainer.explain_instance(all_features_array[0], model.predict_proba, num_features=len(all_features_array[0]))

        features = [f[0] for f in explanation.as_list()]
        weights = [f[1] for f in explanation.as_list()]

        total_weight = sum(weights)
        percentages = [(w / total_weight) * 100 for w in weights]

        top_features = features[:5] + features[-5:]
        top_percentages = percentages[:5] + percentages[-5:]

        text_summary = "Based on the provided input features, the model predicts the transaction to be "
        if prediction == 1:
            text_summary += "fraudulent.\n\n"
        else:
            text_summary += "non-fraudulent.\n\n"

        text_summary += "The top influential features and their contributions to the prediction are as follows:\n\n"
        for feature, percentage in zip(top_features, top_percentages):
            if percentage > 0:
                text_summary += f"The feature '{feature}' contributes positively, indicating a higher likelihood of fraud. "
            else:
                text_summary += f"The feature '{feature}' contributes negatively, indicating a lower likelihood of fraud. "
            text_summary += f"The contribution percentage is {abs(percentage):.2f}%.\n"

        st.subheader('Prediction Result')
        if prediction == 1:
            st.write('**Prediction:** Fraudulent Transaction', unsafe_allow_html=True)
        else:
            st.write('**Prediction:** Non-fraudulent Transaction', unsafe_allow_html=True)

        st.subheader('Feature Contribution')
        plt.figure(figsize=(10, 6))
        plt.barh(top_features, top_percentages, color=['green' if w >= 0 else 'red' for w in weights])
        plt.xlabel('Contribution (%)')
        plt.ylabel('Feature')
        plt.title('Top Influential Features')
        plt.grid(axis='x')
        plt.tight_layout()
        st.pyplot(plt)

        st.subheader('Explanation')
        st.write(text_summary)

if __name__ == '__main__':
    main()
