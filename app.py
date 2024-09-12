import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('customer_churn_data.csv')

# User feature selection
st.title("Customer Churn Prediction")
st.header("Select features to train the model")

selected_features = st.multiselect('Select features', df.columns.tolist(), default=['tenure', 'MonthlyCharges'])

# Separate target and features
target = 'Churn'
X = df[selected_features]
y = df[target]

# Preprocessing: Label encoding for categorical features
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# User selects models to train
st.header("Select machine learning models to apply")
model_options = ['Logistic Regression', 'Random Forest', 'XGBoost']
selected_models = st.multiselect('Choose Models', model_options, default=['Logistic Regression', 'Random Forest'])

# Model initialization
def train_and_predict(models, X_train, y_train, X_test):
    predictions = {}
    
    if 'Logistic Regression' in models:
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        predictions['Logistic Regression'] = y_pred_lr
    
    if 'Random Forest' in models:
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        predictions['Random Forest'] = y_pred_rf
    
    if 'XGBoost' in models:
        xgb_model = XGBClassifier()
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        predictions['XGBoost'] = y_pred_xgb

    return predictions

# Train models and get predictions
if st.button('Train and Predict'):
    model_predictions = train_and_predict(selected_models, X_train, y_train, X_test)
    
    # Display model accuracies
    st.subheader("Model Performance")
    for model, pred in model_predictions.items():
        acc = accuracy_score(y_test, pred)
        st.write(f'{model}: {acc * 100:.2f}% accuracy')

    # Aggregating predictions
    final_prediction = []
    for i in range(len(X_test)):
        # Get majority vote prediction
        churn_votes = sum([model_predictions[model][i] for model in model_predictions])
        final_prediction.append(1 if churn_votes > len(model_predictions) / 2 else 0)
    
    st.subheader("Final Aggregated Prediction")
    st.write(f'Churn: {"Yes" if final_prediction[0] == 1 else "No"}')

# To deploy: streamlit run app.py
