import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('customer_churn_data.csv')

# Exclude "customer ID" and "Churn" from selectable features
excluded_columns = ['customerID', 'Churn']
feature_columns = [col for col in df.columns if col not in excluded_columns]

# User feature selection
st.title("Customer Churn Prediction")
st.header("Select features to train the model")

selected_features = st.multiselect('Select features', feature_columns, default=['tenure', 'MonthlyCharges'])

# Display a snapshot of selected features with the 'Churn' column
if selected_features:
    st.subheader("Data Preview")
    st.write(df[selected_features + ['Churn']].head(5))

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

# Convert target variable 'Churn' from 'Yes', 'No' to 1, 0 using LabelEncoder
y = LabelEncoder().fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# User selects models to train
st.header("Select machine learning models to apply")
model_options = ['Logistic Regression', 'Random Forest', 'Decision Tree']
selected_models = st.multiselect('Choose Models', model_options, default=['Logistic Regression', 'Random Forest'])

# Model initialization and training
def train_and_predict(models, X_train, y_train, X_test):
    predictions = {}
    trained_models = {}
    
    if 'Logistic Regression' in models:
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        predictions['Logistic Regression'] = y_pred_lr
        trained_models['Logistic Regression'] = lr_model
    
    if 'Random Forest' in models:
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        predictions['Random Forest'] = y_pred_rf
        trained_models['Random Forest'] = rf_model
    
    if 'Decision Tree' in models:
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, y_train)
        y_pred_dt = dt_model.predict(X_test)
        predictions['Decision Tree'] = y_pred_dt
        trained_models['Decision Tree'] = dt_model

    return predictions, trained_models

# Train models and get predictions
if st.button('Train and Predict'):
    model_predictions, trained_models = train_and_predict(selected_models, X_train, y_train, X_test)
    
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

    # User Input for New Prediction
    st.subheader("Predict Churn for Custom Input")
    custom_input = []
    for feature in selected_features:
        # Provide example values from the dataset
        example_values = df[feature].dropna().sample(3).tolist()
        example_text = ", ".join(map(str, example_values))
        # Get user input for each selected feature without min/max constraints
        user_input = st.text_input(f"Enter value for {feature} (e.g., {example_text})", value="")
        custom_input.append(float(user_input))
    
    # Encode categorical inputs
    custom_input_df = pd.DataFrame([custom_input], columns=selected_features)
    for col in custom_input_df.select_dtypes(include=['object']).columns:
        le = label_encoders[col]
        custom_input_df[col] = le.transform(custom_input_df[col])

    # Scale the custom input
    custom_input_scaled = scaler.transform(custom_input_df)
    
    # Predict using the trained models
    custom_predictions = {}
    for model_name, model in trained_models.items():
        custom_predictions[model_name] = model.predict(custom_input_scaled)[0]

    # Show custom prediction results
    st.subheader("Custom Input Prediction Results")
    for model_name, prediction in custom_predictions.items():
        st.write(f'{model_name} predicts: {"Churn" if prediction == 1 else "No Churn"}')

    # Aggregate custom predictions
    churn_votes = sum([prediction for prediction in custom_predictions.values()])
    final_custom_prediction = 1 if churn_votes >= 2 else 0
    st.write(f'Final Aggregated Prediction for Custom Input: {"Churn" if final_custom_prediction == 1 else "No Churn"}')

# To deploy: streamlit run app.py
