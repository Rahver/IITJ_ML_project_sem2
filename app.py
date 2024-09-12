import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('customer_churn_data.csv')

# Exclude 'customerID' and 'Churn' from feature selection
excluded_features = ['customerID', 'Churn']
available_features = [col for col in df.columns if col not in excluded_features]

# User feature selection
st.title("Customer Churn Prediction Model")

st.write("""

This ML Project by Rahul Verma-(G23AI2039), Vidyut Bhaskar-(G23AI2128), Atul Singh-(G23AI2104), Mayank Goyal-(G23AI2120), Suraj Mourya-(G23AI2116) allows you to predict customer churn based on selected features.

Choose the features that you believe are most relevant to predicting whether a customer will churn.
The model will use the selected features to train and provide a prediction based on your inputs. This will help the sales team to verify whether their prospect/potential customer is going to churn after their first billing cycle.
""")

st.header("Select Features to Train the Model")

selected_features = st.multiselect('Select features', available_features, default=['gender','Partner','Dependents','PhoneService','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges'])

# Display snapshot of data based on selected features
if selected_features:
    st.subheader("Training Data Snapshot")
    st.write(df[selected_features + ['Churn']].head())

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

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training function
def train_and_predict(X_train, y_train, X_test):
    predictions = {}
    accuracies = {}
    
    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    predictions['Logistic Regression'] = y_pred_lr
    accuracies['Logistic Regression'] = accuracy_score(y_test, y_pred_lr)
    
    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    predictions['Random Forest'] = y_pred_rf
    accuracies['Random Forest'] = accuracy_score(y_test, y_pred_rf)
    
    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    predictions['Decision Tree'] = y_pred_dt
    accuracies['Decision Tree'] = accuracy_score(y_test, y_pred_dt)

    return predictions, accuracies

# Train models and get predictions
model_predictions, model_accuracies = train_and_predict(X_train, y_train, X_test)

# User input for new prediction
st.header("Enter Values for Prediction")

# Create a layout for input fields in a tabular format
num_columns = 3
columns = st.columns(num_columns)

user_input = {}
for idx, feature in enumerate(selected_features):
    col = columns[idx % num_columns]
    
    if df[feature].dtype == 'object':
        # For categorical features, create a dropdown with unique values
        unique_vals = df[feature].unique().tolist()
        user_input[feature] = col.selectbox(f"Select {feature}", unique_vals)
    else:
        # For numerical features, use a slider or number input
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        user_input[feature] = col.slider(f"Enter {feature}", min_val, max_val, value=(min_val + max_val) / 2)

# Convert user input into a DataFrame
user_df = pd.DataFrame([user_input])

# Encode user input with the same label encoders
for col, le in label_encoders.items():
    if col in user_df.columns:
        # Ensure the encoder does not fail with unseen values by using .fit() on the user_df too
        user_df[col] = le.transform(user_df[col].astype(str).values)

# Predict for user input
st.subheader("Prediction for Entered Values")

if st.button('Predict Churn'):
    # Predict using all models
    lr_pred = model_predictions['Logistic Regression']
    rf_pred = model_predictions['Random Forest']
    dt_pred = model_predictions['Decision Tree']
    
    # Display individual model accuracies
    st.write(f"Customer will   Logistic Regression Accuracy: {model_accuracies['Logistic Regression'] * 100:.2f}%")
    st.write(f"Random Forest Accuracy: {model_accuracies['Random Forest'] * 100:.2f}%")
    st.write(f"Decision Tree Accuracy: {model_accuracies['Decision Tree'] * 100:.2f}%")
    st.write([lr_pred[0])
    st.write([rf_pred[0])
    st.write([dt_pred[0])    
    # Aggregate predictions from all models
    churn_votes = sum([lr_pred[0], rf_pred[0], dt_pred[0]])
    final_prediction = "Churn" if churn_votes > 1 else "Not Churn"
    
    st.subheader("Final Aggregated Prediction")
    st.write(f'Final Prediction: Custommer will {final_prediction}')

# To deploy: streamlit run app.py
