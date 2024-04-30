import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.title("TAAPMAAN.ai")

# Load the dataset
df = pd.read_csv('train_x.csv')

# Drop rows with missing values in the 'Temperature (C)' column
df.drop([2698, 19868, 21449, 46441, 47873, 57666], axis=0, inplace=True)

# Fill missing values in Precip Type columns with "No"
precip_columns = [f"Hour {i}: Precip Type" for i in range(6)]
df[precip_columns] = df[precip_columns].fillna("No")

# Encode categorical columns using LabelEncoder
le = LabelEncoder()
summary_columns = [f"Hour {i}: Summary" for i in range(6)]
precip_columns = [f"Hour {i}: Precip Type" for i in range(6)]
df[summary_columns + precip_columns] = df[summary_columns + precip_columns].apply(le.fit_transform)

# Split the data into features (X) and target variable (y)
X = df.iloc[:, :42]
y = df.iloc[:, 42]

# Model training
model = RandomForestRegressor(warm_start=True, min_samples_split=0.00001, max_features="sqrt",
                               oob_score=True, n_jobs=-1, n_estimators=700)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.sidebar.header('Input Parameters')

# Display model parameters
st.sidebar.subheader('Model Parameters')
st.sidebar.write(f'Number of Estimators: {model.n_estimators}')
st.sidebar.write(f'Min Samples Split: {model.min_samples_split}')
st.sidebar.write(f'Max Features: {model.max_features}')

# Display dataset info
st.sidebar.subheader('Dataset Info')
st.sidebar.write(f'Number of rows: {df.shape[0]}')
st.sidebar.write(f'Number of columns: {df.shape[1]}')

# Predict temperature
st.subheader('Temperature Prediction')
# User inputs
inputs = {}
for i in range(6):
    st.sidebar.subheader(f'Hour {i} Inputs')
    col_name_summary = f'Hour {i}: Summary'
    col_name_precip = f'Hour {i}: Precip Type'
    inputs[col_name_summary] = st.sidebar.number_input(f'{col_name_summary}', key=f'summary_{i}', value=0.0)
    inputs[col_name_precip] = st.sidebar.number_input(f'{col_name_precip}', key=f'precip_{i}', value=0.0)

# Make prediction
inputs_df = pd.DataFrame([inputs])
prediction = model.predict(inputs_df)

st.write(f'Predicted Temperature (C): {prediction[0]}')

# Display correlation plot
st.subheader('Correlation Plot')
corr = df.corr()
plt.figure(figsize=(12, 8))
sn.heatmap(corr, annot=True, cmap='coolwarm')
st.pyplot()

# Display actual vs predicted graph
st.subheader('Actual vs Predicted Temperature (C) - First 50 Values (Training Set)')
y_pred_test = pd.DataFrame(model.predict(X_test))
results_train_df = pd.concat([y_test.reset_index(drop=True), y_pred_test], axis=1)
plt.figure(figsize=(10, 6))
plt.plot(results_train_df.index[:50], results_train_df[0].iloc[:50], label="Actual", marker='o')
plt.plot(results_train_df.index[:50], results_train_df[0].iloc[:50], label="Predicted", marker='x')
plt.xlabel("Index")
plt.ylabel("Temperature (C)")
plt.title("Actual vs Predicted Temperature (C) - First 50 Values (Training Set)")
plt.legend()
st.pyplot()
