import streamlit as st
import joblib
import numpy as np
import pandas as pd

# === LOAD MODEL AND RELEVANT FILES ===
model = joblib.load("models/final_random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
misto_columns = joblib.load("models/misto_columns.pkl")
lokalita_columns = joblib.load("models/lokalita_columns.pkl")

st.title("Predikce ceny bytu za m²")
st.write("Zadejte parametry bytu a aplikace odhadne cenu za m².")

# === INPUTS FROM USER ===
plocha = st.number_input("Obytná plocha (m²)", min_value=20.0, max_value=300.0, step=1.0)
mistnosti = st.selectbox("Počet místností", [1, 2, 3, 4, 5])
podlazi = st.number_input("Podlaží", min_value=-1, max_value=20, step=1)
rok_prodeje = st.number_input("Rok prodeje", min_value=2000, max_value=2025, step=1, value=2022)

misto = st.selectbox("Místo", [col.replace("Místo_", "") for col in misto_columns])
lokalita = st.selectbox("Lokalita", [col.replace("Lokalita_", "") for col in lokalita_columns])

# === PREPARE INPUT FOR THE MODEL ===
input_data = {
    "Obytná plocha": plocha,
    "Počet místností": mistnosti,
    "Podlaží": podlazi,
    "Datum prodeje": rok_prodeje
}
X = pd.DataFrame([input_data])

# Add one-hot encoded place and locality
for col in misto_columns:
    X[col] = 1 if col == f"Místo_{misto}" else 0
for col in lokalita_columns:
    X[col] = 1 if col == f"Lokalita_{lokalita}" else 0

# Manually define expected columns
final_columns = ["Obytná plocha", "Počet místností", "Podlaží", "Datum prodeje"] + misto_columns + lokalita_columns

# Fill missing columns with zeros
for col in final_columns:
    if col not in X.columns:
        X[col] = 0

# Ensure column order
X = X[final_columns]

# === SCALE NUMERICAL COLUMNS ===
num_cols = ["Obytná plocha", "Počet místností", "Podlaží"]
X[num_cols] = scaler.transform(X[num_cols])

# === PREDICTION ===
if st.button("Spočítej odhad ceny za m²"):
    prediction = model.predict(X)[0]
    st.success(f"Odhadovaná cena za m²: {prediction:,.2f} Kč")
