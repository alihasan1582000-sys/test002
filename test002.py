# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample data for demonstration
data = pd.DataFrame({
    "residence": [0, 1, 1, 0, 1],
    "total_childern_died": [2, 0, 1, 3, 1],
    "motheredu": ["None", "Primary", "Secondary", "None", "Higher"],
    "drinking_water": ["Improved", "Unimproved", "Improved", "Improved", "Unimproved"],
    "fuel_type": ["Clean", "Unclean", "Clean", "Unclean", "Unclean"],
    "wealth_index": ["Low", "Medium", "High", "Medium", "Low"]
})

data["treatment_urban"] = data["residence"]
X = data.drop(columns=["residence", "treatment_urban"])
y = data["treatment_urban"]

# Preprocessing pipeline
cat_cols = ["motheredu", "drinking_water", "fuel_type", "wealth_index"]
num_cols = ["total_childern_died"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), cat_cols)
], remainder='passthrough')

model = Pipeline(steps=[
    ("pre", preprocessor),
    ("logreg", LogisticRegression())
])

model.fit(X, y)

# Streamlit UI
st.title("Urban Residence Prediction App (Propensity Estimation)")
st.write("This app predicts if a mother lives in an urban area (Treatment = 1) using her profile.")

motheredu = st.selectbox("Mother's Education", data["motheredu"].unique())
drinking_water = st.selectbox("Drinking Water Source", data["drinking_water"].unique())
fuel_type = st.selectbox("Fuel Type", data["fuel_type"].unique())
wealth_index = st.selectbox("Wealth Index", data["wealth_index"].unique())
total_childern_died = st.slider("Total Children Died", 0, 10, 1)

user_input = pd.DataFrame([{
    "motheredu": motheredu,
    "drinking_water": drinking_water,
    "fuel_type": fuel_type,
    "wealth_index": wealth_index,
    "total_childern_died": total_childern_died
}])

# Prediction
pred_prob = model.predict_proba(user_input)[0][1]
pred_class = model.predict(user_input)[0]

st.markdown("### Results")
st.write(f"**Predicted Urban Probability:** {pred_prob:.2f}")
st.write(f"**Predicted Class (0 = Rural, 1 = Urban):** {pred_class}")
