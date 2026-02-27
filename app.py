import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "model.pkl"

# Carregar modelo
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic – Previsão de Sobrevivência")
st.markdown("Simulação usando Machine Learning (Random Forest)")

st.sidebar.header("Dados do Passageiro")

pclass = st.sidebar.selectbox("Classe", [1, 2, 3])
sex = st.sidebar.selectbox("Sexo", ["male", "female"])
age = st.sidebar.slider("Idade", 0, 80, 30)
sibsp = st.sidebar.slider("Irmãos/Cônjuges a bordo", 0, 5, 0)
parch = st.sidebar.slider("Pais/Filhos a bordo", 0, 5, 0)
fare = st.sidebar.slider("Tarifa paga", 0.0, 500.0, 50.0)

sex = 0 if sex == "male" else 1

input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare
}])

st.subheader("📋 Dados do passageiro")
st.dataframe(input_df)

if st.button("🔮 Prever"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("🟢 Sobreviveu")
    else:
        st.error("🔴 Não sobreviveu")
