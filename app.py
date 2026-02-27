import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Titanic ML", layout="centered")
st.title("🚢 Titanic – Previsão de Sobrevivência")

st.info("Inicializando aplicação...")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"

@st.cache_resource
def train_model():
    st.write("🔄 Treinando modelo (primeira execução)...")

    df = pd.read_csv(DATA_PATH)

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    df = df[features + ["Survived"]].copy()

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    X = df[features]
    y = df["Survived"]

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )
    model.fit(X, y)

    return model

model = train_model()

st.success("✅ Modelo carregado com sucesso!")

# ---------- UI ----------
st.subheader("Dados do Passageiro")

pclass = st.selectbox("Classe", [1, 2, 3])
sex = st.selectbox("Sexo", ["male", "female"])
age = st.slider("Idade", 0, 80, 30)
sibsp = st.slider("Irmãos/Cônjuges", 0, 5, 0)
parch = st.slider("Pais/Filhos", 0, 5, 0)
fare = st.slider("Tarifa paga", 0.0, 500.0, 50.0)

sex_num = 0 if sex == "male" else 1

input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex_num,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare
}])

if st.button("🔮 Prever"):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("🟢 Sobreviveu")
    else:
        st.error("🔴 Não sobreviveu")
