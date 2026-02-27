import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Titanic ML",
    layout="wide",
    page_icon="🚢"
)

st.markdown(
    """
    <style>
        .big-title {
            font-size:40px;
            font-weight:700;
        }
        .subtitle {
            font-size:18px;
            color:#666;
        }
        .result-box {
            padding:20px;
            border-radius:12px;
            text-align:center;
            font-size:26px;
            font-weight:600;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- HEADER ----------------
st.markdown('<div class="big-title">🚢 Titanic – Previsão de Sobrevivência</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Aplicação de Machine Learning com Streamlit (Random Forest)</div>', unsafe_allow_html=True)
st.divider()

# ---------------- DATA / MODEL ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"

@st.cache_resource
def train_model():
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

with st.spinner("🔄 Treinando modelo (primeira execução)..."):
    model = train_model()

st.success("✅ Modelo carregado com sucesso")

# ---------------- LAYOUT ----------------
col_inputs, col_result = st.columns([1.2, 1])

# ---------- INPUTS ----------
with col_inputs:
    st.subheader("📋 Dados do Passageiro")

    pclass = st.selectbox("Classe", [1, 2, 3])
    sex = st.selectbox("Sexo", ["male", "female"])
    age = st.slider("Idade", 0, 80, 30)
    sibsp = st.slider("Irmãos / Cônjuges", 0, 5, 0)
    parch = st.slider("Pais / Filhos", 0, 5, 0)
    fare = st.slider("Tarifa paga", 0.0, 500.0, 50.0)

    st.caption("Ajuste os dados e clique em **Prever**")

sex_num = 0 if sex == "male" else 1

input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex_num,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare
}])

# ---------- RESULT ----------
with col_result:
    st.subheader("🔮 Resultado da Previsão")

    if st.button("Prever sobrevivência", use_container_width=True):
        pred = model.predict(input_df)[0]

        if pred == 1:
            st.markdown(
                '<div class="result-box" style="background-color:#d1f5d3;">🟢 Sobreviveu</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box" style="background-color:#f8d7da;">🔴 Não sobreviveu</div>',
                unsafe_allow_html=True
            )

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Projeto educacional baseado no dataset Titanic (Kaggle). "
    "Objetivo: demonstrar Machine Learning em produção com Streamlit."
)
