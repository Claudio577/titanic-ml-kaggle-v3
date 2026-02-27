import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
MODEL_PATH = BASE_DIR / "outputs" / "model.pkl"

@st.cache_resource
def load_or_train_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    # Treinar modelo se não existir
    df = pd.read_csv(DATA_PATH)

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    df = df[features + ["Survived"]]

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    X = df[features]
    y = df["Survived"]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model


model = load_or_train_model()
