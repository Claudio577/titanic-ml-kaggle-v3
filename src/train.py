import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from preprocess import load_data
from utils import OUTPUTS

def main():
    df, _ = load_data()

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    df = df[features + ["Survived"]]

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    X = df[features]
    y = df["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    OUTPUTS.mkdir(exist_ok=True)
    joblib.dump(model, OUTPUTS / "model.pkl")

    print(f"Modelo treinado com accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
