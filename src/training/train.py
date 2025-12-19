import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

mlflow.set_tracking_uri("http://localhost:5000")
print("MLFLOW_TRACKING_URI =", mlflow.get_tracking_uri())

def main(n_estimators: int, max_depth: int):
    if mlflow.active_run() is not None:
        mlflow.end_run()
    # 1. Load data (simple, deterministic dataset for now)
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 3. Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # 4. MLflow tracking (explicit experiment)
    mlflow.set_experiment("iris-classifier")


    # 5. Save & log model artifact
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, artifact_path="model")



    print(f"Training complete. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)

    args = parser.parse_args()

    main(args.n_estimators, args.max_depth)
