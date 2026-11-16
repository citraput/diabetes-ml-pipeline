"""Tahap 3: evaluasi model HANYA pada data uji yang tidak ikut dilatih."""
import pickle

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, recall_score

load_dotenv()

TARGET = "Outcome"


def evaluate(data_path: str, model_path: str) -> None:
    df = pd.read_csv(data_path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X)

    metrics = {
        "test_accuracy": accuracy_score(y, y_pred),
        "test_recall": recall_score(y, y_pred),
        "test_f1": f1_score(y, y_pred),
    }

    mlflow.set_experiment("diabetes-pipeline")
    with mlflow.start_run(run_name="evaluate"):
        mlflow.set_tag("stage", "evaluate")
        mlflow.log_metrics(metrics)

    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["evaluate"]
    evaluate(params["data"], params["model"])