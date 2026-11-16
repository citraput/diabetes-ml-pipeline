"""Tahap 2: split data, tuning Random Forest, catat ke MLflow, simpan model."""
import os
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

# Baca MLFLOW_TRACKING_URI / USERNAME / PASSWORD dari file .env (kalau ada).
# Kalau .env belum ada, MLflow menyimpan catatan di laptop (mlflow.db / mlruns)
load_dotenv()

TARGET = "Outcome"


def train(p: dict) -> None:
    df = pd.read_csv(p["data"])
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # stratify=y menjaga proporsi diabetes/tidak sama di data latih & uji
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=p["test_size"], random_state=p["random_state"], stratify=y
    )

    # Imputer di DALAM pipeline: median dihitung dari data latih saja (tidak bocor)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(random_state=p["random_state"])),
    ])
    # Nama parameter di Pipeline harus diawali "rf__"
    param_grid = {f"rf__{k}": v for k, v in p["param_grid"].items()}

    mlflow.set_experiment("diabetes-pipeline")
    with mlflow.start_run(run_name="train"):
        grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        metrics = {
            "cv_best_f1": grid.best_score_,
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("test_size", p["test_size"])
        mlflow.log_metrics(metrics)
        mlflow.log_text(str(confusion_matrix(y_test, y_pred)), "confusion_matrix.txt")
        mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")
        mlflow.sklearn.log_model(
            best_model,
            name="model",
            signature=infer_signature(X_train, best_model.predict(X_train)),
            input_example=X_train.head(),
            # MLflow versi baru menyimpan model sklearn dengan format "skops" yang
            # menolak tipe objek yang tidak dikenal; tipe ini aman karena model buatan sendiri
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )

    # Simpan model (dilacak DVC) dan data uji (dipakai evaluate.py)
    os.makedirs(os.path.dirname(p["model"]), exist_ok=True)
    with open(p["model"], "wb") as f:
        pickle.dump(best_model, f)
    X_test.assign(**{TARGET: y_test}).to_csv(p["test_data"], index=False)

    print(f"Parameter terbaik: {grid.best_params_}")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")
    print(f"Model disimpan ke {p['model']}, data uji ke {p['test_data']}")


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]
    train(params)