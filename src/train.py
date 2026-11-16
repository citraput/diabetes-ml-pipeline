"""Tahap 2: bandingkan beberapa model, catat ke MLflow, simpan model terbaik."""
import os
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.models import infer_signature
from skops.io import dumps, get_untrusted_types
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Baca MLFLOW_TRACKING_URI / USERNAME / PASSWORD dari .env (kalau ada).
# Kalau .env belum ada, MLflow menyimpan catatan di laptop (mlflow.db / mlruns)
load_dotenv()

TARGET = "Outcome"


def build_pipeline(model_name: str, random_state: int) -> Pipeline:
    """Setiap model diawali imputer (median dari data latih saja)."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "logistic_regression":
        # Logistic Regression butuh fitur berskala sama; model pohon tidak
        steps.append(("scaler", StandardScaler()))
        steps.append(("clf", LogisticRegression(max_iter=1000, random_state=random_state)))
    elif model_name == "random_forest":
        steps.append(("clf", RandomForestClassifier(random_state=random_state)))
    elif model_name == "gradient_boosting":
        steps.append(("clf", HistGradientBoostingClassifier(random_state=random_state)))
    else:
        raise ValueError(f"Model tidak dikenal: {model_name}")
    return Pipeline(steps)


def evaluate_on_test(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def log_sklearn_model(model, X_train) -> None:
    # MLflow versi baru menyimpan model sklearn dengan format "skops" yang hanya
    # menerima tipe objek yang "dipercaya". Model ini buatan sendiri, jadi tipenya aman.
    trusted = get_untrusted_types(data=dumps(model))
    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=infer_signature(X_train, model.predict(X_train)),
        input_example=X_train.head(),
        skops_trusted_types=trusted,
    )


def train(p: dict) -> None:
    df = pd.read_csv(p["data"])
    X = df.drop(columns=[TARGET]).astype(float)
    y = df[TARGET]

    # stratify=y menjaga proporsi diabetes/tidak sama di data latih & uji
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=p["test_size"], random_state=p["random_state"], stratify=y
    )

    mlflow.set_experiment("diabetes-pipeline")
    results = {}

    # Run induk = satu sesi perbandingan; tiap model = run anak (nested)
    with mlflow.start_run(run_name="train"):
        mlflow.log_params({"test_size": p["test_size"], "cv": p["cv"], "scoring": p["scoring"]})

        for model_name, grid_values in p["models"].items():
            with mlflow.start_run(run_name=model_name, nested=True):
                pipe = build_pipeline(model_name, p["random_state"])
                # Nama parameter di Pipeline diawali "clf__"
                param_grid = {f"clf__{k}": v for k, v in grid_values.items()}

                grid = GridSearchCV(pipe, param_grid, cv=p["cv"],
                                    scoring=p["scoring"], n_jobs=-1)
                grid.fit(X_train, y_train)
                best = grid.best_estimator_
                metrics = evaluate_on_test(best, X_test, y_test)
                metrics[f"cv_best_{p['scoring']}"] = grid.best_score_

                mlflow.set_tag("model_type", model_name)
                mlflow.log_params(grid.best_params_)
                mlflow.log_metrics(metrics)
                y_pred = best.predict(X_test)
                mlflow.log_text(str(confusion_matrix(y_test, y_pred)), "confusion_matrix.txt")
                mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")
                log_sklearn_model(best, X_train)

                results[model_name] = {"model": best, "cv_score": grid.best_score_,
                                       "metrics": metrics}
                print(f"\n[{model_name}] best params: {grid.best_params_}")
                for k, v in metrics.items():
                    print(f"  {k}: {v:.3f}")

        # Pilih model terbaik berdasarkan skor CV (data latih),
        # BUKAN skor data uji, supaya data uji tetap "belum pernah dilihat"
        best_name = max(results, key=lambda n: results[n]["cv_score"])
        best_model = results[best_name]["model"]
        mlflow.set_tag("best_model", best_name)
        mlflow.log_metrics({f"best_{k}": v for k, v in results[best_name]["metrics"].items()})

    # Simpan model terbaik (dilacak DVC) dan data uji (dipakai evaluate.py)
    os.makedirs(os.path.dirname(p["model"]), exist_ok=True)
    with open(p["model"], "wb") as f:
        pickle.dump(best_model, f)
    X_test.assign(**{TARGET: y_test}).to_csv(p["test_data"], index=False)

    print(f"\nModel terbaik (berdasarkan CV {p['scoring']}): {best_name}")
    print(f"Model disimpan ke {p['model']}, data uji ke {p['test_data']}")


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]
    train(params)