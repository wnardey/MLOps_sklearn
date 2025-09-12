"""
TRAIN
- Descarga el artifact 'breast_cancer-processed:latest'
- Entrena LogisticRegression en train, evalúa en val y test
- Loguea métricas a W&B y publica un artifact de modelo 'breast_cancer-logreg'

Uso:
  python train.py --IdExecution 003 --C 1.0 --max_iter 500

Requiere:
  - WANDB_API_KEY en el entorno (en Actions como Secret)
Opcional:
  - WANDB_PROJECT (default: MLOps-sklearn)
"""

import os
import argparse
import json
import numpy as np
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from joblib import dump

PROC_ART_NAME = "breast_cancer-processed:latest"
MODEL_ART_NAME = "breast_cancer-logreg"

def read_npz(dirpath: str, split: str):
    fp = os.path.join(dirpath, f"{split}.npz")
    data = np.load(fp)
    return data["X"], data["y"]

def evaluate_binary(y_true, y_pred, y_proba=None):
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "f1":  f1_score(y_true, y_pred),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            pass
    return metrics

def train_and_log(IdExecution: str, C: float, max_iter: int, seed: int):
    project = os.getenv("WANDB_PROJECT", "MLOps-sklearn")
    run_name = f"Train LogReg ExecId-{IdExecution}" if IdExecution else "Train LogReg"

    with wandb.init(project=project, name=run_name, job_type="train") as run:
        # Config para traza
        wandb.config.update({"model": "logreg", "C": C, "max_iter": max_iter, "seed": seed})

        # 1) Consumir artifact PROCESADO
        proc_art = run.use_artifact(PROC_ART_NAME)
        proc_dir = proc_art.download()

        # 2) Cargar splits (ya escalados)
        X_tr, y_tr = read_npz(proc_dir, "train")
        X_val, y_val = read_npz(proc_dir, "val")
        X_te,  y_te  = read_npz(proc_dir, "test")

        # 3) Modelo (sin scaler, ya preprocesado)
        clf = LogisticRegression(
            solver="liblinear",  
            C=C,
            max_iter=max_iter,
            random_state=seed
        )
        clf.fit(X_tr, y_tr)

        # 4) Evaluación en VAL
        y_val_pred = clf.predict(X_val)
        y_val_proba = None
        if hasattr(clf, "predict_proba"):
            y_val_proba = clf.predict_proba(X_val)[:, 1]
        val_metrics = evaluate_binary(y_val, y_val_pred, y_val_proba)

        # 5) Evaluación en TEST (opcional pero útil)
        y_te_pred = clf.predict(X_te)
        y_te_proba = None
        if hasattr(clf, "predict_proba"):
            y_te_proba = clf.predict_proba(X_te)[:, 1]
        test_metrics = evaluate_binary(y_te, y_te_pred, y_te_proba)

        # 6) Log de métricas a W&B
        wandb.log({
            "val/acc": val_metrics.get("acc"),
            "val/f1":  val_metrics.get("f1"),
            "val/roc_auc": val_metrics.get("roc_auc", None),
            "test/acc": test_metrics.get("acc"),
            "test/f1":  test_metrics.get("f1"),
            "test/roc_auc": test_metrics.get("roc_auc", None),
        })

        # 7) Guardar modelo localmente
        os.makedirs("artifacts_model", exist_ok=True)
        model_path = os.path.join("artifacts_model", "model.joblib")
        dump(clf, model_path)

        # 8) Publicar artifact de modelo
        meta = {
            "source_artifact": PROC_ART_NAME,
            "model": "logistic_regression",
            "hyperparams": {"C": C, "max_iter": max_iter, "solver": "liblinear", "random_state": seed},
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        model_art = wandb.Artifact(
            name=MODEL_ART_NAME,
            type="model",
            description="LogReg entrenado sobre breast_cancer procesado",
            metadata=meta
        )
        model_art.add_file(model_path, name="model.joblib")

        # guardar también un JSON con métricas/params
        with open(os.path.join("artifacts_model", "report.json"), "w") as f:
            json.dump(meta, f, indent=2)
        model_art.add_file(os.path.join("artifacts_model", "report.json"), name="report.json")

        run.log_artifact(model_art, aliases=["latest"])
        print(f"[W&B] Modelo publicado como artifact '{MODEL_ART_NAME}:latest'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--IdExecution", type=str, default="", help="ID humano de ejecución")
    parser.add_argument("--C", type=float, default=1.0, help="Regularización de LogisticRegression (C)")
    parser.add_argument("--max_iter", type=int, default=500, help="Iteraciones máximas del solver")
    parser.add_argument("--seed", type=int, default=42, help="Semilla")
    args = parser.parse_args()

    if args.IdExecution:
        print(f"IdExecution: {args.IdExecution}")

    train_and_log(
        IdExecution=args.IdExecution,
        C=args.C,
        max_iter=args.max_iter,
        seed=args.seed
    )
