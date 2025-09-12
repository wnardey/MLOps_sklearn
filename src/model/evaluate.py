"""
EVALUATE
- Descarga artifacts:
    * Dataset procesado: 'breast_cancer-processed:latest'
    * Modelo entrenado:  'breast_cancer-logreg:latest'
- Evalúa en TEST (accuracy, f1, roc_auc si hay probas)
- Loguea métricas y matriz de confusión a W&B (job_type="evaluate")

Uso:
  python evaluate.py --IdExecution 004

Requiere:
  - WANDB_API_KEY (Secret en GitHub Actions)
Opcional:
  - WANDB_PROJECT (default: MLOps-sklearn)
"""

import os
import argparse
import numpy as np
import wandb
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

PROC_ART_NAME  = "breast_cancer-processed:latest"
MODEL_ART_NAME = "breast_cancer-logreg:latest"

def read_npz(dirpath: str, split: str):
    fp = os.path.join(dirpath, f"{split}.npz")
    data = np.load(fp)
    return data["X"], data["y"]

def evaluate_binary(y_true, y_pred, y_proba=None):
    metrics = {
        "test/acc": accuracy_score(y_true, y_pred),
        "test/f1":  f1_score(y_true, y_pred),
    }
    if y_proba is not None:
        try:
            metrics["test/roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            pass
    return metrics

def main(IdExecution: str):
    project = os.getenv("WANDB_PROJECT", "MLOps-sklearn")
    run_name = f"Evaluate ExecId-{IdExecution}" if IdExecution else "Evaluate"

    with wandb.init(project=project, name=run_name, job_type="evaluate") as run:
        # 1) Dataset procesado
        proc_art = run.use_artifact(PROC_ART_NAME)
        proc_dir = proc_art.download()
        X_te, y_te = read_npz(proc_dir, "test")

        # 2) Modelo
        model_art = run.use_artifact(MODEL_ART_NAME)
        model_dir = model_art.download()
        clf = load(os.path.join(model_dir, "model.joblib"))

        # 3) Predicciones
        y_pred = clf.predict(X_te)
        y_proba = None
        if hasattr(clf, "predict_proba"):
            # tomar prob. de la clase positiva (1)
            pos_idx = np.where(clf.classes_ == 1)[0][0] if hasattr(clf, "classes_") else 1
            y_proba = clf.predict_proba(X_te)[:, pos_idx]

        # 4) Métricas
        metrics = evaluate_binary(y_te, y_pred, y_proba)
        wandb.log(metrics)

        # 5) Matriz de confusión
        class_names = [str(int(c)) for c in np.unique(y_te)]
        cm = confusion_matrix(y_te, y_pred)
        wandb.log({
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_te, preds=y_pred, class_names=class_names
            )
        })

        # imprime resumen a stdout
        print("== Test Metrics ==")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("Confusion matrix:\n", cm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--IdExecution", type=str, default="", help="ID humano de ejecución")
    args = parser.parse_args()
    if args.IdExecution:
        print(f"IdExecution: {args.IdExecution}")
    main(args.IdExecution)
