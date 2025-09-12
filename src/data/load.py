"""
LOAD
Carga el dataset breast_cancer de sklearn, lo divide en train/val/test
y sube los tres splits a Weights & Biases como un Artifact con archivos .npz.

Ejemplo:
  python src/data/load.py --IdExecution 001

Requiere:
  - WANDB_API_KEY en el entorno (en GitHub Actions irá como Secret).
Opcional:
  - WANDB_PROJECT (default: MLOps-sklearn)
  - WANDB_ENTITY
"""

import os
import argparse
import numpy as np
import wandb
from typing import Tuple, List
from sklearn import datasets
from sklearn.model_selection import train_test_split

def load_breast_cancer(
    train_size: float = 0.8,
    val_size: float = 0.1,
    seed: int = 42
) -> Tuple[List[tuple], dict]:
    """
    Devuelve [(name, X, y), ...] para train/val/test y un dict de metadata.
    """
    X, y = datasets.load_breast_cancer(return_X_y=True)
    task = "cls"

    # 1) train vs tmp
    strat = y if task == "cls" else None
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, train_size=train_size, random_state=seed, stratify=strat
    )

    # 2) val vs test dentro de tmp
    tmp_size = 1.0 - train_size
    val_rel = val_size / max(tmp_size, 1e-12)   # proporción de val dentro del tmp
    strat_tmp = y_tmp if task == "cls" else None
    # usamos test_size = 1 - val_rel para que la "parte train" sea val_rel (nuestro 'val')
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=1.0 - val_rel, random_state=seed, stratify=strat_tmp
    )

    splits = [("train", X_tr, y_tr), ("val", X_val, y_val), ("test", X_te, y_te)]

    meta = {
        "dataset": "breast_cancer",
        "task": task,
        "n_samples_total": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "train_size": train_size,
        "val_size": val_size,
        "seed": seed,
        "splits_shapes": {n: [int(a.shape[0]), int(a.shape[1])] for n, a, _ in splits},
    }
    vals, counts = np.unique(y, return_counts=True)
    meta["class_distribution_total"] = {int(v): int(c) for v, c in zip(vals, counts)}
    return splits, meta

def load_and_log(IdExecution: str = "", train_size: float = 0.8, val_size: float = 0.1, seed: int = 42):
    project = os.getenv("WANDB_PROJECT", "MLOps-sklearn")

    run_name = f"Load breast_cancer ExecId-{IdExecution}" if IdExecution else "Load breast_cancer"
    with wandb.init(project=project, name=run_name, job_type="load-data") as run:
        splits, meta = load_breast_cancer(train_size=train_size, val_size=val_size, seed=seed)

        artifact = wandb.Artifact(
            name="breast_cancer-raw",
            type="dataset",
            description="raw sklearn breast_cancer split into train/val/test",
            metadata=meta
        )

        # Guardamos cada split como .npz con arrays nombrados
        for name, X, y in splits:
            fname = f"{name}.npz"
            with artifact.new_file(fname, mode="wb") as f:
                np.savez(f, X=X, y=y)

        run.log_artifact(artifact)
        print(f"[W&B] Artifact 'breast_cancer-raw' registrado en proyecto '{project}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--IdExecution", type=str, default="", help="ID humano de ejecución")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.IdExecution:
        print(f"IdExecution: {args.IdExecution}")

    load_and_log(
        IdExecution=args.IdExecution,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )
