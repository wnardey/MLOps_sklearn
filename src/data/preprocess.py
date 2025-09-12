"""
PREPROCESS
- Descarga el artifact 'breast_cancer-raw:latest'
- Aplica StandardScaler (fit en train, transform en val/test)
- Sube un nuevo artifact 'breast_cancer-processed' con splits escalados y params del scaler.

Ejemplo:
  python src/data/preprocess.py --IdExecution 002

Requiere:
  - WANDB_API_KEY (Secret en GitHub Actions)
Opcional:
  - WANDB_PROJECT (default: MLOps-sklearn)
"""

import os
import argparse
import json
import numpy as np
import wandb
from sklearn.preprocessing import StandardScaler

RAW_ART_NAME = "breast_cancer-raw:latest"
PROC_ART_NAME = "breast_cancer-processed"

def read_npz(dirpath: str, split: str):
    """Carga X,y desde <dirpath>/<split>.npz"""
    fp = os.path.join(dirpath, f"{split}.npz")
    data = np.load(fp)
    return data["X"], data["y"]

def preprocess_and_log(IdExecution: str = "", scale: bool = True, with_mean: bool = True, with_std: bool = True):
    project = os.getenv("WANDB_PROJECT", "MLOps-sklearn")
    entity  = os.getenv("WANDB_ENTITY")  # opcional
    run_name = f"Preprocess ExecId-{IdExecution}" if IdExecution else "Preprocess"

    steps = {"scale": bool(scale), "scaler": "standard", "with_mean": bool(with_mean), "with_std": bool(with_std)}

    with wandb.init(project=project, entity=entity, name=run_name, job_type="preprocess-data") as run:
        # 1) Consumir artifact RAW
        raw_art = run.use_artifact(RAW_ART_NAME)
        raw_dir = raw_art.download()

        # 2) Leer splits
        X_tr, y_tr = read_npz(raw_dir, "train")
        X_val, y_val = read_npz(raw_dir, "val")
        X_te, y_te = read_npz(raw_dir, "test")

        # 3) Preprocesar (escalado estándar)
        if scale:
            scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
            X_tr_p = scaler.fit_transform(X_tr)
            X_val_p = scaler.transform(X_val)
            X_te_p  = scaler.transform(X_te)
        else:
            scaler = None
            X_tr_p, X_val_p, X_te_p = X_tr, X_val, X_te

        # 4) Armar metadata
        all_y = np.concatenate([y_tr, y_val, y_te])
        vals, counts = np.unique(all_y, return_counts=True)
        class_dist = {int(v): int(c) for v, c in zip(vals, counts)}

        meta = {
            "source_artifact": RAW_ART_NAME,
            "steps": steps,
            "n_features": int(X_tr.shape[1]),
            "splits_shapes": {
                "train": [int(X_tr_p.shape[0]), int(X_tr_p.shape[1])],
                "val":   [int(X_val_p.shape[0]), int(X_val_p.shape[1])],
                "test":  [int(X_te_p.shape[0]),  int(X_te_p.shape[1])]
            },
            "class_distribution_total": class_dist
        }

        # 5) Crear artifact PROCESADO y anexar archivos
        proc_art = wandb.Artifact(
            name=PROC_ART_NAME,
            type="dataset",
            description="breast_cancer procesado (StandardScaler en train)",
            metadata=meta
        )

        # Guardar splits procesados
        for name, X, y in [("train", X_tr_p, y_tr), ("val", X_val_p, y_val), ("test", X_te_p, y_te)]:
            with proc_art.new_file(f"{name}.npz", mode="wb") as f:
                np.savez(f, X=X, y=y)

        # Guardar parámetros del scaler (si aplica)
        if scaler is not None:
            with proc_art.new_file("scaler_params.npz", mode="wb") as f:
                np.savez(f,
                         mean=(scaler.mean_ if hasattr(scaler, "mean_") else np.array([])),
                         scale=(scaler.scale_ if hasattr(scaler, "scale_") else np.array([])),
                         var=(scaler.var_ if hasattr(scaler, "var_") else np.array([])))
            # y un JSON con la configuración
            with proc_art.new_file("preprocess_config.json", mode="w") as f:
                json.dump(steps, f, indent=2)

        # 6) Subir artifact
        run.log_artifact(proc_art)
        print(f"[W&B] Artifact '{PROC_ART_NAME}' registrado (origen: {RAW_ART_NAME}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--IdExecution", type=str, default="", help="ID humano de ejecución")
    parser.add_argument("--no_scale", action="store_true", help="Desactiva el escalado")
    parser.add_argument("--with_mean", action="store_true", help="(opcional) centrar media (default False si usas --no_scale no aplica)")
    parser.add_argument("--with_std", action="store_true", help="(opcional) escalar varianza (default False si usas --no_scale no aplica)")
    args = parser.parse_args()

    scale = not args.no_scale
    # por defecto queremos with_mean/with_std True si escalamos; si el usuario pasa flags, se respetan:
    with_mean = True if scale else False
    with_std  = True if scale else False
    if args.with_mean: with_mean = True
    if args.with_std:  with_std  = True

    if args.IdExecution:
        print(f"IdExecution: {args.IdExecution}")

    preprocess_and_log(
        IdExecution=args.IdExecution,
        scale=scale,
        with_mean=with_mean,
        with_std=with_std
    )
