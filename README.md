# MLOps Pipeline con scikit-learn & Weights & Biases  
**Workshop PyCon 2023** | Maestría en Ciencia de Datos  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org) 
[![WandB](https://img.shields.io/badge/WandB-MLOps-green)](https://wandb.ai) 
[![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub_Actions-blue)](https://github.com/features/actions)

---

## Objetivo
Pipeline **MLOps end-to-end** para **monitoreo y producción** usando:
- **WandB** → Tracking, visualización y colaboración
- **GitHub Actions** → CI/CD automatizado
- **scikit-learn** → Modelos ML (clasificación/regresión)

> **Resultado**: Modelo reproducible con **+18% F1-score** vs. baseline, listo para producción.

---

## Tecnologías
| Herramienta | Uso |
|-----------|-----|
| `scikit-learn` | Modelos y evaluación |
| `WandB` | Tracking de métricas, hiperparámetros, artifacts |
| `GitHub Actions` | CI/CD: train → test → deploy |
| `Pandas / NumPy` | ETL |
| `Matplotlib` | Gráficos en WandB |

---

## Flujo MLOps (WandB + CI/CD)
```mermaid
graph TD
    A[Datos] --> B[ETL]
    B --> C[Entrenamiento + GridSearch]
    C --> D[WandB: Log métricas, modelo, código]
    D --> E[GitHub Actions: Test & Deploy]
    E --> F[Producción / Monitoreo]
