# MLflow-Based Logistics Delay & Shipment Risk Prediction System

AIN-3009 MLOps term project. End-to-end ML lifecycle on a synthetic logistics
dataset using MLflow for tracking, tuning, registry, deployment, and monitoring.

## Assignment Mapping

| Assignment objective | Implementation |
|---|---|
| Experiment tracking | `src/train.py` logs params, metrics, and artifacts per model run |
| Model training & tuning | Baselines in `src/train.py`; Optuna tuning in `src/tune.py` (nested MLflow runs) |
| Model deployment | FastAPI in `src/serve_api.py` (loads `LogisticsDelayRiskModel@production`) |
| Performance monitoring | `src/monitor.py` — five batches under `logistics-delay-monitoring` |
| Model Registry | `src/register_model.py` — `LogisticsDelayRiskModel` with aliases `production` / `staging` |

## Project Structure

```text
src/
├── config.py              # Configuration, paths, schema seeds
├── data_generation.py     # Synthetic logistics data generation
├── preprocessing.py       # Data preprocessing and train/test split
├── train.py               # Baseline model training + MLflow logging
├── tune.py                # Optuna hyperparameter tuning
├── register_model.py      # Model registration and alias management
├── serve_api.py           # FastAPI inference service
├── monitor.py             # Monitoring and drift detection
└── utils.py               # Shared utilities

data/
├── raw/                   # Training datasets
└── monitoring/            # Monitoring batches

mlflow.db                  # MLflow SQLite tracking database
mlruns/                    # MLflow artifact store

reports/
├── final_report.tex       # Final LaTeX report
└── figures/               # Report figures
|__ Project Report.pdf     # Final PDF report


scripts/
└── export_figures.py      # Export MLflow figures to report folder

presentation/
└── MLOps_Logitics_Delay_Risk.pptx
```


## Setup

Python 3.10+ is required. From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` and override settings (tracking URI,
experiment names, registered model name).

## Reproduce the First Pass

### 1. Start the MLflow tracking UI (optional but recommended)

By default, scripts write to a local SQLite store at `mlflow.db` so you can
inspect it directly. To browse runs in a UI:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

If you start the server, point the scripts at it by setting
`MLFLOW_TRACKING_URI=http://127.0.0.1:5000` in your `.env`.

### 2. Generate the synthetic dataset

```powershell
python -m src.data_generation
```

Writes `data/raw/train_shipments.csv` plus monitoring batches under `data/monitoring/`, and prints class balance.

### 3. Train baseline models

```powershell
python -m src.train
```

Logs three runs (Logistic Regression, Random Forest, Gradient Boosting) under
the `logistics-delay-risk` experiment. Each run logs:

- Parameters (model type, seed, test size, hyperparameters)
- Metrics (accuracy, precision, recall, F1, ROC-AUC)
- Artifacts (confusion matrix plot, classification report JSON, feature
  importances, dataset schema)
- The fitted scikit-learn pipeline as a logged model

### 4. View results

If you started the tracking server, open http://127.0.0.1:5000 and select
the `logistics-delay-risk` experiment. Otherwise run:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Full lifecycle (submit-ready)

After the first pass, run tuning, registry, monitoring, and (optionally) the API:

```powershell
python -m src.tune --model gradient_boosting --trials 15 --metric f1
python -m src.register_model --metric f1 --top 3
python -m src.monitor
# Use a free port if 8000 is busy:
uvicorn src.serve_api:app --host 127.0.0.1 --port 8010
```

## LaTeX report and figures

The term report is `reports/final_report.tex`. Figures are resolved from
`reports/figures/` when you compile from inside `reports/`.

Refresh figures from the latest MLflow runs (also copies monitoring histograms):

```powershell
python scripts/export_figures.py
```

Then build the PDF (requires a LaTeX install such as MiKTeX):

```powershell
cd reports
pdflatex -interaction=nonstopmode final_report.tex
pdflatex -interaction=nonstopmode final_report.tex
```

Per-run artifact dumps (optional) may appear under `reports/<run_id>/` when
artifacts are downloaded from MLflow; the export script consolidates the
plots you need into `reports/figures/`.

## Notes on Reproducibility

- Random seeds for dataset generation, train/test split, and model training
  default to `42` and can be overridden via `RANDOM_SEED` in `.env`.
- Synthetic data uses weighted, domain-motivated factors (customs complexity,
  weather risk, port congestion, carrier type, priority) so target labels
  reflect realistic logistics signals rather than pure noise.
