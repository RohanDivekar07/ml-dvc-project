# ML Experiment Pipeline with Git + DVC

Reproducible ML pipeline (Breast Cancer classification) using Git for code and DVC for data, models and metrics.

## Pipeline
preprocess -> train -> evaluate (defined in `dvc.yaml`)

## Structure
- `data/` dataset (tracked by DVC)
- `src/` preprocess.py, train.py, evaluate.py
- `params.yaml` hyperparameters
- `metrics/metrics.json` evaluation results
- `dvc.yaml`, `dvc.lock` pipeline definition and locked versions

## Reproduce
```bash
git clone https://github.com/RohanDivekar07/ml-dvc-project.git
cd ml-dvc-project
pip install -r requirements.txt
dvc pull
dvc repro
dvc metrics show
```

## Experiments
| Tag | Dataset | n_estimators | max_depth |
|---|---|---|---|
| exp1 | v1 (400 rows) | 100 | 5 |
| exp2 | v1 (400 rows) | 200 | 10 |
| exp3 | v2 (569 rows) | 200 | 10 |

Compare: `dvc metrics diff exp1 exp2`

## Author
Your Name | Roll No. | Course
