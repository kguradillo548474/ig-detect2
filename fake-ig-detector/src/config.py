"""Central configuration: paths, feature names, hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Instagram.csv"
MODEL_PATH = ROOT / "models" / "xgb_model.json"
OUT_DIR = ROOT / "outputs"
METRICS_PATH = OUT_DIR / "metrics.json"

TARGET = "fake"
TEST_SIZE, SEED = 0.20, 42

# Best configuration found by GridSearchCV (243 combos x 5 folds, F1) in the Colab notebook
BEST_PARAMS = dict(n_estimators=500, max_depth=5, learning_rate=0.1, subsample=1.0, colsample_bytree=0.8)
BASE_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", random_state=SEED)
GRID = {"n_estimators": [200, 300, 500], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.7, 0.8, 1.0]}

RISK_LOW, RISK_HIGH = 0.30, 0.70   # <30% low, 30-70% medium, >=70% high
