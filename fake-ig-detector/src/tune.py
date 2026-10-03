"""Optional Optuna hyperparameter search (alternative to the grid search).

    python src/tune.py --trials 50
"""
import argparse, json
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import config as C
from utils import load_clean, split, get_logger

log = get_logger("tune")


def main(trials):
    X_tr, _, y_tr, _ = split(load_clean())
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)

    def objective(t):
        p = dict(n_estimators=t.suggest_int("n_estimators", 100, 600, step=100), max_depth=t.suggest_int("max_depth", 3, 8),
                 learning_rate=t.suggest_float("learning_rate", 0.03, 0.3, log=True), subsample=t.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0))
        return cross_val_score(XGBClassifier(**p, **C.BASE_PARAMS), X_tr, y_tr, cv=cv, scoring="f1", n_jobs=-1).mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(objective, n_trials=trials)
    log.info("Best CV F1 %.4f | %s", study.best_value, study.best_params)
    C.OUT_DIR.mkdir(exist_ok=True); json.dump(study.best_params, open(C.OUT_DIR / "best_params_optuna.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=30); main(ap.parse_args().trials)
