"""Train + evaluate the XGBoost model. Saves models/xgb_model.json and outputs/*.png|json

    python src/train.py            # uses BEST_PARAMS from the notebook's grid search
    python src/train.py --grid     # re-run the full grid search (slow)
"""
import argparse, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import config as C
from utils import load_clean, split, get_logger

log = get_logger("train")


def main(grid=False):
    C.OUT_DIR.mkdir(exist_ok=True); C.MODEL_PATH.parent.mkdir(exist_ok=True)
    df = load_clean(); X_tr, X_te, y_tr, y_te = split(df)
    log.info("Train %d | Test %d | Features %d", len(X_tr), len(X_te), X_tr.shape[1])
    if grid:
        gs = GridSearchCV(XGBClassifier(**C.BASE_PARAMS), C.GRID, scoring="f1", cv=5, n_jobs=-1, verbose=1).fit(X_tr, y_tr)
        model, params = gs.best_estimator_, gs.best_params_
    else:
        params = C.BEST_PARAMS; model = XGBClassifier(**params, **C.BASE_PARAMS).fit(X_tr, y_tr)
    model.save_model(C.MODEL_PATH)

    prob = model.predict_proba(X_te)[:, 1]; pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y_te, pred)
    m = {"accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred), "recall": recall_score(y_te, pred),
         "f1": f1_score(y_te, pred), "roc_auc": roc_auc_score(y_te, prob), "pr_auc": average_precision_score(y_te, prob),
         "confusion_matrix": cm.tolist(), "n_total": len(df), "n_train": len(X_tr), "n_test": len(X_te),
         "params": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in params.items()},
         "features": list(X_tr.columns), "importance": dict(zip(X_tr.columns, map(float, model.feature_importances_)))}
    json.dump(m, open(C.METRICS_PATH, "w"), indent=2)
    log.info({k: round(v, 4) for k, v in m.items() if isinstance(v, float)})

    ConfusionMatrixDisplay(cm, display_labels=["Genuine", "Fake"]).plot(cmap="Blues"); plt.title("Confusion Matrix - XGBoost")
    plt.tight_layout(); plt.savefig(C.OUT_DIR / "confusion_matrix.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y_te, prob); ax[0].plot(fpr, tpr, lw=2, label=f"AUC = {m['roc_auc']:.4f}"); ax[0].plot([0, 1], [0, 1], "--")
    ax[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate"); ax[0].legend(loc="lower right")
    p, r, _ = precision_recall_curve(y_te, prob); ax[1].plot(r, p, lw=2, label=f"AP = {m['pr_auc']:.4f}")
    ax[1].set(title="Precision-Recall curve", xlabel="Recall", ylabel="Precision", ylim=(0.9, 1.005)); ax[1].legend(loc="lower left")
    plt.tight_layout(); plt.savefig(C.OUT_DIR / "roc_pr_curves.png", dpi=130); plt.close()

    imp = sorted(m["importance"].items(), key=lambda kv: kv[1])
    plt.figure(figsize=(9, 5.5)); plt.barh([k for k, _ in imp], [v for _, v in imp], color="#e74c3c")
    plt.title("XGBoost feature importance"); plt.xlabel("Importance"); plt.tight_layout()
    plt.savefig(C.OUT_DIR / "feature_importance.png", dpi=130); plt.close()

    plt.figure(figsize=(8, 4.5)); bins = np.linspace(0, 1, 41)
    plt.hist(prob[y_te == 0], bins, alpha=.7, label="Genuine", color="#2ecc71"); plt.hist(prob[y_te == 1], bins, alpha=.7, label="Fake", color="#e74c3c")
    plt.yscale("log"); plt.title("Predicted fake-probability distribution (test set)"); plt.xlabel("P(fake)"); plt.legend()
    plt.tight_layout(); plt.savefig(C.OUT_DIR / "score_distribution.png", dpi=130); plt.close()
    log.info("Saved model + plots")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--grid", action="store_true"); main(ap.parse_args().grid)
