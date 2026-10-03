"""Inference: Python API + CLI.

    python src/predict.py --csv data/Instagram.csv --out outputs/predictions.csv
    python src/predict.py --username user83920174 --followers 12 --follows 900
"""
import argparse
import pandas as pd
from xgboost import XGBClassifier
import config as C
from utils import engineer, risk_level, load_clean


def load_model():
    m = XGBClassifier(); m.load_model(C.MODEL_PATH); return m


def predict_proba(model, df):
    return model.predict_proba(df[model.get_booster().feature_names])[:, 1]


def predict_profile(model, **raw):
    X = pd.DataFrame([engineer(**raw)]); p = float(predict_proba(model, X)[0])
    return {"fake_probability": p, "verdict": "FAKE" if p >= .5 else "GENUINE", "risk": risk_level(p)[0]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv"); ap.add_argument("--out")
    ap.add_argument("--username", default=""); ap.add_argument("--fullname", default=""); ap.add_argument("--bio", default="")
    ap.add_argument("--nopic", action="store_true"); ap.add_argument("--url", action="store_true"); ap.add_argument("--private", action="store_true")
    ap.add_argument("--posts", type=int, default=0); ap.add_argument("--followers", type=int, default=0); ap.add_argument("--follows", type=int, default=0)
    a = ap.parse_args(); model = load_model()
    if a.csv:
        d = pd.read_csv(a.csv); d["fake_probability"] = predict_proba(model, d)
        d["prediction"] = (d.fake_probability >= .5).map({True: "fake", False: "genuine"})
        (d.to_csv(a.out, index=False) if a.out else print(d.head(20).to_string()))
    else:
        print(predict_profile(model, username=a.username, fullname=a.fullname, bio=a.bio, pic=not a.nopic, url=a.url,
                              private=a.private, posts=a.posts, followers=a.followers, follows=a.follows))
