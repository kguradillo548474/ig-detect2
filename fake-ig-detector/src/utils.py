"""Data loading, preprocessing, feature engineering and logging helpers."""
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import config as C


def get_logger(name):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(name)


def load_clean(path=C.DATA_PATH):
    """Load the CSV, drop duplicate rows and incomplete rows (as in the notebook)."""
    df = pd.read_csv(path)
    return df.drop_duplicates().dropna().copy()


def split(df):
    """Stratified 80/20 split (no scaling: XGBoost is tree-based)."""
    X, y = df.drop(columns=[C.TARGET]), df[C.TARGET]
    return train_test_split(X, y, test_size=C.TEST_SIZE, random_state=C.SEED, stratify=y)


def engineer(username, fullname, bio, pic, url, private, posts, followers, follows):
    """Raw profile inputs -> the dataset's 12 model features."""
    digits = lambda s: sum(c.isdigit() for c in s)
    u, f = username.strip(), fullname.strip()
    return {
        "profile pic": int(pic),
        "nums/length username": digits(u) / len(u) if u else 0.0,
        "fullname words": len(f.split()),
        "nums/length fullname": digits(f) / len(f) if f else 0.0,
        "name==username": int(bool(f) and f.lower().replace(" ", "") == u.lower()),
        "description length": len(bio),
        "external URL": int(url), "private": int(private),
        "#posts": posts, "#followers": followers, "#follows": follows,
        "followers_following_ratio": followers / (follows + 1),
    }


def risk_level(p):
    return ("HIGH RISK", "high") if p >= C.RISK_HIGH else ("MEDIUM RISK", "med") if p >= C.RISK_LOW else ("LOW RISK", "low")
