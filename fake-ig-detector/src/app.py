"""Streamlit dashboard:  streamlit run src/app.py"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import streamlit as st
import xgboost as xgb
import config as C
from predict import load_model, predict_proba
from utils import engineer, risk_level

st.set_page_config(page_title="Fake Instagram Profile Detection", page_icon="🛡️", layout="wide")
st.markdown("""<style>
.banner{background:linear-gradient(90deg,#1b1f2a,#2a1418);border:1px solid #3a2a2e;border-radius:12px;padding:18px 22px;margin-bottom:14px}
.banner h1{margin:0;font-size:1.7rem}.banner p{margin:4px 0 0;color:#9aa0ab;font-size:.9rem}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#2ecc71;margin-right:6px;box-shadow:0 0 8px #2ecc71}
.risk{border-radius:12px;padding:18px;text-align:center;border:1px solid}
.risk .lvl{font-size:.8rem;letter-spacing:.2em;opacity:.85}.risk .score{font-size:3rem;font-weight:800;line-height:1.1}
.low{background:#0f2a1a;border-color:#1f7a45;color:#4ee38a}.med{background:#2a2410;border-color:#a8861f;color:#ffd24d}.high{background:#2e1214;border-color:#b3262e;color:#ff6b6b}
</style>""", unsafe_allow_html=True)

NICE = {"profile pic": "Profile picture", "nums/length username": "Digits in username", "fullname words": "Words in full name",
        "nums/length fullname": "Digits in full name", "name==username": "Name equals username", "description length": "Bio length",
        "external URL": "External URL", "private": "Private account", "#posts": "Post count", "#followers": "Followers",
        "#follows": "Following", "followers_following_ratio": "Follower/following ratio"}


@st.cache_resource
def get_model():
    return load_model(), json.load(open(C.METRICS_PATH))


def explain(model, X):
    """Per-profile feature contributions (SHAP-style, log-odds) from XGBoost."""
    c = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    return pd.Series(c, index=X.columns).sort_values(key=abs, ascending=False)


def show_result(model, X, raw, user):
    p = float(predict_proba(model, X)[0]); lvl, cls = risk_level(p)
    a, b = st.columns([1, 2])
    with a:
        st.markdown(f'<div class="risk {cls}"><div class="lvl">{lvl}</div><div class="score">{p:.0%}</div>'
                    f'<div class="lvl">FAKE PROBABILITY</div></div>', unsafe_allow_html=True)
        st.markdown(f"**Verdict:** {'🚩 Flagged as FAKE' if p >= 0.5 else '✅ Appears GENUINE'}")
    with b:
        st.markdown("**Why this result: top signals**")
        for f, v in explain(model, X).head(5).items():
            icon, txt = ("🔴", "pushes toward FAKE") if v > 0 else ("🟢", "pushes toward GENUINE")
            st.markdown(f"{icon} **{NICE[f]}** = `{raw[f]:.3g}`: {txt}")
    st.session_state.setdefault("scans", []).insert(0, {"Username": user or "-", "Fake prob.": f"{p:.1%}", "Risk": lvl,
                                                        "Verdict": "FAKE" if p >= 0.5 else "GENUINE"})


if not (C.MODEL_PATH.exists() and C.METRICS_PATH.exists()):
    st.error("Model not found. Run `python src/train.py` first."); st.stop()
model, M = get_model()

st.markdown('<div class="banner"><h1>🛡️ Fake Instagram Profile Detection System</h1>'
            '<p><span class="dot"></span>SYSTEM ONLINE · XGBoost classifier · structural &amp; behavioral profile analysis · Group 17 · CST9</p></div>',
            unsafe_allow_html=True)
t1, t2, t3, t4 = st.tabs(["🔍 Detect profile", "📄 Batch scan", "📊 Model performance", "🧪 Dataset insights"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        username = st.text_input("Username", "john_doe92"); fullname = st.text_input("Full name", "John Doe")
        bio = st.text_area("Bio", "", height=100)
        pic = st.checkbox("Has profile picture", True); url = st.checkbox("Has external URL in bio", False)
        private = st.checkbox("Private account", False)
    with c2:
        posts = st.number_input("Number of posts", 0, 100000, 12)
        followers = st.number_input("Followers", 0, 1_000_000_000, 150); follows = st.number_input("Following", 0, 1_000_000, 300)
    feats = engineer(username, fullname, bio, pic, url, private, posts, followers, follows)
    if st.button("🔎 Scan profile", type="primary"):
        X = pd.DataFrame([feats])[M["features"]]
        show_result(model, X, feats, username)
        with st.expander("Features computed from your input"):
            st.dataframe(X.T.rename(columns={0: "value"}))
    if st.session_state.get("scans"):
        st.markdown("**Scan log (this session)**")
        st.dataframe(pd.DataFrame(st.session_state["scans"]), width="stretch", hide_index=True)

with t2:
    st.write("Upload a CSV with the same columns as `Instagram.csv` (the `fake` column is optional).")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        d = pd.read_csv(up); miss = [c for c in M["features"] if c not in d.columns]
        if miss: st.error(f"Missing columns: {miss}")
        else:
            d["fake_probability"] = predict_proba(model, d); d["prediction"] = (d.fake_probability >= .5).map({True: "fake", False: "genuine"})
            d["risk_level"] = d.fake_probability.apply(lambda p: risk_level(p)[0])
            m = st.columns(4); m[0].metric("Profiles scanned", len(d))
            for col, (lbl, ic) in zip(m[1:], [("HIGH RISK", "🔴"), ("MEDIUM RISK", "🟡"), ("LOW RISK", "🟢")]):
                col.metric(f"{ic} {lbl.title()}", int((d.risk_level == lbl).sum()))
            st.dataframe(d, width="stretch")
            st.download_button("Download results", d.to_csv(index=False), "predictions.csv", "text/csv")

with t3:
    cols = st.columns(5)
    for c, (k, n) in zip(cols, [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score"), ("roc_auc", "ROC-AUC")]):
        c.metric(n, f"{M[k]:.4f}")
    st.caption(f"{M['n_total']:,} profiles after cleaning · {M['n_train']:,} train / {M['n_test']:,} test · params: {M['params']}")
    for pair in [("confusion_matrix.png", "roc_pr_curves.png"), ("feature_importance.png", "score_distribution.png")]:
        for col, f in zip(st.columns(2), pair):
            if (C.OUT_DIR / f).exists(): col.image(str(C.OUT_DIR / f))
    st.info("Scores come from a single dataset and may not generalize to current Instagram accounts. This is a classroom demo, not a moderation system.")

with t4:
    for f in ["eda_class_balance.png", "eda_key_features.png", "eda_correlation.png"]:
        if (C.OUT_DIR / f).exists(): st.image(str(C.OUT_DIR / f))
