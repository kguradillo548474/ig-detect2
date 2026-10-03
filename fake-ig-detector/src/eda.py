"""Exploratory data analysis -> outputs/eda_*.png"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import config as C
from utils import load_clean, get_logger

log = get_logger("eda")


def main():
    C.OUT_DIR.mkdir(exist_ok=True)
    df = load_clean(); lab = df[C.TARGET].map({0: "Genuine", 1: "Fake"})
    log.info("%d profiles after cleaning", len(df))

    ax = lab.value_counts().plot.bar(color=["#2ecc71", "#e74c3c"], rot=0)
    ax.set_title("Class balance"); ax.set_ylabel("Profiles")
    plt.tight_layout(); plt.savefig(C.OUT_DIR / "eda_class_balance.png", dpi=130); plt.close()

    plt.figure(figsize=(11, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation between all features"); plt.tight_layout()
    plt.savefig(C.OUT_DIR / "eda_correlation.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    df.groupby(lab)["profile pic"].mean().plot.bar(ax=ax[0], color=["#e74c3c", "#2ecc71"], rot=0)
    ax[0].set_title("Share with profile picture")
    sns.boxplot(x=lab, y=df["nums/length username"], ax=ax[1], palette=["#2ecc71", "#e74c3c"])
    ax[1].set_title("Digit ratio in username")
    sns.boxplot(x=lab, y=df["description length"], ax=ax[2], palette=["#2ecc71", "#e74c3c"])
    ax[2].set_title("Bio length")
    plt.tight_layout(); plt.savefig(C.OUT_DIR / "eda_key_features.png", dpi=130); plt.close()
    log.info("EDA plots saved to %s", C.OUT_DIR)


if __name__ == "__main__":
    main()
