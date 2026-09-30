"""Metrics and charts on the test set. Run: python -m src.evaluate"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve  # noqa: E402

from src import config  # noqa: E402
from src.config import APPROVAL_RATE, BEHAVIORAL_FEATURES, SEED, TARGET  # noqa: E402
from src.features import thin_file_mask  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"baseline": "#2a78d6", "enhanced": "#eb6834"}
TRADITIONAL_BAR = "#a9a8a2"
LABELS = {"baseline": "Baseline (credit only)", "enhanced": "Enhanced (+ behavioral)"}
SEGMENTS = {"full": "Full test set", "thin": "Thin-file"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK,
        "font.size": 10,
    }
)


def recall_at_approval_rate(y, score, approval_rate: float = APPROVAL_RATE) -> float:
    """Approve the `approval_rate` lowest-risk applicants; return the share of actual defaulters that fall in the rejected group."""
    y, score = np.asarray(y), np.asarray(score)
    n_reject = int(round(len(y) * (1 - approval_rate)))
    rejected = np.argsort(-score, kind="stable")[:n_reject]
    return y[rejected].sum() / max(y.sum(), 1)


def bad_rate_at_approval_rate(y, score, approval_rate: float = APPROVAL_RATE) -> float:
    """Default rate among the approved applicants (lower is better)."""
    y, score = np.asarray(y), np.asarray(score)
    n_approve = int(round(len(y) * approval_rate))
    approved = np.argsort(score, kind="stable")[:n_approve]
    return y[approved].mean()


def segment_metrics(y, score) -> dict:
    return {
        "n": len(y),
        "default_rate": float(np.mean(y)),
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "recall_at_approval": recall_at_approval_rate(y, score),
        "bad_rate_approved": bad_rate_at_approval_rate(y, score),
    }


def bootstrap_auc_delta(y, s_base, s_enh, n_boot: int = 500, seed: int = SEED):
    """95% bootstrap CI for ROC-AUC(enhanced) - ROC-AUC(baseline), paired resampling."""
    rng = np.random.default_rng(seed)
    y, s_base, s_enh = map(np.asarray, (y, s_base, s_enh))
    deltas = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        deltas.append(roc_auc_score(y[i], s_enh[i]) - roc_auc_score(y[i], s_base[i]))
    return np.percentile(deltas, [2.5, 97.5])


def compute_metrics(scorers: dict, test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    masks = {"full": np.ones(len(test), bool), "thin": thin_file_mask(test).to_numpy()}
    scores = {name: s.raw_score(test) for name, s in scorers.items()}
    y = test[TARGET].to_numpy()

    rows = []
    for name in scorers:
        feature_set, algo = name.split("_")
        for seg, m in masks.items():
            rows.append({"model": algo, "features": feature_set, "segment": seg,
                         **segment_metrics(y[m], scores[name][m])})
    metrics = pd.DataFrame(rows)

    cis = {}
    for algo in ("logreg", "lgbm"):
        for seg, m in masks.items():
            cis[(algo, seg)] = bootstrap_auc_delta(
                y[m], scores[f"baseline_{algo}"][m], scores[f"enhanced_{algo}"][m]
            )
    return metrics, {"scores": scores, "masks": masks, "y": y, "auc_delta_ci": cis}


def results_markdown(metrics: pd.DataFrame, cis: dict) -> str:
    pct = f"{APPROVAL_RATE:.0%}"
    lines = [
        f"| Model | Segment | n | Default rate | ROC-AUC (base → enh) | PR-AUC (base → enh) "
        f"| Recall @ {pct} approval (base → enh) | ΔROC-AUC [95% CI] |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for algo, algo_label in (("lgbm", "LightGBM"), ("logreg", "Logistic reg.")):
        for seg in ("full", "thin"):
            b = metrics.query("model == @algo and segment == @seg and features == 'baseline'").iloc[0]
            e = metrics.query("model == @algo and segment == @seg and features == 'enhanced'").iloc[0]
            lo, hi = cis[(algo, seg)]
            lines.append(
                f"| {algo_label} | {SEGMENTS[seg]} | {b.n:,} | {b.default_rate:.1%} "
                f"| {b.roc_auc:.3f} → **{e.roc_auc:.3f}** "
                f"| {b.pr_auc:.3f} → **{e.pr_auc:.3f}** "
                f"| {b.recall_at_approval:.1%} → **{e.recall_at_approval:.1%}** "
                f"| {e.roc_auc - b.roc_auc:+.3f} [{lo:+.3f}, {hi:+.3f}] |"
            )
    return "\n".join(lines) + "\n"


def update_readme(results_md: str, readme=config.ROOT / "README.md"):
    """Replace the block between the RESULTS markers in the README, if present."""
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if not readme.exists() or start not in (text := readme.read_text()):
        return
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    readme.write_text(f"{head}{start}\n{results_md}{end}{tail}")


def plot_roc(ctx: dict, algo: str = "lgbm"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, (seg, m) in zip(axes, ctx["masks"].items()):
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, ls="--", zorder=1)
        for fs in ("baseline", "enhanced"):
            s = ctx["scores"][f"{fs}_{algo}"][m]
            fpr, tpr, _ = roc_curve(ctx["y"][m], s)
            auc = roc_auc_score(ctx["y"][m], s)
            ax.plot(fpr, tpr, color=COLORS[fs], lw=2, label=f"{LABELS[fs]}  AUC {auc:.3f}", zorder=3)
        ax.set_title(f"{SEGMENTS[seg]}  (n={m.sum():,})", loc="left")
        ax.set_xlabel("False positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_aspect("equal")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("True positive rate")
    fig.suptitle("ROC curves - LightGBM, baseline vs enhanced", x=0.02, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def plot_segment_comparison(metrics: pd.DataFrame, algo: str = "lgbm"):
    panels = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
              ("recall_at_approval", f"Recall @ {APPROVAL_RATE:.0%} approval")]
    m = metrics[metrics.model == algo]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    x = np.arange(2)
    w = 0.36
    for ax, (col, title) in zip(axes, panels):
        for k, fs in enumerate(("baseline", "enhanced")):
            vals = [m.query("segment == @seg and features == @fs")[col].iloc[0] for seg in SEGMENTS]
            bars = ax.bar(x + (k - 0.5) * (w + 0.02), vals, w, color=COLORS[fs], label=LABELS[fs], zorder=3)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.set_xticks(x, list(SEGMENTS.values()))
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        top = m[col].max()
        ax.set_ylim(0, min(1.0, top * 1.18) if top > 0.5 else top * 1.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Baseline vs enhanced by segment - LightGBM", x=0.02, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def shap_values(scorer, test: pd.DataFrame, n: int = 5000):
    sample = test.sample(min(n, len(test)), random_state=SEED)
    X = scorer.pipeline[:-1].transform(sample)
    return scorer.contributions(sample), X


def plot_feature_importance(contrib: pd.DataFrame):
    imp = contrib.abs().mean().sort_values()
    colors = [COLORS["enhanced"] if f in BEHAVIORAL_FEATURES else TRADITIONAL_BAR for f in imp.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index, imp.values, color=colors, height=0.7, zorder=3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP value|  (impact on log-odds of default)")
    ax.set_title("Feature importance - enhanced LightGBM", loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS["enhanced"], label="Behavioral (simulated)"),
                       Patch(color=TRADITIONAL_BAR, label="Traditional credit")], loc="lower right")
    fig.tight_layout()
    return fig


def plot_shap_beeswarm(contrib: pd.DataFrame, X: pd.DataFrame):
    import shap

    shap.summary_plot(contrib.to_numpy(), X, show=False, max_display=15, plot_size=(9, 6.5))
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP summary - enhanced LightGBM", loc="left")
    fig.tight_layout()
    return fig


def main():
    scorers = joblib.load(config.MODELS_DIR / "models.joblib")
    test = joblib.load(config.MODELS_DIR / "test_set.joblib")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics, ctx = compute_metrics(scorers, test)
    metrics.round(4).to_csv(config.REPORTS_DIR / "metrics.csv", index=False)
    md = results_markdown(metrics, ctx["auc_delta_ci"])
    (config.REPORTS_DIR / "results.md").write_text(md)
    if config.REPORTS_DIR == config.ROOT / "reports":
        update_readme(md)
    print(md)

    figs = {
        "roc_curves.png": plot_roc(ctx),
        "segment_comparison.png": plot_segment_comparison(metrics),
    }
    contrib, X = shap_values(scorers["enhanced_lgbm"], test)
    figs["feature_importance.png"] = plot_feature_importance(contrib)
    for name, fig in figs.items():
        fig.savefig(config.FIGURES_DIR / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
    fig = plot_shap_beeswarm(contrib, X)
    fig.savefig(config.FIGURES_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics to {config.REPORTS_DIR}/ and charts to {config.FIGURES_DIR}/")


if __name__ == "__main__":
    main()
