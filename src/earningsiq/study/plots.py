"""Figures for the text-price study."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3de"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

FEATURE_LABELS = {
    "kw_tone": "Keyword tone",
    "finbert_tone": "FinBERT tone",
    "llm_tone": "LLM tone",
    "d_kw_tone": "Change in keyword tone",
    "d_finbert_tone": "Change in FinBERT tone",
    "d_llm_tone": "Change in LLM tone",
    "mdna_change": "MD&A rewritten",
    "risk_change": "Risk Factors rewritten",
}

TARGET_LABELS = {
    "car_announce": "Earnings release\ndays 0 to +1",
    "car_filing": "Filing\ndays 0 to +1",
    "car_filing_lagged": "Filing published after\nthe release, days 0 to +1",
    "car_drift": "Month after filing\ndays +2 to +21",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path):
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _interval_panels(table: pd.DataFrame, panel_column: str, row_column: str, value: str, low: str, high: str,
                     panel_labels: dict, row_labels: dict, reference: float, reference_label: str, title: str,
                     subtitle: str, xlabel: str, path: Path):
    """Dot-and-interval chart: one panel per target, one row per feature."""
    panels = list(dict.fromkeys(table[panel_column]))
    rows = list(dict.fromkeys(table[row_column]))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels) + 2.2, 0.42 * len(rows) + 2.0), sharex=True, sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    fig.patch.set_facecolor(SURFACE)

    for ax, panel in zip(axes, panels):
        part = table[table[panel_column] == panel].set_index(row_column)
        _style(ax)
        ax.axvline(reference, color=TEXT_SECONDARY, linewidth=1)
        for y, row in enumerate(rows):
            if row not in part.index:
                continue
            ax.plot([part.loc[row, low], part.loc[row, high]], [y, y], color=BLUE, linewidth=2, solid_capstyle="round")
            ax.plot(part.loc[row, value], y, "o", color=BLUE, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_title(panel_labels.get(panel, panel), fontsize=9.5, color=TEXT, loc="left", pad=8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([row_labels.get(row, row) for row in rows], color=TEXT, fontsize=9.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))

    axes[0].annotate(reference_label, xy=(reference, len(rows) - 0.5), xytext=(3, 3), textcoords="offset points",
                     fontsize=8, color=TEXT_SECONDARY)
    fig.supxlabel(xlabel, fontsize=9.5, color=TEXT_SECONDARY)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=TEXT, fontweight="bold")
    fig.text(0.01, 0.985 - 0.36 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=9.5, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.42 / fig.get_figheight()), w_pad=2.0)
    _save(fig, path)


def association_figure(association: pd.DataFrame, path: Path):
    _interval_panels(
        association, "target", "feature", "rho", "ci_low", "ci_high",
        TARGET_LABELS, FEATURE_LABELS, reference=0.0, reference_label="no relation",
        title="Text measures and abnormal returns",
        subtitle="Rank correlation with 95% interval. An interval that crosses the line is compatible with no relation.",
        xlabel="Spearman correlation", path=path,
    )


def prediction_figure(prediction: pd.DataFrame, path: Path):
    table = prediction.copy()
    table["low"] = table["auc_ci"].map(lambda interval: interval[0])
    table["high"] = table["auc_ci"].map(lambda interval: interval[1])
    labels = {"keywords": "Keyword tone only", "all text": "All text measures",
              "past return": "Past returns only", "text + past return": "Text and past returns"}
    _interval_panels(
        table, "target", "features", "auc", "low", "high",
        TARGET_LABELS, labels, reference=0.5, reference_label="chance",
        title="Predicting the direction of the abnormal return",
        subtitle="AUC on years the model was not fitted on, with 95% interval. 0.5 is a coin flip.",
        xlabel="AUC", path=path,
    )


def car_path_figure(data: pd.DataFrame, paths: pd.DataFrame, feature: str, high_label: str, low_label: str, path: Path):
    """Average cumulative abnormal return around the filing, for the highest and lowest third of a feature."""
    rank = data.groupby("ticker")[feature].rank(pct=True)
    groups = {
        high_label: (data.loc[rank > 2 / 3, "accession"], BLUE),
        low_label: (data.loc[rank <= 1 / 3, "accession"], ORANGE),
    }

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (2, 3)))

    for label, (accessions, color) in groups.items():
        group = paths.loc[paths.index.intersection(accessions)]
        cumulative = group.cumsum(axis=1).mean() * 100
        ax.plot(cumulative.index, cumulative.to_numpy(), color=color, linewidth=2, label=f"{label} (n = {len(group)})")
        ax.annotate(label, xy=(cumulative.index[-1], cumulative.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TEXT)

    ax.annotate("filing", xy=(0, ax.get_ylim()[1]), xytext=(4, -10), textcoords="offset points", fontsize=8.5, color=TEXT_SECONDARY)
    ax.set_xlabel("Trading days relative to the filing", fontsize=9.5, color=TEXT_SECONDARY)
    ax.set_ylabel("Cumulative abnormal return, %", fontsize=9.5, color=TEXT_SECONDARY)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2, labelcolor=TEXT)
    ax.set_title(f"Abnormal returns around the filing, by {FEATURE_LABELS.get(feature, feature)}",
                 fontsize=12.5, color=TEXT, fontweight="bold", loc="left", pad=22)
    ax.text(0, 1.03, "Thirds are formed within each company. Day 0 is the first session that can react to the filing.",
            transform=ax.transAxes, fontsize=9.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    _save(fig, path)
