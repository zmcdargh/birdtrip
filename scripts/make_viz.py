"""
Four visualizations of the precompute results, designed against Tufte's
principles (see skills/tufte_principles): maximal data-ink, no chartjunk,
grayscale with a single purposeful accent, direct labels over legends,
range-frame axes, small multiples, and a slopegraph. Honest axes throughout.

  fig1_vagrant_unmasked   - why pooled frequency lies about the Varied Thrush
  fig2_phenology          - small multiples: detection probability across the year
  fig3_shrinkage          - low-sample estimates pulled toward the regional mean
  fig4_alpha_slopegraph   - how the site ranking reorders as rarity-weighting rises

Outputs PNG + SVG into viz/.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from birdtrip.precompute import load, ebird_week  # reuse loaders

VIZ = ROOT / "viz"; VIZ.mkdir(exist_ok=True)
INK = "#222222"; MUTE = "#999999"; FAINT = "#cccccc"; ACCENT = "#c0392b"  # one accent only

# --- minimal Tufte-leaning style: erase non-data ink -------------------------
mpl.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
    "font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.titlesize": 11, "axes.titlelocation": "left",
    "axes.titlepad": 10, "figure.facecolor": "white", "savefig.facecolor": "white",
})

def save(fig, name):
    fig.savefig(VIZ / f"{name}.png", bbox_inches="tight")
    fig.savefig(VIZ / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)

def range_frame(ax, x, y):
    """Tufte range-frame: spines span only the data range."""
    ax.spines["left"].set_bounds(min(y), max(y))
    ax.spines["bottom"].set_bounds(min(x), max(x))

MONTH_STARTS = [(m - 1) * 4 + 1 for m in range(1, 13)]
MONTH_LABELS = list("JFMAMJJASOND")

# load data once
sed, ebd = load(ROOT / "data/sample/ebd_sample.txt",
                ROOT / "data/sample/ebd_sample_sampling.txt", current_year=2026)
pre = pd.read_csv(ROOT / "data/precomputed.csv")

# ============================================================================
# FIG 1 — Vagrant unmasked: per-year detection at Central Park, winter weeks
# ============================================================================
def fig1():
    winter = lambda w: (w <= 8) | (w >= 44)
    sw = sed[winter(sed["week"]) & (sed["LOCALITY"] == "Central Park")]
    ew = ebd[winter(ebd["week"]) & (ebd["LOCALITY"] == "Central Park")]
    years = sorted(sw["year"].unique())
    den = sw.groupby("year")["SAMPLING EVENT IDENTIFIER"].nunique()

    def yearly_freq(common):
        d = (ew[ew["COMMON NAME"] == common].groupby("year")["SAMPLING EVENT IDENTIFIER"]
             .nunique().reindex(years).fillna(0))
        return (d / den.reindex(years)).values

    series = [("Northern Cardinal", "present every winter"),
              ("Varied Thrush", "one winter only — a single over-wintering bird")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.1), sharey=True)
    for ax, (sp, note) in zip(axes, series):
        f = yearly_freq(sp)
        pooled = f.mean()  # pooling the 5 winters ~ the eBird bar-chart number
        ax.axhline(pooled, color=ACCENT, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.plot(years, f, "o", color=INK, ms=7, zorder=3)
        for yr, val in zip(years, f):              # direct value labels, not a y-grid
            if val > 0.001:
                ax.annotate(f"{val:.0%}", (yr, val), textcoords="offset points",
                            xytext=(0, 8), ha="center", color=INK, fontsize=8.5)
        ax.set_title(f"{sp}\n{note}", fontsize=10)
        ax.set_xticks(years); ax.set_xticklabels([str(y) for y in years], fontsize=8.5)
        ax.set_ylim(-0.04, 0.85)
        range_frame(ax, years, [0, 0.8])
        ax.tick_params(length=3)
    axes[0].set_ylabel("detected on … % of winter checklists")
    # one accent annotation, on the right panel, clear of the dots
    axes[1].annotate("pooled frequency\n(what a bar chart shows)", xy=(years[-1], series_pooled := \
                     yearly_freq("Varied Thrush").mean()), xytext=(years[1] - 0.3, 0.46),
                     color=ACCENT, fontsize=8.5, va="center",
                     arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8))
    fig.subplots_adjust(top=0.72, bottom=0.2, wspace=0.12)
    fig.suptitle("A single vagrant inflates the pooled frequency", x=0.005, y=0.99, ha="left",
                 fontsize=13, weight="bold")
    fig.text(0.005, 0.02,
             "Pooling five winters, the Varied Thrush looks ~16% findable. But it appeared in only 1 of 5 years "
             "(occupancy 0.2):\nthe planner down-weights it accordingly. The Cardinal's pooled number is honest "
             "because every year resembles it.", fontsize=8.3, color=MUTE)
    save(fig, "fig1_vagrant_unmasked")

# ============================================================================
# FIG 2 — Phenology small multiples: P(lifer per checklist) across the year
# ============================================================================
def fig2():
    panels = [("Saltmarsh Sparrow", "Montauk Point", "coastal specialty, summer breeder"),
              ("American Robin", "Central Park", "resident, dips in midwinter"),
              ("Northern Cardinal", "Central Park", "resident, flat year-round")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 6.0), sharex=True)
    for ax, (sp, loc, note) in zip(axes, panels):
        d = (pre[(pre["COMMON NAME"] == sp) & (pre["LOCALITY"] == loc)]
             .set_index("week")["p_lifer_1"].reindex(range(1, 49)).fillna(0))
        ax.fill_between(d.index, d.values, color=FAINT, zorder=1)
        ax.plot(d.index, d.values, color=INK, lw=1.4, zorder=2)
        peak = d.idxmax()
        ax.plot([peak], [d[peak]], "o", color=ACCENT, ms=5, zorder=3)
        ax.annotate(f"peak {d[peak]:.0%}", (peak, d[peak]), textcoords="offset points",
                    xytext=(6, -1), color=ACCENT, fontsize=8.5, va="center")
        ax.set_title(f"{sp} · {loc}  —  {note}", fontsize=9.5)
        ax.set_ylim(0, 0.62); ax.set_yticks([0, 0.5])
        ax.set_yticklabels(["0", "50%"], fontsize=8)
        ax.spines["left"].set_bounds(0, 0.5); ax.tick_params(length=3)
    axes[-1].set_xticks(MONTH_STARTS); axes[-1].set_xticklabels(MONTH_LABELS, fontsize=8.5)
    axes[-1].spines["bottom"].set_bounds(1, 48)
    fig.suptitle("When to go: detection probability across the year", x=0.005, ha="left",
                 fontsize=13, weight="bold")
    fig.text(0.005, 0.005, "Shared vertical scale (chance of detection on one checklist). "
             "Same axes in every panel — only the species changes.", fontsize=8.3, color=MUTE)
    fig.subplots_adjust(hspace=0.45)
    save(fig, "fig2_phenology")

# ============================================================================
# FIG 3 — Shrinkage: raw -> shrunk estimate vs sample size
# ============================================================================
def fig3():
    d = pre.copy()
    d = d[d["freq_raw"] > 0]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    # segment from raw to shrunk for each cell; movement shrinks as n grows
    for _, r in d.iterrows():
        ax.plot([r["n_checklists"], r["n_checklists"]], [r["freq_raw"], r["freq_shrunk"]],
                color=FAINT, lw=0.8, zorder=1)
    ax.scatter(d["n_checklists"], d["freq_raw"], s=12, color=MUTE, zorder=2, label="raw")
    ax.scatter(d["n_checklists"], d["freq_shrunk"], s=12, color=INK, zorder=3, label="shrunk")
    ax.set_xscale("log")
    ax.set_xlabel("complete checklists in the cell (log scale)", labelpad=6)
    ax.set_ylabel("detection frequency")
    ax.set_ylim(0, 1.02)
    ax.tick_params(length=3)
    # direct labels, stacked top-left, clear of the data and the axes
    xmin = d["n_checklists"].min()
    ax.scatter([xmin], [0.97], s=12, color=MUTE); ax.text(xmin * 1.12, 0.97, "raw frequency",
            color=MUTE, fontsize=9, va="center")
    ax.scatter([xmin], [0.90], s=12, color=INK); ax.text(xmin * 1.12, 0.90, "after shrinkage",
            color=INK, fontsize=9, va="center")
    fig.subplots_adjust(top=0.88, bottom=0.24)
    fig.suptitle("Low-sample estimates get pulled toward the regional mean", x=0.005, y=0.98, ha="left",
                 fontsize=13, weight="bold")
    fig.text(0.005, 0.02, "Each vertical stub is one (species, place, week) cell: raw frequency at one end, "
             "the Beta-Binomial estimate at the other.\nCells built on few checklists (left) move a lot; "
             "well-sampled cells (right) barely budge.", fontsize=8.3, color=MUTE)
    save(fig, "fig3_shrinkage")

# ============================================================================
# FIG 4 — alpha slopegraph: site ranking at alpha=0 vs alpha=1.5
# ============================================================================
def fig4():
    d = pre[pre["trusted"]].copy()
    allbest = d.groupby("COMMON NAME")["p_lifer_1"].max()
    wmap = {sp: 1.0 / max(allbest.get(sp, 0.0), 0.05) for sp in allbest.index}
    def ranked(alpha, n=6):
        d["score"] = d.apply(lambda r: (wmap[r["COMMON NAME"]] ** alpha) * r["p_lifer_1"], axis=1)
        g = (d.groupby(["LOCALITY", "week"])["score"].sum().sort_values(ascending=False)
             .head(n).reset_index())
        g["label"] = g["LOCALITY"] + ", wk " + g["week"].astype(str)
        return g
    left, right = ranked(0.0), ranked(1.5)
    labels = list(dict.fromkeys(left["label"].tolist() + right["label"].tolist()))
    lp = {l: i for i, l in enumerate(left["label"])}
    rp = {l: i for i, l in enumerate(right["label"])}

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for lab in labels:
        li, ri = lp.get(lab), rp.get(lab)
        newcomer = li is None and ri is not None     # promoted into the top list by rarity-weighting
        dropped = li is not None and ri is None       # pushed out of the top list
        col = ACCENT if newcomer else (FAINT if dropped else INK)
        if li is not None and ri is not None:
            ax.plot([0, 1], [li, ri], color=col, lw=1.3, zorder=2)
        if li is not None:
            ax.plot(0, li, "o", color=col, ms=4)
            ax.text(-0.03, li, lab, ha="right", va="center", fontsize=8.6, color=col)
        if ri is not None:
            ax.plot(1, ri, "o", color=col, ms=4)
            ax.text(1.03, ri, lab, ha="left", va="center", fontsize=8.6,
                    color=col, weight=("bold" if newcomer else "normal"))
    ax.set_xlim(-0.78, 1.78); ax.set_ylim(-1.4, max(len(left), len(right)) - 0.4)
    ax.invert_yaxis(); ax.axis("off")
    ax.text(0, -1.15, "rank at α = 0\n(most birds)", ha="center", va="bottom", fontsize=9.5, weight="bold")
    ax.text(1, -1.15, "rank at α = 1.5\n(favor specialties)", ha="center", va="bottom", fontsize=9.5, weight="bold")
    fig.subplots_adjust(top=0.82, bottom=0.12)
    fig.suptitle("One knob reorders the recommendations", x=0.005, y=0.99, ha="left",
                 fontsize=13, weight="bold")
    fig.text(0.005, 0.01, "Each line is a (place, week). Raising rarity-weighting (α) promotes coastal-specialty "
             "and spring-migration weeks (red)\ninto the top list, displacing the busiest sites (grey). "
             "α is continuous; 0 and 1.5 are just two stops on the slider.", fontsize=8.3, color=MUTE)
    save(fig, "fig4_alpha_slopegraph")

fig1(); fig2(); fig3(); fig4()
print("done ->", VIZ)
