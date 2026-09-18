"""Generate the two figures the analysis needs that the pipeline did not produce.

  tsocial_validation_curve.png  CARE-GNN's T-Social run was still improving when
                                the session cap stopped it. The curve is the
                                clearest way to show that it was a time wall
                                rather than a model that failed to learn.
  efficiency_tradeoff.png       Per-epoch time against peak GPU memory. The two
                                models sit at opposite corners, and a scatter
                                carries that far better than two tables.

Both read from ../tables and ../evidence. Run:  python build_figures.py
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DELIVERABLE = os.path.dirname(HERE)
TABLES = os.path.join(DELIVERABLE, "tables")
FIGURES = os.path.join(DELIVERABLE, "figures")
EVIDENCE = os.path.join(DELIVERABLE, "evidence")

CARE = "#1F3B63"
GHRN = "#C2571A"
GRID = "#D8DEE8"
INK = "#22272E"


def style(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------
def tsocial_curve():
    path = os.path.join(EVIDENCE, "tsocial", "final_seed2", "validation_metrics.csv")
    if not os.path.isfile(path):
        print("skip tsocial curve: no validation_metrics.csv")
        return None
    rows = read_csv(path)
    epochs = [int(r["epoch"]) for r in rows]
    auprc = [float(r["auprc"]) for r in rows]
    auroc = [float(r["auroc"]) for r in rows]
    best = max(range(len(auprc)), key=lambda i: auprc[i])

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.2, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.16})

    top.plot(epochs, auprc, color=CARE, linewidth=2, marker="o", markersize=4)
    top.scatter([epochs[best]], [auprc[best]], s=110, facecolor="white",
                edgecolor=CARE, linewidth=2, zorder=5)
    # Below-left of the marker: the point sits top-right, so anything above it
    # collides with the subtitle.
    top.annotate(f"best  {auprc[best]:.4f}\nepoch {epochs[best]}",
                 xy=(epochs[best], auprc[best]), xytext=(-84, -34),
                 textcoords="offset points", fontsize=9, color=CARE, weight="bold",
                 arrowprops=dict(arrowstyle="-", color=CARE, linewidth=1, alpha=0.6))
    top.set_ylabel("Validation AUPRC", fontsize=10, color=INK)
    top.set_ylim(min(auprc) - 0.008, max(auprc) + 0.012)
    style(top)

    bottom.plot(epochs, auroc, color=GHRN, linewidth=2, marker="o", markersize=4)
    bottom.set_ylabel("Validation AUROC", fontsize=10, color=INK)
    bottom.set_xlabel("Training epoch", fontsize=10, color=INK)
    style(bottom)

    for ax in (top, bottom):
        ax.axvspan(epochs[-1] + 0.35, epochs[-1] + 1.3, color="#F0D9D2", alpha=0.85)
    top.annotate("session cap\nreached here",
                 xy=(epochs[-1] + 0.85, max(auprc) * 0.62),
                 ha="center", fontsize=8.5, color="#8A3B1E", weight="bold")
    top.set_xlim(-0.6, epochs[-1] + 1.4)

    fig.suptitle("CARE-GNN on T-Social: still improving when the run was stopped",
                 fontsize=12.5, color=INK, weight="bold", x=0.512, y=0.985)
    fig.text(0.512, 0.930,
             "16 epochs over 1 h 55 m on a Tesla T4. No test evaluation was ever "
             "reached, so no T-Social result is reported.",
             ha="center", fontsize=9, color="#55606E")
    fig.subplots_adjust(top=0.875, bottom=0.10, left=0.095, right=0.975, hspace=0.16)
    out = os.path.join(FIGURES, "tsocial_validation_curve.png")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
def efficiency_tradeoff():
    rows = read_csv(os.path.join(TABLES, "efficiency.csv"))
    agg = {(r["model"], r["dataset"]): r
           for r in read_csv(os.path.join(TABLES, "master_mean_sd.csv"))}

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for model, colour in (("CARE-GNN", CARE), ("GHRN", GHRN)):
        xs, ys, names = [], [], []
        for r in rows:
            if r["model"] != model:
                continue
            raw = agg[(r["model"], r["dataset"])]
            xs.append(float(raw["mean_epoch_s"]))
            ys.append(float(raw["peak_gpu_mb"]))
            names.append(r["dataset"])
        marker = "o" if model == "CARE-GNN" else "s"
        ax.scatter(xs, ys, s=95, color=colour, marker=marker, label=model,
                   edgecolor="white", linewidth=1.2, zorder=4)
        # Points within a model cluster sit close together on the log axes, so
        # alternate the label above and below to stop neighbours colliding.
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        for rank, i in enumerate(order):
            dy = 10 if rank % 2 == 0 else -17
            align = "left" if rank == 0 else "center"
            dx = -6 if rank == 0 else 0
            ax.annotate(names[i], xy=(xs[i], ys[i]), xytext=(dx, dy),
                        textcoords="offset points", ha=align, fontsize=8,
                        color=colour)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean training time per epoch (s, log scale)", fontsize=10, color=INK)
    ax.set_ylabel("Peak GPU memory (MB, log scale)", fontsize=10, color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    ax.set_title("The trade is time against memory",
                 fontsize=12.5, color=INK, weight="bold", loc="left", pad=26)
    ax.text(0, 1.035,
            "GHRN buys speed with memory; CARE-GNN buys memory with time. "
            "Lower-left is cheaper on both axes.",
            transform=ax.transAxes, fontsize=9, color="#55606E")
    fig.tight_layout()
    out = os.path.join(FIGURES, "efficiency_tradeoff.png")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def main():
    os.makedirs(FIGURES, exist_ok=True)
    for produced in (tsocial_curve(), efficiency_tradeoff()):
        if produced:
            print("wrote", os.path.relpath(produced, DELIVERABLE))


if __name__ == "__main__":
    main()
