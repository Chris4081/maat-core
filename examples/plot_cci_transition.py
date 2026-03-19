#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_cci_transition.py
----------------------
Creates a publication-ready plot from:
cci_critical_transition_results.csv

Outputs:
    cci_transition_plot.png
    cci_transition_plot.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("cci_critical_transition_results.csv")

    limit = df["limit"]
    cci = df["cci"]
    margin = df["margin"]

    # Transition point = CCI maximum
    idx_peak = cci.idxmax()
    limit_peak = float(limit.iloc[idx_peak])
    cci_peak = float(cci.iloc[idx_peak])

    fig, ax1 = plt.subplots(figsize=(8.6, 5.2))

    # --- Left axis: CCI ---
    line1 = ax1.plot(
        limit,
        cci,
        marker="o",
        linewidth=2.4,
        markersize=6,
        label="CCI"
    )
    ax1.set_xlabel("Constraint limit", fontsize=12)
    ax1.set_ylabel("Critical Coherence Index (CCI)", fontsize=12)
    ax1.set_title("CCI Response to Constraint-Induced Transition", fontsize=15, pad=12)
    ax1.grid(True, alpha=0.3)

    # Highlight transition point
    ax1.axvline(limit_peak, linestyle="--", alpha=0.6, label="Transition point")
    ax1.annotate(
        f"CCI peak\n(limit ≈ {limit_peak:.3f})",
        xy=(limit_peak, cci_peak),
        xytext=(limit_peak + 0.01, cci_peak - 0.03),
        arrowprops=dict(arrowstyle="->", lw=1.2),
        fontsize=10
    )

    # --- Right axis: margin ---
    ax2 = ax1.twinx()
    line2 = ax2.plot(
        limit,
        margin,
        linestyle="--",
        marker="x",
        linewidth=2.0,
        markersize=7,
        label="Constraint margin"
    )
    ax2.set_ylabel("Constraint margin", fontsize=12)

    # Critical margin threshold
    crit_margin = 0.03
    ax2.axhline(crit_margin, linestyle=":", linewidth=2, label="Critical margin threshold")

    # Regime shading
    ax1.axvspan(limit.min(), limit_peak, alpha=0.08)
    ax1.axvspan(limit_peak, limit.max(), alpha=0.04)

    # Regime labels
    ax1.text(limit.min() + 0.01, max(cci) * 0.92, "Constraint-dominated regime", fontsize=10)
    ax1.text(limit_peak + 0.005, max(cci) * 0.25, "Relaxed interior regime", fontsize=10)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    extra_labels = ["Transition point", "Critical margin threshold"]
    handles = [line1[0], line2[0]]
    handles += [
        plt.Line2D([0], [0], linestyle="--"),
        plt.Line2D([0], [0], linestyle=":")
    ]
    labels += extra_labels

    ax1.legend(handles, labels, loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig("cci_transition_plot.png", dpi=300, bbox_inches="tight")
    plt.savefig("cci_transition_plot.pdf", bbox_inches="tight")

    print("\nSaved plot: cci_transition_plot.png")
    print("Saved plot: cci_transition_plot.pdf")

    plt.show()


if __name__ == "__main__":
    main()