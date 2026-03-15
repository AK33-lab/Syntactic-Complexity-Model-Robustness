"""Scatter plots: complexity metrics (X) vs model error rate (Y)."""
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── load data ────────────────────────────────────────────────────────────────
complex_df = pd.read_csv(os.path.join(SCRIPT_DIR, "complex.csv"))
perf_df = pd.read_csv(os.path.join(SCRIPT_DIR, "perf.csv"))

# Pivot complexity to wide: index = perturbation method, columns = metric
complexity = complex_df.pivot(
    index="perturbation method", columns="metric type", values="value"
).reset_index()

# Compute error rate
perf_df["error_rate"] = 1.0 - perf_df["performance"]

# Merge on perturbation method (inner join drops 'pp' if not in complex.csv)
merged = perf_df.merge(complexity, on="perturbation method")

metrics = ["clause_count", "subj_verb_dist", "cfg_depth"]
metric_labels = {
    "clause_count": "Clause Count",
    "subj_verb_dist": "Subject–Verb Distance",
    "cfg_depth": "CFG Tree Depth",
}

models = sorted(merged["model"].unique())
colors = {"mlp": "#e15759", "rnn": "#f28e2b", "roberta": "#4e79a7", "bart": "#76b7b2"}
markers = {"mlp": "o", "rnn": "s", "roberta": "^", "bart": "D"}

# ── one figure with 3 subplots ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Complexity Metrics vs Model Error Rate", fontsize=14, fontweight="bold")

for ax, metric in zip(axes, metrics):
    for model in models:
        subset = merged[merged["model"] == model]
        ax.scatter(
            subset[metric],
            subset["error_rate"],
            label=model.upper(),
            color=colors.get(model, "gray"),
            marker=markers.get(model, "o"),
            s=90,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        # Annotate each point with the perturbation method name
        for _, row in subset.iterrows():
            ax.annotate(
                row["perturbation method"],
                xy=(row[metric], row["error_rate"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6.5,
                color=colors.get(model, "gray"),
                alpha=0.85,
            )

    ax.set_xlabel(metric_labels[metric], fontsize=11)
    ax.set_ylabel("Error Rate (1 − Accuracy)", fontsize=11)
    ax.set_title(metric_labels[metric], fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)

# Single shared legend outside the subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    title="Model",
    loc="lower center",
    ncol=len(models),
    bbox_to_anchor=(0.5, -0.08),
    frameon=True,
)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "complexity_vs_error.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
