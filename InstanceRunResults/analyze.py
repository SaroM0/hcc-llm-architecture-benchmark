"""Analysis and visualizations for HCC LLM Architecture Benchmark."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import Counter
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent / "hcc-results" / "results"
OUT  = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

RUNS = {
    "A2\nGPT-5.2\n(large)": BASE / "arm=A2/run=20260328_041556_A2_gpt52/model=gpt52.jsonl",
    "A3\nQwen-30B\n(best·RAG)": BASE / "arm=A3/run=20260329_230113_A3_qwen3_vl_30b/model=qwen3_vl_30b.jsonl",
    "A3\nQwen-30B\nreasoning=high": BASE / "arm=A3/run=20260322_205349_A3_qwen_reasoning_high/model=qwen3_vl_30b.jsonl",
    "A3\nQwen-30B\nreasoning=low": BASE / "arm=A3/run=20260322_205349_A3_qwen_reasoning_low/model=qwen3_vl_30b.jsonl",
}

SCORES = ["-2", "-1", "0", "+1", "+2"]
COLORS = {
    "A2\nGPT-5.2\n(large)":          "#2563EB",
    "A3\nQwen-30B\n(best·RAG)":       "#DC2626",
    "A3\nQwen-30B\nreasoning=high":   "#D97706",
    "A3\nQwen-30B\nreasoning=low":    "#16A34A",
}
PALETTE = ["#2563EB", "#DC2626", "#D97706", "#16A34A"]

# ── Load data ─────────────────────────────────────────────────────────────────
def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]

data = {name: load(path) for name, path in RUNS.items()}

# ── Helpers ───────────────────────────────────────────────────────────────────
def metrics(records):
    n = len(records)
    correct  = sum(1 for r in records if r["scoring"]["is_correct"] == 1)
    partial  = sum(1 for r in records if r["scoring"]["is_partial"] == 1)
    parse_ok = sum(1 for r in records if not r["errors"]["parse_error"])
    costs    = [r["cost_usd"] for r in records if r["cost_usd"]]
    lats_s   = [r["latency_ms"] / 1000 for r in records if r["latency_ms"]]
    return {
        "n": n,
        "accuracy":       correct / n,
        "within_one":     partial / n,
        "parse_rate":     parse_ok / n,
        "cost_total":     sum(costs),
        "cost_per_item":  sum(costs) / len(costs) if costs else 0,
        "lat_avg":        sum(lats_s) / len(lats_s) if lats_s else 0,
        "lat_p95":        sorted(lats_s)[int(0.95 * len(lats_s))] if lats_s else 0,
    }

def confusion_matrix(records):
    mat = np.zeros((5, 5), dtype=int)
    idx = {s: i for i, s in enumerate(SCORES)}
    for r in records:
        g = r["gold"]["label"]
        p = r["scoring"]["pred_label"]
        if g in idx and p in idx:
            mat[idx[g]][idx[p]] += 1
    return mat

def per_score_acc(records):
    accs = {}
    for s in SCORES:
        sub = [r for r in records if r["gold"]["label"] == s]
        accs[s] = (sum(1 for r in sub if r["scoring"]["is_correct"] == 1) / len(sub)
                   if sub else np.nan)
    return accs

stats = {name: metrics(recs) for name, recs in data.items()}
names = list(data.keys())

# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — Main comparison: accuracy + within-1 + parse rate
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("HCC LLM Architecture Benchmark — Main Metrics", fontsize=14, fontweight="bold", y=1.01)

short = [n.replace("\n", " ") for n in names]
x = np.arange(len(names))
w = 0.6

for ax, key, title, fmt in [
    (axes[0], "accuracy",   "Exact Match Accuracy",    ".1%"),
    (axes[1], "within_one", "Within-1 Accuracy\n(|pred−gold| ≤ 1)", ".1%"),
    (axes[2], "parse_rate", "Parse Rate\n(score extracted successfully)", ".1%"),
]:
    vals = [stats[n][key] for n in names]
    bars = ax.bar(x, vals, width=w, color=PALETTE, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, labels=[f"{v:{fmt}}" for v in vals], padding=4, fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

plt.tight_layout()
fig.savefig(OUT / "01_main_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 01_main_metrics.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — Confusion matrices (A2_gpt52 and A3_qwen_low, the two main runs)
# ═════════════════════════════════════════════════════════════════════════════
key_runs = ["A2\nGPT-5.2\n(large)", "A3\nQwen-30B\nreasoning=low"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Confusion Matrices — Predicted vs Gold Likert Score", fontsize=13, fontweight="bold")

for ax, name in zip(axes, key_runs):
    mat = confusion_matrix(data[name])
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(SCORES); ax.set_yticklabels(SCORES)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Gold", fontsize=11)
    ax.set_title(name.replace("\n", " "), fontsize=11, fontweight="bold")
    for i in range(5):
        for j in range(5):
            v = mat[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=11 if v > 0 else 9,
                    color="white" if v > mat.max() * 0.6 else "black",
                    fontweight="bold" if i == j else "normal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
fig.savefig(OUT / "02_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 02_confusion_matrices.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — Per-score accuracy breakdown (grouped bars)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle("Per-Score Exact Match Accuracy by Run", fontsize=13, fontweight="bold")

n_runs = len(names)
x = np.arange(5)
total_w = 0.75
w = total_w / n_runs
offsets = np.linspace(-(total_w - w) / 2, (total_w - w) / 2, n_runs)

for i, (name, color) in enumerate(zip(names, PALETTE)):
    accs_d = per_score_acc(data[name])
    vals = [accs_d[s] for s in SCORES]
    bars = ax.bar(x + offsets[i], vals, width=w, color=color,
                  label=name.replace("\n", " "), edgecolor="white", linewidth=0.5, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(SCORES, fontsize=12)
ax.set_xlabel("Gold Likert Score", fontsize=11)
ax.set_ylabel("Exact Match Accuracy", fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5, label="50% baseline")
ax.legend(fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

# annotate gold class sizes (from A2 run as reference)
gold_counts = Counter(r["gold"]["label"] for r in data["A2\nGPT-5.2\n(large)"])
for xi, s in enumerate(SCORES):
    ax.text(xi, -0.07, f"n={gold_counts[s]}", ha="center", fontsize=8, color="grey",
            transform=ax.get_xaxis_transform())

plt.tight_layout()
fig.savefig(OUT / "03_per_score_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 03_per_score_accuracy.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 — Score distribution: Gold vs Predicted
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Score Distribution: Gold vs Predicted", fontsize=13, fontweight="bold")
axes = axes.flatten()

for ax, (name, recs) in zip(axes, data.items()):
    gold_c = Counter(r["gold"]["label"] for r in recs if r["gold"]["label"])
    pred_c = Counter(r["scoring"]["pred_label"] for r in recs
                     if r["scoring"]["pred_label"] and r["scoring"]["pred_label"] != "None")
    x = np.arange(5)
    w = 0.35
    gold_vals = [gold_c.get(s, 0) for s in SCORES]
    pred_vals = [pred_c.get(s, 0) for s in SCORES]
    ax.bar(x - w/2, gold_vals, w, label="Gold (ground truth)", color="#374151", alpha=0.85)
    ax.bar(x + w/2, pred_vals, w, label="Predicted", color=COLORS[name], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(SCORES)
    ax.set_title(name.replace("\n", " ") + f"  (n={len(recs)})", fontsize=10, fontweight="bold")
    ax.set_ylabel("Count"); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "04_score_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 04_score_distributions.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 5 — Cost vs Accuracy scatter + latency
# ═════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Cost & Latency vs Performance", fontsize=13, fontweight="bold")

# Cost per item vs accuracy
for name, color in zip(names, PALETTE):
    s = stats[name]
    ax1.scatter(s["cost_per_item"], s["accuracy"], s=160, color=color,
                zorder=5, edgecolors="white", linewidths=1.5)
    ax1.annotate(name.replace("\n", " "), (s["cost_per_item"], s["accuracy"]),
                 textcoords="offset points", xytext=(8, 4), fontsize=7.5)

ax1.set_xlabel("Cost per Item (USD)", fontsize=11)
ax1.set_ylabel("Exact Match Accuracy", fontsize=11)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax1.set_title("Cost-Efficiency", fontsize=11, fontweight="bold")
ax1.spines[["top", "right"]].set_visible(False)

# Avg latency bar
lat_vals = [stats[n]["lat_avg"] for n in names]
lat_p95  = [stats[n]["lat_p95"] for n in names]
x = np.arange(len(names))
bars = ax2.bar(x, lat_vals, 0.6, color=PALETTE, edgecolor="white", label="Avg latency")
ax2.errorbar(x, lat_vals, yerr=[np.zeros(len(names)),
             [p - a for p, a in zip(lat_p95, lat_vals)]],
             fmt="none", color="black", capsize=5, linewidth=1.5, label="P95")
ax2.set_xticks(x)
ax2.set_xticklabels([n.replace("\n", " ") for n in names], fontsize=8)
ax2.set_ylabel("Latency (seconds)", fontsize=11)
ax2.set_title("Avg Latency per Item (+ P95)", fontsize=11, fontweight="bold")
ax2.bar_label(bars, labels=[f"{v:.0f}s" for v in lat_vals], padding=4, fontsize=9)
ax2.legend(fontsize=9)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "05_cost_latency.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 05_cost_latency.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 6 — Summary dashboard
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("HCC LLM Benchmark — Summary Dashboard", fontsize=15, fontweight="bold", y=1.01)

# 6a. Accuracy + Within-1 dual bar
ax = fig.add_subplot(gs[0, :2])
x = np.arange(len(names))
w = 0.35
b1 = ax.bar(x - w/2, [stats[n]["accuracy"]   for n in names], w, color=PALETTE, alpha=1.0,   label="Exact Match")
b2 = ax.bar(x + w/2, [stats[n]["within_one"] for n in names], w, color=PALETTE, alpha=0.45,  label="Within-1", hatch="//")
ax.set_xticks(x); ax.set_xticklabels([n.replace("\n", " ") for n in names], fontsize=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("Accuracy vs Within-1", fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=9)
for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                      f"{b.get_height():.1%}", ha="center", fontsize=8, fontweight="bold")

# 6b. Cost total
ax2 = fig.add_subplot(gs[0, 2])
costs = [stats[n]["cost_total"] for n in names]
bars = ax2.bar(range(len(names)), costs, color=PALETTE, edgecolor="white")
ax2.set_xticks(range(len(names)))
ax2.set_xticklabels([n.split("\n")[0]+"\n"+n.split("\n")[1] for n in names], fontsize=8)
ax2.bar_label(bars, labels=[f"${v:.1f}" for v in costs], padding=3, fontsize=9, fontweight="bold")
ax2.set_title("Total API Cost (USD)", fontsize=11, fontweight="bold")
ax2.set_ylabel("USD")
ax2.spines[["top", "right"]].set_visible(False)

# 6c. Confusion (A2 only — best full run)
ax3 = fig.add_subplot(gs[1, 0])
mat = confusion_matrix(data["A2\nGPT-5.2\n(large)"])
im = ax3.imshow(mat, cmap="Blues")
ax3.set_xticks(range(5)); ax3.set_yticks(range(5))
ax3.set_xticklabels(SCORES, fontsize=9); ax3.set_yticklabels(SCORES, fontsize=9)
ax3.set_xlabel("Predicted", fontsize=9); ax3.set_ylabel("Gold", fontsize=9)
ax3.set_title("A2 GPT-5.2 — Confusion", fontsize=10, fontweight="bold")
for i in range(5):
    for j in range(5):
        v = mat[i, j]
        ax3.text(j, i, str(v), ha="center", va="center", fontsize=9,
                 color="white" if v > mat.max() * 0.6 else "black",
                 fontweight="bold" if i == j else "normal")

# 6d. Confusion (A3 qwen low — best A3)
ax4 = fig.add_subplot(gs[1, 1])
mat4 = confusion_matrix(data["A3\nQwen-30B\nreasoning=low"])
im4 = ax4.imshow(mat4, cmap="Oranges")
ax4.set_xticks(range(5)); ax4.set_yticks(range(5))
ax4.set_xticklabels(SCORES, fontsize=9); ax4.set_yticklabels(SCORES, fontsize=9)
ax4.set_xlabel("Predicted", fontsize=9); ax4.set_ylabel("Gold", fontsize=9)
ax4.set_title("A3 Qwen reasoning=low — Confusion", fontsize=10, fontweight="bold")
for i in range(5):
    for j in range(5):
        v = mat4[i, j]
        ax4.text(j, i, str(v), ha="center", va="center", fontsize=9,
                 color="white" if v > mat4.max() * 0.6 else "black",
                 fontweight="bold" if i == j else "normal")

# 6e. Stats table
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
table_data = [
    ["Run", "N", "Acc", "±1", "Cost"],
    ["A2 GPT-5.2", "382", "57.3%", "90.8%", "$18.2"],
    ["A3 Qwen best", "151*", "33.1%", "87.4%", "$9.0"],
    ["A3 Qwen high", "396†", "51.8%", "80.1%", "$8.2"],
    ["A3 Qwen low", "396†", "54.8%", "80.6%", "$9.9"],
]
tbl = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
tbl.scale(1.1, 1.6)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1E3A5F"); cell.set_text_props(color="white", fontweight="bold")
    elif r == 1:
        cell.set_facecolor("#DBEAFE")
    elif r in (2, 4):
        cell.set_facecolor("#FEF3C7")
    cell.set_edgecolor("#E5E7EB")
ax5.set_title("* incomplete  † old dataset (396 items)", fontsize=7, color="grey", pad=2)

plt.savefig(OUT / "06_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 06_dashboard.png")

# ═════════════════════════════════════════════════════════════════════════════
# Print summary
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("BENCHMARK SUMMARY")
print("="*60)
for name in names:
    s = stats[name]
    print(f"\n{name.replace(chr(10),' ')}")
    print(f"  Items:     {s['n']}")
    print(f"  Accuracy:  {s['accuracy']:.1%}  |  Within-1: {s['within_one']:.1%}")
    print(f"  Parse:     {s['parse_rate']:.1%}")
    print(f"  Cost:      ${s['cost_total']:.2f} total  /  ${s['cost_per_item']:.4f} per item")
    print(f"  Latency:   {s['lat_avg']:.0f}s avg  /  {s['lat_p95']:.0f}s P95")

print(f"\nFigures saved to: {OUT}/")
