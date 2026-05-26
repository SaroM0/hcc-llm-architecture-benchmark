"""Analysis and visualization of A2 vs A3 benchmark results on real HCC SCT dataset."""

import json
import pathlib
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

BASE = pathlib.Path(__file__).parent
MATRIX = BASE / "matrix"
FIGS = BASE / "figures"
FIGS.mkdir(exist_ok=True)

SCORES = ["+2", "+1", "0", "-1", "-2"]
SCORE_VALS = {"+2": 2, "+1": 1, "0": 0, "-1": -1, "-2": -2}

# ── colours ──────────────────────────────────────────────────────────────────
C_A2 = "#2563EB"   # blue
C_A3 = "#DC2626"   # red
C_GOLD = "#6B7280" # grey

# ── load data ─────────────────────────────────────────────────────────────────
def load_run(run_dir: pathlib.Path):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    preds = [json.loads(l) for l in (run_dir / "predictions.jsonl").read_text().splitlines() if l.strip()]
    return manifest, preds

A2_DIR = MATRIX / "20260401_155744_A2_gpt52"
A3_DIR = MATRIX / "20260401_155822_A3_qwen3_vl_30b"

m2, p2 = load_run(A2_DIR)
m3, p3 = load_run(A3_DIR)

RUNS = [
    ("A2  GPT-5.2", m2, p2, C_A2),
    ("A3  Qwen-30B", m3, p3, C_A3),
]

# ── helpers ───────────────────────────────────────────────────────────────────
def gold(pred): return pred.get("expected_answer")
def pred(pred): return pred.get("predicted_answer")

def confusion(preds):
    mat = np.zeros((5, 5), dtype=int)
    for p in preds:
        g, pr = gold(p), pred(p)
        if g in SCORES and pr in SCORES:
            mat[SCORES.index(g)][SCORES.index(pr)] += 1
    return mat

def score_accuracy(preds):
    by_score = {s: {"n": 0, "correct": 0} for s in SCORES}
    for p in preds:
        g, pr = gold(p), pred(p)
        if g in SCORES:
            by_score[g]["n"] += 1
            if pr == g:
                by_score[g]["correct"] += 1
    return by_score

# ── figure 1: main metrics ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
fig.suptitle("A2 vs A3 — HCC SCT Benchmark (88 items)", fontsize=14, fontweight="bold", y=1.02)

metric_keys = ["accuracy", "partial_accuracy", "parse_rate"]
metric_labels = ["Exact Accuracy", "Within-1 Accuracy", "Parse Rate"]

for ax, key, label in zip(axes, metric_keys, metric_labels):
    vals = [m["metrics"][key] for _, m, _, _ in RUNS]
    colors = [c for _, _, _, c in RUNS]
    labels = [name for name, _, _, _ in RUNS]
    bars = ax.bar(labels, [v * 100 for v in vals], color=colors, width=0.45, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("%" if key != "parse_rate" else "%")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(50, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)

plt.tight_layout()
plt.savefig(FIGS / "01_main_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 01_main_metrics.png")

# ── figure 2: confusion matrices ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Confusion Matrices (rows = gold, cols = predicted)", fontsize=13, fontweight="bold")

for ax, (name, manifest, preds, color) in zip(axes, RUNS):
    mat = confusion(preds)
    n = mat.sum()
    norm = mat / n if n > 0 else mat
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=0.5)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(SCORES); ax.set_yticklabels(SCORES)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Gold", fontsize=11)
    acc = manifest["metrics"]["accuracy"]
    ax.set_title(f"{name}\nAccuracy {acc*100:.1f}%", fontsize=11, fontweight="bold", color=color)
    for i in range(5):
        for j in range(5):
            v = mat[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if norm[i, j] > 0.25 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(FIGS / "02_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 02_confusion_matrices.png")

# ── figure 3: per-score accuracy ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
gold_counts = collections.Counter(gold(p) for p in p2 if gold(p) in SCORES)
active_scores = [s for s in SCORES if gold_counts[s] > 0]
x = np.arange(len(active_scores))
width = 0.35

for i, (name, manifest, preds, color) in enumerate(RUNS):
    sa = score_accuracy(preds)
    accs = [sa[s]["correct"] / sa[s]["n"] * 100 if sa[s]["n"] > 0 else 0 for s in active_scores]
    bars = ax.bar(x + (i - 0.5) * width, accs, width, label=name, color=color,
                  edgecolor="white", linewidth=1)
    for bar, acc in zip(bars, accs):
        if acc > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{acc:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([f"{s}\n(n={gold_counts[s]})" for s in active_scores], fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_ylim(0, 120)
ax.set_title("Per-Score Accuracy (n = gold class size)", fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.axhline(50, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)

plt.tight_layout()
plt.savefig(FIGS / "03_per_score_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 03_per_score_accuracy.png")

# ── figure 4: score distributions ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Score Distributions", fontsize=13, fontweight="bold")

datasets = [
    ("Gold (ground truth)", collections.Counter(gold(p) for p in p2), C_GOLD),
    ("A2  GPT-5.2 predictions", collections.Counter(pred(p) for p in p2 if pred(p)), C_A2),
    ("A3  Qwen-30B predictions", collections.Counter(pred(p) for p in p3 if pred(p)), C_A3),
]

for ax, (title, counts, color) in zip(axes, datasets):
    vals = [counts.get(s, 0) for s in SCORES]
    total = sum(vals)
    bars = ax.bar(SCORES, vals, color=color, edgecolor="white", linewidth=1.2, alpha=0.85)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v}\n({v/total*100:.0f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.35 + 2)
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(FIGS / "04_score_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 04_score_distributions.png")

# ── figure 5: cost & latency ──────────────────────────────────────────────────
def cost_stats(preds):
    costs, latencies = [], []
    for p in preds:
        dbg = p.get("debug") or {}
        c = dbg.get("total_cost_usd", 0) or 0
        events = p.get("events") or []
        lat = sum(e.get("latency_ms", 0) or 0 for e in events if isinstance(e, dict))
        costs.append(c)
        latencies.append(lat / 1000)
    return costs, latencies

c2, l2 = cost_stats(p2)
c3, l3 = cost_stats(p3)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Cost & Latency per Item", fontsize=13, fontweight="bold")

# Cost
ax = axes[0]
for name, costs, color in [("A2 GPT-5.2", c2, C_A2), ("A3 Qwen-30B", c3, C_A3)]:
    valid = [c for c in costs if c > 0]
    if valid:
        ax.hist(valid, bins=20, color=color, alpha=0.6, label=f"{name} (mean ${np.mean(valid):.3f})")
ax.set_xlabel("Cost per item (USD)")
ax.set_ylabel("Count")
ax.set_title("Cost Distribution per Item")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Latency
ax = axes[1]
total_h = [m["metrics"]["total_items"] for _, m, _, _ in RUNS]
lat_h = [m["latency_seconds"] / 3600 for _, m, _, _ in RUNS]
names = [name for name, _, _, _ in RUNS]
colors = [c for _, _, _, c in RUNS]
bars = ax.bar(names, lat_h, color=colors, edgecolor="white", linewidth=1.2)
for bar, h, n in zip(bars, lat_h, total_h):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{h:.1f}h\n({h*60/n:.0f}m/item)", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Total wall-clock time (hours)")
ax.set_title("Total Run Duration")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(FIGS / "05_cost_latency.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 05_cost_latency.png")

# ── figure 6: summary dashboard ───────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor("#F9FAFB")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Title
fig.text(0.5, 0.97, "HCC LLM Benchmark — A2 vs A3 Results",
         ha="center", va="top", fontsize=16, fontweight="bold")
fig.text(0.5, 0.93, "Multi-Agent Consensus RAG · Real SCT Ground Truth · 88 items",
         ha="center", va="top", fontsize=11, color="#6B7280")

# Metric cards
metric_data = [
    ("Exact Accuracy", "accuracy", "%", 100),
    ("Within-1 Accuracy", "partial_accuracy", "%", 100),
    ("Parse Rate", "parse_rate", "%", 100),
]

for col, (label, key, unit, scale) in enumerate(metric_data):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor("white")
    vals = [m["metrics"][key] * scale for _, m, _, _ in RUNS]
    colors = [c for _, _, _, c in RUNS]
    names = [name for name, _, _, _ in RUNS]
    bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_title(label, fontsize=10, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

# Confusion A2
ax = fig.add_subplot(gs[1, 0])
mat = confusion(p2)
norm = mat / mat.sum()
im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=0.5)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(SCORES, fontsize=8); ax.set_yticklabels(SCORES, fontsize=8)
ax.set_title("A2 GPT-5.2  confusion", fontsize=10, fontweight="bold", color=C_A2)
for i in range(5):
    for j in range(5):
        if mat[i,j] > 0:
            ax.text(j, i, str(mat[i,j]), ha="center", va="center", fontsize=9,
                    color="white" if norm[i,j] > 0.25 else "black")

# Confusion A3
ax = fig.add_subplot(gs[1, 1])
mat = confusion(p3)
norm = mat / mat.sum()
im = ax.imshow(norm, cmap="Reds", vmin=0, vmax=0.5)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(SCORES, fontsize=8); ax.set_yticklabels(SCORES, fontsize=8)
ax.set_title("A3 Qwen-30B  confusion", fontsize=10, fontweight="bold", color=C_A3)
for i in range(5):
    for j in range(5):
        if mat[i,j] > 0:
            ax.text(j, i, str(mat[i,j]), ha="center", va="center", fontsize=9,
                    color="white" if norm[i,j] > 0.25 else "black")

# Summary table
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
rows = [
    ["Metric", "A2 GPT-5.2", "A3 Qwen-30B"],
    ["Items", "88", "88"],
    ["Exact Acc.", "67.1%", "42.1%"],
    ["Within-1", "90.9%", "73.9%"],
    ["Parse Rate", "100%", "88.6%"],
    ["MAE", "0.48", "0.74"],
    ["Duration", "10.3 h", "21.1 h"],
]
table = ax.table(cellText=rows[1:], colLabels=rows[0],
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#E5E7EB")
    if r == 0:
        cell.set_facecolor("#1E3A5F")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == 1 and r > 0:
        cell.set_facecolor("#DBEAFE")
    elif c == 2 and r > 0:
        cell.set_facecolor("#FEE2E2")
ax.set_title("Summary", fontsize=10, fontweight="bold", pad=8)

plt.savefig(FIGS / "06_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 06_dashboard.png")

# ── console summary ───────────────────────────────────────────────────────────
print()
print("=" * 60)
print("RESULTS — HCC SCT Benchmark (88 real items)")
print("=" * 60)
print(f"\nGold distribution: {dict(sorted(collections.Counter(gold(p) for p in p2 if gold(p)).items()))}")
for name, manifest, preds, _ in RUNS:
    m = manifest["metrics"]
    lat_h = manifest["latency_seconds"] / 3600
    print(f"\n{name}")
    print(f"  Accuracy:    {m['accuracy']*100:.1f}%   (exact: {m['exact_matches']}/{m['total_items']})")
    print(f"  Within-1:    {m['partial_accuracy']*100:.1f}%   ({m['within_one']}/{m['total_items']})")
    print(f"  Parse rate:  {m['parse_rate']*100:.1f}%   ({m['parsed_items']}/{m['total_items']})")
    print(f"  MAE:         {m['mean_absolute_error']:.3f}")
    print(f"  Duration:    {lat_h:.1f} h  ({lat_h*60/m['total_items']:.0f} min/item)")
    print(f"  Pred dist.:  {m['score_distribution']}")

print(f"\nFigures → {FIGS}/")
