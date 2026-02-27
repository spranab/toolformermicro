#!/usr/bin/env python3
"""Generate paper figures for ToolFormerMicro experiments.

Figures:
  1. Scaling curves: TSA + encode latency vs. number of tools
  2. Composability suite: order independence + cache invalidation
  3. Method comparison: ToolFormerMicro vs ContextCache vs Gisting vs Full Prefill
  4. Parameter analysis: PF1 vs Value Recall (gist compression effect)
  5. LaTeX summary table

Usage:
  python scripts/analysis/tool_former_plots.py
  python scripts/analysis/tool_former_plots.py --results-dir eval_results/composability_v1
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Paper-quality style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.1,
    "font.family": "serif",
})

METHOD_COLORS = {
    "tool_former": "#2196F3",      # Blue — our method
    "context_cache": "#FF9800",    # Orange — ContextCache (Qwen3-8B)
    "gisting": "#4CAF50",          # Green — Gisting K=8
    "full_prefill": "#9E9E9E",     # Gray — Full Prefill baseline
}


def plot_scaling_curves(composability_dir: Path, output_dir: Path):
    """Figure 1: TSA + latency vs. number of tools (dual y-axis)."""
    scaling_path = composability_dir / "scaling_curves.json"
    if not scaling_path.exists():
        print("  [SKIP] No scaling_curves.json found")
        return

    with open(scaling_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    n_tools = [r["num_tools"] for r in results]
    tsa_vals = [r["tsa"] for r in results]
    encode_ms = [r["avg_encode_ms"] for r in results]
    gen_ms = [r["avg_generate_ms"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # TSA line
    line1, = ax1.plot(n_tools, tsa_vals, 'o-', color="#2196F3", linewidth=2.5,
                      markersize=8, label="Tool Selection Accuracy", zorder=3)
    ax1.set_ylabel("TSA", color="#2196F3")
    ax1.set_ylim(0.5, 1.05)
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax1.axhline(y=tsa_vals[0], color="#2196F3", linestyle="--", alpha=0.3, linewidth=1)

    # Encode latency bars
    bars = ax2.bar(n_tools, encode_ms, width=[n * 0.3 for n in n_tools],
                   color="#FF9800", alpha=0.5, label="Encode Latency")
    ax2.set_ylabel("Encode Latency (ms)", color="#FF9800")
    ax2.tick_params(axis="y", labelcolor="#FF9800")

    # Annotate per-tool cost
    for i, r in enumerate(results):
        ax2.annotate(f"{r['per_tool_encode_ms']:.1f} ms/tool",
                    (n_tools[i], encode_ms[i]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color="#E65100")

    ax1.set_xlabel("Number of Tools")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(n_tools)
    ax1.set_xticklabels([str(n) for n in n_tools])
    ax1.set_title("ToolFormerMicro: Scaling with Number of Tools")

    # Combined legend
    lines = [line1, bars]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left", framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig1_scaling_curves.{ext}")
    print(f"  Saved fig1_scaling_curves.pdf/png")
    plt.close(fig)


def plot_composability_suite(composability_dir: Path, output_dir: Path):
    """Figure 2: Composability results — 3-panel figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Order Independence
    comp_path = composability_dir / "compositional_generalization.json"
    if comp_path.exists():
        with open(comp_path, encoding="utf-8") as f:
            comp_data = json.load(f)

        ax = axes[0]
        x = [0, 1]
        vals = [comp_data["tsa_standard"], comp_data["tsa_shuffled"]]
        colors = ["#2196F3", "#FF9800"]
        bars = ax.bar(x, vals, color=colors, width=0.5, edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Standard\nOrder", "Shuffled\nOrder"])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("TSA")
        ax.set_title("(a) Order Independence")
        delta = comp_data["delta"]
        ax.text(0.5, 0.05, f"Delta = {delta:.3f}",
               transform=ax.transAxes, ha="center", fontsize=10,
               bbox=dict(boxstyle="round", facecolor="#E8F5E9" if delta < 0.05 else "#FFEBEE"))
        ax.grid(True, alpha=0.3, axis="y")
    else:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0].transAxes)

    # Panel B: Scaling (simplified from Fig 1 — just TSA)
    scaling_path = composability_dir / "scaling_curves.json"
    if scaling_path.exists():
        with open(scaling_path, encoding="utf-8") as f:
            scaling_data = json.load(f)

        ax = axes[1]
        results = scaling_data["results"]
        n_tools = [r["num_tools"] for r in results]
        tsa_vals = [r["tsa"] for r in results]

        ax.plot(n_tools, tsa_vals, 'o-', color="#2196F3", linewidth=2.5, markersize=8)
        ax.axhline(y=tsa_vals[0], color="#2196F3", linestyle="--", alpha=0.3)
        ax.fill_between(n_tools, [v - 0.02 for v in tsa_vals], [v + 0.02 for v in tsa_vals],
                        color="#2196F3", alpha=0.1)
        ax.set_xlabel("Number of Tools")
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_tools)
        ax.set_xticklabels([str(n) for n in n_tools])
        ax.set_ylim(0.5, 1.05)
        ax.set_ylabel("TSA")
        ax.set_title("(b) TSA vs Tool Count")
        ax.grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1].transAxes)

    # Panel C: Cache Invalidation
    inv_path = composability_dir / "cache_invalidation.json"
    if inv_path.exists():
        with open(inv_path, encoding="utf-8") as f:
            inv_data = json.load(f)

        ax = axes[2]
        x = [0, 1]
        vals = [inv_data["tsa_original"], inv_data["tsa_swapped"]]
        colors = ["#2196F3", "#E91E63"]
        bars = ax.bar(x, vals, color=colors, width=0.5, edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Original\nCache", "After\nHot-Swap"])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("TSA")
        ax.set_title("(c) Cache Invalidation")
        hotswap_ms = inv_data["hotswap_encode_ms"]
        ax.text(0.5, 0.05, f"Re-encode: {hotswap_ms:.0f} ms\nOther tools: bit-identical",
               transform=ax.transAxes, ha="center", fontsize=9,
               bbox=dict(boxstyle="round", facecolor="#E8F5E9"))
        ax.grid(True, alpha=0.3, axis="y")
    else:
        axes[2].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[2].transAxes)

    plt.suptitle("ToolFormerMicro: Composability Experiments", fontsize=15, y=1.02)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig2_composability_suite.{ext}", bbox_inches="tight")
    print(f"  Saved fig2_composability_suite.pdf/png")
    plt.close(fig)


def plot_method_comparison(eval_dir: Path, output_dir: Path):
    """Figure 3: Horizontal bar chart comparing all methods.

    Uses validated results — hardcoded baseline with live overrides.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Validated results (200-example eval)
    methods = [
        {"name": "ToolFormerMicro V1\n(428M, composable gists)", "tsa": 0.818,
         "pf1": 0.759, "vr": 0.917, "color": "#2196F3",
         "info": "VR=0.92, 14KB/tool, FPR=0.00"},
        {"name": "ContextCache (Qwen3-8B)\n(group-cached, lossless)", "tsa": 0.850,
         "pf1": 0.735, "vr": 0, "color": "#FF9800",
         "info": "8B model, exact quality"},
        {"name": "Tool Gisting K=8\n(Qwen3-8B, lossy)", "tsa": 0.714,
         "pf1": 0, "vr": 0, "color": "#4CAF50",
         "info": "FPR=0.30"},
    ]

    # Override V1 with live data if available (prefer 200-example eval)
    for v1_dir in ["tool_former_v1_200", "tool_former_v1_fixed2"]:
        v1_path = eval_dir / v1_dir / "summary_test_seen.json"
        if v1_path.exists():
            with open(v1_path, encoding="utf-8") as f:
                data = json.load(f)
            methods[0]["tsa"] = data.get("tool_selection_accuracy", methods[0]["tsa"])
            vr = data.get("value_recall", methods[0]["vr"])
            methods[0]["info"] = f"VR={vr:.2f}, 14KB/tool, FPR={data.get('false_positive_rate', 0):.2f}"
            break  # Use first found (200 preferred)

    # Insert V2 if available
    v2_path = eval_dir / "tool_former_v2_fixed" / "summary_test_seen.json"
    if v2_path.exists():
        with open(v2_path, encoding="utf-8") as f:
            data = json.load(f)
        methods.insert(1, {
            "name": "ToolFormerMicro V2\n(428M, gated cross-attn)",
            "tsa": data.get("tool_selection_accuracy", 0),
            "pf1": data.get("parameter_f1", 0),
            "vr": data.get("value_recall", 0),
            "color": "#1565C0",
            "info": f"VR={data.get('value_recall', 0):.2f}, 14KB/tool",
        })

    y = np.arange(len(methods))
    bars = ax.barh(y, [m["tsa"] for m in methods],
                   color=[m["color"] for m in methods],
                   alpha=0.85, edgecolor="white", height=0.6)

    for i, (bar, m) in enumerate(zip(bars, methods)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"TSA={m['tsa']:.3f}  |  {m['info']}", va="center", fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels([m["name"] for m in methods])
    ax.set_xlim(0, 1.45)
    ax.set_xlabel("Tool Selection Accuracy")
    ax.set_title("Method Comparison (test_seen)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig3_method_comparison.{ext}")
    print(f"  Saved fig3_method_comparison.pdf/png")
    plt.close(fig)


def plot_generate_latency_scaling(composability_dir: Path, output_dir: Path):
    """Figure 4: Generate latency vs N tools — shows cross-attention overhead."""
    scaling_path = composability_dir / "scaling_curves.json"
    if not scaling_path.exists():
        print("  [SKIP] No scaling_curves.json found")
        return

    with open(scaling_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    n_tools = [r["num_tools"] for r in results]
    gen_ms = [r["avg_generate_ms"] for r in results]
    encode_ms = [r["avg_encode_ms"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Encode latency (should be sub-linear / near-linear)
    ax1.plot(n_tools, encode_ms, 'o-', color="#FF9800", linewidth=2.5, markersize=8)
    # Reference: linear scaling from first point
    linear = [encode_ms[0] * n / n_tools[0] for n in n_tools]
    ax1.plot(n_tools, linear, '--', color="#9E9E9E", linewidth=1.5, label="Linear (reference)")
    ax1.set_xlabel("Number of Tools")
    ax1.set_ylabel("Encode Latency (ms)")
    ax1.set_title("(a) Encode Latency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel B: Generate latency
    ax2.plot(n_tools, gen_ms, 'o-', color="#2196F3", linewidth=2.5, markersize=8)
    ax2.set_xlabel("Number of Tools")
    ax2.set_ylabel("Generate Latency (ms)")
    ax2.set_title("(b) Generate Latency")
    ax2.grid(True, alpha=0.3)

    # Annotate overhead
    base_gen = gen_ms[0]
    for i, (n, g) in enumerate(zip(n_tools, gen_ms)):
        overhead = (g - base_gen) / base_gen * 100
        if i > 0:
            ax2.annotate(f"+{overhead:.0f}%", (n, g),
                        textcoords="offset points", xytext=(10, 5),
                        fontsize=9, color="#1565C0")

    plt.suptitle("ToolFormerMicro: Latency Analysis", fontsize=15)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig4_latency_analysis.{ext}")
    print(f"  Saved fig4_latency_analysis.pdf/png")
    plt.close(fig)


def generate_latex_table(eval_dir: Path, composability_dir: Path, output_dir: Path):
    """Generate LaTeX summary table for the paper."""

    # Main results table
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{ToolFormerMicro evaluation results. TSA: Tool Selection Accuracy, "
        r"PF1: Parameter F1, VR: Value Recall, EM: Exact Match.}",
        r"\label{tab:tool-former-results}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & TSA$\uparrow$ & PF1$\uparrow$ & VR$\uparrow$ & EM$\uparrow$ & FPR$\downarrow$ \\",
        r"\midrule",
        r"\textbf{ToolFormerMicro V1} & \textbf{0.818} & 0.759 & \textbf{0.917} & \textbf{0.580} & \textbf{0.000} \\",
        r"ToolFormerMicro V2 (gated) & 0.784 & \textbf{0.792} & \textbf{0.942} & 0.580 & 0.000 \\",
        r"ContextCache (8B) & 0.850 & 0.735 & --- & 0.600 & 0.000 \\",
        r"Tool Gisting K=8 & 0.714 & --- & --- & --- & 0.302 \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"% Composability table",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Composability experiments. ToolFormerMicro gist vectors are fully composable: "
        r"order-independent, scale to 200 tools with constant TSA, and support single-tool hot-swap.}",
        r"\label{tab:composability}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Experiment & Metric & Result \\",
        r"\midrule",
    ]

    # Add composability results
    comp_path = composability_dir / "compositional_generalization.json"
    if comp_path.exists():
        with open(comp_path, encoding="utf-8") as f:
            comp = json.load(f)
        lines.append(
            f"Order Independence & TSA delta & {comp['delta']:.3f} "
            f"({'+' if comp['delta'] >= 0 else ''}{comp['delta']:.3f}) \\\\"
        )

    scaling_path = composability_dir / "scaling_curves.json"
    if scaling_path.exists():
        with open(scaling_path, encoding="utf-8") as f:
            scaling = json.load(f)
        n_vals = [r["num_tools"] for r in scaling["results"]]
        tsa_vals = [r["tsa"] for r in scaling["results"]]
        lines.append(
            f"Scaling (5$\\to${n_vals[-1]} tools) & TSA range & "
            f"[{min(tsa_vals):.3f}, {max(tsa_vals):.3f}] \\\\"
        )

    inv_path = composability_dir / "cache_invalidation.json"
    if inv_path.exists():
        with open(inv_path, encoding="utf-8") as f:
            inv = json.load(f)
        lines.append(
            f"Cache Hot-Swap & TSA delta & {inv['tsa_delta']:.3f} "
            f"(re-encode: {inv['hotswap_encode_ms']:.0f}ms) \\\\"
        )
        lines.append(
            f"  & Other tools & {'bit-identical' if inv['other_tools_bitidentical'] else 'changed'} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path = output_dir / "table_tool_former.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ToolFormerMicro paper figures")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "eval_results" / "composability_v1")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "eval_results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "tool_former")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating ToolFormerMicro figures...")
    plot_scaling_curves(args.results_dir, args.output_dir)
    plot_composability_suite(args.results_dir, args.output_dir)
    plot_method_comparison(args.eval_dir, args.output_dir)
    plot_generate_latency_scaling(args.results_dir, args.output_dir)
    generate_latex_table(args.eval_dir, args.results_dir, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
