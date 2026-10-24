"""
Tạo bảng và plot tổng hợp kết quả benchmark, phục vụ báo cáo / paper.
"""

import csv
from pathlib import Path


def comparison_table_markdown(comparison):
    """
    comparison: output của benchmark.compare_configs()
        {metric_name: {config_name: score, ...}, ...}
    Trả về string markdown table.
    """
    if not comparison:
        return "(không có dữ liệu)"

    config_names = sorted({c for scores in comparison.values() for c in scores})

    header = "| Metric | " + " | ".join(config_names) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(config_names)) + " |"
    rows = [header, sep]

    for metric, scores in sorted(comparison.items()):
        row = [metric] + [f"{scores.get(c, float('nan')):.4f}" for c in config_names]
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join(rows)


def comparison_table_csv(comparison, save_path):
    """Lưu bảng so sánh ra file CSV."""
    if not comparison:
        return None

    config_names = sorted({c for scores in comparison.values() for c in scores})

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + config_names)
        for metric, scores in sorted(comparison.items()):
            writer.writerow([metric] + [scores.get(c, "") for c in config_names])

    return save_path


def plot_comparison_bar_chart(comparison, save_path, metrics_to_plot=None, title="Benchmark Comparison"):
    """
    Vẽ bar chart so sánh các config theo từng metric.
    metrics_to_plot: list metric muốn vẽ (None = vẽ hết, có thể rối nếu nhiều metric).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = metrics_to_plot or sorted(comparison.keys())
    metrics = [m for m in metrics if m in comparison]
    if not metrics:
        return None

    config_names = sorted({c for m in metrics for c in comparison[m]})
    x = np.arange(len(metrics))
    width = 0.8 / max(len(config_names), 1)

    fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 1.2), 5))

    for i, config in enumerate(config_names):
        values = [comparison[m].get(config, 0) for m in metrics]
        ax.bar(x + i * width, values, width, label=config)

    ax.set_xticks(x + width * (len(config_names) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


def per_case_csv(report, save_path):
    """Xuất điểm từng test case (per_case) của 1 report ra CSV, tiện lọc/sort trong Excel."""
    per_case = report.get("per_case", [])
    if not per_case:
        return None

    metric_keys = sorted({k for r in per_case for k, v in r["scores"].items() if isinstance(v, (int, float))})

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "case_type", "latency_sec"] + metric_keys)
        for r in per_case:
            row = [r["id"], r["case_type"], f"{r['latency_sec']:.4f}"]
            row += [r["scores"].get(k, "") for k in metric_keys]
            writer.writerow(row)

    return save_path
