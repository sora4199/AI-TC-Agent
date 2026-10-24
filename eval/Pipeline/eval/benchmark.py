"""
Chạy benchmark giữa nhiều configuration/baseline khác nhau trên CÙNG 1 dataset,
rồi tổng hợp lại để so sánh (VD: so sánh model A vs model B, hoặc top_k=5 vs top_k=10).
"""

from .pipeline import run_pipeline


def run_benchmark(dataset, configs, evaluator, module_name, save_dir=None):
    """
    configs: dict[str, callable] -> {"config_name": predict_fn, ...}
        VD: {"baseline_gpt4": predict_fn_v1, "finetuned_v2": predict_fn_v2}

    Trả về:
        {
          "config_name_1": <report từ run_pipeline>,
          "config_name_2": <report từ run_pipeline>,
          ...
        }
    """
    results = {}
    for config_name, predict_fn in configs.items():
        save_path = f"{save_dir}/{module_name}_{config_name}.json" if save_dir else None
        report = run_pipeline(
            dataset=dataset,
            predict_fn=predict_fn,
            evaluator=evaluator,
            module_name=f"{module_name}[{config_name}]",
            save_path=save_path,
        )
        results[config_name] = report
    return results


def compare_configs(benchmark_results):
    """
    So sánh điểm aggregate giữa các config.
    benchmark_results: output của run_benchmark()

    Trả về dict[metric_name][config_name] = score, tiện để in bảng hoặc vẽ plot.
    """
    comparison = {}
    for config_name, report in benchmark_results.items():
        for metric, value in report["aggregate"].items():
            if isinstance(value, (int, float)):
                comparison.setdefault(metric, {})[config_name] = value
    return comparison
