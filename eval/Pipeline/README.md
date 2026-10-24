# Evaluation Pipeline - TC-Agent

Bao phủ toàn bộ phần **code được** trong tài liệu "Benchmark + Evaluation":
đủ 8 module, dataset validation/coverage, benchmark so sánh config, report
(bảng + plot), và error analysis tự động.

Phần **không code được** (cần bạn chủ động làm), tài liệu gốc có nhắc nhưng
KHÔNG nằm trong bộ code này:
- Thu thập test case thật từ các member khác (đây là việc giao tiếp/con người)
- Tự viết nội dung case cụ thể cho normal/edge/adversarial (code chỉ *kiểm tra*
  bạn có đủ loại case, không tự nghĩ ra case hộ bạn)
- Phân tích NGUYÊN NHÂN gốc rễ của failure case (code chỉ *liệt kê* case fail,
  còn đọc và suy luận tại sao sai vẫn cần con người)

## Cấu trúc

```
eval/
  metrics/
    text_metrics.py            # exact_match, token_f1, rouge_l, bleu1-4        -> Agent/RAG
    retrieval_metrics.py       # recall@k, precision@k, mrr, ndcg@k             -> Retriever/Reranker/Memory/Few-shot
    classification_metrics.py  # accuracy, precision/recall/f1, confusion_matrix -> Router
    tool_use_metrics.py        # tool_selection_accuracy, argument_match         -> Tool-use
    calculation_metrics.py     # numeric exact match, relative_error            -> Calculation
    multistep_metrics.py       # step_sequence_score, task_success              -> Agent/Multi-step Task
  evaluators/                  # 1 evaluator / module, đăng ký trong EVALUATOR_REGISTRY
    text_evaluator.py, retriever_evaluator.py, reranker_evaluator.py,
    memory_evaluator.py, router_evaluator.py, tool_use_evaluator.py,
    calculation_evaluator.py, multistep_evaluator.py, fewshot_evaluator.py
  dataset_loader.py            # đọc file JSON test case
  dataset_manager.py           # validate format + coverage (normal/edge/adversarial)
  pipeline.py                  # load -> chạy module -> tính điểm -> tổng hợp -> lưu report
  benchmark.py                 # chạy NHIỀU config/baseline trên cùng dataset để so sánh
  report.py                    # bảng markdown/CSV + bar chart PNG (cho báo cáo/paper)
  error_analysis.py            # tự động lọc failure case theo ngưỡng metric
data/
  router_sample.json, retriever_sample.json, rag_sample.json,
  tool_use_sample.json, calculation_sample.json, multistep_sample.json,
  fewshot_sample.json
  (reranker/memory dùng chung format với retriever_sample.json, bạn tự thêm file
  khi có dataset thật)
run_eval.py                    # demo chạy nhanh nhiều module, module GIẢ LẬP
run_benchmark_demo.py          # demo validate + benchmark 2 config + report + error analysis
```

## Bảng module <-> evaluator <-> ground_truth format

| Module (theo docx)      | key trong EVALUATOR_REGISTRY | ground_truth format                                 |
| ------------------------ | ----------------------------- | ---------------------------------------------------- |
| RAG / Retrieval           | `retriever`                   | `{"relevant_doc_ids": [...]}`                        |
| Reranking                 | `reranker`                    | `{"relevant_doc_ids": [...]}`                        |
| Router / OOD Detection    | `router`                      | `{"route": "rag"\|"agent"\|"ood"}`                    |
| Memory / Multi-turn       | `memory`                      | `{"relevant_memory_ids": [...]}`                      |
| Tool-use                  | `tool_use`                    | `{"tool_calls": [{"tool_name":..., "arguments":{}}]}` |
| Agent / Multi-step Task   | `multistep`                   | `{"steps": [{"action":...}], "answer": "..."}`        |
| Calculation                | `calculation`                 | `{"result": number, "tolerance": float (tuỳ chọn)}`   |
| Dynamic Few-shot            | `fewshot`                      | `{"relevant_example_ids": [...]}`                     |
| (Agent/RAG trả lời text)  | `agent` / `rag`                | `{"answer": "...", "source_ids": [...] (tuỳ chọn)}`   |

## Chạy thử ngay

```bash
pip install nltk matplotlib --break-system-packages
python3 run_eval.py               # demo nhanh 3 module
python3 run_benchmark_demo.py     # demo đầy đủ: validate + benchmark + report + error analysis
```

## Tích hợp Ollama

Pipeline không phụ thuộc LLM cụ thể nào - chỉ cần `predict_fn` gọi model rồi
map output về đúng schema. Đã có sẵn adapter cho Ollama:

```
eval/adapters/ollama_adapter.py   # call_ollama_text / call_ollama_json / call_ollama_chat
run_with_ollama.py                # ví dụ THẬT: Router + RAG chạy qua Ollama
```

**Chuẩn bị:**
```bash
ollama serve                           # nếu chưa chạy sẵn
ollama pull llama3.1                   # hoặc model bạn đang dùng
pip install requests --break-system-packages
```

**Chạy:**
```bash
python3 run_with_ollama.py
```

**Cách viết predict_fn của riêng bạn** - 2 kiểu:

```python
from eval.adapters.ollama_adapter import call_ollama_text, call_ollama_json

# Module trả lời text tự do (Agent/RAG -> answer: string)
def predict_fn_rag(input_data):
    answer = call_ollama_text(f"Trả lời: {input_data['query']}", model="llama3.1")
    return {"answer": answer, "sources": [], "metadata": {}}

# Module cần output có cấu trúc (Router, Tool-use, Calculation...) - ép JSON
def predict_fn_router(input_data):
    prompt = f'Phân loại route cho: "{input_data["query"]}". Trả JSON: {{"route": "rag"|"agent"|"ood"}}'
    result = call_ollama_json(prompt, model="llama3.1")
    return {"route": result["route"], "confidence": result.get("confidence", 0.0), "reason": None}
```

Với module Memory/Multi-turn cần giữ lịch sử hội thoại, dùng `call_ollama_chat(messages, ...)`
thay vì `call_ollama_text`.

Lưu ý: Ollama chạy model local nên **độ trễ mỗi request cao hơn API cloud** -
nếu dataset lớn (hàng trăm/nghìn case), cân nhắc chạy pipeline theo batch nhỏ
hoặc thêm giới hạn concurrency (mặc định pipeline chạy tuần tự, không parallel).

## Cắm module THẬT vào (nếu không dùng Ollama)

`run_eval.py` hiện dùng các hàm `predict_fn_router`, `predict_fn_retriever`,
`predict_fn_rag` **giả lập** (mock), chỉ để chứng minh pipeline chạy được đầu-cuối.

Khi đồng đội bạn có module thật, thay các hàm này bằng hàm gọi module thật,
miễn là tuân đúng input/output schema trong `MODULE_SCHEMA.md`. Ví dụ:

```python
from my_team.router import RouterModule

router = RouterModule()

def predict_fn_router(input_data):
    # input_data đã đúng format Query {query, session_id, user_id, metadata}
    decision = router.route(input_data["query"])
    return {"route": decision.route, "confidence": decision.confidence, "reason": decision.reason}
```

Chỉ cần đổi `predict_fn`, phần load dataset / tính metric / tổng hợp report
giữ nguyên không cần sửa.

## Format dataset (input cho pipeline)

Mỗi file JSON là 1 list test case:

```json
[
  {
    "id": "case_001",
    "case_type": "normal",        // "normal" | "edge" | "adversarial"
    "input": { ... },              // đúng Request schema của module (xem MODULE_SCHEMA.md)
    "ground_truth": { ... }        // đáp án đúng, format tuỳ module (xem bên dưới)
  }
]
```

Xem bảng ground_truth format đầy đủ ở mục "Bảng module <-> evaluator" phía trên.

## Chuẩn hóa format + kiểm tra coverage (`dataset_manager.py`)

```python
from eval.dataset_loader import load_dataset
from eval.dataset_manager import validate_dataset, coverage_report

dataset = load_dataset("data/retriever_sample.json")
errors = validate_dataset(dataset, "retriever")     # kiểm tra field bắt buộc, case_type hợp lệ
coverage = coverage_report(dataset)                  # đếm case theo normal/edge/adversarial
```

Dùng hàm này mỗi khi nhận dataset mới từ member khác, để chắc chắn họ gửi
đúng format trước khi đưa vào pipeline.

## Benchmark nhiều config/baseline (`benchmark.py`)

```python
from eval.benchmark import run_benchmark, compare_configs

configs = {"bm25": predict_fn_bm25, "embedding": predict_fn_embedding}
results = run_benchmark(dataset, configs, evaluator, module_name="retriever", save_dir="reports")
comparison = compare_configs(results)   # {metric: {config_name: score}}
```

## Report - bảng + plot cho báo cáo/paper (`report.py`)

```python
from eval.report import comparison_table_markdown, comparison_table_csv, plot_comparison_bar_chart

print(comparison_table_markdown(comparison))
comparison_table_csv(comparison, "reports/comparison.csv")
plot_comparison_bar_chart(comparison, "reports/comparison.png", metrics_to_plot=["recall@5", "mrr"])
```

## Error analysis - tìm failure case (`error_analysis.py`)

```python
from eval.error_analysis import find_low_score_cases, save_error_report

failures = find_low_score_cases(report, metric_key="recall@5", threshold=0.5)
save_error_report(report, "recall@5", threshold=0.5, save_path="reports/errors.json")
```

Xem ví dụ chạy đầy đủ 4 bước trên (validate → benchmark → report → error
analysis) trong `run_benchmark_demo.py`.

## Đối chiếu với yêu cầu trong "Benchmark + Evaluation.docx"

| Yêu cầu trong docx                              | Trạng thái | File tương ứng                     |
| ------------------------------------------------ | ---------- | ----------------------------------- |
| Dataset cho 8 thành phần                          | ✅ Code xong (khung + validate) | `eval/evaluators/*`, `eval/metrics/*` |
| Evaluation pipeline tự động                       | ✅          | `eval/pipeline.py`                  |
| Chuẩn hóa format dataset/ground truth/metrics      | ✅          | `eval/dataset_manager.py`           |
| Thu thập test case từ member khác                  | ⏳ Bạn tự làm | (chỉ có validator hỗ trợ kiểm tra)   |
| Chạy benchmark giữa config/baseline                | ✅          | `eval/benchmark.py`                 |
| Tổng hợp bảng/plot cho báo cáo, paper              | ✅          | `eval/report.py`                    |
| Error analysis                                    | ✅ (liệt kê failure case) | `eval/error_analysis.py`            |
| Đảm bảo đủ normal/edge/adversarial                | ✅ (kiểm tra coverage), ⏳ (tự viết nội dung case vẫn cần bạn) | `eval/dataset_manager.py::coverage_report` |

## Thêm module mới / thêm metric mới

- Metric mới → thêm hàm vào file tương ứng trong `eval/metrics/`.
- Module mới → tạo file trong `eval/evaluators/`, kế thừa `BaseEvaluator`,
  implement `evaluate_case()`, đăng ký vào `EVALUATOR_REGISTRY`, thêm
  `REQUIRED_GT_FIELDS` trong `dataset_manager.py`.

## Việc bạn cần làm tiếp

1. Thay các `predict_fn_*` giả lập bằng module thật (hoặc API call module thật).
2. Thu thập dataset thật từ teammate theo đúng format, chạy `validate_dataset()`
   + `coverage_report()` để kiểm tra trước khi dùng.
3. Chạy `run_benchmark_demo.py` với config/module thật để có bảng/plot thật
   cho báo cáo hoặc paper.
4. Khi có failure case từ `error_analysis.py`, tự đọc và phân tích nguyên nhân
   (phần này máy không suy luận hộ được).
