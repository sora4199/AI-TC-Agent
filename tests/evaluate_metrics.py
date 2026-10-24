import json
import time
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.tools.retriever.retriever import Retriever
from src.tools.reranker.reranker import Reranker
from src.agent.tools import ToolRegistry, build_retriever_tool, build_reranker_tool
from src.agent.agent import NativeAgent
from src.agent.executor import AgentExecutor

from eval.metrics import rouge_l_score, compute_bleu_scores

def run_metric_eval(num_samples=None):
    # Đường dẫn
    # câu hỏi và đáp án kỳ vọng
    test_file = ROOT_DIR / "dataset" / "test.json"
    
    # Retriever tìm kiếm
    index_path = ROOT_DIR / "database" / "index.faiss"
    contexts_path = ROOT_DIR / "database" / "contexts.json"
    
    log_dir = ROOT_DIR / "tests" / "results"
    log_file = log_dir / "test_newdata.txt"
    os.makedirs(log_dir, exist_ok=True)
    
    def log_print(message):
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"--- BẮT ĐẦU ĐÁNH GIÁ (TEST RUN): {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")

    print(f"Đang khởi tạo hệ thống. Kết quả sẽ lưu tại {log_file}...")
    log_print("1. Khởi tạo hệ thống (Model: qwen2.5:7b)...")
    
    # Khởi tạo Retriever
    retriever = Retriever(
        index_path=str(index_path), 
        contexts_path=str(contexts_path)
    ) 
    reranker = Reranker(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    
    # Đăng ký 2 tool độc lập
    registry = ToolRegistry()
    registry.register(build_retriever_tool(retriever))
    registry.register(build_reranker_tool(reranker))
    
    agent = NativeAgent(registry=registry, model_name="qwen2.5:7b")
    executor = AgentExecutor(agent=agent, max_iterations=4) # Tăng lặp vì LLM phải gọi 2 tool

    log_print(f"2. Tải dữ liệu từ {test_file}...")
    with open(test_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    total_rouge_f1 = 0
    total_rouge_recall = 0
    total_bleu = {"bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0}
    
    samples_run = len(dataset) if num_samples is None else min(num_samples, len(dataset))
    log_print(f"Bắt đầu chấm điểm {samples_run} mẫu...\n")
    
    for i, item in enumerate(dataset[:samples_run], 1):
        print(f"Đang xử lý: {i}/{samples_run}...", end="\r", flush=True)
        
        question = item.get("question", "")
        ground_truth = str(item.get("answer", ""))
        
        response = executor.execute(
            query=question,
            session_id=f"eval_{i}",
            metadata={"route": "agent"}
        )
        
        prediction = str(response.get("answer", ""))
        sources = response.get("sources", [])
        
        r_scores = rouge_l_score(prediction, ground_truth)
        b_scores = compute_bleu_scores(prediction, ground_truth)
        
        total_rouge_f1 += r_scores["f1"]
        total_rouge_recall += r_scores["recall"]
        for k in total_bleu:
            total_bleu[k] += b_scores[k]
            
        log_print(f"[{i}/{samples_run}] Hỏi: {question}")
        log_print(f" - Kỳ vọng : {ground_truth}")
        log_print(f" - Thực tế : {prediction}")
        log_print(f" - Điểm    : ROUGE-L(F1)={r_scores['f1']:.4f} | Recall={r_scores['recall']:.4f} | BLEU-1={b_scores['bleu1']:.4f}")
        
        log_print(" - Tài liệu tham chiếu (Retrieved Context):")
        if sources:
            for idx, doc in enumerate(sources):
                content = doc.get("content", "")[:300].replace("\n", " ")
                log_print(f"   [{idx+1}] {content}...")
        else:
            log_print("   (Không tìm thấy tài liệu nào từ FAISS)")
        log_print("\n")

    log_print("="*50)
    log_print("KẾT QUẢ TRUNG BÌNH TOÀN BỘ BATCH:")
    log_print(f"Average ROUGE-L (F1)     : {total_rouge_f1 / samples_run:.4f}")
    log_print(f"Average ROUGE-L (Recall) : {total_rouge_recall / samples_run:.4f}")
    for k in total_bleu:
        log_print(f"Average {k.upper():<16} : {total_bleu[k] / samples_run:.4f}")

    print(f"\nHoàn tất! Đã lưu toàn bộ kết quả vào {log_file}.")

if __name__ == "__main__":
    NUM_SAMPLES = None
    run_metric_eval(num_samples=NUM_SAMPLES)