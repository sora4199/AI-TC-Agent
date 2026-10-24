import sys
import time
import uuid
from pathlib import Path

# Trỏ đường dẫn về thư mục gốc
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.tools.retriever.retriever import Retriever
from src.tools.reranker.reranker import Reranker
from src.agent.tools import ToolRegistry, build_retriever_tool, build_reranker_tool
from src.agent.agent import NativeAgent
from src.agent.executor import AgentExecutor

def main():
    print("Đang khởi tạo E-SPIRIT Agent (Qwen 2.5 7B)...")
    
    # Trỏ đúng vào kho sách giáo khoa của dự án
    index_path = ROOT_DIR / "database" / "index.faiss"
    contexts_path = ROOT_DIR / "database" / "contexts.json"
    
    # 1. Cài đặt hệ thống Tool
    retriever = Retriever(index_path=str(index_path), contexts_path=str(contexts_path))
    reranker = Reranker(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    
    registry = ToolRegistry()
    registry.register(build_retriever_tool(retriever))
    registry.register(build_reranker_tool(reranker))
    
    # 2. Khởi tạo Agent Executor
    agent = NativeAgent(registry=registry, model_name="qwen2.5:7b")
    executor = AgentExecutor(agent=agent, max_iterations=4)
    
    print("Hệ thống đã sẵn sàng!\n")
    print("="*60)
    print("CHAT TRỰC TIẾP VỚI E-SPIRIT AI")
    print("Gõ 'exit' hoặc 'quit' để thoát chương trình.")
    print("="*60)
    
    session_id = f"chat_{uuid.uuid4().hex[:8]}"
    
    while True:
        try:
            # Nhận câu hỏi từ người dùng
            user_input = input("\nBạn: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("Tạm biệt!")
                break
                
            print("Agent đang suy nghĩ và tra cứu tài liệu...", end="\r", flush=True)
            
            # Giao việc cho Agent
            start_time = time.time()
            response = executor.execute(
                query=user_input,
                session_id=session_id,
                metadata={"route": "agent"}
            )
            latency = time.time() - start_time
            
            answer = response.get("answer", "")
            sources = response.get("sources", [])
            
            # Xóa dòng trạng thái chờ
            print(" " * 60, end="\r") 
            
            # In kết quả
            print(f"Agent: {answer}")
            
            # In minh bạch các tài liệu Agent đã đọc để trả lời
            if sources:
                print(f"\n   [!] Đã tham khảo {len(sources)} tài liệu (Thời gian xử lý: {latency:.1f}s):")
                for idx, doc in enumerate(sources, 1):
                    content = doc.get('content', '')[:150].replace('\n', ' ')
                    #print(f"       - Nguồn {idx}: {content}...")
                    
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f"\nLỗi hệ thống: {e}")

if __name__ == "__main__":
    main()