import time
import ollama
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# ==========================================
# IMPORT MODULES
# ==========================================
from src.query_router.router_interface import QueryRouter
from src.tools.memory.memory import Memory
from src.tools.retriever.retriever import Retriever
from src.tools.reranker.reranker import Reranker
from src.tools.calculator.calculator import Calculator
from src.tools.datetime.datetime import DateTime
from src.tools.web_search.web_search import WebSearchTool

from src.agent.agent import NativeAgent
from src.agent.executor import AgentExecutor

# Import các hàm bọc Tool (Đảm bảo bạn đã dán 3 hàm build mới vào file tools.py)
from src.agent.tools import (
    ToolRegistry, 
    build_retriever_tool, 
    build_reranker_tool,
    build_calculator_tool,
    build_datetime_tool,
    build_web_search_tool
)

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU (DATA PATHS)
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent
# Trỏ chính xác vào kho dữ liệu thật của dự án
FAISS_INDEX_PATH = str(ROOT_DIR / "database" / "index.faiss")
CONTEXTS_JSON_PATH = str(ROOT_DIR / "database" / "contexts.json")

# ==========================================
# 2. API SCHEMAS (CONTRACT V0)
# ==========================================
class QueryRequest(BaseModel):
    query: str
    session_id: str
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

class SourceModel(BaseModel):
    document_id: str
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceModel]
    metadata: Dict[str, Any]

# ==========================================
# 3. KHỞI TẠO CÁC THÀNH PHẦN GLOBAL
# ==========================================
app = FastAPI(title="E-SPIRIT API", version="1.0", description="TC-Agent Unified API (Router + RAG + Agent)")

# Khởi tạo Router và Memory
query_router = QueryRouter(backend="llm", model_name="qwen2.5:7b")
memory_manager = Memory(mode="retriever_based")

# Khởi tạo các công cụ gốc (Ép Retriever đọc đúng data thật)
retriever_instance = Retriever(index_path=FAISS_INDEX_PATH, contexts_path=CONTEXTS_JSON_PATH)
reranker_instance = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
calc_instance = Calculator()
dt_instance = DateTime()
web_instance = WebSearchTool()

# Đăng ký toàn bộ Tools vào Registry cho Agent
registry = ToolRegistry()
registry.register(build_retriever_tool(retriever_instance))
registry.register(build_reranker_tool(reranker_instance))
registry.register(build_calculator_tool(calc_instance))
registry.register(build_datetime_tool(dt_instance))
registry.register(build_web_search_tool(web_instance))

# Khởi tạo Agent
agent = NativeAgent(registry=registry, model_name="qwen2.5:7b")
executor = AgentExecutor(agent=agent, max_iterations=4)

# ==========================================
# 4. ENDPOINT CỐT LÕI (/api/v1/query)
# ==========================================
@app.post("/api/v1/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    start_time = time.perf_counter()
    
    try:
        # Bộ định tuyến phân loại câu hỏi
        decision = query_router.route(request.query)
        route_name = decision.route
        
        # ---------------------------------------------------------
        # NHÁNH 1: OOD HANDLER (Từ chối hoặc yêu cầu làm rõ)
        # ---------------------------------------------------------
        if decision.is_ood or decision.requires_clarification:
            answer_text = (
                "Xin lỗi, tôi là trợ lý tư vấn phong tục tâm linh E-SPIRIT. Tôi không thể hỗ trợ chủ đề này." 
                if decision.is_ood 
                else "Bạn có thể nói rõ hơn bạn cần chuẩn bị mâm lễ cho dịp gì và ngân sách khoảng bao nhiêu không?"
            )
            return QueryResponse(
                answer=answer_text,
                sources=[],
                metadata={
                    "route": route_name,
                    "confidence": decision.confidence,
                    "pipeline": "OOD_Handler",
                    "latency": round(time.perf_counter() - start_time, 2)
                }
            )

        # ---------------------------------------------------------
        # NHÁNH 2: RAG PIPELINE (Tra cứu kiến thức trực tiếp)
        # ---------------------------------------------------------
        if route_name == "knowledge_qa":
            # 1. Tìm kiếm tài liệu thô
            retrieval_res = retriever_instance.retrieve(query=request.query, top_k=20)
            candidates = retrieval_res.get("documents", [])
            
            # 2. Rerank tài liệu
            rerank_res = reranker_instance.rerank(query=request.query, documents=candidates, top_k=5)
            top_docs = rerank_res.get("documents", [])
            
            # 3. Tổng hợp đáp án bằng LLM
            context_text = "\n\n".join([f"Tài liệu {i+1}: {d['document']['content']}" for i, d in enumerate(top_docs)])
            system_prompt = (
                "Bạn là chuyên gia tư vấn tâm linh. Dựa vào các tài liệu sau, hãy trả lời câu hỏi của người dùng ngắn gọn, chính xác.\n"
                "Tuyệt đối không tự bịa đặt thông tin. Nếu không có thông tin, trả lời 'Không có thông tin'.\n\n"
                f"TÀI LIỆU:\n{context_text}"
            )
            
            response = ollama.chat(
                model="qwen2.5:7b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.query}
                ],
                options={"temperature": 0.1}
            )
            
            mapped_sources = [
                SourceModel(
                    document_id=doc["document"].get("id", "unknown"),
                    content=doc["document"].get("content", ""),
                    metadata=doc["document"].get("metadata", {})
                ) for doc in top_docs
            ]
            
            return QueryResponse(
                answer=response.get("message", {}).get("content", "").strip(),
                sources=mapped_sources,
                metadata={
                    "route": route_name,
                    "pipeline": "RAG_Pipeline",
                    "latency": round(time.perf_counter() - start_time, 2)
                }
            )

        # ---------------------------------------------------------
        # NHÁNH 3: AGENT MULTI-STEP REASONING (Suy luận phức tạp)
        # ---------------------------------------------------------
        
        # 1. Trích xuất ngữ cảnh trí nhớ
        recent_history = memory_manager.search(
            query=request.query, 
            session_id=request.session_id, 
            top_k=3
        )
        history_text = "\n".join([mem["content"] for mem in recent_history.get("memories", [])])

        # 2. Khởi chạy luồng Agent (Sẽ tự động gọi Tools)
        agent_result = executor.execute(
            query=request.query,
            session_id=request.session_id,
            metadata={"route": route_name, "history": history_text}
        )
        
        # 3. Ghi nhớ lượt tương tác
        memory_manager.add_turn(
            session_id=request.session_id,
            user_query=request.query,
            assistant_response=agent_result.get("answer", "")
        )

        mapped_sources = [
            SourceModel(
                document_id=src.get("document_id", src.get("id", "unknown")),
                content=src.get("content", ""),
                metadata=src.get("metadata", {})
            ) for src in agent_result.get("sources", [])
        ]

        return QueryResponse(
            answer=agent_result.get("answer", ""),
            sources=mapped_sources,
            metadata={
                "route": route_name,
                "confidence": decision.confidence,
                "pipeline": "Agent_Reasoning",
                "iterations": agent_result.get("metadata", {}).get("iterations", 0),
                "latency": round(time.perf_counter() - start_time, 2)
            }
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")