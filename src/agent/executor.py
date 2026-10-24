import logging
import uuid
import json
import time
from typing import List, Dict, Any, Optional
from .schemas import AgentState, ToolStatus, ToolResult
from .agent import NativeAgent
from .prompt import AGENT_SYSTEM_PROMPT

class AgentExecutor:
    def __init__(self, agent: NativeAgent, max_iterations: int = 5, request_timeout_sec: int = 45):
        self.agent = agent
        self.max_iterations = max_iterations
        self.request_timeout_sec = request_timeout_sec
        self.logger = logging.getLogger(__name__)

    def _validate_and_build_history(self, query: str, history: List[Dict[str, str]]) -> List[Dict]:
        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
        
        for msg in history:
            role = msg.get("role")
            if role in ["user", "assistant"]:
                messages.append({"role": role, "content": str(msg.get("content", ""))})
                
        messages.append({"role": "user", "content": query})
        return messages

    def execute(self, query: str, session_id: str = "", user_id: Optional[str] = None, metadata: Dict = None, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Lưu ý Contract v0: 'history' là tham số internal runtime orchestration, 
        KHÔNG PHẢI là một field thuộc public schema Query.
        """
        req_id = str(uuid.uuid4())
        start_time = time.time()
        
        if not query or not query.strip() or len(query) > 1000:
            self.logger.warning(f"[{req_id}] Invalid query rejected.")
            return {"answer": "", "sources": [], "metadata": {"status": "ERROR", "error_code": "INVALID_QUERY_INPUT"}}

        clean_query = query.strip()
        state = AgentState(
            user_query=clean_query, 
            request_id=req_id,
            messages=self._validate_and_build_history(clean_query, history or [])
        )

        executed_tools = set()

        try:
            while state.iteration < self.max_iterations:
                if time.time() - start_time > self.request_timeout_sec:
                    self.logger.warning(f"[{req_id}] Request Deadline Exceeded.")
                    return {"answer": "", "sources": [], "metadata": {"status": "ERROR", "error_code": "REQUEST_TIMEOUT"}}

                self.logger.info(f"[{req_id}] Iteration {state.iteration + 1}")
                state = self.agent.step(state)

                if state.error:
                    self.logger.error(f"[{req_id}] Agent Error: {state.error}")
                    return {"answer": "", "sources": [], "metadata": {"status": "ERROR", "error_code": "AGENT_EXECUTION_FAILURE"}}

                if state.final_answer:
                    # --- CHỐT CHẶN ÉP BUỘC DÙNG TOOL ---
                    if state.iteration == 0 and not state.sources:
                        self.logger.warning(f"[{req_id}] Agent bypassed tool. Forcing tool call.")
                        state.final_answer = None # Hủy câu trả lời tự biên tự diễn
                        
                        # Bơm thêm lời nhắc thép vào luồng chat để chấn chỉnh LLM
                        state.messages.append({
                            "role": "user",
                            "content": "SYSTEM ALERT: Bạn đã vi phạm quy tắc. BẮT BUỘC phải gọi công cụ 'knowledge_retriever' để tra cứu thông tin trước khi đưa ra đáp án cuối cùng."
                        })
                        state.iteration += 1
                        continue # Ép chạy lại vòng lặp
                    # -----------------------------------

                    return {
                        "answer": state.final_answer, 
                        "sources": state.sources,
                        "metadata": {
                            "status": "SUCCESS",
                            "iterations": state.iteration + 1, 
                            "latency": round(time.time() - start_time, 2),
                            "session_id": session_id,
                            "user_id": user_id,
                            "agent_metadata": metadata or {}
                        }
                    }

                last_message = state.messages[-1]
                if last_message.get("tool_calls"):
                    for tool_call in last_message["tool_calls"]:
                        call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")
                        func = tool_call["function"]
                        t_name = func["name"]
                        t_args = func["arguments"]
                        
                        canonical_args = json.dumps(t_args, sort_keys=True, ensure_ascii=False)
                        call_signature = f"{t_name}_{canonical_args}"
                        
                        if call_signature in executed_tools:
                            tool_res = ToolResult(status=ToolStatus.ERROR, error="Duplicate tool call detected.")
                        else:
                            tool_res = self.agent.registry.execute(t_name, t_args)
                            if tool_res.status == ToolStatus.SUCCESS:
                                executed_tools.add(call_signature)

                        if tool_res.status == ToolStatus.SUCCESS:
                            content_str = json.dumps(tool_res.data, ensure_ascii=False)
                            
                            if tool_res.data and "documents" in tool_res.data:
                                for doc in tool_res.data.get("documents", []):
                                    state.sources.append({
                                        "document_id": doc.get("id", ""),
                                        "content": doc.get("content", ""),
                                        "metadata": doc.get("metadata", {})
                                    })
                                
                                unique_sources = {}
                                for source in state.sources:
                                    doc_id = source.get("document_id")
                                    if doc_id:
                                        unique_sources[doc_id] = source
                                state.sources = list(unique_sources.values())
                                
                        else:
                            content_str = json.dumps({
                                "status": tool_res.status.value,
                                "error": tool_res.error or "Unknown error.",
                                "metadata": tool_res.metadata
                            }, ensure_ascii=False)

                        state.messages.append({
                            "role": "tool",
                            "name": t_name,
                            "content": content_str,
                            "tool_call_id": call_id
                        })

                state.iteration += 1

            self.logger.warning(f"[{req_id}] Max iterations reached.")
            return {"answer": "", "sources": [], "metadata": {"status": "ERROR", "error_code": "MAX_ITERATIONS_REACHED"}}

        except Exception as e:
            self.logger.exception(f"[{req_id}] Critical system failure: {e}")
            return {"answer": "", "sources": [], "metadata": {"status": "ERROR", "error_code": "AGENT_INTERNAL_ERROR"}}