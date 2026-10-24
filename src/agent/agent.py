import ollama
import logging
import time
from typing import List, Dict, Any
from httpx import ReadTimeout, ConnectError
from .schemas import AgentState
from .tools import ToolRegistry

class NativeAgent:
    def __init__(self, registry: ToolRegistry, model_name: str = "qwen2.5:3b"):
        self.registry = registry
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

    def _call_llm_with_retry(self, messages: List[Dict], max_attempts: int = 3) -> Dict[str, Any]:
        """Cơ chế Exponential Backoff có phân loại ngoại lệ."""
        for attempt in range(max_attempts):
            try:
                # Bổ sung Keep-alive config giả lập timeout nếu Ollama API wrapper cập nhật
                return ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    tools=self.registry.get_all_definitions(),
                    options={"temperature": 0.1}
                )
            except (ReadTimeout, ConnectError) as e:
                self.logger.warning(f"LLM Network Error (Attempt {attempt+1}/{max_attempts}): {e}")
                if attempt == max_attempts - 1:
                    raise e
                time.sleep(2 ** attempt) # 1s, 2s, 4s...
            except Exception as e:
                self.logger.error(f"Non-retryable LLM Error: {e}")
                raise e # Không retry với các lỗi logic (vd: sai định dạng prompt)

    def step(self, state: AgentState) -> AgentState:
        try:
            response = self._call_llm_with_retry(state.messages)
            message = response.get("message", {})
            
            state.messages.append(message)

            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    func = tool_call.get("function", {})
                    state.intermediate_steps.append({
                        "type": "tool_call",
                        "id": tool_call.get("id", ""), # Hỗ trợ Tool Call ID
                        "name": func.get("name"),
                        "args": func.get("arguments", {})
                    })
            elif message.get("content"):
                state.final_answer = message["content"]
            else:
                state.error = "Agent generated empty response."

        except Exception as e:
            state.error = f"Agent Execution Error: {str(e)}"
            
        return state