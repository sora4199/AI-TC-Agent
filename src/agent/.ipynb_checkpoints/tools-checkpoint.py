import logging
import jsonschema
from typing import Callable, Dict, Any, List
from .schemas import ToolResult, ToolStatus
from src.tools.calculator.calculator import Calculator
from src.tools.datetime.datetime import DateTime
from src.tools.web_search.web_search import WebSearchTool
logger = logging.getLogger(__name__)

class Tool:
    def __init__(self, name: str, description: str, func: Callable[[Dict[str, Any]], ToolResult], parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters
        
    def get_ollama_definition(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def run(self, arguments: Dict[str, Any]) -> ToolResult:
        try:
            jsonschema.validate(instance=arguments, schema=self.parameters)
            return self.func(arguments)
        except jsonschema.exceptions.ValidationError as e:
            logger.warning(f"Tool {self.name} validation failed: {e.message}")
            return ToolResult(status=ToolStatus.ERROR, error="Validation Error: Tham số không hợp lệ.")
        except Exception as e:
            logger.exception(f"Internal execution error in Tool {self.name}")
            return ToolResult(
                status=ToolStatus.ERROR, 
                error="Internal tool failure.", 
                metadata={"error_code": "TOOL_INTERNAL_ERROR"}
            )

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools.get(name)

    def get_all_definitions(self) -> List[Dict[str, Any]]:
        return [t.get_ollama_definition() for t in self._tools.values()]

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(status=ToolStatus.ERROR, error=f"Tool {name} không tồn tại trong hệ thống.")
        return tool.run(arguments)

def build_retriever_tool(retriever: Any) -> Tool:
    def retriever_wrapper(args: Dict[str, Any]) -> ToolResult:
        query = args.get("query", "")
        top_k = args.get("top_k", 20) 
        
        result = retriever.retrieve(query=query, top_k=top_k)
        status = result.get("metadata", {}).get("num_results", 0)
        
        if status > 0:
            return ToolResult(
                status=ToolStatus.SUCCESS, 
                data={
                    "documents": result.get("documents", []),
                    "metadata": result.get("metadata", {})
                }
            )
        
        return ToolResult(
            status=ToolStatus.NO_RESULT, 
            error="Không tìm thấy dữ liệu.",
            metadata={"status": "NO_RESULTS"}
        )

    return Tool(
        name="knowledge_retriever",
        description="Tra cứu tài liệu từ cơ sở dữ liệu dựa trên câu hỏi. Gọi đầu tiên để lấy danh sách ứng viên.",
        func=retriever_wrapper,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Câu hỏi cần tra cứu."},
                "top_k": {"type": "integer", "default": 20, "minimum": 1}
            },
            "required": ["query"], # CHỈ YÊU CẦU QUERY
            "additionalProperties": False 
        }
    )

def build_reranker_tool(reranker: Any) -> Tool:
    def reranker_wrapper(args: Dict[str, Any]) -> ToolResult:
        query = args.get("query", "")
        documents = args.get("documents", [])
        top_k = args.get("top_k", 5)
        
        if not documents:
            return ToolResult(status=ToolStatus.ERROR, error="Danh sách tài liệu rỗng.")
            
        result = reranker.rerank(query=query, documents=documents, top_k=top_k)
        
        return ToolResult(
            status=ToolStatus.SUCCESS, 
            data={
                "documents": result.get("documents", []),
                "metadata": {"status": "SUCCESS"}
            }
        )

    return Tool(
        name="knowledge_reranker",
        description="Chấm điểm và sắp xếp lại tài liệu. CHỈ sử dụng sau khi đã gọi knowledge_retriever.",
        func=reranker_wrapper,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "metadata": {"type": "object"}
                        },
                        "required": ["id", "content", "metadata"],
                        "additionalProperties": False
                    }
                },
                "top_k": {"type": "integer", "default": 5, "minimum": 1}
            },
            "required": ["query", "documents"], # YÊU CẦU CẢ QUERY VÀ DOCUMENTS
            "additionalProperties": False 
        }
    )
def build_calculator_tool(calculator: Any) -> Tool:
    def calculator_wrapper(args: Dict[str, Any]) -> ToolResult:
        expression = args.get("expression", "")
        res = calculator.calculate(expression)
        
        if "error" in res:
            return ToolResult(status=ToolStatus.ERROR, error=res["error"])
        return ToolResult(status=ToolStatus.SUCCESS, data=res)

    return Tool(
        name="calculator",
        description="Thực hiện các phép tính toán học (cộng, trừ, nhân, chia) để kiểm tra ngân sách, tính tiền.",
        func=calculator_wrapper,
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Biểu thức toán học, ví dụ: (1500 - 500) * 2"}
            },
            "required": ["expression"],
            "additionalProperties": False 
        }
    )

def build_datetime_tool(datetime_instance: Any) -> Tool:
    def datetime_wrapper(args: Dict[str, Any]) -> ToolResult:
        res = datetime_instance.get_datetime()
        return ToolResult(status=ToolStatus.SUCCESS, data=res)

    return Tool(
        name="get_current_datetime",
        description="Lấy thông tin ngày, giờ, thứ hiện tại của hệ thống để xác định thời gian cúng lễ.",
        func=datetime_wrapper,
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False 
        }
    )

def build_web_search_tool(web_search: Any) -> Tool:
    def web_search_wrapper(args: Dict[str, Any]) -> ToolResult:
        query = args.get("query", "")
        # Gọi execute của WebSearchTool (đã trả về chuẩn ToolResult)
        return web_search.execute(query=query)

    return Tool(
        name="web_search",
        description="Tìm kiếm thông tin cập nhật trên internet. Dùng khi thông tin nội bộ không đủ.",
        func=web_search_wrapper,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Nội dung cần tìm kiếm trên internet"}
            },
            "required": ["query"],
            "additionalProperties": False 
        }
    )