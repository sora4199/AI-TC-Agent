import logging
from typing import Optional
from src.agent.schemas import ToolResult, ToolStatus

logger = logging.getLogger("WebSearchTool")
logger.setLevel(logging.INFO)

MAX_RESULTS_CAP = 20  

class WebSearchTool:
    """
    Công cụ thực hiện tìm kiếm trên Web.
    Yêu cầu đầu vào là một chuỗi truy vấn (query) và số lượng kết quả tối đa.

    Lưu ý: hiện tại dùng mock engine, chưa gọi API thật.
    api_key được giữ lại cho lần tích hợp search engine thật sau này.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        if api_key is None:
            logger.warning("Khởi tạo WebSearchTool không có api_key (đang chạy mock engine).")

    def execute(self, query: str, max_results: int = 5) -> ToolResult:
        if not isinstance(query, str) or not query.strip():
            logger.error("Đầu vào 'query' rỗng hoặc không phải chuỗi.")
            return ToolResult(
                status=ToolStatus.ERROR,
                error="Query string cannot be empty or whitespace.",
                metadata={"provided_query": query}
            )

        if not isinstance(max_results, int) or max_results <= 0:
            logger.error(f"Đầu vào 'max_results' không hợp lệ: {max_results}")
            return ToolResult(
                status=ToolStatus.ERROR,
                error="max_results must be a positive integer greater than 0."
            )

        if max_results > MAX_RESULTS_CAP:
            logger.warning(f"max_results={max_results} vượt cap, giới hạn còn {MAX_RESULTS_CAP}.")
            max_results = MAX_RESULTS_CAP

        try:
            logger.info(f"Đang tìm kiếm web cho truy vấn: '{query}'")

            mock_search_responses = [
                {"title": "Kết quả 1", "url": "https://example.com/1", "snippet": f"Thông tin chi tiết về {query}"},
                {"title": "Kết quả 2", "url": "https://example.com/2", "snippet": f"Một góc nhìn khác về {query}"},
                {"title": "Kết quả 3", "url": "https://example.com/3", "snippet": f"Tài liệu tham khảo cho {query}"}
            ]

            results = mock_search_responses[:max_results]

            if not results:
                logger.info("Không tìm thấy kết quả nào.")
                return ToolResult(
                    status=ToolStatus.NO_RESULT,
                    error="No search results found.",
                    metadata={"query": query, "engine": "mock_engine"}
                )

            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"search_results": results},
                metadata={
                    "query_used": query,
                    "results_count": len(results),
                    "max_requested": max_results
                }
            )

        except Exception as e:
            logger.exception(f"Lỗi không xác định khi tìm kiếm web: {str(e)}")
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Web search failed due to internal error: {str(e)}",
                metadata={"query": query}
            )