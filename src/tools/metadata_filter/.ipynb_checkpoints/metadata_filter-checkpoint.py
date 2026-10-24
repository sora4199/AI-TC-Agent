import logging
from dataclasses import asdict
from typing import List, Dict, Any
from src.agent.schemas import Source, ToolResult, ToolStatus

logger = logging.getLogger("MetadataFilterTool")
logger.setLevel(logging.INFO)

class MetadataFilterTool:
    """
    Công cụ lọc danh sách các Source (tài liệu) dựa trên Metadata.
    So khớp là exact-match (==) trên từng key trong filters — không hỗ trợ
    range, list-in hay partial match.

    filters=None hoặc filters={} đều được coi là "không lọc" và trả về
    nguyên danh sách gốc.
    """

    @staticmethod
    def execute(sources: List[Source], filters: Dict[str, Any]) -> ToolResult:
        if not isinstance(sources, list):
            logger.error("Đầu vào 'sources' không phải là một danh sách (list).")
            return ToolResult(
                status=ToolStatus.ERROR,
                error="Input 'sources' must be a list of Source objects."
            )

        if not sources:
            logger.info("Danh sách nguồn (sources) rỗng, không có gì để lọc.")
            return ToolResult(
                status=ToolStatus.NO_RESULT,
                error="Input sources list is empty.",
                metadata={"applied_filters": filters}
            )

        if filters is None or (isinstance(filters, dict) and len(filters) == 0):
            logger.info("Không có điều kiện lọc (filters rỗng/None). Trả về danh sách gốc.")
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"filtered_sources": [asdict(s) for s in sources]},
                metadata={"applied_filters": "NONE", "output_count": len(sources)}
            )

        if not isinstance(filters, dict):
            logger.error("Đầu vào 'filters' không phải dict hợp lệ.")
            return ToolResult(
                status=ToolStatus.ERROR,
                error="Input 'filters' must be a dict."
            )

        try:
            filtered_sources: List[Source] = []
            skipped_count = 0

            for source in sources:
                if not hasattr(source, 'metadata') or not isinstance(source.metadata, dict):
                    logger.warning(f"Bỏ qua source {getattr(source, 'document_id', 'Unknown')} vì metadata không hợp lệ.")
                    skipped_count += 1
                    continue

                is_match = True
                for filter_key, filter_value in filters.items():
                    if source.metadata.get(filter_key) != filter_value:
                        is_match = False
                        break

                if is_match:
                    filtered_sources.append(source)

            if not filtered_sources:
                logger.info(f"Không có Source nào khớp với bộ lọc: {filters}")
                return ToolResult(
                    status=ToolStatus.NO_RESULT,
                    metadata={
                        "applied_filters": filters,
                        "input_count": len(sources),
                        "skipped_count": skipped_count
                    }
                )

            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"filtered_sources": [asdict(s) for s in filtered_sources]},
                metadata={
                    "applied_filters": filters,
                    "input_count": len(sources),
                    "output_count": len(filtered_sources),
                    "skipped_count": skipped_count
                }
            )

        except Exception as e:
            logger.exception(f"Lỗi trong quá trình lọc metadata: {str(e)}")
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Metadata filtering failed: {str(e)}",
                metadata={"filters": filters}
            )