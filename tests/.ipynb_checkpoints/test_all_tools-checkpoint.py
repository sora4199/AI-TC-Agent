import sys
from pathlib import Path
from pprint import pprint

# Đảm bảo import được từ thư mục gốc
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import các tools
from src.tools.calculator.calculator import Calculator
from src.tools.datetime.datetime import DateTime
from src.tools.memory.memory import Memory
from src.tools.web_search.web_search import WebSearchTool
from src.tools.metadata_filter.metadata_filter import MetadataFilterTool
from src.agent.schemas import Source # Dùng cho filter test

def run_tests():
    print("🚀 BẮT ĐẦU TEST CÁC TOOLS...\n" + "="*40)

    # 1. Test Calculator
    print("\n[1] Testing Calculator...")
    calc = Calculator()
    calc_res = calc.calculate("(1500 - 500) * 2")
    print(f" -> Kết quả: {calc_res}")

    # 2. Test DateTime
    print("\n[2] Testing DateTime...")
    dt = DateTime()
    dt_res = dt.get_datetime()
    print(f" -> Thời gian hiện tại: {dt_res.get('datetime')}")

    # 3. Test Web Search
    print("\n[3] Testing Web Search...")
    web = WebSearchTool()
    web_res = web.execute("Ngày lễ Phật Đản", max_results=2)
    print(f" -> Status: {web_res.status} | Số kết quả: {len(web_res.data.get('search_results', []))}")

    # 4. Test Metadata Filter
    print("\n[4] Testing Metadata Filter...")
    mock_sources = [
        Source(document_id="1", content="Text 1", metadata={"type": "buddhist"}),
        Source(document_id="2", content="Text 2", metadata={"type": "catholic"})
    ]
    filter_res = MetadataFilterTool.execute(mock_sources, filters={"type": "buddhist"})
    print(f" -> Status: {filter_res.status} | Filtered: {len(filter_res.data.get('filtered_sources', []))} docs")

    # 5. Test Memory
    print("\n[5] Testing Memory...")
    mem = Memory(mode="retriever_based")
    mem.add_turn("session_test", "Chào bạn", "Chào bạn, tôi có thể giúp gì?")
    mem_res = mem.search(query="Chào", session_id="session_test", top_k=2)
    print(f" -> Số bản ghi lấy được: {mem_res.get('metadata', {}).get('num_results')}")

    print("\n" + "="*40 + "\n✅ TẤT CẢ CÁC TOOL CHẠY THÀNH CÔNG, KHÔNG CÓ LỖI CRASH!")

if __name__ == "__main__":
    run_tests()