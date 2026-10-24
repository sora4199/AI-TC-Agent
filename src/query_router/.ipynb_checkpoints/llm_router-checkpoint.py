import json
import ollama

class LLMRouter:
    def __init__(self, model_name: str = "qwen2.5:7b"):
        self.model_name = model_name
        self.system_prompt = """BẠN LÀ QUERY ROUTER CHUYÊN BIỆT CHO HỆ THỐNG 'TC-AGENT' (TƯ VẤN ĐỒ CÚNG, NGHI LỄ VIỆT NAM).

Phân loại query của người dùng vào đúng 1 trong 5 nhóm:
1. "knowledge_qa": Hỏi kiến thức, ý nghĩa phong tục, cách bày mâm lễ, văn khấn CỤ THỂ một dịp lễ rõ ràng.
2. "product_search": Tìm kiếm sản phẩm đơn lẻ hoặc hỏi giá một mặt hàng cụ thể.
3. "multi_step": Yêu cầu tính toán, gom đơn, kiểm tra tồn kho CÓ ĐẦY ĐỦ THÔNG TIN dịp lễ/mặt hàng.
4. "ambiguous": Câu hỏi chung chung, cộc lốc, muốn đặt hàng/mua đồ cúng nhưng KHÔNG NÓI RÕ: cúng dịp gì? ngân sách bao nhiêu? cần bao nhiêu món? (Ví dụ: "cho tôi mâm cúng", "đặt đồ lễ gấp", "mua đồ cúng").
5. "ood": Hoàn toàn không liên quan đồ cúng (thời tiết, code, toán) HOẶC câu hỏi nghiên cứu lịch sử triều đại quá sâu.

BẮT BUỘC TRẢ VỀ DUY NHẤT 1 ĐỐI TƯỢNG JSON:
{
  "route": "knowledge_qa" | "product_search" | "multi_step" | "ambiguous" | "ood",
  "confidence": 0.95,
  "is_ood": false,
  "requires_clarification": false
}"""

    def route(self, query: str) -> dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ]
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                format="json",
                options={"temperature": 0.0}
            )
            content = response.get("message", {}).get("content", "{}")
            result = json.loads(content)
            
            # Chuẩn hóa giá trị route
            route_val = result.get("route", "").lower()
            result["route"] = route_val
            
            # Đồng bộ cờ is_ood
            if route_val == "ood":
                result["is_ood"] = True
            else:
                result["is_ood"] = result.get("is_ood", False)
                
            return result
        except Exception as e:
            return {
                "route": "ood",
                "confidence": 0.0,
                "is_ood": True,
                "requires_clarification": False,
                "error": str(e)
            }
