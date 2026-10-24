
import re
import json

class RuleBasedRouter:
    def __init__(self):
        # Từ khóa nhận diện Out-Of-Domain
        self.ood_keywords = ["thời tiết", "python", "lập trình", "sort array", "code", "bóng đá"]
        
        # Từ khóa nhận diện Multi-step (cần tính toán, kiểm tra nhiều bước)
        self.multi_step_keywords = ["tính tổng", "tính tiền", "kiểm tra xem còn", "rồi tính"]
        
        # Từ khóa tra cứu sản phẩm đơn lẻ
        self.search_keywords = ["có bán", "giá bao nhiêu", "giá là bao nhiêu", "tìm sản phẩm"]
        
        # Từ khóa kiến thức / văn khấn / phong tục
        self.qa_keywords = ["gồm những gì", "như thế nào", "đọc thế nào", "văn khấn", "chuẩn bị những gì"]

    def route(self, query: str) -> dict:
        q = query.lower()
        
        # 1. Quét OOD trước
        if any(kw in q for kw in self.ood_keywords):
            return {
                "route": "ood",
                "confidence": 0.9,
                "is_ood": True,
                "requires_clarification": False
            }
        
        # 2. Quét Ambiguous (câu cộc lốc, thiếu ngữ cảnh)
        if q.strip() in ["cho tôi một cái mâm cúng", "tôi muốn đặt đồ lễ gấp", "cho tôi mâm cúng", "mua đồ cúng"]:
            return {
                "route": "ambiguous",
                "confidence": 0.85,
                "is_ood": False,
                "requires_clarification": True,
                "missing_information": ["occasion", "budget"]
            }
            
        # 3. Quét Multi-step
        if any(kw in q for kw in self.multi_step_keywords):
            return {
                "route": "multi_step",
                "confidence": 0.8,
                "is_ood": False,
                "requires_clarification": False
            }

        # 4. Quét Product Search
        if any(kw in q for kw in self.search_keywords):
            return {
                "route": "product_search",
                "confidence": 0.8,
                "is_ood": False,
                "requires_clarification": False
            }

        # 5. Quét Knowledge QA
        if any(kw in q for kw in self.qa_keywords):
            return {
                "route": "knowledge_qa",
                "confidence": 0.8,
                "is_ood": False,
                "requires_clarification": False
            }

        # Mặc định nếu không khớp rule nào
        return {
            "route": "knowledge_qa",
            "confidence": 0.5,
            "is_ood": False,
            "requires_clarification": False
        }
