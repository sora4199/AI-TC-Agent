
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityRouter:
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold
        
        # Mẫu đại diện (Anchors) cho từng intent
        self.exemplars = {
            "knowledge_qa": [
                "mâm cúng nhập trạch nhà mới cần chuẩn bị những gì",
                "bài văn khấn khai trương cửa hàng đọc thế nào",
                "ý nghĩa nghi thức cúng rằm tháng bảy",
                "hướng dẫn chuẩn bị lễ vật cúng giao thừa"
            ],
            "product_search": [
                "bên bạn có bán gà luộc cánh tiên không",
                "giá giỏ hoa quả ngũ quả là bao nhiêu",
                "tìm mua xôi chè cúng đầy tháng",
                "báo giá bộ đồ lễ cúng mụ"
            ],
            "multi_step": [
                "tìm các món cần mua cho lễ thôi nôi rồi tính tổng tiền",
                "lên danh sách mâm cúng rằm và kiểm tra xem còn đủ hàng không",
                "chọn combo cúng khai trương và cộng tổng chi phí"
            ],
            "ambiguous": [
                "cho tôi một cái mâm cúng",
                "tôi muốn đặt đồ lễ gấp",
                "tôi cần đồ cúng",
                "mua giúp mâm cúng"
            ]
        }
        
        # Gom toàn bộ văn bản mẫu để fit TF-IDF
        self.corpus = []
        self.doc_labels = []
        for intent, texts in self.exemplars.items():
            for t in texts:
                self.corpus.append(t)
                self.doc_labels.append(intent)
                
        self.vectorizer = TfidfVectorizer()
        self.exemplar_vectors = self.vectorizer.fit_transform(self.corpus)

    def route(self, query: str) -> dict:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.exemplar_vectors)[0]
        
        max_idx = int(np.argmax(sims))
        max_sim = float(sims[max_idx])
        best_intent = self.doc_labels[max_idx]
        
        # Nếu độ tương đồng cao nhất dưới ngưỡng threshold -> OOD
        if max_sim < self.threshold:
            return {
                "route": "ood",
                "confidence": round(1.0 - max_sim, 2),
                "is_ood": True,
                "requires_clarification": False
            }
            
        return {
            "route": best_intent,
            "confidence": round(max_sim, 2),
            "is_ood": False,
            "requires_clarification": (best_intent == "ambiguous")
        }
