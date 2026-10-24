import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class ClassifierRouter:
    def __init__(self, training_data: list, ood_threshold: float = 0.4):
        self.ood_threshold = ood_threshold
        
        texts = [item["query"] for item in training_data]
        labels = [item["label"] for item in training_data]
        
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(texts)
        
        # Train một classifier nhẹ trên tập mẫu
        self.clf = LogisticRegression(max_iter=200)
        self.clf.fit(X, labels)

    def route(self, query: str) -> dict:
        x_vec = self.vectorizer.transform([query])
        probs = self.clf.predict_proba(x_vec)[0]
        max_prob = float(np.max(probs))
        pred_label = self.clf.classes_[np.argmax(probs)]
        
        # Nếu xác suất phân loại cao nhất quá thấp -> đẩy sang OOD
        if max_prob < self.ood_threshold:
            return {
                "route": "ood",
                "confidence": round(1.0 - max_prob, 2),
                "is_ood": True,
                "requires_clarification": False
            }
            
        return {
            "route": pred_label,
            "confidence": round(max_prob, 2),
            "is_ood": (pred_label == "ood"),
            "requires_clarification": (pred_label == "ambiguous")
        }