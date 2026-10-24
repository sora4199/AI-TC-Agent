
class NoRouter:
    """Baseline 1: Mọi query đều bị ném thẳng vào Knowledge QA (hoặc Agent)"""
    def route(self, query: str) -> dict:
        return {
            "route": "knowledge_qa",
            "confidence": 1.0,
            "is_ood": False,
            "requires_clarification": False
        }
