
from typing import List, Optional
from pydantic import BaseModel
from src.query_router.llm_router import LLMRouter
class RouterDecision(BaseModel):
    route: str
    confidence: float
    is_ood: bool
    requires_clarification: bool
    missing_information: Optional[List[str]] = []

class QueryRouter:
    """Unified Router Interface cho TC-Agent pipeline."""
    def __init__(self, backend: str = "llm", model_name: str = "qwen2.5:7b"):
        self.backend = backend
        if backend == "llm":
            self.engine = LLMRouter(model_name=model_name)
        else:
            raise NotImplementedError(f"Backend {backend} chưa được hỗ trợ.")

    def route(self, query: str) -> RouterDecision:
        raw_res = self.engine.route(query)
        route_name = raw_res.get("route", "ood").lower()
        
        # Đồng bộ logic Ambiguous theo Mục 7
        missing_info = []
        requires_clarify = (route_name == "ambiguous")
        if requires_clarify:
            missing_info = ["occasion", "budget", "target_entity"]

        return RouterDecision(
            route=route_name,
            confidence=float(raw_res.get("confidence", 0.0)),
            is_ood=(route_name == "ood"),
            requires_clarification=requires_clarify,
            missing_information=missing_info
        )
