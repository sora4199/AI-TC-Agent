from .text_evaluator import TextEvaluator
from .retriever_evaluator import RetrieverEvaluator
from .reranker_evaluator import RerankerEvaluator
from .memory_evaluator import MemoryEvaluator
from .router_evaluator import RouterEvaluator
from .tool_use_evaluator import ToolUseEvaluator
from .calculation_evaluator import CalculationEvaluator
from .multistep_evaluator import MultiStepEvaluator
from .fewshot_evaluator import FewShotEvaluator

# 8 module đúng theo Benchmark + Evaluation.docx:
# RAG/Retrieval, Reranking, Router/OOD, Memory/Multi-turn, Tool-use,
# Agent/Multi-step Task, Calculation, Dynamic Few-shot
EVALUATOR_REGISTRY = {
    "router": RouterEvaluator,
    "retriever": RetrieverEvaluator,
    "reranker": RerankerEvaluator,
    "memory": MemoryEvaluator,
    "agent": TextEvaluator,
    "rag": TextEvaluator,
    "tool_use": ToolUseEvaluator,
    "multistep": MultiStepEvaluator,
    "calculation": CalculationEvaluator,
    "fewshot": FewShotEvaluator,
}
