from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

class ToolStatus(str, Enum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    NO_RESULT = "NO_RESULT"

@dataclass
class ToolResult:
    status: ToolStatus
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Source:
    document_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentState:
    user_query: str
    request_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[str] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    iteration: int = 0