# Module Communication Schema

Tài liệu này định nghĩa **input/output contract giữa các module**.

Mỗi thành viên khi implement module của mình cần đảm bảo input/output đúng theo schema bên dưới.

---

# 1. Router

## Input

Router nhận:

```text
Query
├── query: string
├── session_id: string
├── user_id: string | null
└── metadata: object
```

## Output

Router phải trả:

```text
RouteDecision
├── route: "agent" | "rag" | "ood"
├── confidence: float
└── reason: string | null
```

Example:

```json
{
  "route": "rag",
  "confidence": 0.91,
  "reason": "Query có thể được trả lời từ knowledge base"
}
```

---

# 2. Agent

## Input

Agent nhận:

```text
Query
├── query: string
├── session_id: string
├── user_id: string | null
└── metadata: object
```

Trong quá trình xử lý, Agent có thể gọi các Tools.

## Retriever Tool Input

```text
RetrievalRequest
├── query: string
└── top_k: integer
```

## Retriever Tool Output

```text
RetrievalResult
├── documents: Document[]
└── metadata: object
```

### Document

```text
Document
├── id: string
├── content: string
└── metadata: object
```

---

## Reranker Tool Input

```text
RerankRequest
├── query: string
├── documents: Document[]
└── top_k: integer
```

## Reranker Tool Output

```text
RerankResult
└── documents: RankedDocument[]
```

### RankedDocument

```text
RankedDocument
├── document: Document
├── score: float
└── rank: integer
```

---

## Memory Tool Input

Memory chỉ cung cấp khả năng **read** cho Agent.

```text
MemoryRequest
├── query: string
├── session_id: string
└── top_k: integer
```

## Memory Tool Output

```text
MemoryResult
├── memories: Memory[]
└── metadata: object
```

### Memory

```text
Memory
├── id: string
├── content: string
└── metadata: object
```

Agent **không trực tiếp write vào Memory**.

Việc tạo/cập nhật Memory được hệ thống Memory tự quản lý.

---

## Output

Agent phải trả:

```text
Response
├── answer: string
├── sources: Source[]
└── metadata: object
```

---

# 3. RAG

## Input

RAG nhận:

```text
Query
├── query: string
├── session_id: string
├── user_id: string | null
└── metadata: object
```

## Internal Communication

RAG có thể sử dụng Retriever.

### Retriever Input

```text
RetrievalRequest
├── query: string
└── top_k: integer
```

### Retriever Output

```text
RetrievalResult
├── documents: Document[]
└── metadata: object
```

### Document

```text
Document
├── id: string
├── content: string
└── metadata: object
```

---

RAG có thể sử dụng Reranker.

### Reranker Input

```text
RerankRequest
├── query: string
├── documents: Document[]
└── top_k: integer
```

### Reranker Output

```text
RerankResult
└── documents: RankedDocument[]
```

### RankedDocument

```text
RankedDocument
├── document: Document
├── score: float
└── rank: integer
```

## Output

RAG phải trả:

```text
Response
├── answer: string
├── sources: Source[]
└── metadata: object
```

---

# 4. Retriever

## Input

Retriever nhận:

```text
RetrievalRequest
├── query: string
└── top_k: integer
```

Example:

```json
{
  "query": "cách chuẩn bị lễ giỗ",
  "top_k": 20
}
```

## Output

Retriever phải trả:

```text
RetrievalResult
├── documents: Document[]
└── metadata: object
```

Example:

```json
{
  "documents": [
    {
      "id": "doc_001",
      "content": "Trong ngày giỗ...",
      "metadata": {}
    }
  ],
  "metadata": {
    "num_results": 20
  }
}
```

### Document

```text
Document
├── id: string
├── content: string
└── metadata: object
```

### Requirement

Retriever **không tự rerank** kết quả.

Retriever chỉ chịu trách nhiệm:

```text
Query + top_k
      ↓
Candidate Documents
```

---

# 5. Reranker

## Input

Reranker nhận:

```text
RerankRequest
├── query: string
├── documents: Document[]
└── top_k: integer
```

Example:

```json
{
  "query": "cách chuẩn bị lễ giỗ",
  "documents": [
    {
      "id": "doc_001",
      "content": "...",
      "metadata": {}
    }
  ],
  "top_k": 5
}
```

## Output

Reranker phải trả:

```text
RerankResult
└── documents: RankedDocument[]
```

### RankedDocument

```text
RankedDocument
├── document: Document
├── score: float
└── rank: integer
```

Example:

```json
{
  "documents": [
    {
      "document": {
        "id": "doc_001",
        "content": "...",
        "metadata": {}
      },
      "score": 0.94,
      "rank": 1
    }
  ]
}
```

### Requirement

Reranker **không thực hiện retrieval**.

Reranker chỉ chịu trách nhiệm:

```text
Query + Candidate Documents
            ↓
      Ranked Documents
```

---

# 6. Memory

Memory cung cấp khả năng **read** thông tin cho Agent.

Việc write/update Memory được hệ thống Memory tự quản lý và **không nằm trong public interface của Agent**.

## Input

Memory nhận:

```text
MemoryRequest
├── query: string
├── session_id: string
└── top_k: integer
```

Example:

```json
{
  "query": "previous user preferences",
  "session_id": "session_001",
  "top_k": 5
}
```

## Output

Memory phải trả:

```text
MemoryResult
├── memories: Memory[]
└── metadata: object
```

### Memory

```text
Memory
├── id: string
├── content: string
└── metadata: object
```

Example:

```json
{
  "memories": [
    {
      "id": "memory_001",
      "content": "User prefers concise answers",
      "metadata": {}
    }
  ],
  "metadata": {
    "num_results": 1
  }
}
```

### Requirement

Agent chỉ có quyền:

```text
Query → Read Memory
```

Agent không chịu trách nhiệm:

```text
Agent → Write Memory
```

---

# 7. Final Response

Agent và RAG sử dụng chung `Response`.

```text
Response
├── answer: string
├── sources: Source[]
└── metadata: object
```

### Source

```text
Source
├── document_id: string
├── content: string
└── metadata: object
```

Example:

```json
{
  "answer": "Theo tài liệu...",
  "sources": [
    {
      "document_id": "doc_001",
      "content": "Trong ngày giỗ...",
      "metadata": {}
    }
  ],
  "metadata": {
    "route": "rag"
  }
}
```

---

# 8. Interface Summary

| Module    | Input              | Output            |
| --------- | ------------------ | ----------------- |
| Router    | `Query`            | `RouteDecision`   |
| Agent     | `Query`            | `Response`        |
| RAG       | `Query`            | `Response`        |
| Retriever | `RetrievalRequest` | `RetrievalResult` |
| Reranker  | `RerankRequest`    | `RerankResult`    |
| Memory    | `MemoryRequest`    | `MemoryResult`    |

---

# 9. Implementation Rules

1. **Không thay đổi field của Input/Output nếu chưa thống nhất với team.**

2. Có thể tự do thay đổi implementation bên trong module.

3. Output phải đúng schema đã định nghĩa.

4. Nếu cần thêm field mới, thông báo với leader.

5. Các module khác chỉ được sử dụng **public interface**, không phụ thuộc vào implementation nội bộ.

6. Retriever chỉ retrieval, **không rerank**.

7. Reranker chỉ rerank, **không retrieval**.

8. Agent chỉ **read Memory**, không trực tiếp write Memory.

---

# 10. Current Contract

```text
Router

    Query → RouteDecision


Agent

    Query → Response

    RetrievalRequest → RetrievalResult

    RerankRequest → RerankResult

    MemoryRequest → MemoryResult


RAG

    Query → Response

    RetrievalRequest → RetrievalResult

    RerankRequest → RerankResult


Retriever

    RetrievalRequest → RetrievalResult


Reranker

    RerankRequest → RerankResult


Memory

    MemoryRequest → MemoryResult
```

**Contract version: v0**

