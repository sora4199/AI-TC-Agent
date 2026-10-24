from sentence_transformers import CrossEncoder


class Reranker:
    """
    Rerank candidate documents based on query-document relevance.

    Input:
        {
            "query": str,
            "documents": [
                {
                    "id": str,
                    "content": str,
                    "metadata": dict
                }
            ],
            "top_k": int
        }

    Output:
        {
            "documents": [
                {
                    "document": {
                        "id": str,
                        "content": str,
                        "metadata": dict
                    },
                    "score": float,
                    "rank": int
                }
            ]
        }

    Note:
        Reranker does NOT perform retrieval.
        It only scores and ranks candidate documents.
    """

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=5):
        """
        Rerank candidate documents.

        Args:
            query: User query.
            documents: Candidate documents from Retriever.
            top_k: Number of documents to return.

        Returns:
            RerankResult
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if not isinstance(documents, list):
            raise ValueError("documents must be a list")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if not documents:
            return {
                "documents": []
            }

        # Không trả nhiều hơn số candidate documents
        top_k = min(top_k, len(documents))

        # ====================================================
        # CREATE QUERY-DOCUMENT PAIRS
        # ====================================================

        pairs = [
            [query, document["content"]]
            for document in documents
        ]

        # ====================================================
        # SCORE DOCUMENTS
        # ====================================================

        scores = self.model.predict(pairs)

        # ====================================================
        # SORT BY SCORE
        # ====================================================

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        # ====================================================
        # BUILD RESULT
        # ====================================================

        ranked_documents = []

        for rank, (document, score) in enumerate(
            ranked[:top_k],
            start=1,
        ):
            ranked_documents.append(
                {
                    "document": document,
                    "score": float(score),
                    "rank": rank,
                }
            )

        return {
            "documents": ranked_documents
        }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":
    from src.tools.retriever import Retriever

    # ========================================================
    # Initialize Retriever + Reranker
    # ========================================================

    retriever = Retriever()
    reranker = Reranker()

    # ========================================================
    # Query
    # ========================================================

    query = "Đội bóng Boca Juniors mà Diego Maradona đã chơi có trụ sở tại La Boca, gần Buenos Aires phải không?"

    # Retriever lấy 20 candidate documents
    retrieval_result = retriever.retrieve(
        query=query,
        top_k=20,
    )

    candidates = retrieval_result["documents"]

    print(f"\nRetrieved {len(candidates)} candidates")

    # ========================================================
    # Rerank
    # ========================================================

    result = reranker.rerank(
        query=query,
        documents=candidates,
        top_k=5,
    )

    # ========================================================
    # Print Result
    # ========================================================

    print("\n=== RERANK RESULT ===")

    for item in result["documents"]:
        document = item["document"]

        print(f"\nRank: {item['rank']}")
        print(f"Score: {item['score']:.4f}")
        print(f"ID: {document['id']}")
        print(f"Content: {document['content']}")