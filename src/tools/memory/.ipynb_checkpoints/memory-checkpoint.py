from typing import List, Dict, Any
import re


class Memory:

    def __init__(
        self,
        mode: str = "retriever_based",
        recent_k: int = 2
    ):
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        self.mode = mode
        self.recent_k = recent_k

        if mode not in [
            "no_memory",
            "full_history",
            "retriever_based"
        ]:
            raise ValueError(
                "mode phải là: no_memory, full_history hoặc retriever_based"
            )

    # =====================================================
    # INTERNAL WRITE
    # Hệ thống Memory tự lưu hội thoại
    # Agent không gọi hàm này
    # =====================================================

    def add_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str
    ):
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        turn_id = len(self.sessions[session_id]) + 1

        memory = {
            "id": f"memory_{session_id}_{turn_id}",
            "content": (
                f"User: {user_query}\n"
                f"Assistant: {assistant_response}"
            ),
            "metadata": {
                "session_id": session_id,
                "turn": turn_id
            }
        }

        self.sessions[session_id].append(memory)

    # =====================================================
    # INTERNAL
    # Xóa Memory của một session
    # =====================================================

    def clear_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    # =====================================================
    # PUBLIC INTERFACE
    #
    # MemoryRequest
    #   query
    #   session_id
    #   top_k
    # ->
    # MemoryResult
    #   memories
    #   metadata
    # =====================================================

    def search(
        self,
        query: str,
        session_id: str,
        top_k: int
    ) -> Dict[str, Any]:

        if not isinstance(query, str):
            raise TypeError("query phải là string")

        if not isinstance(session_id, str):
            raise TypeError("session_id phải là string")

        if not isinstance(top_k, int):
            raise TypeError("top_k phải là integer")

        if top_k < 0:
            raise ValueError("top_k phải >= 0")

        memories = self.sessions.get(session_id, [])

        # -------------------------------------------------
        # NO MEMORY
        # -------------------------------------------------

        if self.mode == "no_memory":

            return {
                "memories": [],
                "metadata": {
                    "num_results": 0
                }
            }

        # -------------------------------------------------
        # FULL HISTORY
        # -------------------------------------------------

        if self.mode == "full_history":

            selected = memories[-top_k:] if top_k > 0 else []

            return {
                "memories": selected,
                "metadata": {
                    "num_results": len(selected)
                }
            }

        # -------------------------------------------------
        # RETRIEVER BASED
        # -------------------------------------------------

        selected = self._retrieve(
            query,
            memories,
            top_k
        )

        return {
            "memories": selected,
            "metadata": {
                "num_results": len(selected)
            }
        }

    # =====================================================
    # RETRIEVER BASED
    # =====================================================

    def _retrieve(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:

        if not memories or top_k == 0:
            return []

        if len(memories) <= top_k:
            return memories.copy()

        # Lấy một số memory gần nhất để giữ context
        recent = memories[-self.recent_k:]

        older = memories[:-self.recent_k]

        # Nếu query có từ chỉ context
        context_words = [
            "cái này",
            "cái đó",
            "món này",
            "món đó",
            "nó",
            "đó",
            "này",
            "như lúc nãy",
            "như trước",
            "vừa nãy"
        ]

        search_query = query.lower()

        if any(
            word in search_query
            for word in context_words
        ):
            search_query += " " + " ".join(
                memory["content"]
                for memory in recent
            )

        query_tokens = self._tokenize(search_query)

        scored = []

        for memory in older:

            memory_tokens = self._tokenize(
                memory["content"]
            )

            score = self._similarity(
                query_tokens,
                memory_tokens
            )

            scored.append(
                (score, memory)
            )

        # Memory liên quan nhất trước
        scored.sort(
            key=lambda x: x[0],
            reverse=True
        )

        number_of_old_memories = max(
            0,
            top_k - len(recent)
        )

        selected_old = [
            memory
            for score, memory in scored[
                :number_of_old_memories
            ]
            if score > 0
        ]

        result = selected_old + recent

        return result[:top_k]

    # =====================================================
    # TOKENIZE
    # =====================================================
    @staticmethod
    def _tokenize(text: str) -> set:

        words = re.findall(
            r"\b\w+\b",
            text.lower(),
            flags=re.UNICODE
        )

        return set(words)

    # =====================================================
    # SIMILARITY
    # =====================================================

    @staticmethod
    def _similarity(
        query_words: set,
        memory_words: set
    ) -> float:

        if not query_words or not memory_words:
            return 0.0

        intersection = query_words & memory_words
        union = query_words | memory_words

        return len(intersection) / len(union)


# =========================================================
# TEST MEMORY
# =========================================================

if __name__ == "__main__":

    memory = Memory(
        mode="retriever_based",
        recent_k=2
    )

    session_id = "session_001"

    # Hệ thống tự lưu hội thoại
    memory.add_turn(
        session_id,
        "Tôi đang học Python",
        "Bạn nên học function và loop trước."
    )

    memory.add_turn(
        session_id,
        "Tôi muốn học list",
        "List dùng để lưu nhiều phần tử."
    )

    memory.add_turn(
        session_id,
        "Cho tôi ví dụ",
        "Ví dụ: numbers = [1, 2, 3]."
    )

    # Agent chỉ READ Memory
    result = memory.search(
        query="cái này",
        session_id=session_id,
        top_k=3
    )

    print(result)      