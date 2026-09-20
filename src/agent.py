from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Answers a user question based strictly on retrieved context from the knowledge base.
        """
        # Nếu store rỗng, trả thông báo an toàn, không gọi LLM vô ích
        if self.store.get_collection_size() == 0:
            return "Cơ sở tri thức hiện chưa có tài liệu nào để trả lời câu hỏi."

        # 1. Truy xuất top-k chunk liên quan nhất
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin phù hợp trong tài liệu để trả lời câu hỏi."

        # 2. Dựng ngữ cảnh có đánh số [1], [2], ... kèm nguồn để hỗ trợ Source Traceability
        context_blocks = []
        for i, r in enumerate(results, 1):
            doc_id = r.get("metadata", {}).get("doc_id") or r.get("id", "unknown")
            content = r.get("content", "").strip()
            context_blocks.append(f"[{i}] (Nguồn: {doc_id})\n{content}")

        context_str = "\n\n".join(context_blocks)

        # 3. Dựng prompt chặt chẽ chống ảo giác (hallucination)
        prompt = (
            "Bạn là trợ lý AI trả lời câu hỏi dựa trên tài liệu được cung cấp.\n"
            "Chỉ sử dụng thông tin trong phần Ngữ cảnh dưới đây để trả lời. "
            "Nếu thông tin không có trong ngữ cảnh, hãy nói rõ là không tìm thấy, tuyệt đối không suy đoán hay bịa đặt thông tin. "
            "Trích dẫn số thứ tự nguồn [1], [2] tương ứng khi sử dụng thông tin.\n\n"
            f"--- NGỮ CẢNH ---\n{context_str}\n\n"
            f"--- CÂU HỎI ---\n{question}\n\n"
            "--- CÂU TRẢ LỜI ---"
        )

        return self.llm_fn(prompt)
