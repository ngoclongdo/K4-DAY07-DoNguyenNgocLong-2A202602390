"""
bench.py - Công cụ đo lường và đánh giá chất lượng truy xuất RAG (Lab 07 - K4-L3B).

Quy trình thực hiện:
1. Đọc từng file .md trong data/e_comercial/, tách frontmatter thành metadata và body thành content.
2. Chunk phần thân văn bản. Mỗi chunk tạo thành một Document:
       Document(id=f"{path.stem}#{i}", content=chunk,
                metadata={**frontmatter, "doc_id": path.stem})
3. Nạp vào EmbeddingStore, chạy 5 câu hỏi benchmark qua search_with_filter().
4. In top-3 kết quả kèm score và doc_id để đối chiếu với Gold Answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable

# Đảm bảo UTF-8 cho terminal Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    MockEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

# ---------------------------------------------------------------------------
# 1. Cấu hình Chunker của bạn (Mỗi thành viên chỉ đổi 1 dòng này để so sánh)
# Gợi ý:
#   - FixedSizeChunker(chunk_size=300, overlap=50)
#   - SentenceChunker(max_sentences_per_chunk=3)
#   - RecursiveChunker(chunk_size=350)
# ---------------------------------------------------------------------------
ACTIVE_CHUNKER = RecursiveChunker(chunk_size=350)
CHUNKER_NAME = "RecursiveChunker(chunk_size=350)"

DATA_DIR = Path("data/e_comercial")

# ---------------------------------------------------------------------------
# 2. Bộ câu hỏi Benchmark của nhóm
# ---------------------------------------------------------------------------
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Màn hình điện thoại mua tại Revibe được bảo hành trong bao lâu và có những điều kiện loại trừ nào?",
        "gold_doc": "revibe-terms-and-warranty",
        "gold_answer": "Màn hình chỉ được bảo hành trong 10 ngày đầu tiên kể từ ngày mua. Sau 10 ngày, bảo hành chỉ áp dụng cho các linh kiện khác ngoại trừ màn hình. Các trường hợp như rơi vỡ vật lý, vào nước hoặc tự ý sửa chữa sẽ bị từ chối bảo hành.",
        "filter": None,
    },
    {
        "id": 2,
        "query": "Khi mua hàng tại Newegg, phí hoàn kho (restocking fee) áp dụng cho những mặt hàng nào và mức phí là bao nhiêu?",
        "gold_doc": "newegg-return-policy",
        "gold_answer": "Mức phí hoàn kho là 15%, áp dụng cho các sản phẩm đã bị bóc hộp/mở seal thuộc các danh mục: ổ cứng (HD), bo mạch chủ (MB), card đồ họa (VGA), máy chiếu (Projector) và tivi (TV). Sản phẩm còn nguyên seal hoặc bị lỗi/hỏng do vận chuyển không bị tính phí này.",
        "filter": None,
    },
    {
        "id": 3,
        "query": "Khi đơn hàng giao thành công mà hai bên không phát sinh khiếu nại, sau bao nhiêu ngày tiền thanh toán sẽ được giải ngân vào Số Dư Tài Khoản?",
        "gold_doc": "shopee-returns-seller",
        "gold_answer": "Nếu Người Mua không bấm 'Đã nhận được hàng' và không yêu cầu trả hàng, Shopee sẽ giải ngân tiền cho Người Bán nhanh nhất vào ngày thứ 04 (bốn) kể từ khi đơn hàng cập nhật trạng thái 'Giao Hàng Thành Công'.",
        "filter": {"audience": "seller"},
        "is_ab_test": True,  # Đánh dấu để chạy cả 2 chế độ: Có filter và Không filter
    },
    {
        "id": 4,
        "query": "Khách hàng mua hàng tại Refurbed cần làm những bước nào trước khi đóng gói gửi trả thiết bị?",
        "gold_doc": "refurbed-return-policy",
        "gold_answer": "Khách hàng cần: (1) Sao lưu dữ liệu và khôi phục cài đặt gốc (factory settings); (2) Đăng xuất khỏi mọi tài khoản bảo mật cá nhân (iCloud, Google); (3) Đóng gói bằng hộp carton chèn đệm lót, không dùng phong bì đệm khí; (4) In và dán nhãn hoàn trả mới, bóc nhãn cũ; (5) Tuyệt đối không gửi thiết bị có pin bị phồng (swollen battery).",
        "filter": None,
    },
    {
        "id": 5,
        "query": "Theo quy định Shopee, Người Bán phải chịu những loại phí cố định và phí giao dịch nào trên mỗi đơn hàng thành công?",
        "gold_doc": "shopee-returns-seller",
        "gold_answer": "Người Bán chịu 2 loại phí chính: (1) Phí Xử Lý Giao Dịch là 6% (đã bao gồm thuế GTGT) áp dụng cho tất cả phương thức thanh toán; (2) Phí Cố Định (hoa hồng sàn) tính theo tỷ lệ phần trăm tùy theo ngành hàng trên giá bán sản phẩm.",
        "filter": None,
    },
]


def parse_markdown_file(path: Path) -> tuple[dict, str]:
    """Tách frontmatter YAML và phần thân văn bản Markdown."""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            raw_fm, body = parts[1], parts[2]
            metadata = {}
            for line in raw_fm.strip().split("\n"):
                line = line.strip()
                if ":" in line:
                    k, v = line.split(":", 1)
                    metadata[k.strip()] = v.strip().strip("\"'")
            return metadata, body.strip()
    return {}, text.strip()


def get_cached_embedder() -> Callable[[str], list[float]]:
    """Khởi tạo Embedder và tích hợp bộ nhớ đệm (Cache) theo hash để tiết kiệm chi phí/thời gian."""
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    base_embedder = _mock_embed

    if provider == "openai":
        try:
            base_embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
            print("[INFO] Su dung OpenAIEmbedder")
        except Exception as e:
            print(f"[CANH BAO] Khong tai duoc OpenAIEmbedder ({e}), chuyen sang MockEmbedder")
    elif provider == "gemini":
        try:
            base_embedder = GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
            print("[INFO] Su dung GeminiEmbedder")
        except Exception as e:
            print(f"[CANH BAO] Khong tai duoc GeminiEmbedder ({e}), chuyen sang MockEmbedder")
    elif provider == "local":
        try:
            base_embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
            print("[INFO] Su dung LocalEmbedder")
        except Exception as e:
            print(f"[CANH BAO] Khong tai duoc LocalEmbedder ({e}), chuyen sang MockEmbedder")
    else:
        print("[INFO] Su dung MockEmbedder (deterministic hash)")

    cache_file = Path(".embedding_cache.json")
    cache: dict[str, list[float]] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    def cached_embedder(text: str) -> list[float]:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in cache:
            return cache[text_hash]
        vec = base_embedder(text)
        cache[text_hash] = vec
        return vec

    import atexit

    def save_cache():
        try:
            cache_file.write_text(json.dumps(cache), encoding="utf-8")
        except Exception:
            pass

    atexit.register(save_cache)
    return cached_embedder


def build_knowledge_base(chunker, embedder) -> tuple[EmbeddingStore, list[Document]]:
    """Đọc file từ data/e_comercial, chunk ngoài store, nạp Document có đủ metadata."""
    store = EmbeddingStore(collection_name="benchmark_kb", embedding_fn=embedder)
    all_chunks: list[Document] = []

    md_files = sorted(DATA_DIR.glob("*.md"))
    print(f"\n[BUOC 1 & 2] Nap va Chunking tai lieu tu {DATA_DIR}...")

    for path in md_files:
        fm, body = parse_markdown_file(path)
        chunks_text = chunker.chunk(body)

        for i, chunk in enumerate(chunks_text):
            chunk_id = f"{path.stem}#{i}"
            # Lan tỏa frontmatter vào từng chunk, doc_id trỏ về stem của file gốc
            metadata = {
                **fm,
                "doc_id": path.stem,
                "chunk_index": i,
                "source_file": str(path),
            }
            doc = Document(id=chunk_id, content=chunk, metadata=metadata)
            all_chunks.append(doc)

    print(f"-> Da nap {len(md_files)} file .md, sinh ra tong cong {len(all_chunks)} chunks.")
    store.add_documents(all_chunks)
    return store, all_chunks


def run_benchmark(store: EmbeddingStore):
    """Chạy 5 query đánh giá và in bảng kết quả chi tiết."""
    print("\n" + "=" * 90)
    print(f"KET QUA BENCHMARK RETRIEVAL - CHIEN LUOC: {CHUNKER_NAME}")
    print("=" * 90)

    for q in BENCHMARK_QUERIES:
        qid = q["id"]
        query = q["query"]
        gold_doc = q["gold_doc"]
        meta_filter = q["filter"]

        print(f"\n[CAU {qid}] {query}")
        print(f"  * Gold Doc   : {gold_doc}")
        print(f"  * Filter     : {meta_filter}")
        print(f"  * Gold Answer: {q['gold_answer']}")

        results = store.search_with_filter(query, top_k=3, metadata_filter=meta_filter)

        print("  * Top-3 Retrieved Chunks:")
        for rank, r in enumerate(results, 1):
            r_doc_id = r.get("metadata", {}).get("doc_id", "N/A")
            score = r.get("score", 0.0)
            snippet = r.get("content", "").replace("\n", " ").strip()
            if len(snippet) > 110:
                snippet = snippet[:110] + "..."
            match_icon = "[V] (GOLD DOC)" if r_doc_id == gold_doc else "[X]"
            print(f"    [{rank}] Score: {score:+.4f} | doc_id: {r_doc_id:30} {match_icon}")
            print(f"        Trich doan: \"{snippet}\"")

        # Nếu là câu A/B test: Chạy đối chiếu thêm 1 lần khi KHÔNG CÓ filter
        if q.get("is_ab_test"):
            print("  --- [A/B TEST DOI CHUNG: KHONG DUNG FILTER] ---")
            unfiltered_results = store.search_with_filter(query, top_k=3, metadata_filter=None)
            for rank, r in enumerate(unfiltered_results, 1):
                r_doc_id = r.get("metadata", {}).get("doc_id", "N/A")
                score = r.get("score", 0.0)
                match_icon = "[V] (GOLD DOC)" if r_doc_id == gold_doc else "[X]"
                print(f"    [{rank}] Score: {score:+.4f} | doc_id: {r_doc_id:30} {match_icon}")


if __name__ == "__main__":
    embedder = get_cached_embedder()
    store, docs = build_knowledge_base(ACTIVE_CHUNKER, embedder)
    run_benchmark(store)
