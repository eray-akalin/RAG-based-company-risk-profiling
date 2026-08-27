"""
Semantic Retriever with Optional Cross-Encoder Reranking

Retrieves relevant evidence chunks from the FAISS index using
semantic search, with optional cross-encoder reranking.
"""

import os
import re
import pickle
import numpy as np
import faiss  # type: ignore[import-not-found]
from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore[import-not-found]

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    EMBEDDINGS_DIR, EMBEDDING_MODEL,
    TOP_K, RETRIEVE_TOP_K, RERANK_ENABLED, RERANKER_MODEL,
    HYBRID_ENABLED, BM25_TOP_K, RRF_K,
    RELEVANCE_KEEP_RATIO, RELEVANCE_FLOOR,
    get_embeddings_dir,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function/question words carry no topical signal but, in natural-language
# query templates ("What does the company face?"), they let BM25 match
# off-topic chunks and inject noise into the fusion. Dropping them sharpens
# BM25 toward content terms.
_STOPWORDS = frozenset("""
a an the this that these those of to in on at by for from with without within
and or but nor so as is are was were be been being am do does did doing done
have has had having will would shall should can could may might must
what which who whom whose when where why how
we our us you your they their them it its he she his her i me my
not no any some all each other such than then there here about into over under
company companies business operations operating results financial condition
risk risks could would may also include including related due
""".split())


def _tokenize(text: str) -> list[str]:
    """Lowercase content-word tokenizer for BM25 lexical matching.
    Drops stopwords and single characters so BM25 keys on topical terms."""
    return [t for t in _TOKEN_RE.findall(text.lower())
            if len(t) > 1 and t not in _STOPWORDS]


def _sigmoid(x: float) -> float:
    """Map a cross-encoder logit to a [0, 1] relevance probability."""
    return float(1.0 / (1.0 + np.exp(-x)))


class SemanticRetriever:
    """
    Retrieves relevant chunks from the FAISS index.
    Supports:
    - Basic semantic retrieval (embedding similarity)
    - Cross-encoder reranking for improved precision
    - Metadata filtering (by company, year)
    """

    def __init__(self, index_dir: str = None, year: int = None):
        """
        Initialize the retriever by loading the FAISS index and metadata.

        Args:
            index_dir: Directory containing the FAISS index files
            year: Fiscal year to load index for (uses year-specific dir)
        """
        if index_dir is None:
            if year is not None:
                # Try year-specific directory first
                year_dir = get_embeddings_dir(year)
                if os.path.exists(os.path.join(year_dir, "faiss_index.bin")):
                    index_dir = year_dir
                else:
                    index_dir = EMBEDDINGS_DIR
            else:
                index_dir = EMBEDDINGS_DIR

        self.index_dir = index_dir
        self.year = year
        self.index = None
        self.chunk_metadata = None
        self.embedding_model = None
        self.reranker_model = None
        self.bm25 = None
        self._bm25_ids = []  # FAISS idx aligned with BM25 corpus order

        self._load_index()
        if HYBRID_ENABLED:
            self._build_bm25()

    def _load_index(self):
        """Load the FAISS index and chunk metadata."""
        # Load FAISS index
        index_path = os.path.join(self.index_dir, "faiss_index.bin")
        self.index = faiss.read_index(index_path)
        print(f"Loaded FAISS index: {self.index.ntotal} vectors")

        # Load chunk metadata
        meta_path = os.path.join(self.index_dir, "chunk_metadata.pkl")
        with open(meta_path, "rb") as f:
            self.chunk_metadata = pickle.load(f)
        print(f"Loaded metadata for {len(self.chunk_metadata)} chunks")

    def _get_embedding_model(self):
        """Lazy-load the embedding model (process-wide cached singleton)."""
        if self.embedding_model is None:
            from src.model_cache import get_embedding_model
            self.embedding_model = get_embedding_model(EMBEDDING_MODEL)
        return self.embedding_model

    def _get_reranker_model(self):
        """Lazy-load the cross-encoder reranker (process-wide cached singleton)."""
        if self.reranker_model is None and RERANK_ENABLED:
            from src.model_cache import get_reranker_model
            self.reranker_model = get_reranker_model(RERANKER_MODEL)
        return self.reranker_model

    def _build_bm25(self):
        """Build an in-memory BM25 index over the chunk corpus."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("WARNING: rank_bm25 not installed; falling back to dense-only.")
            self.bm25 = None
            return

        self._bm25_ids = list(self.chunk_metadata.keys())
        corpus = [
            _tokenize(self.chunk_metadata[i].get("text", ""))
            for i in self._bm25_ids
        ]
        self.bm25 = BM25Okapi(corpus)
        print(f"Built BM25 index over {len(corpus)} chunks")

    def _dense_search(self, query: str, k: int) -> list[tuple]:
        """Return [(faiss_idx, score), ...] from the dense FAISS index."""
        query_embedding = self.embed_query(query)
        scores, indices = self.index.search(query_embedding, k)
        return [
            (int(idx), float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    def _bm25_search(self, query: str, k: int) -> list[tuple]:
        """Return [(faiss_idx, score), ...] from the BM25 lexical index."""
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [
            (self._bm25_ids[j], float(scores[j]))
            for j in top
            if scores[j] > 0
        ]

    @staticmethod
    def _rrf_fuse(rankings: list[list[tuple]], rrf_k: int) -> list[int]:
        """
        Reciprocal Rank Fusion: combine multiple ranked lists into one
        ordering. Each list is [(idx, score), ...] sorted best-first.
        Returns idx values ordered by fused score (descending).
        """
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, (idx, _score) in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        return [idx for idx, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string.
        For bge models, queries should be prefixed with "Represent this sentence: "
        """
        model = self._get_embedding_model()

        # BGE models recommend a query prefix
        if "bge" in EMBEDDING_MODEL.lower():
            query = "Represent this sentence: " + query

        embedding = model.encode(
            [query],
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        company_filter: str = None,
        year_filter: str = None,
        rerank: bool = None,
    ) -> list[dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The search query
            top_k: Number of results to return (default from config)
            company_filter: Only return chunks from this company (ticker)
            year_filter: Only return chunks from this year
            rerank: Whether to use cross-encoder reranking (default from config)

        Returns:
            List of dicts with chunk text, metadata, and scores
        """
        if top_k is None:
            top_k = TOP_K
        if rerank is None:
            rerank = RERANK_ENABLED

        # BM25 presence is the single source of truth: production builds it only
        # when HYBRID_ENABLED, while the evaluator can force-build it to ablate.
        use_hybrid = self.bm25 is not None

        # Determine how many candidates to pull before reranking/truncation.
        initial_k = RETRIEVE_TOP_K if rerank else top_k
        if company_filter or year_filter:
            # Retrieve more to survive metadata filtering.
            initial_k = min(initial_k * 3, self.index.ntotal)

        # 1) Gather candidates (dense, plus lexical when hybrid is on).
        dense_hits = self._dense_search(query, initial_k)
        dense_scores = {idx: score for idx, score in dense_hits}

        if use_hybrid:
            bm25_k = max(BM25_TOP_K, initial_k)
            bm25_hits = self._bm25_search(query, bm25_k)
            bm25_scores = {idx: score for idx, score in bm25_hits}
            ordered_idx = self._rrf_fuse([dense_hits, bm25_hits], RRF_K)
        else:
            bm25_scores = {}
            ordered_idx = [idx for idx, _ in dense_hits]

        # 2) Build results with metadata + filters, preserving fused order.
        results = []
        for idx in ordered_idx:
            meta = self.chunk_metadata.get(int(idx), {})

            if company_filter and meta.get("company") != company_filter:
                continue
            if year_filter and meta.get("year") != year_filter:
                continue

            results.append({
                "chunk_id": meta.get("chunk_id", ""),
                "company": meta.get("company", ""),
                "company_name": meta.get("company_name", ""),
                "year": meta.get("year", ""),
                "section": meta.get("section", ""),
                "chunk_index": meta.get("chunk_index", -1),
                "text": meta.get("text", ""),
                "faiss_score": dense_scores.get(idx, 0.0),
                "bm25_score": bm25_scores.get(idx, 0.0),
                "retrieval_mode": "hybrid" if use_hybrid else "dense",
                "faiss_rank": len(results) + 1,
            })

        # 3) Rerank (cross-encoder) or truncate.
        if rerank and results:
            results = self._rerank(query, results, top_k)
        else:
            results = results[:top_k]

        # 4) Attach a normalized [0, 1] relevance score for gating downstream.
        for r in results:
            if "rerank_score" in r:
                r["relevance"] = _sigmoid(r["rerank_score"])
            else:
                # Cosine similarity from normalized embeddings is already ~[0, 1].
                r["relevance"] = max(0.0, min(1.0, r.get("faiss_score", 0.0)))

        return results

    def _rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        """
        Rerank results using a cross-encoder model.
        Cross-encoders jointly encode query-document pairs for more accurate scoring.
        """
        reranker = self._get_reranker_model()
        if reranker is None:
            return results[:top_k]

        # Prepare query-document pairs
        pairs = [(query, r["text"]) for r in results]

        # Score with cross-encoder
        rerank_scores = reranker.predict(pairs)

        # Add rerank scores and sort
        for i, score in enumerate(rerank_scores):
            results[i]["rerank_score"] = float(score)

        results.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Update ranks
        for i, r in enumerate(results[:top_k]):
            r["final_rank"] = i + 1

        return results[:top_k]

    def retrieve_for_risk_category(
        self,
        risk_category: dict,
        company: str = None,
        top_k: int = None,
        keep_ratio: float = None,
        floor: float = None,
    ) -> list[dict]:
        """
        Retrieve evidence for a specific risk category using multiple query templates.
        Aggregates and deduplicates results across all queries.

        Gating is RELATIVE to the best chunk for this category (cross-encoder
        scores are not comparable across categories): keep chunks whose
        relevance is >= keep_ratio * best_relevance, provided the best chunk
        clears the absolute `floor`. Returns a variable number of chunks
        (0..top_k) based on evidence strength.

        Args:
            risk_category: Risk category dict from config (with query_templates)
            company: Optional company ticker filter
            top_k: Maximum number of final results to return
            keep_ratio: Fraction of the best relevance a chunk must reach.
                Defaults to RELEVANCE_KEEP_RATIO. Pass 0.0 to disable gating.
            floor: Absolute noise floor for the best chunk. Defaults to
                RELEVANCE_FLOOR.

        Returns:
            Deduplicated, relevance-gated, ranked list of evidence chunks
            (may be empty when the category has no real evidence).
        """
        if top_k is None:
            top_k = TOP_K
        if keep_ratio is None:
            keep_ratio = RELEVANCE_KEEP_RATIO
        if floor is None:
            floor = RELEVANCE_FLOOR

        all_results = {}

        for query_template in risk_category["query_templates"]:
            results = self.retrieve(
                query=query_template,
                top_k=top_k,
                company_filter=company,
                rerank=RERANK_ENABLED,
            )
            for r in results:
                chunk_id = r["chunk_id"]
                if chunk_id not in all_results:
                    all_results[chunk_id] = r
                    all_results[chunk_id]["query_matches"] = 1
                else:
                    # Chunk appeared in multiple queries — boost its score
                    all_results[chunk_id]["query_matches"] += 1
                    # Keep the highest score
                    score_key = "rerank_score" if "rerank_score" in r else "faiss_score"
                    if r[score_key] > all_results[chunk_id].get(score_key, 0):
                        all_results[chunk_id][score_key] = r[score_key]

        if not all_results:
            return []

        # Relative gating: drop chunks far below this category's best chunk.
        best_rel = max(r.get("relevance", 0.0) for r in all_results.values())
        if best_rel < floor:
            return []  # no real evidence for this category
        min_keep = keep_ratio * best_rel
        gated = [
            r for r in all_results.values()
            if r.get("relevance", 0.0) >= min_keep
        ]

        # Sort by query_matches (more = better), then by relevance
        sorted_results = sorted(
            gated,
            key=lambda x: (
                x.get("query_matches", 1),
                x.get("relevance", x.get("rerank_score", 0)),
            ),
            reverse=True,
        )

        return sorted_results[:top_k]

    def keyword_search(
        self,
        query: str,
        top_k: int = None,
        company_filter: str = None,
    ) -> list[dict]:
        """
        Baseline keyword search using simple term matching.
        Used as a comparison baseline for evaluation.
        """
        if top_k is None:
            top_k = TOP_K

        query_terms = set(query.lower().split())
        results = []

        for idx, meta in self.chunk_metadata.items():
            if company_filter and meta.get("company") != company_filter:
                continue

            text_lower = meta.get("text", "").lower()
            # Count matching terms
            matches = sum(1 for term in query_terms if term in text_lower)

            if matches > 0:
                results.append({
                    "chunk_id": meta.get("chunk_id", ""),
                    "company": meta.get("company", ""),
                    "company_name": meta.get("company_name", ""),
                    "year": meta.get("year", ""),
                    "section": meta.get("section", ""),
                    "chunk_index": meta.get("chunk_index", -1),
                    "text": meta.get("text", ""),
                    "keyword_score": matches / len(query_terms),
                    "keyword_matches": matches,
                })

        results.sort(key=lambda x: x["keyword_score"], reverse=True)
        return results[:top_k]


def test_retrieval():
    """Quick test of the retrieval pipeline."""
    retriever = SemanticRetriever()

    test_queries = [
        "supply chain disruption risk",
        "cybersecurity data breach",
        "government regulation compliance",
        "competition and market share",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        results = retriever.retrieve(query, top_k=3)
        for r in results:
            score_type = "rerank" if "rerank_score" in r else "faiss"
            score = r.get("rerank_score", r.get("faiss_score", 0))
            print(f"\n  [{r['company']}] Score={score:.4f} ({score_type})")
            print(f"  {r['text'][:200]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("Semantic Retriever Test")
    print("=" * 60)
    test_retrieval()
