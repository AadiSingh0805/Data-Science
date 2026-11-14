"""Quantitative Evaluation Engine for RAG Triad Metrics.
Benchmarks Hybrid GraphRAG against Naive Vector RAG across:
1. Context Recall (Multi-Hop Entity & Relational Coverage)
2. Faithfulness (|Groundable Claims| / |Total Claims|)
3. Answer Relevance (Semantic Cosine Proximity)
4. Query Latency & Token Profiling
"""

import time
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from src.models.vector import VectorEngine
from src.models.hybrid import HybridRetriever


GOLDEN_BENCHMARK_DATASET = [
    {
        "id": "q1",
        "query": "What macro risk at ASML could interrupt server deliveries from Supermicro to Microsoft Azure?",
        "required_entities": ["company:asml_holding_nv", "company:tsmc", "company:nvidia_corporation", "company:super_micro_computer_inc", "company:microsoft_azure"],
        "required_hops": [
            ("company:asml_holding_nv", "company:tsmc"),
            ("company:tsmc", "company:nvidia_corporation"),
            ("company:nvidia_corporation", "company:super_micro_computer_inc"),
            ("company:super_micro_computer_inc", "company:microsoft_azure")
        ],
        "ground_truth_claims": [
            "ASML supplies EUV lithography machines to TSMC.",
            "TSMC is the sole foundry manufacturing GPUs for NVIDIA.",
            "NVIDIA supplies HGX platforms to Supermicro.",
            "Supermicro delivers complete server clusters to Microsoft Azure.",
            "ASML faces extended EUV lead times and export control restrictions."
        ]
    },
    {
        "id": "q2",
        "query": "Which contract manufacturer's Zhengzhou facility poses an assembly concentration risk for Apple?",
        "required_entities": ["company:hon_hai_precision_foxconn", "facility:foxconn_zhengzhou_mega-facility", "company:apple_inc"],
        "required_hops": [
            ("company:hon_hai_precision_foxconn", "company:apple_inc"),
            ("company:hon_hai_precision_foxconn", "facility:foxconn_zhengzhou_mega-facility")
        ],
        "ground_truth_claims": [
            "Hon Hai Precision (Foxconn) performs final assembly for Apple iPhones.",
            "Foxconn assembly is heavily concentrated in Zhengzhou, China."
        ]
    },
    {
        "id": "q3",
        "query": "How does seismic activity at TSMC's Tainan facilities propagate downstream to Supermicro?",
        "required_entities": ["risk_event:seismic_&_earthquake_hazards", "facility:tainan_science_park_fab", "company:tsmc", "company:nvidia_corporation", "company:super_micro_computer_inc"],
        "required_hops": [
            ("company:tsmc", "risk_event:seismic_&_earthquake_hazards"),
            ("company:tsmc", "company:nvidia_corporation"),
            ("company:nvidia_corporation", "company:super_micro_computer_inc")
        ],
        "ground_truth_claims": [
            "TSMC operates fabrication facilities in Tainan exposed to seismic hazards.",
            "TSMC manufactures Blackwell and Hopper GPUs exclusively for NVIDIA.",
            "NVIDIA allocates GPUs to Supermicro for AI server manufacturing."
        ]
    },
    {
        "id": "q4",
        "query": "Who is the sole supplier of optical components to ASML and what happens if their delivery halts?",
        "required_entities": ["company:carl_zeiss_smt", "company:asml_holding_nv", "company:tsmc"],
        "required_hops": [
            ("company:carl_zeiss_smt", "company:asml_holding_nv"),
            ("company:asml_holding_nv", "company:tsmc")
        ],
        "ground_truth_claims": [
            "Carl Zeiss SMT is the sole supplier of precision EUV projection optics to ASML.",
            "If Zeiss deliveries halt, ASML cannot build EUV scanners, disrupting TSMC fab capacity."
        ]
    }
]


class RAGTriadEvaluator:
    """Evaluates and compares RAG pipelines using quantitative Triad metrics."""

    def __init__(self, vector_engine: VectorEngine, hybrid_retriever: HybridRetriever):
        self.vector_engine = vector_engine
        self.hybrid_retriever = hybrid_retriever
        self.encoder = vector_engine.encoder

    def evaluate_context_recall(
        self,
        retrieved_texts: List[str],
        required_hops: List[Tuple[str, str]]
    ) -> float:
        """Calculates multi-hop relational context recall:
        proportion of required hops whose endpoints are covered in retrieved chunks.
        """
        if not required_hops:
            return 1.0

        full_context = " ".join(retrieved_texts).lower()
        hops_found = 0

        for src, tgt in required_hops:
            src_name = src.split(":")[-1].replace("_", " ")
            tgt_name = tgt.split(":")[-1].replace("_", " ")

            # Check if both endpoints are mentioned or supported in context
            src_in = any(token in full_context for token in src_name.split() if len(token) > 3)
            tgt_in = any(token in full_context for token in tgt_name.split() if len(token) > 3)

            if src_in and tgt_in:
                hops_found += 1

        return hops_found / len(required_hops)

    def evaluate_faithfulness(
        self,
        generated_claims: List[str],
        retrieved_texts: List[str]
    ) -> float:
        """Calculates Faithfulness: |Groundable Claims| / |Total Claims|."""
        if not generated_claims:
            return 1.0

        full_context = " ".join(retrieved_texts).lower()
        grounded_count = 0

        for claim in generated_claims:
            clean_words = [re.sub(r"[^\w]", "", w.lower()) for w in claim.split()]
            key_words = [w for w in clean_words if len(w) >= 4]
            if not key_words:
                continue
            matches = sum(
                1 for w in key_words
                if w in full_context or (len(w) >= 6 and (w[:5] in full_context or w[:-2] in full_context))
            )
            if matches / len(key_words) >= 0.5:
                grounded_count += 1

        return grounded_count / len(generated_claims)

    def evaluate_answer_relevance(self, query: str, answer_text: str) -> float:
        """Calculates Answer Relevance via Cosine Similarity between Query and Answer."""
        if not answer_text.strip():
            return 0.0

        embs = self.encoder.encode([query, answer_text], normalize_embeddings=True)
        cos_sim = float(np.dot(embs[0], embs[1]))
        return max(0.0, cos_sim)

    def synthesize_answer(self, query: str, retrieved_chunks: List[str], paths_found: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Generates grounded answer and distinct factual claims."""
        claims = []
        if paths_found:
            for p in paths_found:
                claims.append(f"Supply chain dependency traces: {p['summary']}.")

        for chunk in retrieved_chunks[:2]:
            sentences = [s.strip() for s in chunk.split(".") if len(s.strip()) > 20]
            if sentences:
                claims.append(sentences[0] + ".")

        answer_text = " ".join(claims) if claims else "Insufficient information in retrieved context."
        return answer_text, claims

    def run_comparative_benchmark(self, top_k: int = 3) -> Dict[str, Any]:
        """Runs comparative benchmark of Naive Vector RAG vs Hybrid GraphRAG."""
        results = {
            "naive_vector": {
                "context_recall": [],
                "faithfulness": [],
                "answer_relevance": [],
                "latency_ms": []
            },
            "hybrid_graph": {
                "context_recall": [],
                "faithfulness": [],
                "answer_relevance": [],
                "latency_ms": []
            }
        }

        for item in GOLDEN_BENCHMARK_DATASET:
            query = item["query"]
            required_hops = item["required_hops"]
            ground_truth_claims = item["ground_truth_claims"]

            # ---- 1. Evaluate Naive Vector RAG ----
            t0 = time.perf_counter()
            vec_hits = self.vector_engine.search(query, top_k=top_k)
            t_vec = (time.perf_counter() - t0) * 1000

            vec_texts = [h["content"] for h in vec_hits]
            vec_answer, vec_claims = self.synthesize_answer(query, vec_texts, paths_found=[])

            recall_vec = self.evaluate_context_recall(vec_texts, required_hops)
            faith_vec = self.evaluate_faithfulness(ground_truth_claims, vec_texts)
            rel_vec = self.evaluate_answer_relevance(query, vec_answer)

            results["naive_vector"]["context_recall"].append(recall_vec)
            results["naive_vector"]["faithfulness"].append(faith_vec)
            results["naive_vector"]["answer_relevance"].append(rel_vec)
            results["naive_vector"]["latency_ms"].append(t_vec)

            # ---- 2. Evaluate Hybrid GraphRAG ----
            t1 = time.perf_counter()
            hybrid_res = self.hybrid_retriever.hybrid_search(query, top_k=top_k)
            t_hybrid = (time.perf_counter() - t1) * 1000

            hybrid_texts = [h["content"] for h in hybrid_res["fused_results"]]
            hybrid_answer, hybrid_claims = self.synthesize_answer(
                query, hybrid_texts, hybrid_res["paths_found"]
            )

            recall_hyb = self.evaluate_context_recall(hybrid_texts, required_hops)
            faith_hyb = self.evaluate_faithfulness(ground_truth_claims, hybrid_texts)
            rel_hyb = self.evaluate_answer_relevance(query, hybrid_answer)

            results["hybrid_graph"]["context_recall"].append(recall_hyb)
            results["hybrid_graph"]["faithfulness"].append(faith_hyb)
            results["hybrid_graph"]["answer_relevance"].append(rel_hyb)
            results["hybrid_graph"]["latency_ms"].append(t_hybrid)

        summary = {
            "naive_vector": {
                "mean_context_recall": float(np.mean(results["naive_vector"]["context_recall"])),
                "mean_faithfulness": float(np.mean(results["naive_vector"]["faithfulness"])),
                "mean_answer_relevance": float(np.mean(results["naive_vector"]["answer_relevance"])),
                "mean_latency_ms": float(np.mean(results["naive_vector"]["latency_ms"]))
            },
            "hybrid_graph": {
                "mean_context_recall": float(np.mean(results["hybrid_graph"]["context_recall"])),
                "mean_faithfulness": float(np.mean(results["hybrid_graph"]["faithfulness"])),
                "mean_answer_relevance": float(np.mean(results["hybrid_graph"]["answer_relevance"])),
                "mean_latency_ms": float(np.mean(results["hybrid_graph"]["latency_ms"]))
            }
        }
        return summary
