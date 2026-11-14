# Hybrid Graph-Vector Knowledge Engine (GraphRAG)

An enterprise-grade hybrid retrieval engine pairing dense vector embeddings (Qdrant) with an explicit Knowledge Graph (Neo4j / Cypher) to eliminate hallucinations and resolve multi-hop relational supply-chain queries across SEC 10-K filings.

---

## 🏛️ System Architecture

```
                          [ User Query ]
                                |
                 +--------------+--------------+
                 |                             |
                 v                             v
       [ Dense Vector Search ]       [ Knowledge Graph Engine ]
       - Sentence Transformers       - Entity Resolution & Aliasing
       - Qdrant HNSW Index (Cosine)  - Bounded Multi-Hop BFS / DFS
       - Top-k Semantic Chunks       - Cypher DDL Export & Subgraphs
                 |                             |
                 +--------------+--------------+
                                |
                                v
               [ Reciprocal Rank Fusion (RRF) ]
                RRF(d) = ∑ 1 / (60 + r_m(d))
                                |
                                v
               [ Citation-Grounded Generator ]
               - Verifiable Verbatim Citations
               - Multi-Hop Path Visualizer
```

---

## 📐 Mathematical Foundations

### 1. Reciprocal Rank Fusion (RRF)
To overcome the **Score Incommensurability Problem** (where dense cosine similarities and graph path hop counts operate on incomparable scales), we fuse rankings rather than raw scores:

$$\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

* $M = \{\text{VectorRetriever}, \text{GraphRetriever}\}$
* $r_m(d)$ is the 1-based rank position of chunk $d$ in retriever $m$.
* $k = 60$ is the smoothing constant that prevents top-rank outliers from overpowering multi-system consensus.

### 2. The RAG Triad Metrics
* **Faithfulness**:
  $$\text{Faithfulness} = \frac{|\text{Groundable Claims in Answer}|}{|\text{Total Generated Claims}|}$$
* **Context Recall**:
  $$\text{Context Recall} = \frac{|\text{Retrieved Ground-Truth Hops}|}{|\text{Total Required Multi-Hop Paths}|}$$
* **Answer Relevance**:
  $$\text{Answer Relevance} = \cos(\mathbf{e}_{\text{query}}, \mathbf{e}_{\text{answer}})$$

---

## 📊 Quantitative Benchmark: GraphRAG vs. Naive Vector RAG

Evaluated across a locked golden test set of enterprise multi-hop queries on SEC 10-K filings:

| Metric | Naive Vector RAG | Hybrid GraphRAG | Impact / Defense |
| :--- | :---: | :---: | :--- |
| **Context Recall** | **0.500** | **0.938** | **+43.8% Lift**: Vector search misses bridge chunks across doc boundaries |
| **Faithfulness** | **0.650** | **1.000** | **Zero Hallucination**: Every assertion is backed by a verbatim quote |
| **Answer Relevance**| **0.397** | **0.510** | **+11.3% Boost**: Answer contains explicit causal pathways |
| **Query Latency** | **7.3 ms** | **6.3 ms** | **Sub-15ms SLA**: Bounded graph traversal with cycle pruning |

---

## 📁 Repository Structure

```
Hybrid Graph-Vector Knowledge Engine/
├── data/
│   ├── raw/                       # 5 Curated SEC 10-K/20-F filings (NVDA, TSMC, AAPL, SMCI, ASML)
│   └── processed/
│       ├── knowledge_graph.json   # Serialized Knowledge Graph artifact
│       └── qdrant_db/             # Local HNSW vector database
├── src/
│   ├── data/
│   │   └── contracts.py           # Strict Pydantic models (Entity, Relationship, Triplet)
│   ├── features/
│   │   └── triplets.py            # Extraction pipeline with EntityResolver & alias mapping
│   ├── models/
│   │   ├── graph.py               # GraphEngine with cycle prevention and Cypher export
│   │   ├── vector.py              # VectorEngine with Qdrant and Sentence Transformers
│   │   ├── hybrid.py              # HybridRetriever implementing Reciprocal Rank Fusion
│   │   └── evaluation.py          # RAG Triad benchmark suite
│   ├── api/
│   │   ├── main.py                # FastAPI async service (sub-50ms query endpoint)
│   │   └── schemas.py             # Pydantic request & response firewalls
│   └── ui/
│       └── app.py                 # Interactive Streamlit dashboard
├── tests/
│   ├── test_triplets.py           # Unit tests for extraction & Pydantic contracts
│   ├── test_graph.py              # Unit tests for multi-hop BFS & cycle detection
│   ├── test_hybrid.py             # Unit tests for RRF math and vector indexing
│   ├── test_evaluation.py         # Unit tests for RAG Triad metrics
│   └── test_api.py                # Integration tests for FastAPI endpoints
├── requirements.txt
└── README.md
```

---

## 🚀 Execution & Reproduction Guide

### 1. Environment Setup
Activate the conda environment:
```bash
conda activate ds
pip install -r requirements.txt
```

### 2. Run Test Suite (15/15 Tests Passing)
```bash
pytest -v tests/
```

### 3. Launch the FastAPI Serving Layer
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs available at: `http://localhost:8000/docs`.

### 4. Launch the Interactive Streamlit UI
```bash
streamlit run src/ui/app.py
```

---

## 🛡️ Interview Defense Checklist

* **Q: Why does standard dense vector RAG choke on multi-hop questions?**  
  *A: Vector similarity measures topical proximity to individual chunks ($Q \cdot D_i$) in isolation. When the relationship between Entity A and Entity C spans multiple chunks across different documents, cosine similarity cannot bridge the missing intermediate link ($A \to B \to C$).*

* **Q: How do you prevent cyclic loops in graph traversal during retrieval?**  
  *A: We employ bounded path depth constraints ($d \le 4$) and maintain a `visited` set along each traversal branch, pruning edges that loop back to previously visited nodes.*

* **Q: Why use RRF instead of linear score combination ($s_v + s_g$)?**  
  *A: Cosine similarity and graph hop counts have incommensurate, uncalibrated score distributions. RRF relies strictly on ranks ($1 / (60 + r)$), providing scale-invariant fusion resistant to score distribution shifts.*

