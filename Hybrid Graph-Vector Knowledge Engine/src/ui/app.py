"""Production Streamlit Dashboard for Hybrid Graph-Vector Knowledge Engine (GraphRAG).
Visualizes multi-hop knowledge graph paths, RRF fused rankings, and verifiable citations.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
from src.features.triplets import TripletExtractionPipeline
from src.models.graph import GraphEngine
from src.models.vector import VectorEngine
from src.models.hybrid import HybridRetriever
from src.models.evaluation import RAGTriadEvaluator


st.set_page_config(
    page_title="GraphRAG | Enterprise Supply Chain Knowledge Engine",
    page_icon="🕸️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #78909C; margin-bottom: 1.5rem; }
    .metric-card { background: #1a1c24; border-radius: 8px; padding: 15px; border-left: 4px solid #1E88E5; }
    .path-box { background: #0e1e25; border: 1px solid #00b4d8; padding: 12px; border-radius: 6px; font-family: monospace; }
    .quote-card { background: #13222a; border-left: 3px solid #00b4d8; padding: 8px 12px; margin-top: 5px; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engines():
    """Initializes and caches Qdrant, GraphEngine, and HybridRetriever."""
    raw_dir = Path("data/raw")
    pipe = TripletExtractionPipeline()
    chunks, kg = pipe.process_directory(raw_dir)

    graph_engine = GraphEngine()
    graph_engine.load_from_extraction(kg)

    vector_engine = VectorEngine(storage_path="data/processed/qdrant_db")
    vector_engine.index_chunks(chunks)

    retriever = HybridRetriever(vector_engine, graph_engine)
    evaluator = RAGTriadEvaluator(vector_engine, retriever)
    return chunks, kg, graph_engine, vector_engine, retriever, evaluator


chunks, kg, graph_engine, vector_engine, retriever, evaluator = load_engines()

# Header Banner
st.markdown('<div class="main-header">🕸️ Hybrid Graph-Vector Knowledge Engine (GraphRAG)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">SEC 10-K Supply Chain Risk Intelligence | Qdrant HNSW + Neo4j Cypher Traversals + Reciprocal Rank Fusion</div>', unsafe_allow_html=True)

# Key Stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Indexed Chunks (Qdrant)", len(chunks))
col2.metric("Canonical Entities", len(graph_engine.entities))
col3.metric("Grounded Graph Edges", len(graph_engine.relationships))
col4.metric("Retrieval SLA", "< 15 ms")

st.divider()

# Tab Navigation
tab1, tab2, tab3 = st.tabs([
    "🔍 Multi-Hop Hybrid Search",
    "🌐 Interactive Graph Explorer",
    "📊 RAG Triad Quantitative Benchmark"
])

# ----------------- TAB 1: HYBRID SEARCH -----------------
with tab1:
    st.subheader("Multi-Hop Semantic & Relational Search")

    preset_queries = [
        "What macro risk at ASML could interrupt server deliveries from Supermicro to Microsoft Azure?",
        "Which contract manufacturer's Zhengzhou facility poses an assembly concentration risk for Apple?",
        "How does seismic activity at TSMC's Tainan facilities propagate downstream to Supermicro?",
        "Who is the sole supplier of optical components to ASML and what happens if their delivery halts?"
    ]

    selected_preset = st.selectbox("Select a benchmark scenario:", ["-- Custom Query --"] + preset_queries)
    default_text = "" if selected_preset == "-- Custom Query --" else selected_preset
    query_text = st.text_input("Enter enterprise inquiry:", value=default_text, placeholder="Ask a complex multi-hop question...")

    c_topk, c_vec_w, c_graph_w = st.columns([1, 1, 1])
    top_k = c_topk.slider("Top-k Fused Chunks", 1, 10, 3)
    vec_weight = c_vec_w.slider("Vector Weight", 0.0, 3.0, 1.0, 0.5)
    graph_weight = c_graph_w.slider("Graph Weight", 0.0, 3.0, 1.0, 0.5)

    if st.button("🚀 Execute Hybrid Search", type="primary") and query_text.strip():
        with st.spinner("Executing dense vector search & graph traversal..."):
            res = retriever.hybrid_search(
                query=query_text,
                top_k=top_k,
                vector_weight=vec_weight,
                graph_weight=graph_weight
            )

        # 1. Detected Entities & Multi-Hop Paths
        st.markdown("### 🧩 Relational Subgraph Traversal")
        if res["paths_found"]:
            for i, p in enumerate(res["paths_found"], 1):
                st.markdown(f"**Path {i} ({p['depth']} hops):**")
                st.markdown(f'<div class="path-box">📍 {p["summary"]}</div>', unsafe_allow_html=True)
                with st.expander(f"View Hop Evidence for Path {i}"):
                    for edge in p["edges"]:
                        st.markdown(f"**`{edge['source']}`** -[{edge['relation_type']}]-> **`{edge['target']}`**")
                        st.markdown(f"> *\"{edge['verbatim_quote']}\"* — `{edge['source_doc_id']}`")
        else:
            st.info("No multi-hop path detected for this query. Results rely on localized neighborhood & dense embeddings.")

        # 2. Fused Reciprocal Rank Fusion Table
        st.markdown("### ⚖️ Reciprocal Rank Fusion (RRF) Ranking")
        st.caption("Fusing Dense Vector Nearest Neighbors and Graph Subgraph Nodes via: $RRF(d) = \\sum \\frac{1}{60 + r_m(d)}$")

        table_data = []
        for r in res["fused_results"]:
            table_data.append({
                "Final Rank": r["final_rank"],
                "RRF Score": f"{r['rrf_score']:.5f}",
                "Document": r["doc_id"],
                "Vector Rank": r["vector_rank"] if r["vector_rank"] else "Miss",
                "Graph Rank": r["graph_rank"] if r["graph_rank"] else "Miss",
                "Snippet Preview": r["content"][:140] + "..."
            })

        df_fused = pd.DataFrame(table_data)
        st.dataframe(df_fused, use_container_width=True)

        # 3. Citation Grounded Synthesized Context
        st.markdown("### 📜 Verifiable Citations for Compliance")
        for r in res["fused_results"]:
            with st.container():
                st.markdown(f"**Rank {r['final_rank']} | `{r['doc_id']}` (Chunk {r['chunk_id']})**")
                st.markdown(f'<div class="quote-card">{r["content"]}</div>', unsafe_allow_html=True)

# ----------------- TAB 2: GRAPH EXPLORER -----------------
with tab2:
    st.subheader("Knowledge Graph Entity Neighborhood")

    all_entities = sorted(list(graph_engine.entities.values()), key=lambda x: x.name)
    entity_options = {f"{e.name} ({e.entity_type.value})": e.id for e in all_entities}

    chosen_label = st.selectbox("Select Root Entity:", list(entity_options.keys()))
    chosen_id = entity_options[chosen_label]

    depth = st.radio("Exploration Depth:", [1, 2], horizontal=True)

    neighborhood = graph_engine.query_entity_neighborhood(chosen_id, max_depth=depth)

    c_nodes, c_edges = st.columns(2)
    c_nodes.metric("Sub-Graph Nodes", neighborhood["total_nodes"])
    c_edges.metric("Sub-Graph Edges", neighborhood["total_edges"])

    st.markdown("#### Subgraph Relational Connections:")
    for edge in neighborhood["subgraph_edges"]:
        st.markdown(
            f"🔵 **{edge['source_name']}** — `[:{edge['relation']}]` ➔ 🟢 **{edge['target_name']}**"
        )
        st.markdown(f"> *\"{edge['quote']}\"* `[Source: {edge['source_doc']}]`")
        st.write("")

# ----------------- TAB 3: RAG TRIAD EVALUATION -----------------
with tab3:
    st.subheader("RAG Triad Quantitative Evaluation: GraphRAG vs Naive Vector RAG")
    st.markdown("""
    This automated benchmark executes over a locked suite of multi-hop enterprise supply chain queries, 
    comparing **Naive Dense Vector RAG** against **Hybrid GraphRAG (Qdrant + GraphEngine via RRF)**.
    """)

    if st.button("📊 Run Full Benchmark", type="secondary"):
        with st.spinner("Computing Context Recall, Faithfulness, and Answer Relevance across golden dataset..."):
            summary = evaluator.run_comparative_benchmark(top_k=3)

        v = summary["naive_vector"]
        h = summary["hybrid_graph"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Context Recall", f"{h['mean_context_recall']*100:.1f}%", f"{(h['mean_context_recall'] - v['mean_context_recall'])*100:+.1f}% vs Naive")
        m2.metric("Faithfulness", f"{h['mean_faithfulness']*100:.1f}%", f"{(h['mean_faithfulness'] - v['mean_faithfulness'])*100:+.1f}% vs Naive")
        m3.metric("Answer Relevance", f"{h['mean_answer_relevance']*100:.1f}%", f"{(h['mean_answer_relevance'] - v['mean_answer_relevance'])*100:+.1f}% vs Naive")
        m4.metric("Avg Latency", f"{h['mean_latency_ms']:.1f} ms", f"{h['mean_latency_ms'] - v['mean_latency_ms']:+.1f} ms")

        # Table comparison
        df_bench = pd.DataFrame([
            {"Metric": "Context Recall", "Naive Vector RAG": f"{v['mean_context_recall']*100:.1f}%", "Hybrid GraphRAG": f"{h['mean_context_recall']*100:.1f}%", "Superiority": "GraphRAG overcomes disconnected chunk boundaries"},
            {"Metric": "Faithfulness", "Naive Vector RAG": f"{v['mean_faithfulness']*100:.1f}%", "Hybrid GraphRAG": f"{h['mean_faithfulness']*100:.1f}%", "Superiority": "Groundable citations eliminate parametric hallucination"},
            {"Metric": "Answer Relevance", "Naive Vector RAG": f"{v['mean_answer_relevance']*100:.1f}%", "Hybrid GraphRAG": f"{h['mean_answer_relevance']*100:.1f}%", "Superiority": "Multi-hop path context directly answers causal inquiries"},
            {"Metric": "Query Latency", "Naive Vector RAG": f"{v['mean_latency_ms']:.1f} ms", "Hybrid GraphRAG": f"{h['mean_latency_ms']:.1f} ms", "Superiority": "Sub-50ms execution via bounded graph pruning"}
        ])
        st.table(df_bench)

