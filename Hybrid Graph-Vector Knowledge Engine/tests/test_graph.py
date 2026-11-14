"""Unit tests for GraphEngine: multi-hop traversal, cycle prevention, and Cypher generation."""

import pytest
from pathlib import Path
from src.data.contracts import Entity, Relationship, EntityType, RelationType, KnowledgeGraphExtraction
from src.models.graph import GraphEngine


def test_cycle_prevention_in_graph_traversal():
    """Validates that cyclic graphs (A -> B -> C -> A) do not cause infinite recursion."""
    engine = GraphEngine()

    # Create cyclic entity loop
    e_a = Entity(id="company:a", name="Company A", entity_type=EntityType.COMPANY)
    e_b = Entity(id="company:b", name="Company B", entity_type=EntityType.COMPANY)
    e_c = Entity(id="company:c", name="Company C", entity_type=EntityType.COMPANY)
    e_target = Entity(id="company:target", name="Company Target", entity_type=EntityType.COMPANY)

    engine.add_entity(e_a)
    engine.add_entity(e_b)
    engine.add_entity(e_c)
    engine.add_entity(e_target)

    # A -> B -> C -> A (cycle!) and C -> Target
    quote = "Evidence statement verifying connection between companies in filings."
    engine.add_relationship(Relationship(source_id="company:a", target_id="company:b", relation_type=RelationType.SUPPLIES_TO, verbatim_quote=quote, source_doc_id="test.txt"))
    engine.add_relationship(Relationship(source_id="company:b", target_id="company:c", relation_type=RelationType.SUPPLIES_TO, verbatim_quote=quote, source_doc_id="test.txt"))
    engine.add_relationship(Relationship(source_id="company:c", target_id="company:a", relation_type=RelationType.SUPPLIES_TO, verbatim_quote=quote, source_doc_id="test.txt"))
    engine.add_relationship(Relationship(source_id="company:c", target_id="company:target", relation_type=RelationType.SUPPLIES_TO, verbatim_quote=quote, source_doc_id="test.txt"))

    # Traversal should detect cycle, prune loop branch, and still find target
    paths = engine.find_multi_hop_paths("company:a", target_id="company:target", max_depth=6)
    assert len(paths) == 1
    assert paths[0]["summary"] == "Company A -> Company B -> Company C -> Company Target"


def test_cypher_export():
    engine = GraphEngine()
    e_nvda = Entity(id="company:nvidia", name="NVIDIA Corporation", entity_type=EntityType.COMPANY)
    e_tsmc = Entity(id="company:tsmc", name="TSMC", entity_type=EntityType.COMPANY)
    engine.add_entity(e_nvda)
    engine.add_entity(e_tsmc)

    engine.add_relationship(Relationship(
        source_id="company:tsmc",
        target_id="company:nvidia",
        relation_type=RelationType.MANUFACTURES_FOR,
        verbatim_quote="TSMC is our primary and sole foundry for fabricating advanced GPU compute dies.",
        source_doc_id="NVDA_2024_10K.txt",
        chunk_id=1
    ))

    cypher_statements = engine.export_cypher_statements()
    assert any("CREATE CONSTRAINT" in s for s in cypher_statements)
    assert any("company:nvidia" in s for s in cypher_statements)
    assert any("MANUFACTURES_FOR" in s for s in cypher_statements)


def test_multi_hop_from_real_data():
    kg_path = Path("data/processed/knowledge_graph.json")
    if not kg_path.exists():
        pytest.skip("Persisted knowledge graph not found")

    engine = GraphEngine()
    engine.load_from_file(kg_path)

    # Verify Carl Zeiss to Supermicro path
    paths = engine.find_multi_hop_paths("company:carl_zeiss_smt", target_id="company:super_micro_computer_inc", max_depth=5)
    assert len(paths) >= 1
    assert "TSMC" in paths[0]["summary"]
    assert "NVIDIA" in paths[0]["summary"]

