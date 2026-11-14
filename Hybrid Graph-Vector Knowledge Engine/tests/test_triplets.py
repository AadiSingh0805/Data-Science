"""Unit tests for TripletExtractionPipeline and EntityResolver."""

import pytest
from pathlib import Path
from src.data.contracts import EntityType, RelationType, KnowledgeGraphExtraction, DocumentChunk
from src.features.triplets import EntityResolver, TripletExtractionPipeline


def test_entity_resolver_canonical_mapping():
    resolver = EntityResolver()
    eid, cname, etype = resolver.resolve("TSMC")
    assert eid == "company:tsmc"
    assert cname == "TSMC"
    assert etype == EntityType.COMPANY

    # Variant spelling resolution
    eid2, cname2, etype2 = resolver.resolve("Taiwan Semiconductor Manufacturing Company")
    assert eid2 == "company:tsmc"
    assert cname2 == "TSMC"

    # ASML resolution
    eid3, cname3, _ = resolver.resolve("ASML Holding N.V.")
    assert eid3 == "company:asml_holding_nv"
    assert cname3 == "ASML Holding N.V."


def test_no_self_loops_validation():
    pipeline = TripletExtractionPipeline()
    chunk = DocumentChunk(
        chunk_id=0,
        doc_id="test_doc.txt",
        content="Hon Hai Precision (Foxconn) operates its facility in Zhengzhou.",
        metadata={}
    )
    kg = pipeline.extract_from_chunk(chunk)
    for rel in kg.relationships:
        assert rel.source_id != rel.target_id


def test_referential_integrity():
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        pytest.skip("data/raw directory not found")

    pipeline = TripletExtractionPipeline()
    chunks, kg = pipeline.process_directory(raw_dir)

    assert len(chunks) > 0
    assert len(kg.entities) > 0
    assert len(kg.relationships) > 0

    # Ensure every relationship endpoint exists in entities
    entity_ids = {e.id for e in kg.entities}
    for rel in kg.relationships:
        assert rel.source_id in entity_ids
        assert rel.target_id in entity_ids
        assert len(rel.verbatim_quote) >= 10

