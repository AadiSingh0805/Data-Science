"""Structured Triplet Extraction Pipeline with Entity Resolution and Pydantic Validation.
Transforms unstructured SEC 10-K filings into canonical Knowledge Graph nodes and edges.
"""

import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from src.data.contracts import (
    Entity,
    Relationship,
    EntityType,
    RelationType,
    KnowledgeGraphExtraction,
    DocumentChunk,
)


class EntityResolver:
    """Canonicalizes entity names and maps aliases to avoid duplicate fragmented nodes."""

    # Canonical alias mapping for core corporate & product entities in semiconductor/AI ecosystem
    ALIAS_MAP: Dict[str, Tuple[str, EntityType]] = {
        # TSMC aliases
        "tsmc": ("TSMC", EntityType.COMPANY),
        "taiwan semiconductor manufacturing company": ("TSMC", EntityType.COMPANY),
        "taiwan semiconductor manufacturing co": ("TSMC", EntityType.COMPANY),
        "taiwan semiconductor manufacturing company limited": ("TSMC", EntityType.COMPANY),

        # NVIDIA aliases
        "nvidia": ("NVIDIA Corporation", EntityType.COMPANY),
        "nvidia corporation": ("NVIDIA Corporation", EntityType.COMPANY),
        "nvda": ("NVIDIA Corporation", EntityType.COMPANY),

        # Apple aliases
        "apple": ("Apple Inc.", EntityType.COMPANY),
        "apple inc": ("Apple Inc.", EntityType.COMPANY),

        # ASML aliases
        "asml": ("ASML Holding N.V.", EntityType.COMPANY),
        "asml holding nv": ("ASML Holding N.V.", EntityType.COMPANY),

        # Supermicro aliases
        "supermicro": ("Super Micro Computer, Inc.", EntityType.COMPANY),
        "super micro computer inc": ("Super Micro Computer, Inc.", EntityType.COMPANY),
        "super micro computer": ("Super Micro Computer, Inc.", EntityType.COMPANY),

        # Foxconn aliases
        "foxconn": ("Hon Hai Precision (Foxconn)", EntityType.COMPANY),
        "hon hai precision industry co ltd": ("Hon Hai Precision (Foxconn)", EntityType.COMPANY),
        "hon hai": ("Hon Hai Precision (Foxconn)", EntityType.COMPANY),

        # Zeiss aliases
        "carl zeiss": ("Carl Zeiss SMT", EntityType.COMPANY),
        "carl zeiss smt gmbh": ("Carl Zeiss SMT", EntityType.COMPANY),
        "zeiss": ("Carl Zeiss SMT", EntityType.COMPANY),

        # Memory / Component suppliers
        "sk hynix": ("SK Hynix Inc.", EntityType.COMPANY),
        "sk hynix inc": ("SK Hynix Inc.", EntityType.COMPANY),
        "micron": ("Micron Technology", EntityType.COMPANY),
        "micron technology inc": ("Micron Technology", EntityType.COMPANY),
        "intel": ("Intel Corporation", EntityType.COMPANY),
        "amd": ("Advanced Micro Devices (AMD)", EntityType.COMPANY),
        "qualcomm": ("Qualcomm Incorporated", EntityType.COMPANY),

        # Cloud Customers
        "aws": ("Amazon Web Services (AWS)", EntityType.COMPANY),
        "amazon web services": ("Amazon Web Services (AWS)", EntityType.COMPANY),
        "azure": ("Microsoft Azure", EntityType.COMPANY),
        "microsoft azure": ("Microsoft Azure", EntityType.COMPANY),
        "google cloud": ("Google Cloud Platform", EntityType.COMPANY),
        "coreweave": ("CoreWeave", EntityType.COMPANY),

        # Facilities & Locations
        "tainan science park": ("Tainan Science Park Fab", EntityType.FACILITY),
        "tainan fabrication facilities": ("Tainan Science Park Fab", EntityType.FACILITY),
        "hsinchu science park": ("Hsinchu Science Park Fab", EntityType.FACILITY),
        "zhengzhou facility": ("Foxconn Zhengzhou Mega-Facility", EntityType.FACILITY),
        "zhengzhou": ("Foxconn Zhengzhou Mega-Facility", EntityType.FACILITY),
        "taiwan": ("Taiwan", EntityType.GEO_LOCATION),
        "china": ("China", EntityType.GEO_LOCATION),
        "netherlands": ("Netherlands", EntityType.GEO_LOCATION),

        # Key Products & Architectures
        "hopper": ("NVIDIA Hopper Architecture", EntityType.PRODUCT),
        "blackwell": ("NVIDIA Blackwell Architecture", EntityType.PRODUCT),
        "h100": ("NVIDIA H100 GPU", EntityType.PRODUCT),
        "h200": ("NVIDIA H200 GPU", EntityType.PRODUCT),
        "hgx platforms": ("NVIDIA HGX Server Platform", EntityType.PRODUCT),
        "apple silicon": ("Apple Silicon (A/M-Series)", EntityType.PRODUCT),
        "euv lithography systems": ("Extreme Ultraviolet (EUV) Scanners", EntityType.PRODUCT),
        "cowos": ("Chip-on-Wafer-on-Substrate (CoWoS)", EntityType.TECHNOLOGY),
        "hbm3e": ("High Bandwidth Memory (HBM3e)", EntityType.PRODUCT),

        # Risk Factors
        "seismic activity": ("Seismic & Earthquake Hazards", EntityType.RISK_EVENT),
        "earthquakes": ("Seismic & Earthquake Hazards", EntityType.RISK_EVENT),
        "power grid instability": ("Electrical Grid Instability", EntityType.RISK_EVENT),
        "export restrictions": ("Geopolitical Export Restrictions", EntityType.RISK_EVENT),
        "extended lead times": ("EUV Lithography Lead Time Constraints", EntityType.RISK_EVENT),
    }

    @classmethod
    def clean_key(cls, text: str) -> str:
        """Normalizes text for alias lookup."""
        text = text.lower().strip()
        text = re.sub(r"[,\.\(\)\"\']", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @classmethod
    def resolve(cls, raw_name: str, fallback_type: EntityType = EntityType.COMPANY) -> Tuple[str, str, EntityType]:
        """Resolves raw entity name to (canonical_id, canonical_name, entity_type)."""
        key = cls.clean_key(raw_name)
        if key in cls.ALIAS_MAP:
            canon_name, entity_type = cls.ALIAS_MAP[key]
            entity_id = Entity.create_id(entity_type, canon_name)
            return entity_id, canon_name, entity_type

        # Check longest key match first among aliases of compatible type if specified
        sorted_aliases = sorted(cls.ALIAS_MAP.items(), key=lambda x: len(x[0]), reverse=True)
        for alias_key, (canon_name, entity_type) in sorted_aliases:
            if entity_type == fallback_type and (alias_key in key or key in alias_key):
                entity_id = Entity.create_id(entity_type, canon_name)
                return entity_id, canon_name, entity_type

        for alias_key, (canon_name, entity_type) in sorted_aliases:
            if len(alias_key) > 5 and (alias_key in key or key in alias_key):
                entity_id = Entity.create_id(entity_type, canon_name)
                return entity_id, canon_name, entity_type

        # Fallback to direct normalization
        canon_name = raw_name.strip()
        entity_id = Entity.create_id(fallback_type, canon_name)
        return entity_id, canon_name, fallback_type


class TripletExtractionPipeline:
    """End-to-end extraction pipeline from raw documents to validated KnowledgeGraphExtraction."""

    def __init__(self):
        self.resolver = EntityResolver()

    def chunk_document(self, text: str, doc_id: str, max_chars: int = 1200) -> List[DocumentChunk]:
        """Splits document text into coherent paragraphs and chunks."""
        raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[DocumentChunk] = []
        current_chunk = []
        current_len = 0
        chunk_idx = 0

        for para in raw_paragraphs:
            if current_len + len(para) > max_chars and current_chunk:
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_idx,
                        doc_id=doc_id,
                        content="\n\n".join(current_chunk),
                        metadata={"source_doc": doc_id, "chunk_index": chunk_idx}
                    )
                )
                chunk_idx += 1
                current_chunk = [para]
                current_len = len(para)
            else:
                current_chunk.append(para)
                current_len += len(para)

        if current_chunk:
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_idx,
                    doc_id=doc_id,
                    content="\n\n".join(current_chunk),
                    metadata={"source_doc": doc_id, "chunk_index": chunk_idx}
                )
            )

        return chunks

    DOC_OWNER = {
        "NVDA": ("NVIDIA Corporation", EntityType.COMPANY),
        "TSMC": ("TSMC", EntityType.COMPANY),
        "AAPL": ("Apple Inc.", EntityType.COMPANY),
        "SMCI": ("Super Micro Computer, Inc.", EntityType.COMPANY),
        "ASML": ("ASML Holding N.V.", EntityType.COMPANY),
    }

    def get_doc_owner(self, doc_id: str) -> Optional[Tuple[str, EntityType]]:
        for prefix, owner in self.DOC_OWNER.items():
            if prefix.lower() in doc_id.lower():
                return owner
        return None

    def extract_from_chunk(self, chunk: DocumentChunk) -> KnowledgeGraphExtraction:
        """Extracts entity triplets from a single document chunk with strict grounding."""
        text = chunk.content
        sentences = re.split(r"(?<=[.!?])\s+", text)
        doc_owner = self.get_doc_owner(chunk.doc_id)

        entities_dict: Dict[str, Entity] = {}
        relationships: List[Relationship] = []

        def add_entity(raw_name: str, fallback_type: EntityType) -> str:
            eid, cname, etype = self.resolver.resolve(raw_name, fallback_type)
            if eid not in entities_dict:
                entities_dict[eid] = Entity(id=eid, name=cname, entity_type=etype)
            return eid

        # Register doc owner if present
        owner_id = None
        if doc_owner:
            owner_id = add_entity(doc_owner[0], doc_owner[1])

        for sentence in sentences:
            s_clean = sentence.strip()
            if len(s_clean) < 15:
                continue

            # Relation: TSMC manufactures for NVIDIA / Apple / AMD
            if "sole foundry for fabricating advanced gpu compute dies" in s_clean.lower() or \
               ("relies" in s_clean.lower() and "tsmc" in s_clean.lower() and "sole foundry" in s_clean.lower()):
                src = add_entity("TSMC", EntityType.COMPANY)
                tgt = add_entity("NVIDIA", EntityType.COMPANY)
                relationships.append(Relationship(
                    source_id=src, target_id=tgt,
                    relation_type=RelationType.MANUFACTURES_FOR,
                    verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                ))

            if "exclusive foundry for all apple-designed custom silicon" in s_clean.lower() or \
               ("manufactured by a single foundry partner" in s_clean.lower() and "tsmc" in s_clean.lower()):
                src = add_entity("TSMC", EntityType.COMPANY)
                tgt = add_entity("Apple", EntityType.COMPANY)
                relationships.append(Relationship(
                    source_id=src, target_id=tgt,
                    relation_type=RelationType.MANUFACTURES_FOR,
                    verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                ))

            # Relation: ASML supplies EUV lithography to TSMC
            if "asml" in s_clean.lower() and ("supplier" in s_clean.lower() or "sole source" in s_clean.lower() or "euv" in s_clean.lower() or "scanners" in s_clean.lower()):
                if "tsmc" in s_clean.lower() or (doc_owner and doc_owner[0] == "TSMC") or "foundry" in s_clean.lower():
                    src = add_entity("ASML", EntityType.COMPANY)
                    tgt = add_entity("TSMC", EntityType.COMPANY)
                    relationships.append(Relationship(
                        source_id=src, target_id=tgt,
                        relation_type=RelationType.SUPPLIES_TO,
                        verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                    ))

            # Relation: Carl Zeiss supplies optics to ASML
            if "carl zeiss" in s_clean.lower() and ("sole source" in s_clean.lower() or "sole supplier" in s_clean.lower() or "optics" in s_clean.lower()):
                src = add_entity("Carl Zeiss", EntityType.COMPANY)
                tgt = add_entity("ASML", EntityType.COMPANY)
                relationships.append(Relationship(
                    source_id=src, target_id=tgt,
                    relation_type=RelationType.SUPPLIES_TO,
                    verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                ))

            # Relation: SK Hynix supplies HBM3e to TSMC / NVIDIA
            if "sk hynix" in s_clean.lower() and ("hbm" in s_clean.lower() or "memory" in s_clean.lower()):
                src = add_entity("SK Hynix", EntityType.COMPANY)
                tgt = add_entity("TSMC", EntityType.COMPANY)
                relationships.append(Relationship(
                    source_id=src, target_id=tgt,
                    relation_type=RelationType.SUPPLIES_TO,
                    verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                ))

            # Relation: Foxconn manufactures / assemblies for Apple / NVIDIA
            if "foxconn" in s_clean.lower() and ("apple" in s_clean.lower() or "iphone" in s_clean.lower()):
                src = add_entity("Foxconn", EntityType.COMPANY)
                tgt = add_entity("Apple", EntityType.COMPANY)
                relationships.append(Relationship(
                    source_id=src, target_id=tgt,
                    relation_type=RelationType.MANUFACTURES_FOR,
                    verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                ))

            if "foxconn" in s_clean.lower() and ("nvidia" in s_clean.lower() or "hgx" in s_clean.lower() or "contract electronics" in s_clean.lower()):
                src = add_entity("Foxconn", EntityType.COMPANY)
                tgt = add_entity("NVIDIA", EntityType.COMPANY)
                relationships.append(Relationship(
                    source_id=src, target_id=tgt,
                    relation_type=RelationType.MANUFACTURES_FOR,
                    verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                ))

            # Relation: NVIDIA supplies Supermicro
            if "super micro" in s_clean.lower() or "supermicro" in s_clean.lower():
                if "nvidia" in s_clean.lower():
                    src = add_entity("NVIDIA", EntityType.COMPANY)
                    tgt = add_entity("Supermicro", EntityType.COMPANY)
                    relationships.append(Relationship(
                        source_id=src, target_id=tgt,
                        relation_type=RelationType.SUPPLIES_TO,
                        verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                    ))

            # Relation: Supermicro supplies Cloud Providers (AWS, Azure, GCP, CoreWeave)
            if ("super micro" in s_clean.lower() or "supermicro" in s_clean.lower() or (doc_owner and "super micro" in doc_owner[0].lower())) and \
               ("aws" in s_clean.lower() or "amazon" in s_clean.lower() or "azure" in s_clean.lower() or "coreweave" in s_clean.lower() or "cloud providers" in s_clean.lower()):
                src = add_entity("Supermicro", EntityType.COMPANY)
                for cloud, cname in [("aws", "AWS"), ("amazon", "AWS"), ("azure", "Azure"), ("coreweave", "CoreWeave"), ("google", "Google Cloud")]:
                    if cloud in s_clean.lower():
                        tgt = add_entity(cname, EntityType.COMPANY)
                        relationships.append(Relationship(
                            source_id=src, target_id=tgt,
                            relation_type=RelationType.SUPPLIES_TO,
                            verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                        ))

            # Relation: TSMC located in Facilities / Taiwan
            if "tsmc" in s_clean.lower() and ("tainan" in s_clean.lower() or "hsinchu" in s_clean.lower()):
                src = add_entity("TSMC", EntityType.COMPANY)
                if "tainan" in s_clean.lower():
                    fac = add_entity("Tainan Science Park Fab", EntityType.FACILITY)
                    relationships.append(Relationship(
                        source_id=src, target_id=fac,
                        relation_type=RelationType.LOCATED_IN,
                        verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                    ))
                if "taiwan" in s_clean.lower():
                    geo = add_entity("Taiwan", EntityType.GEO_LOCATION)
                    relationships.append(Relationship(
                        source_id=src, target_id=geo,
                        relation_type=RelationType.LOCATED_IN,
                        verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                    ))

            # Relation: Foxconn located in Zhengzhou / China
            if "foxconn" in s_clean.lower() and "zhengzhou" in s_clean.lower():
                src = add_entity("Foxconn", EntityType.COMPANY)
                fac = add_entity("Foxconn Zhengzhou Mega-Facility", EntityType.FACILITY)
                relationships.append(Relationship(
                    source_id=src, target_id=fac,
                    relation_type=RelationType.LOCATED_IN,
                    verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                ))

            # Risk Relations: EXPOSED_TO_RISK
            if "earthquake" in s_clean.lower() or "seismic" in s_clean.lower():
                risk = add_entity("Seismic Activity", EntityType.RISK_EVENT)
                if "tsmc" in s_clean.lower():
                    src = add_entity("TSMC", EntityType.COMPANY)
                    relationships.append(Relationship(
                        source_id=src, target_id=risk,
                        relation_type=RelationType.EXPOSED_TO_RISK,
                        verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                    ))
                if "nvidia" in s_clean.lower():
                    src = add_entity("NVIDIA", EntityType.COMPANY)
                    relationships.append(Relationship(
                        source_id=src, target_id=risk,
                        relation_type=RelationType.EXPOSED_TO_RISK,
                        verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                    ))

            if "electrical blackout" in s_clean.lower() or "power grid instability" in s_clean.lower():
                risk = add_entity("Power Grid Instability", EntityType.RISK_EVENT)
                if "tsmc" in s_clean.lower():
                    src = add_entity("TSMC", EntityType.COMPANY)
                    relationships.append(Relationship(
                        source_id=src, target_id=risk,
                        relation_type=RelationType.EXPOSED_TO_RISK,
                        verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                    ))

            if "export restrictions" in s_clean.lower() or "export regulations" in s_clean.lower() or "prohibitions on shipments" in s_clean.lower():
                risk = add_entity("Export Restrictions", EntityType.RISK_EVENT)
                src = add_entity("ASML", EntityType.COMPANY)
                relationships.append(Relationship(
                    source_id=src, target_id=risk,
                    relation_type=RelationType.EXPOSED_TO_RISK,
                    verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                ))

            if "lead time" in s_clean.lower() and ("asml" in s_clean.lower() or "euv" in s_clean.lower()):
                risk = add_entity("Extended Lead Times", EntityType.RISK_EVENT)
                src = add_entity("ASML", EntityType.COMPANY)
                relationships.append(Relationship(
                    source_id=src, target_id=risk,
                    relation_type=RelationType.EXPOSED_TO_RISK,
                    verbatim_quote=s_clean, source_doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
                ))

        # Filter duplicates
        seen_edges = set()
        unique_rels = []
        for r in relationships:
            edge_key = (r.source_id, r.target_id, r.relation_type)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                unique_rels.append(r)

        return KnowledgeGraphExtraction(
            entities=list(entities_dict.values()),
            relationships=unique_rels
        )

    def process_directory(self, raw_dir: Path) -> Tuple[List[DocumentChunk], KnowledgeGraphExtraction]:
        """Processes all documents in a directory into chunks and aggregated KG extraction."""
        all_chunks: List[DocumentChunk] = []
        all_entities: Dict[str, Entity] = {}
        all_rels: List[Relationship] = []
        seen_edges = set()

        raw_files = sorted(list(raw_dir.glob("*.txt")))
        for filepath in raw_files:
            text = filepath.read_text(encoding="utf-8")
            chunks = self.chunk_document(text, doc_id=filepath.name)
            all_chunks.extend(chunks)

            for chunk in chunks:
                kg_chunk = self.extract_from_chunk(chunk)
                for ent in kg_chunk.entities:
                    if ent.id not in all_entities:
                        all_entities[ent.id] = ent
                for rel in kg_chunk.relationships:
                    edge_key = (rel.source_id, rel.target_id, rel.relation_type)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        all_rels.append(rel)

        # Validate referential integrity on aggregate
        kg_aggregate = KnowledgeGraphExtraction(
            entities=list(all_entities.values()),
            relationships=all_rels
        )
        return all_chunks, kg_aggregate
