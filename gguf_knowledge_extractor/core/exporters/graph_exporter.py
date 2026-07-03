"""
Knowledge Graph exporter
========================
Outputs the extracted knowledge as a graph in two formats:
  - GraphML  (XML-based, works with NetworkX, Gephi, yEd, Cytoscape)
  - Turtle   (RDF/N3, works with Apache Jena, rdflib, SPARQL endpoints)

Nodes:
  - ModelNode      (the GGUF model itself)
  - DomainNode     (geography, science, programming, etc.)
  - ConceptNode    (a specific concept/skill)
  - FactNode       (an extracted fact)
  - ProbeNode      (a probe that was run)
  - LayerNode      (a model layer, from weight inspection)

Edges:
  - knows_domain     Model -> Domain
  - has_concept      Domain -> Concept
  - mastered_at      Concept -> "expert|proficient|..." (literal)
  - knows_fact       Model -> Fact
  - fact_in_domain   Fact -> Domain
  - has_layer        Model -> Layer
  - validated_by     Fact/Concept -> Probe
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import networkx as nx

from ..extractor import KnowledgeReport


# ---------------------------------------------------------------------- #
# GraphML
# ---------------------------------------------------------------------- #
def export_graphml(report: KnowledgeReport, out_path: Union[str, Path]) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    G = nx.DiGraph()

    model_id = f"model:{report.gguf_filename}"
    md = report.metadata if isinstance(report.metadata, dict) else {}
    G.add_node(model_id, label=md.get("name") or report.gguf_filename,
               type="model", arch=md.get("arch",""),
               quantization=md.get("quantization",""),
               vocab_size=md.get("vocab_size",0),
               context_length=md.get("context_length",0))

    # Domains & concepts
    for c in report.concepts:
        domain = c.get("domain","general")
        domain_id = f"domain:{domain}"
        G.add_node(domain_id, label=domain, type="domain")
        G.add_edge(model_id, domain_id, relation="knows_domain",
                   mastery=c.get("mastery_level","unknown"),
                   pass_rate=c.get("pass_rate",0))

        concept_id = f"concept:{domain}:{c.get('mastery_level','?')}"
        G.add_node(concept_id, label=f"{domain} ({c.get('mastery_level','')})",
                   type="concept", n_probes=c.get("n_probes",0),
                   n_passed=c.get("n_passed",0))
        G.add_edge(domain_id, concept_id, relation="has_concept")
        G.add_edge(concept_id, model_id, relation="validated_by")

    # Facts
    for f in report.facts:
        fact_id = f"fact:{f.get('id','')}"
        G.add_node(fact_id, label=f.get("id",""), type="fact",
                   correct=f.get("correct",False),
                   domain=f.get("domain",""),
                   expected=f.get("expected_answer",""),
                   model_answer=(f.get("model_answer","") or "")[:200])
        G.add_edge(model_id, fact_id, relation="knows_fact",
                   correct=f.get("correct",False))
        if f.get("domain"):
            G.add_edge(fact_id, f"domain:{f['domain']}", relation="fact_in_domain")

    # Behavioral
    bp = report.behavioral_profile
    if bp:
        bnode = f"behavioral:{report.gguf_filename}"
        G.add_node(bnode, label="Behavioral Profile", type="behavioral",
                   refusal_rate=bp.get("refusal_rate",0),
                   n_refusal_probes=bp.get("n_refusal_probes",0),
                   n_refused=bp.get("n_refused",0))
        G.add_edge(model_id, bnode, relation="has_behavioral_profile")

    # Calibration
    cal = report.calibration
    if cal:
        cnode = f"calibration:{report.gguf_filename}"
        G.add_node(cnode, label="Calibration", type="calibration",
                   hallucination_ack_rate=cal.get("hallucination_acknowledgment_rate") or 0,
                   math_accuracy=cal.get("math_accuracy") or 0)
        G.add_edge(model_id, cnode, relation="has_calibration")

    # Layers
    wi = report.weight_inspection if isinstance(report.weight_inspection, dict) else {}
    for ls in wi.get("layer_stats", [])[:50]:
        lnode = f"layer:{ls.get('layer_index',0)}"
        G.add_node(lnode, label=f"Layer {ls.get('layer_index',0)}", type="layer")
        G.add_edge(model_id, lnode, relation="has_layer")

    # v2: Attribution nodes
    attr = report.attribution if isinstance(report.attribution, dict) else {}
    if attr:
        # Fingerprint node
        fp = attr.get("knowledge_fingerprint","")
        if fp:
            fp_node = f"fingerprint:{report.gguf_filename}"
            G.add_node(fp_node, label="Knowledge Fingerprint", type="fingerprint",
                       fingerprint=fp[:32],
                       n_layers_analyzed=(attr.get("stats",{}) or {}).get("n_layers_analyzed_mlp",0))
            G.add_edge(model_id, fp_node, relation="has_fingerprint")

        # Top memory neurons
        mlp = attr.get("mlp_analysis", {}) or {}
        for n in (mlp.get("global_top_neurons", []) or [])[:30]:
            nnode = f"neuron:L{n.get('layer','?')}_N{n.get('neuron_index','?')}"
            top_tokens = n.get("top_activating_tokens", []) or []
            top_tok = top_tokens[0][0] if top_tokens and isinstance(top_tokens[0], list) else ""
            G.add_node(nnode, label=f"Neuron L{n.get('layer','?')}:{n.get('neuron_index','?')}",
                       type="neuron",
                       memory_strength=n.get("memory_strength",0),
                       top_activating_token=str(top_tok)[:30])
            lnode = f"layer:{n.get('layer',0)}"
            if not G.has_node(lnode):
                G.add_node(lnode, label=f"Layer {n.get('layer',0)}", type="layer")
                G.add_edge(model_id, lnode, relation="has_layer")
            G.add_edge(lnode, nnode, relation="contains_neuron")

        # Fact attributions
        for fa in attr.get("fact_attributions", []) or []:
            if fa.get("attributed_layer") is None:
                continue
            attr_node = f"attribution:{fa.get('probe_id','')}"
            G.add_node(attr_node, label=f"Attribution: {fa.get('probe_id','')}",
                       type="attribution",
                       layer=fa.get("attributed_layer"),
                       neuron=fa.get("attributed_neuron"),
                       confidence=fa.get("attribution_confidence",0),
                       method=fa.get("method",""))
            fact_node = f"fact:{fa.get('probe_id','')}"
            if G.has_node(fact_node):
                G.add_edge(fact_node, attr_node, relation="attributed_by")
            nnode = f"neuron:L{fa.get('attributed_layer','?')}_N{fa.get('attributed_neuron','?')}"
            if not G.has_node(nnode):
                G.add_node(nnode, label=f"Neuron L{fa.get('attributed_layer','?')}:{fa.get('attributed_neuron','?')}",
                           type="neuron")
                lnode = f"layer:{fa.get('attributed_layer',0)}"
                if not G.has_node(lnode):
                    G.add_node(lnode, label=f"Layer {fa.get('attributed_layer',0)}", type="layer")
                    G.add_edge(model_id, lnode, relation="has_layer")
                G.add_edge(lnode, nnode, relation="contains_neuron")
            G.add_edge(attr_node, nnode, relation="located_at")

    nx.write_graphml(G, out_path)
    return str(out_path)


# ---------------------------------------------------------------------- #
# RDF/Turtle
# ---------------------------------------------------------------------- #
def export_turtle(report: KnowledgeReport, out_path: Union[str, Path]) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    PREFIX = """@prefix : <http://gguf-extractor.local/ns#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix dc: <http://purl.org/dc/elements/1.1/> .

"""
    lines: list[str] = [PREFIX]

    def esc(s):
        s = str(s) if s is not None else ""
        s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        return s

    model_uri = f":model_{report.gguf_filename.replace('.','_')}"
    md = report.metadata if isinstance(report.metadata, dict) else {}
    lines.append(f"{model_uri} a :GGUFModel ;")
    lines.append(f"    rdfs:label \"{esc(md.get('name') or report.gguf_filename)}\" ;")
    lines.append(f"    :fileName \"{esc(report.gguf_filename)}\" ;")
    lines.append(f"    :architecture \"{esc(md.get('arch',''))}\" ;")
    lines.append(f"    :quantization \"{esc(md.get('quantization',''))}\" ;")
    lines.append(f"    :vocabSize {md.get('vocab_size',0)} ;")
    lines.append(f"    :contextLength {md.get('context_length',0)} ;")
    lines.append(f"    :extractedAt \"{esc(report.extraction_timestamp)}\" .")
    lines.append("")

    # Concepts
    for c in report.concepts:
        d = c.get("domain","general")
        domain_uri = f":domain_{d.replace(' ','_')}"
        lines.append(f"{domain_uri} a :Domain ; rdfs:label \"{esc(d)}\" .")
        lines.append(f"{model_uri} :knowsDomain {domain_uri} ; :masteryLevel \"{esc(c.get('mastery_level',''))}\" .")
        concept_uri = f":concept_{d.replace(' ','_')}_{c.get('mastery_level','?')}"
        lines.append(f"{concept_uri} a :Concept ; rdfs:label \"{esc(d)} ({c.get('mastery_level','')})\" ;")
        lines.append(f"    :probeCount {c.get('n_probes',0)} ; :passedCount {c.get('n_passed',0)} .")
        lines.append(f"{domain_uri} :hasConcept {concept_uri} .")
        lines.append("")

    # Facts
    for f in report.facts:
        fact_uri = f":fact_{f.get('id','').replace(' ','_')}"
        lines.append(f"{fact_uri} a :Fact ;")
        lines.append(f"    rdfs:label \"{esc(f.get('id',''))}\" ;")
        lines.append(f"    :prompt \"{esc(f.get('prompt',''))}\" ;")
        lines.append(f"    :expectedAnswer \"{esc(f.get('expected_answer',''))}\" ;")
        lines.append(f"    :modelAnswer \"{esc(f.get('model_answer',''))}\" ;")
        lines.append(f"    :correct \"{f.get('correct',False)}\"^^xsd:boolean .")
        lines.append(f"{model_uri} :knowsFact {fact_uri} .")
        if f.get("domain"):
            lines.append(f"{fact_uri} :inDomain :domain_{f['domain'].replace(' ','_')} .")
        lines.append("")

    # Behavioral
    bp = report.behavioral_profile
    if bp:
        b_uri = f":behavioral_{report.gguf_filename.replace('.','_')}"
        lines.append(f"{b_uri} a :BehavioralProfile ;")
        lines.append(f"    :refusalRate {bp.get('refusal_rate',0)} ;")
        lines.append(f"    :refusalProbeCount {bp.get('n_refusal_probes',0)} ;")
        lines.append(f"    :refusedCount {bp.get('n_refused',0)} .")
        lines.append(f"{model_uri} :hasBehavioralProfile {b_uri} .")
        lines.append("")

    # Calibration
    cal = report.calibration
    if cal:
        c_uri = f":calibration_{report.gguf_filename.replace('.','_')}"
        lines.append(f"{c_uri} a :Calibration ;")
        if cal.get("hallucination_acknowledgment_rate") is not None:
            lines.append(f"    :hallucinationAckRate {cal['hallucination_acknowledgment_rate']} ;")
        if cal.get("math_accuracy") is not None:
            lines.append(f"    :mathAccuracy {cal['math_accuracy']} .")
        lines.append(f"{model_uri} :hasCalibration {c_uri} .")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(out_path)
