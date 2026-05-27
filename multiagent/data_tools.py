"""
Data Access Layer for OER Catalyst Multi-Agent System.
Provides tools that agents can call to query the pipeline outputs.
"""
import json
import pandas as pd
import os
import re
from pathlib import Path

PROJECT_DIR = Path("/home/hs/oer-catalyst-project")
DATA_DIR = Path("/home/hs/바탕화면/hsdata")
OUTPUT_DIR = PROJECT_DIR / "output"

# ============================================================
# Lab equipment constraints
# ============================================================
LAB_EQUIPMENT = {
    "available": [
        "beaker", "hotplate", "magnetic_stirrer", "syringe_pump",
        "ph_meter", "pipettes", "capping_agents", "centrifuge", "rde",
        "balance", "fume_hood", "drying_oven", "ultrasonic_bath"
    ],
    "unavailable": [
        "autoclave", "furnace", "microwave_oven", "vacuum_oven",
        "glove_box", "ball_mill", "cvd", "sputtering", "ald"
    ]
}

# Intents that require unavailable equipment
UNAVAILABLE_INTENTS = {
    "annealing": "furnace",
    "phase_transformation": "furnace",
    "phosphorization": "furnace",
    "sulfidation": "furnace",
    "calcination": "furnace",
    "cvd_deposition": "cvd",
    "sputtering": "sputtering",
    "hydrothermal": "autoclave",
}

# ============================================================
# Cache for loaded data
# ============================================================
_cache = {}


def _load_sentences():
    """Load the contextual intent CSV."""
    if "sentences" not in _cache:
        path = OUTPUT_DIR / "sentences_contextual_intent.csv"
        df = pd.read_csv(path)
        _cache["sentences"] = df
    return _cache["sentences"]


def _load_json(path):
    """Load a JSON file with caching."""
    key = str(path)
    if key not in _cache:
        with open(path, "r", encoding="utf-8") as f:
            _cache[key] = json.load(f)
    return _cache[key]


# ============================================================
# Tool Functions (callable by agents)
# ============================================================

def search_sentences(query: str, top_n: int = 10) -> str:
    """
    Search the 14,331 extracted sentences for keywords.
    Returns matching sentences with paper metadata.
    """
    df = _load_sentences()
    query_lower = query.lower()
    terms = query_lower.split()

    # Use OR logic: match any term, rank by number of matches
    scores = pd.Series([0] * len(df), dtype=float)
    for term in terms:
        if len(term) >= 2:
            match = df["sentence"].str.lower().str.contains(term, na=False, regex=False)
            scores += match.astype(float)
    
    # Also search in contextual_intent column
    for term in terms:
        if len(term) >= 2:
            match = df["contextual_intent"].str.lower().str.contains(term, na=False, regex=False)
            scores += match.astype(float) * 0.5  # intent match weighted lower
    
    results = df[scores > 0].copy()
    results["_score"] = scores[scores > 0]
    results = results.sort_values("_score", ascending=False).head(top_n)

    if results.empty:
        return f"No sentences found for query: {query}"

    output_lines = [f"Found {len(results)} sentences (score-ranked):"]
    for _, row in results.iterrows():
        score = row.get('_score', 0)
        output_lines.append(
            f"[{row.get('paper_id', 'N/A')}] "
            f"Intent: {row.get('contextual_intent', 'N/A')} "
            f"(score:{score:.1f}) | "
            f"{row['sentence'][:250]}"
        )

    return "\n".join(output_lines)


def search_papers(query: str, top_n: int = 5) -> str:
    """
    Search 905 JSON papers by title, abstract, or content keywords.
    Returns paper titles, journals, and DOIs.
    """
    query_lower = query.lower()
    results = []

    for fpath in DATA_DIR.glob("*.json"):
        try:
            data = _load_json(fpath)
            meta = data.get("metadata", {})
            title = meta.get("title", "").lower()
            keywords = " ".join(meta.get("keywords", [])).lower()

            # Also check first section text as abstract proxy
            doc = data.get("document", {})
            first_text = ""
            for section in list(doc.values())[:1]:
                if isinstance(section, dict):
                    for sub in list(section.values())[:1]:
                        if isinstance(sub, str):
                            first_text = sub[:500].lower()

            if query_lower in title or query_lower in keywords or query_lower in first_text:
                results.append({
                    "file": fpath.name,
                    "title": meta.get("title", "N/A"),
                    "journal": meta.get("journal", "N/A"),
                    "doi": meta.get("doi", "N/A"),
                })
        except Exception:
            continue

        if len(results) >= top_n:
            break

    if not results:
        return f"No papers found for: {query}"

    lines = []
    for r in results:
        lines.append(f"[{r['file']}] {r['title']}\n  Journal: {r['journal']} | DOI: {r['doi']}")

    return "\n\n".join(lines)


def get_synthesis_network() -> str:
    """
    Get the synthesis route network: nodes, edges, transition probabilities.
    """
    path = OUTPUT_DIR / "synthesis_network.json"
    if not path.exists():
        return "Synthesis network file not found."

    data = _load_json(path)
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    lines = ["=== Synthesis Network ==="]
    lines.append(f"Nodes (intents): {len(nodes)}")
    for n in nodes[:30]:
        lines.append(f"  - {n.get('id', 'N/A')} (count: {n.get('count', 0)})")

    lines.append(f"\nTop transitions:")
    sorted_edges = sorted(edges, key=lambda x: x.get("weight", 0), reverse=True)
    for e in sorted_edges[:15]:
        lines.append(
            f"  {e.get('source', 'N/A')} → {e.get('target', 'N/A')} "
            f"(weight: {e.get('weight', 0):.0f}, prob: {e.get('probability', 0):.2f})"
        )

    return "\n".join(lines)


def get_common_routes(target_material: str = "", top_n: int = 10) -> str:
    """
    Get common synthesis routes from the pipeline output.
    Optionally filter by target material keywords.
    """
    path = OUTPUT_DIR / "common_routes.json"
    if not path.exists():
        return "Common routes file not found."

    routes = _load_json(path)

    lines = [f"=== Common Synthesis Routes (top {top_n}) ==="]
    for r in routes[:top_n]:
        route_val = r.get("route", "N/A")
        # route can be string with arrows, or list
        if isinstance(route_val, list):
            route_str = " → ".join(route_val)
        else:
            route_str = str(route_val)
        count = r.get("count", 0)
        papers = r.get("paper_count", r.get("papers", "N/A"))
        lines.append(f"  [{count}x] {route_str} (papers: {papers})")

    return "\n".join(lines)


def get_lab_compatible_routes() -> str:
    """
    Get synthesis routes that are compatible with the lab's equipment.
    """
    path = OUTPUT_DIR / "lab_compatible_routes.csv"
    if not path.exists():
        return "Lab compatible routes file not found."

    df = pd.read_csv(path)
    feasible_col = "feasible" if "feasible" in df.columns else "lab_compatible"
    compatible = df[df[feasible_col] == True]

    lines = [f"=== Lab Compatible Routes ({len(compatible)} total) ==="]
    for _, row in compatible.head(15).iterrows():
        route = row.get("route", row.get("sequence", "N/A"))
        eta = row.get("overpotential_mV", row.get("avg_overpotential", "N/A"))
        lines.append(f"  Route: {route} | η: {eta} mV")

    return "\n".join(lines)


def get_route_performance() -> str:
    """
    Get OER performance data by synthesis route.
    """
    path = OUTPUT_DIR / "route_performance.csv"
    if not path.exists():
        return "Route performance file not found."

    df = pd.read_csv(path)

    lines = ["=== Route Performance Data ==="]
    lines.append(f"Total papers with performance data: {len(df)}")

    if "overpotential_mV" in df.columns:
        valid = df["overpotential_mV"].dropna()
        if len(valid) > 0:
            lines.append(f"Overpotential range: {valid.min():.0f} - {valid.max():.0f} mV")
            lines.append(f"Mean: {valid.mean():.0f} ± {valid.std():.0f} mV")

    return "\n".join(lines)


def get_intent_impact() -> str:
    """
    Get the impact of each synthesis intent on OER performance.
    """
    path = OUTPUT_DIR / "intent_impact.csv"
    if not path.exists():
        return "Intent impact file not found."

    df = pd.read_csv(path)
    lines = ["=== Intent Impact on Performance ==="]
    for _, row in df.iterrows():
        intent = row.get('intent', 'N/A')
        # Handle different column names
        diff = row.get('difference_mV', row.get('delta_eta_mV', 0))
        with_n = row.get('with_n', row.get('count', 0))
        with_mean = row.get('with_mean', 0)
        without_mean = row.get('without_mean', 0)
        try:
            diff_val = float(diff)
            lines.append(
                f"  {intent}: Δη = {diff_val:+.1f} mV "
                f"(n={with_n}, with={with_mean:.0f} vs without={without_mean:.0f} mV)"
            )
        except (ValueError, TypeError):
            lines.append(f"  {intent}: Δη = {diff} mV (n={with_n})")

    return "\n".join(lines)


def get_process_transfer_evidence() -> str:
    """
    Get scientific evidence for process substitution (e.g., autoclave → co-precipitation).
    """
    path = OUTPUT_DIR / "process_transfer_evidence.json"
    if not path.exists():
        return "Process transfer evidence file not found."

    data = _load_json(path)
    lines = ["=== Process Transfer Evidence ==="]

    # Summary stats
    summary = data.get("summary", {})
    total_papers = summary.get("total_nife_papers", "N/A")
    direct_comp = summary.get("direct_comparisons_found", "N/A")
    perf_points = summary.get("performance_data_points", "N/A")
    mech_evidence = summary.get("mechanism_evidence", "N/A")

    lines.append(f"\nTotal NiFe LDH papers analyzed: {total_papers}")
    lines.append(f"Direct comparisons (same paper): {direct_comp}")
    lines.append(f"Performance data points: {perf_points}")
    lines.append(f"Mechanism evidence passages: {mech_evidence}")

    # Key findings
    findings = data.get("key_findings", [])
    if findings:
        lines.append(f"\n--- Key Findings ({len(findings)}) ---")
        for i, f in enumerate(findings, 1):
            finding = f.get("finding", "N/A")
            evidence = f.get("evidence", "N/A")
            lines.append(f"  [{i}] {finding}")
            lines.append(f"      Evidence: {evidence}")

    # Reasoning chain
    chain = data.get("reasoning_chain", [])
    if chain:
        lines.append(f"\n--- Reasoning Chain ---")
        for step in chain:
            s = step.get("step", "?")
            claim = step.get("claim", "N/A")
            ev = step.get("evidence", "N/A")
            lines.append(f"  Step {s}: {claim}")
            lines.append(f"    → Evidence: {ev}")

    return "\n".join(lines)


def get_nife_protocols() -> str:
    """
    Get the 3 pre-generated NiFe LDH protocols.
    """
    path = OUTPUT_DIR / "nife_ldh_protocols.json"
    if not path.exists():
        return "NiFe LDH protocols file not found."

    data = _load_json(path)
    lines = ["=== NiFe LDH Protocols ==="]

    for proto in data.get("protocols", []):
        lines.append(f"\n--- {proto.get('name', 'N/A')} ---")
        lines.append(f"Method: {proto.get('method', 'N/A')}")
        lines.append(f"Expected η: {proto.get('expected_overpotential', 'N/A')}")

        steps = proto.get("steps", [])
        for i, step in enumerate(steps, 1):
            lines.append(f"  Step {i}: {step.get('action', 'N/A')}")

    return "\n".join(lines)


def check_equipment_compatibility(intent_list: list) -> str:
    """
    Check if a list of synthesis intents is compatible with lab equipment.
    Returns which intents are feasible and which need substitution.
    """
    lines = ["=== Equipment Compatibility Check ==="]

    for intent in intent_list:
        # Handle LLM passing dicts or other types
        if isinstance(intent, dict):
            intent = intent.get("intent", intent.get("chemical_intent", str(intent)))
        if not isinstance(intent, str):
            intent = str(intent)
        intent_lower = intent.lower().strip()
        if intent_lower in UNAVAILABLE_INTENTS:
            equip = UNAVAILABLE_INTENTS[intent_lower]
            lines.append(f"  ❌ {intent}: requires {equip} (NOT AVAILABLE)")
            # Suggest alternatives
            alternatives = {
                "annealing": "aging at 80-90°C (Ostwald ripening)",
                "phase_transformation": "co-precipitation + controlled pH aging",
                "phosphorization": "phosphorus salt addition during co-precipitation",
                "sulfidation": "thiourea addition during synthesis",
                "calcination": "thermal aging at 80-120°C on hotplate",
                "hydrothermal": "co-precipitation + aging / urea hydrolysis",
                "cvd_deposition": "electrodeposition (in-situ growth on substrate)",
                "sputtering": "electrodeposition or dip-coating",
            }
            alt = alternatives.get(intent_lower, "consult literature analyst")
            lines.append(f"     → Alternative: {alt}")
        else:
            lines.append(f"  ✅ {intent}: lab compatible")

    return "\n".join(lines)


def check_capabilities_feasibility(capabilities: list) -> str:
    """
    Check if required capabilities can be fulfilled by lab equipment.
    This checks the actual equipment needs (e.g., 'sealed vessel', 'furnace')
    rather than just intent names.
    """
    unavailable_keywords = [
        "autoclave", "sealed vessel", "sealed reactor", "hydrothermal reactor",
        "furnace", "tube furnace", "muffle furnace", "box furnace",
        "microwave", "microwave-assisted", "microwave reactor",
        "vacuum oven", "vacuum chamber",
        "glove box", "glovebox", "inert atmosphere chamber",
        "cvd", "chemical vapor deposition", "sputtering", "atomic layer deposition",
        "high pressure", "supercritical",
        "plasma", "laser",
    ]
    available_keywords = [
        "stirring", "magnetic stirrer", "stirrer",
        "hotplate", "heating", "temperature control",
        "ph", "ph meter", "ph monitoring",
        "pipette", "syringe", "syringe pump",
        "centrifuge", "centrifugation",
        "washing", "filtration",
        "drying oven", "oven", "drying",
        "mixing", "dissolving", "weighing",
        "rde", "rotating disk",
        "electrodeposition", "electrochemical",
        "ultrasonication", "sonication",
        "ambient", "room temperature",
        "liquid mixing", "solution preparation",
        "precursor mixing", "metal ion mixing",
        "aging", "thermal aging",
        "coprecipitation", "co-precipitation",
    ]

    lines = ["=== Required Capabilities Feasibility Check ==="]

    for cap in capabilities:
        if isinstance(cap, dict):
            cap = str(cap)
        cap_str = str(cap).lower().strip()
        cap_display = str(cap)

        is_unavailable = False
        reason = ""
        for kw in unavailable_keywords:
            if kw in cap_str:
                is_unavailable = True
                reason = f"requires '{kw}' → NOT AVAILABLE in our lab"
                break

        if is_unavailable:
            lines.append(f"  ❌ {cap_display}: {reason}")
            alt_map = {
                "autoclave": "open beaker + hotplate aging at 80-90°C (longer time, Ostwald ripening)",
                "sealed vessel": "open beaker with watch glass cover on hotplate",
                "sealed reactor": "open beaker with watch glass cover on hotplate",
                "hydrothermal reactor": "co-precipitation + 80-90°C aging",
                "furnace": "hotplate aging at 80-100°C (slower equivalent crystallization)",
                "tube furnace": "not replaceable — consider alternative synthesis route",
                "microwave": "conventional hotplate heating (longer time)",
                "vacuum oven": "conventional drying oven at 80-100°C",
                "glove box": "not available — work in fume hood",
                "cvd": "electrodeposition or dip-coating",
                "sputtering": "electrodeposition or dip-coating",
                "high pressure": "atmospheric pressure co-precipitation + aging",
            }
            for kw, alt in alt_map.items():
                if kw in cap_str:
                    lines.append(f"     → Alternative: {alt}")
                    break
        else:
            is_available = any(kw in cap_str for kw in available_keywords)
            if is_available:
                lines.append(f"  ✅ {cap_display}: available in lab")
            else:
                lines.append(f"  ⚠️ {cap_display}: needs manual verification")

    return "\n".join(lines)


# ============================================================
# Tool Registry (for AutoGen function calling)
# ============================================================

TOOL_FUNCTIONS = [
    search_sentences,
    search_papers,
    get_synthesis_network,
    get_common_routes,
    get_lab_compatible_routes,
    get_route_performance,
    get_intent_impact,
    get_process_transfer_evidence,
    get_nife_protocols,
    check_equipment_compatibility,
    check_capabilities_feasibility,
]

# Pydantic-style function schemas for OpenAI function calling
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_sentences",
            "description": "Search 14,331 extracted sentences from OER catalyst literature by keywords. Returns matching sentences with paper metadata and chemical intent tags.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Space-separated keywords to search (e.g., 'NiFe LDH co-precipitation')"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search 905 OER catalyst papers by title, keywords, or content. Returns paper metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for paper titles and content"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Max results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_synthesis_network",
            "description": "Get the full synthesis route network with nodes (chemical intents) and edges (transitions between process steps). Shows how synthesis procedures flow.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_common_routes",
            "description": "Get the most common synthesis routes found across 905 papers. Can filter by target material.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_material": {
                        "type": "string",
                        "description": "Material to filter by (e.g., 'NiFe LDH')",
                        "default": ""
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of routes (default: 10)",
                        "default": 10
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_lab_compatible_routes",
            "description": "Get synthesis routes that are compatible with the lab equipment (beaker, hotplate, stirrer, syringe pump, pH meter, centrifuge, RDE).",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_route_performance",
            "description": "Get OER performance data (overpotential) for different synthesis routes.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_intent_impact",
            "description": "Get the impact of each chemical intent (synthesis step type) on OER performance (overpotential change in mV).",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_process_transfer_evidence",
            "description": "Get scientific evidence for substituting unavailable processes (autoclave → co-precipitation, furnace → aging, etc.) from literature.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_nife_protocols",
            "description": "Get 3 pre-generated NiFe LDH synthesis protocols (co-precipitation, electrodeposition, urea hydrolysis).",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_equipment_compatibility",
            "description": "Check if a list of synthesis process steps is compatible with available lab equipment. Identifies which steps need substitution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of chemical intents to check (e.g., ['nucleation', 'crystallization', 'purification'])"
                    }
                },
                "required": ["intent_list"]
            }
        }
    },
]
