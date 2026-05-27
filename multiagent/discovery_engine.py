"""
Materials Discovery Engine — Cross-Domain Novelty Search

핵심: "문제 → Intent 분해 → 다분야 Exploration → Novelty Check → Discovery 제안"

Input:  Material + Problem
        "NiFe LDH, stability bad"

Output: Novel material + synthesis protocol candidates
        (서로 다른 출처의 조합 = Discovery)

Architecture:
  Step 1: Problem Decomposer
          Problem → Chemical Intents + Search Queries
          
  Step 2: Explorer (per Intent, multi-domain search)
          Intent A "Crystallization" → OER, Battery, MOF, Perovskite...
          Intent B "Surface Protection" → Drug Delivery, Semiconductor, Corrosion...
          
  Step 3: Novelty Checker (Semantic Scholar API)
          "NiFe LDH" + "lipid coating" → 0 papers = NOVEL ✅
          "NiFe LDH" + "co-precipitation" → 50 papers = NOT novel ❌
          
  Step 4: Hypothesis Generator
          조합 + 화학적 충돌 검사 + 신뢰도 등급
          
  Step 5: CID Translation
          최종 프로토콜 → CID 6컬럼 + 실험실 제약 검증

Usage:
  python discovery_engine.py "NiFe LDH stability" 
  python discovery_engine.py "CsPbBr3 moisture sensitivity"
"""
import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from data_tools import search_sentences
from cid_agent import CORE_INTENTS, _load_api_key, parse_cid_json

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

MODEL = "gpt-4o-mini"
model_client = OpenAIChatCompletionClient(model=MODEL, api_key=_load_api_key())


# ============================================================
# Step 1: Problem Decomposer
# ============================================================
DECOMPOSER_PROMPT = f"""당신은 재료 과학 문제 분해 전문가입니다.
재료의 문제를 CID Chemical Intent 단위로 분해합니다.

10개 Chemical Intent:
{chr(10).join(f"  {i+1}. {x}" for i, x in enumerate(CORE_INTENTS))}

출력 형식 (JSON):
{{
  "material": "재료명",
  "problem": "문제 설명",
  "decomposed": [
    {{
      "intent": "Chemical Intent명",
      "sub_problem": "이 Intent에서 무엇이 부족한가",
      "domain_A_search": "합성/제조 관점에서 검색할 키워드 (여러 분야)",
      "domain_B_search": "문제 해결 관점에서 검색할 키워드 (여러 분야)"
    }}
  ],
  "novelty_queries": [
    "material + solution 조합이 새로운지 확인할 검색어"
  ]
}}

예시:
문제: "NiFe LDH가 물에서 불안정하다"
→ Intent: Morphology Control (표면 보호 필요)
→ domain_A: "NiFe LDH synthesis coating encapsulation"
→ domain_B: "perovskite stability encapsulation, drug delivery lipid coating, semiconductor passivation"
→ novelty: "NiFe LDH lipid coating", "NiFe LDH polymer encapsulation"

반드시 JSON으로 출력하세요. 한국어로 설명은 쓰되 키/값은 영어로."""


# ============================================================
# Step 2: Explorer — Multi-Domain Search
# ============================================================
DOMAIN_KEYWORDS = {
    "OER/Electrocatalysis": ["OER", "electrocatalyst", "water splitting", "oxygen evolution"],
    "Battery/Energy Storage": ["battery", "supercapacitor", "energy storage", "Li-ion"],
    "MOF/Porous Materials": ["MOF", "ZIF", "COF", "porous", "metal-organic framework"],
    "Perovskite/Solar": ["perovskite", "solar cell", "photovoltaic", "halide"],
    "Drug Delivery/Bio": ["drug delivery", "lipid", "nanoparticle coating", "biocompatible"],
    "Semiconductor": ["semiconductor", "passivation", "ALD", "dielectric"],
    "Corrosion/Protection": ["corrosion", "protective coating", "inhibitor", "anodization"],
    "Polymer Science": ["polymer", "block copolymer", "self-assembly", "encapsulation"],
    "Colloid/Nano": ["colloid", "nanoparticle", "seed-mediated", "core-shell"],
    "Ceramics": ["ceramic", "sintering", "sol-gel", "calcination"],
}

EXPLORER_PROMPT = """당신은 다분야 화학 지식 전문가입니다.
각 Intent에 대해 여러 분야에서의 해결책을 제안합니다.

할 일:
1. 주어진 Intent에 대해, 각 분야(Domain)에서 어떤 Mechanism을 사용하는지 정리
2. 각 Mechanism을 CID 형식으로 간단히 기술:
   - Intent: (10개 중 하나)
   - Mechanism: ①원인 ②과정 ③결과 체인
   - Domain: 어느 분야에서 온 것인가
   - Required Capabilities: 필요한 장비/조건
   - Lab Compatible: 우리 실험실에서 가능한가? (autoclave/furnace/microwave 없음)
3. Novelty 후보: material + mechanism 조합이 새로울 것 같은지 판단

출력: JSON 배열
한국어로 응답."""


# ============================================================
# Step 3: Novelty Check via Semantic Scholar
# ============================================================
async def check_novelty(material: str, solution: str) -> dict:
    """
    Check if a material + solution combination is novel
    using Semantic Scholar API (free, no key required).
    """
    import urllib.request
    import urllib.parse
    
    query = f"{material} {solution}"
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(query)}&limit=5&fields=title,year,abstract"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OER-SDL-Discovery/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        
        total = data.get("total", 0)
        papers = data.get("data", [])
        
        # Check if any paper explicitly combines material + solution
        relevant = 0
        for p in papers[:5]:
            title = (p.get("title") or "").lower()
            abstract = (p.get("abstract") or "").lower()
            text = title + " " + abstract
            
            mat_words = material.lower().split()
            sol_words = solution.lower().split()
            
            mat_match = sum(1 for w in mat_words if w in text)
            sol_match = sum(1 for w in sol_words if w in text)
            
            if mat_match >= 1 and sol_match >= 1:
                relevant += 1
        
        if relevant == 0:
            novelty = "NOVEL"
            confidence = "HIGH"
        elif relevant <= 1:
            novelty = "LIKELY_NOVEL"
            confidence = "MEDIUM"
        else:
            novelty = "NOT_NOVEL"
            confidence = "LOW"
        
        return {
            "query": query,
            "total_results": total,
            "relevant": relevant,
            "novelty": novelty,
            "confidence": confidence,
            "sample_titles": [p.get("title", "N/A") for p in papers[:3]],
        }
        
    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "novelty": "UNKNOWN",
            "confidence": "LOW",
        }


# ============================================================
# Step 5: CID Translation + Protocol
# ============================================================
CID_TRANSLATOR_PROMPT = f"""당신은 CID 전문가입니다.
Discovery 제안을 CID 6컬럼으로 번역합니다.

10개 Chemical Intent:
{chr(10).join(f"  {i+1}. {x}" for i, x in enumerate(CORE_INTENTS))}

6컬럼:
1. step_id
2. raw_step (한국어 설명)
3. chemical_intent (10개 중 하나)
4. mechanism (①원인 ②과정 ③결과 체인)
5. tunable_conditions (변수 + 제안값)
6. required_capabilities (장비/조건)
7. output_form
8. source_domain (어느 분야에서 가져왔는지)
9. novelty_score (NOVEL/LIKELY_NOVEL/NOT_NOVEL)
10. confidence (HIGH/MEDIUM/LOW)

실험실 제약: beaker, hotplate, stirrer, syringe pump, pH meter, centrifuge, RDE만 가능
불가: autoclave, furnace, microwave, vacuum oven, glove box

반드시 JSON 배열로 출력하세요."""


# ============================================================
# Main Pipeline
# ============================================================
async def run_discovery(material_problem: str):
    print(f"\n{'═'*60}")
    print(f"  Materials Discovery Engine v1.0")
    print(f"  Input: {material_problem}")
    print(f"{'═'*60}\n")
    
    # ─── Step 1: Decompose ────────────────────────────────────
    print("┌─────────────────────────────────────────┐")
    print("│ Step 1: Problem Decomposition              │")
    print("└─────────────────────────────────────────┘\n")
    
    decomposer = AssistantAgent(
        name="Decomposer",
        model_client=model_client,
        system_message=DECOMPOSER_PROMPT,
    )
    
    from autogen_core import CancellationToken
    from autogen_agentchat.messages import TextMessage as TMsg
    
    resp = await decomposer.on_messages(
        [TMsg(content=material_problem, source="user")],
        cancellation_token=CancellationToken(),
    )
    
    decomposition_text = getattr(resp.chat_message, 'content', '') if resp.chat_message else ''
    print(decomposition_text[:2000])
    
    # Parse decomposition
    try:
        decomp_json = json.loads(
            decomposition_text.split('```json')[1].split('```')[0]
            if '```json' in decomposition_text
            else decomposition_text
        )
    except:
        decomp_json = {"material": material_problem, "problem": material_problem, "decomposed": []}
    
    material = decomp_json.get("material", material_problem.split()[0])
    print(f"\n  Material: {material}")
    print(f"  Decomposed intents: {len(decomp_json.get('decomposed', []))}")
    
    # ─── Step 2: Explore Multi-Domain ─────────────────────────
    print(f"\n┌─────────────────────────────────────────┐")
    print(f"│ Step 2: Multi-Domain Exploration           │")
    print(f"└─────────────────────────────────────────┘\n")
    
    # Pre-search from internal DB
    search_results = {}
    for intent_info in decomp_json.get("decomposed", []):
        for key in ["domain_A_search", "domain_B_search"]:
            query = intent_info.get(key, "")
            if query:
                result = search_sentences(query, top_n=3)
                search_results[query] = result
    
    explorer = AssistantAgent(
        name="Explorer",
        model_client=model_client,
        system_message=EXPLORER_PROMPT,
    )
    
    explore_task = f"""Material: {material}
Problem: {decomp_json.get('problem', '')}

Decomposed Intents:
{json.dumps(decomp_json.get('decomposed', []), ensure_ascii=False, indent=2)}

Internal DB search results (OER 905 papers):
{json.dumps(search_results, ensure_ascii=False)[:4000]}

Domains to explore:
{json.dumps(list(DOMAIN_KEYWORDS.keys()), ensure_ascii=False)}

위 정보를 바탕으로:
1. 각 Intent에 대해 여러 분야에서 사용하는 Mechanism을 제안
2. 우리 실험실에서 실행 가능한 것만 선별
3. JSON 배열로 출력 (각 제안에 domain, novelty 판단 포함)"""

    resp = await explorer.on_messages(
        [TMsg(content=explore_task, source="user")],
        cancellation_token=CancellationToken(),
    )
    
    exploration_text = getattr(resp.chat_message, 'content', '') if resp.chat_message else ''
    print(exploration_text[:3000])
    
    # ─── Step 3: Novelty Check ────────────────────────────────
    print(f"\n┌─────────────────────────────────────────┐")
    print(f"│ Step 3: Novelty Check (Semantic Scholar)   │")
    print(f"└─────────────────────────────────────────┘\n")
    
    novelty_queries = decomp_json.get("novelty_queries", [])
    
    # Add material + domain combinations
    for intent_info in decomp_json.get("decomposed", []):
        domain_b = intent_info.get("domain_B_search", "")
        if domain_b:
            words = domain_b.split(",")
            for w in words[:2]:
                novelty_queries.append(f"{material} {w.strip()}")
    
    novelty_results = []
    checked = set()
    for query in novelty_queries[:8]:
        q = query.strip().lower()
        if q in checked:
            continue
        checked.add(q)
        
        result = await check_novelty(material, query.replace(material, "").strip())
        novelty_results.append(result)
        
        icon = {"NOVEL": "🆕", "LIKELY_NOVEL": "🟡", "NOT_NOVEL": "❌", "UNKNOWN": "❓"}.get(result["novelty"], "?")
        print(f"  {icon} {query}: {result['novelty']} (relevant={result.get('relevant','?')}, total={result.get('total_results','?')})")
        if result.get("sample_titles"):
            for t in result["sample_titles"][:2]:
                print(f"     → {t[:80]}")
    
    # ─── Step 4 & 5: Generate Discovery Proposals ─────────────
    print(f"\n┌─────────────────────────────────────────┐")
    print(f"│ Step 4+5: Discovery Proposals + CID        │")
    print(f"└─────────────────────────────────────────┘\n")
    
    # Filter novel combinations
    novel_items = [r for r in novelty_results if r["novelty"] in ["NOVEL", "LIKELY_NOVEL"]]
    
    translator = AssistantAgent(
        name="CID_Translator",
        model_client=model_client,
        system_message=CID_TRANSLATOR_PROMPT,
    )
    
    translate_task = f"""Material: {material}
Problem: {decomp_json.get('problem', '')}

Novel combinations found:
{json.dumps(novel_items, ensure_ascii=False, indent=2)}

Exploration results:
{exploration_text[:3000]}

위 정보를 바탕으로:
1. Novel/Likely Novel 조합에 대해 완성된 합성 프로토콜을 CID로 작성
2. 각 step에 source_domain 표시
3. 화학적 충돌 검사 (용매 충돌, 온도 범위 등)
4. 실험실에서 실행 가능한지 판단
5. 최종 Discovery 제안 1~3개를 JSON 배열로 출력"""

    resp = await translator.on_messages(
        [TMsg(content=translate_task, source="user")],
        cancellation_token=CancellationToken(),
    )
    
    discovery_text = getattr(resp.chat_message, 'content', '') if resp.chat_message else ''
    print(discovery_text[:4000])
    
    # ─── Save Results ─────────────────────────────────────────
    result = {
        "input": material_problem,
        "material": material,
        "decomposition": decomp_json,
        "exploration": exploration_text,
        "novelty_check": novelty_results,
        "novel_count": len(novel_items),
        "discovery_proposals": discovery_text,
    }
    
    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"discovery_{material.replace(' ', '_').lower()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n{'═'*60}")
    print(f"  Discovery 완료!")
    print(f"  Novel 조합: {len(novel_items)}개")
    print(f"  저장: {out_path}")
    print(f"{'═'*60}")
    
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", help="Material + problem description")
    args = parser.parse_args()
    asyncio.run(run_discovery(args.problem))
