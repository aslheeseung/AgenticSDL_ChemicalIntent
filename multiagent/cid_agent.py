"""
CID Agent — Chemical Intent Descriptor v3.2
논문 합성 절차를 CID 6컬럼으로 번역

컬럼:
  1. Raw Step     — 논문 원문
  2. Chemical Intent — 이 step의 목적 (닫힌 집합)
  3. Mechanism    — 어떻게 달성하는가 (열린 집합, 화학적 맥락)
  4. Tunable Conditions — 조절 가능한 변수 + 논문 값
  5. Required Capabilities — 필요한 장비/기능
  6. Output Form  — 이 step 산물의 형태

Usage:
  python cid_agent.py "NiFe LDH" --papers 5
"""
import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Optional

# Load .env
def _load_api_key():
    key = os.environ.get("OPENAI_API_KEY", "")
    if key and len(key) > 50:
        return key
    prefix = "OPENAI_API_KEY"
    for _ep in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
        if _ep.exists():
            for line in _ep.read_text().strip().splitlines():
                if line.split("=", 1)[0].strip() == prefix:
                    key = line.split("=", 1)[1].strip()
                    break
    return key

sys.path.insert(0, str(Path(__file__).parent))
from data_tools import TOOL_FUNCTIONS

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

MODEL = "gpt-4o-mini"
model_client = OpenAIChatCompletionClient(model=MODEL, api_key=_load_api_key())
tool_map = {fn.__name__: fn for fn in TOOL_FUNCTIONS}

# ============================================================
# CID Intent Set (v3.2 — 10 core + extended)
# ============================================================
CORE_INTENTS = [
    "Nucleation",
    "Crystallization",
    "Redox Control",
    "Stoichiometry Control",
    "Doping",
    "Morphology Control",
    "Purification",
    "Drying",
    "Catalyst-Electrode Coupling",
    "Electrochemical Activation",
]

# Extended set for broader coverage (can be mapped to core 10 later)
EXTENDED_INTENTS = CORE_INTENTS + [
    "Mixing",
    "Dissolution",
    "Aging",
    "Etching",
    "Phase Transformation",
    "Substrate Preparation",
    "Surface Modification",
    "Intercalation",
    "Exfoliation",
    "Carbon Coating",
    "Calcination",
]

# ============================================================
# Agent Prompts
# ============================================================

EXTRACTOR_PROMPT = """당신은 합성 문헌 추출 전문가입니다.
논문에서 합성(synthesis/experimental) 관련 문장을 찾아 원문 그대로 추출합니다.

필수 절차:
1. **반드시 search_sentences를 먼저 호출하세요.** 이게 가장 중요합니다.
   - 여러 키워드로 나눠서 검색: "nife ldh", "co-precipitation", "hydrothermal", "electrodeposition"
   - 각각 개별 단어로도 검색: "precipitation", "nucleation", "synthesis"
2. search_papers로 관련 논문 메타데이터 보충
3. 합성 절차가 포함된 문장들을 원문 그대로 모음
4. 각 문장에 출처(paper ID)와 기존 Intent 태그를 붙임

중요: 
- 의역하지 말고 원문 그대로 추출하세요
- search_sentences를 반드시 여러 번 호출하세요 (다양한 키워드로)
- 한국어로 응답하되, 원문 인용은 영어 그대로 유지하세요"""

CID_PROMPT = f"""당신은 CID(Chemical Intent Descriptor) v3.2 작성 전문가입니다.
추출된 합성 문장을 CID 6컬럼으로 번역합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CID 6컬럼 정의:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **Raw Step**: 논문 원문 그대로 (절대 수정하지 마세요)
2. **Chemical Intent**: 이 step이 "무엇을 하려는 것인지" (아래 10개에서 선택)
3. **Mechanism**: 그 목적을 "어떻게 화학적으로 달성하는지"
4. **Tunable Conditions**: 조절 가능한 변수 = 논문에 기록된 값 (판단 없이 사실만)
5. **Required Capabilities**: 이 step에 필요한 장비/기능 (자연어)
6. **Output Form**: 이 step이 만들어내는 것 (다음 step과 연결)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chemical Intent 닫힌 집합 (반드시 이 중 하나):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join(f"  {i+1}. {x}" for i, x in enumerate(CORE_INTENTS))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mechanism 작성 기준 (가장 중요):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mechanism은 "왜 이 조건이 필요한지, 원자/분자 수준에서 무슨 일이 일어나는지"를 
서술해야 합니다. 아래 세 가지를 모두 포함하세요:

  ① 원인: 무엇이 이 반응/변화를 일으키는가?
  ② 과정: 원자/이온/분자 수준에서 어떤 변화가 일어나는가?
  ③ 결과: 그 변화가 어떤 구조/조성/상태를 만들어내는가?

❌ 나쁜 예시 (금지):
  - "다양한 방법으로 결정화" → 너무 모호, 화학적 내용 없음
  - "Co-precipitation을 통해 합성" → 방법 라벨만, 메커니즘 없음
  - "결정 구조가 유지됨" → 과정/원인 설명 없음
  - "원문 그대로 복사" → 너무 상세, 메커니즘이 안 보임

✅ 좋은 예시:

[예시 1] Raw: "NH4F slows down the LDH nucleation and growth rate, 
         allowing the formation of pure NiFe-LDH nanosheets"
Intent: Nucleation
Mechanism: "NH4F가 Ni²⁺/Fe³⁺와 배위하여 금속 이온의 방출 속도를 낮춤 →
           핵형성 속도가 감소하여 과포화도 천천히 상승 → 
           균일한 크기의 LDH nanosheet가 선택적으로 성장"
Conditions: {{"additive": "NH4F"}}
Required: ["precursor mixing", "reaction rate control"]
Output: "NiFe-LDH nanosheets"

[예시 2] Raw: "NiFe-LDH was prepared with a hydrothermal method at 120°C for 12h"
Intent: Crystallization
Mechanism: "120°C 고온 + 밀폐 용기에서의 자기발생 압력 → 
           금속 수산화물 층간의 탈수/재배열 가속 → 
           질서정연한 LDH 층상 구조 형성 (12h = 충분한 결정화 시간)"
Conditions: {{"temperature": "120°C", "time": "12h", "method": "hydrothermal"}}
Required: ["sealed vessel", "heating up to 120°C"]
Output: "crystalline NiFe-LDH"

[예시 3] Raw: "epoxide was used as a proton scavenger to slowly release OH⁻"
Intent: Nucleation
Mechanism: "Epoxide가 H⁺와 반응하여 ring-opening → 
           용액 내 H⁺ 농도가 서서히 감소 → 
           OH⁻/H⁺ 비율이 점진적으로 상승하여 
           급격한 과포화 없이 균일 핵형성 유도"
Conditions: {{"additive": "epoxide"}}
Required: ["liquid mixing", "pH monitoring"]
Output: "homogeneously nucleated LDH precursor"

[예시 4] Raw: "Ce–NiFe LDH with Fe:Ce = 1.8:0.2"
Intent: Doping
Mechanism: "Ce⁴⁺가 Fe³⁺ 자리에 부분 치환 → 
           격자 내 이온 반경 차이(Ce⁴⁺ 0.97Å vs Fe³⁺ 0.645Å)로 
           국소적 격자 왜곡 유발 → 
           이 왜곡이 OER 활성점 주변의 전자 구조를 변조"
Conditions: {{"Fe:Ce ratio": "1.8:0.2", "dopant": "Ce"}}
Required: ["precise stoichiometric mixing"]
Output: "Ce-doped NiFe-LDH"

[예시 5] Raw: "the precipitate was collected and washed several times with 
         deionized water and dried at 60°C"
Intent: Purification
Mechanism: "원심분리로 고체 산물을 모액과 분리 → 
           DI water 세척으로 잔류 이온(NO₃⁻, Na⁺) 제거 → 
           60°C 건조로 층간 잔류 수분 증발"
Conditions: {{"wash_solvent": "DI water", "wash_cycles": "several", "dry_temp": "60°C"}}
Required: ["centrifugation", "drying oven"]
Output: "purified NiFe-LDH powder"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
핵심 규칙:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- **1행 = 1 Intent**. 여러 intent가 섞였으면 행을 나눔
- **판단 금지**. "가장 중요하다" "최적이다" 금지. 사실만.
- 합성 방법 라벨(hydrothermal, sol-gel)은 Intent가 아님
- 우리 DB의 기존 intent 태그를 참고하되, 최종은 반드시 10개 core 중 하나로 매핑

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
기존 31개 intent → 10개 core intent 매핑 참고:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  nucleation → Nucleation
  crystallization, crystal_growth → Crystallization
  precipitation → Nucleation 또는 Crystallization (맥락에 따라)
  dissolution, mixing → Stoichiometry Control
  doping → Doping
  etching → Morphology Control
  phase_transformation, annealing → Crystallization 또는 Redox Control
  phosphorization, sulfidation, nitridation → Doping
  purification, washing, centrifugation, separation → Purification
  drying → Drying
  deposition, electrodeposition → Catalyst-Electrode Coupling
  substrate_preparation → Catalyst-Electrode Coupling
  oxidation, reduction → Redox Control
  reference_synthesis, reagent_info → 제외 (합성 step이 아님)
  characterization → 제외 (분석 step)
  exfoliation, intercalation → Morphology Control

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
출력 형식 (반드시 JSON):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```json
[
  {{
    "step_id": "S01",
    "raw_step": "논문 원문 그대로",
    "chemical_intent": "10개 core 중 하나",
    "mechanism": "①원인 ②과정 ③결과 체인으로 서술",
    "tunable_conditions": {{"변수": "값"}},
    "required_capabilities": ["capability"],
    "output_form": "산물 형태"
  }}
]
```

한국어로 응답하되, Raw Step은 영어 그대로, Intent는 영어 이름 그대로.
반드시 JSON 형식으로 출력하세요. """

EXPERIMENTAL_PROMPT = """당신은 Experimental Agent입니다.
CID를 읽고 우리 실험실 장비로 실행 가능한지 판단합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실험실 장비 현황:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 사용 가능: 비커, hotplate, 자석교반기, 시린지 펌프, pH 미터,
   피펫, capping agent, 원심분리기, RDE, 전자저울, 건조 오븐(80-100°C)
❌ 사용 불가: autoclave, furnace, microwave oven, vacuum oven, glove box, CVD, tube furnace

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
핵심 작업:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **CID의 Required Capabilities를 하나씩 점검**
   - **반드시 check_capabilities_feasibility 도구를 먼저 호출하세요!**
   - 각 CID 행의 required_capabilities 리스트를 이 도구에 전달
   - 도구 결과에서 ❌가 나오면 그 행은 실험실에서 불가능
   - 예: ["sealed vessel", "heating up to 120°C"] → ❌ (autoclave 필요)
   - 예: ["stirring", "pH monitoring"] → ✅

2. **❌인 행에 대해 같은 Intent를 다른 Mechanism으로 대체**
   - Intent가 Crystallization이고 Mechanism이 "고온 고압"이면
     → 대체: "co-precipitation + 80-90°C aging (Ostwald ripening)"
   - 이때 get_process_transfer_evidence와 search_sentences로 
     문헌 근거를 반드시 검색하세요!

3. **대체 프로토콜 작성**
   - 모든 행이 ✅가 되도록 수정
   - 정확한 시약, 농도, 온도, 시간, pH 포함
   - 장비 제약 내에서 실제로 실행 가능한지 최종 확인

4. **성능 예측**
   - 대체 후 예상 overpotential
   - 리스크 및 주의사항

check_capabilities_feasibility(필수!), check_equipment_compatibility, get_process_transfer_evidence 도구를 반드시 사용하세요.
마지막에 "CID_COMPLETE" 라고 적으세요.
한국어로 응답하세요."""


# ============================================================
# Build Team
# ============================================================
def build_team(termination=None):
    extractor = AssistantAgent(
        name="Literature_Extractor",
        model_client=model_client,
        system_message=EXTRACTOR_PROMPT,
        tools=[tool_map[n] for n in [
            "search_sentences", "search_papers", "get_route_performance",
            "get_common_routes", "get_intent_impact",
        ] if n in tool_map],
    )

    cid_agent = AssistantAgent(
        name="CID_Agent",
        model_client=model_client,
        system_message=CID_PROMPT,
    )

    experimental = AssistantAgent(
        name="Experimental_Agent",
        model_client=model_client,
        system_message=EXPERIMENTAL_PROMPT,
        tools=[tool_map[n] for n in [
            "check_capabilities_feasibility",
            "check_equipment_compatibility", "get_lab_compatible_routes",
            "get_process_transfer_evidence", "search_sentences",
            "get_route_performance", "get_intent_impact",
        ] if n in tool_map],
    )

    if termination is None:
        termination = TextMentionTermination("CID_COMPLETE") | MaxMessageTermination(10)

    return RoundRobinGroupChat(
        participants=[extractor, cid_agent, experimental],
        termination_condition=termination,
    )


# ============================================================
# Parse CID output
# ============================================================
def parse_cid_json(text: str) -> list:
    """Extract CID JSON array from agent output."""
    # Try to find JSON in the text
    import re
    # Look for JSON array
    matches = re.findall(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Try to find individual JSON objects
    matches = re.findall(r'\{[^{}]*"step_id"[^{}]*\}', text, re.DOTALL)
    results = []
    for match in matches:
        try:
            results.append(json.loads(match))
        except json.JSONDecodeError:
            continue
    
    return results if results else []


# ============================================================
# Main
# ============================================================
async def run(target: str, save_output: bool = True):
    team = build_team()
    
    print(f"\n{'='*60}")
    print(f"  CID Agent — Chemical Intent Descriptor v3.2")
    print(f"  Target: {target}")
    print(f"  Flow: Literature_Extractor → CID_Agent → Experimental_Agent")
    print(f"{'='*60}\n")
    
    task = f"""{target}의 합성 절차를 CID로 번역해주세요.

단계:
1. Literature_Extractor: 논문에서 합성 절차 문장을 원문 그대로 추출
2. CID_Agent: 추출된 문장을 CID 6컬럼(JSON)으로 번역
3. Experimental_Agent: CID를 읽고 우리 실험실에서 실행 가능한지 판단 + 대체 프로토콜 작성"""

    # Collect all messages
    all_messages = []
    stream = team.run_stream(task=task)
    
    async for message in stream:
        msg_type = type(message).__name__
        
        if msg_type == "TextMessage":
            source = getattr(message, 'source', '')
            content = getattr(message, 'content', '')
            if source != 'user':
                all_messages.append({
                    "type": "text",
                    "source": source,
                    "content": content,
                })
                print(f"\n{'─'*50}")
                print(f"  {source}")
                print(f"{'─'*50}")
                print(content[:2000])
                
        elif msg_type == "ToolCallSummaryMessage":
            content = getattr(message, 'content', '')
            all_messages.append({"type": "tool_summary", "content": content})
            print(f"\n  [Tool Result] {content[:500]}")
    
    # Parse CID from CID_Agent output
    cid_rows = []
    for msg in all_messages:
        if msg.get("source") == "CID_Agent":
            parsed = parse_cid_json(msg.get("content", ""))
            if parsed:
                cid_rows = parsed
    
    # Save results
    result = {
        "target": target,
        "cid_rows": cid_rows,
        "messages": all_messages,
    }
    
    if save_output and cid_rows:
        out_dir = Path(__file__).parent.parent / "output"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"cid_{target.replace(' ', '_').lower()}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n\nCID 저장: {out_path}")
        print(f"CID 행 수: {len(cid_rows)}")
    elif save_output:
        out_dir = Path(__file__).parent.parent / "output"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"cid_{target.replace(' ', '_').lower()}_raw.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n\n결과 저장 (CID 파싱 실패, raw): {out_path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CID Agent")
    parser.add_argument("target", help="Target material (e.g., 'NiFe LDH')")
    parser.add_argument("--no-save", action="store_true", help="Don't save output")
    args = parser.parse_args()
    
    asyncio.run(run(args.target, save_output=not args.no_save))
