"""
OER Catalyst Synthesis Protocol Translator
Multi-Agent System using AutoGen + OpenAI

핵심 기능: 문헌의 최적 공정(autoclave/furnace 포함)을 
           실험실 장비 제약 내에서 실현 가능한 공정으로 번역

Flow:
  1. Paper Finder     — 최신 문헌에서 타겟 물질의 최적 합성법 탐색
  2. Intent Decomposer — 논문 공정을 Chemical Intent 단위로 분해
  3. Gap Analyzer      — 실험실 장비 제약과의 갭 식별
  4. Substitution Finder — 불가능한 intent의 대체 수단 탐색 + 근거
  5. Protocol Reassembler — 대체 공정으로 재조립
  6. Reviewer          — 최종 타당성 검증

Usage:
  python run_translator.py "NiFe LDH, η < 200 mV"
  python run_translator.py  # interactive
"""
import os
import sys
import json
from pathlib import Path

# Load .env
for _ep in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
    if _ep.exists():
        for line in _ep.read_text().strip().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        break

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

sys.path.insert(0, str(Path(__file__).parent))
from data_tools import TOOL_FUNCTIONS, LAB_EQUIPMENT, UNAVAILABLE_INTENTS

# ============================================================
# Model
# ============================================================
MODEL = "gpt-4o-mini"

_api_key = os.environ.get("OPENAI_API_KEY", "")
if not _api_key or len(_api_key) < 50:
    for _ep in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
        if _ep.exists():
            for line in _ep.read_text().strip().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    _api_key = line.split("=", 1)[1].strip()
                    break

model_client = OpenAIChatCompletionClient(model=MODEL, api_key=_api_key)

# ============================================================
# Tools
# ============================================================
tool_map = {fn.__name__: fn for fn in TOOL_FUNCTIONS}

paper_finder_tools = [tool_map[n] for n in [
    "search_papers", "search_sentences", "get_route_performance",
    "get_common_routes", "get_intent_impact",
] if n in tool_map]

gap_tools = [tool_map[n] for n in [
    "check_equipment_compatibility", "get_lab_compatible_routes",
] if n in tool_map]

substitution_tools = [tool_map[n] for n in [
    "search_sentences", "search_papers", "get_process_transfer_evidence",
    "get_synthesis_network", "get_intent_impact",
] if n in tool_map]

review_tools = [tool_map[n] for n in [
    "get_process_transfer_evidence", "search_sentences", "get_nife_protocols",
    "get_route_performance", "get_intent_impact",
] if n in tool_map]


# ============================================================
# Agent Prompts
# ============================================================

PAPER_FINDER_PROMPT = """당신은 OER 촉매 문헌 탐색 전문가입니다.

중요: 성능이 가장 좋은 방법을 우선 찾으세요. autoclave, furnace 등의 
장비 제약은 신경 쓰지 마세요 — 나중에 다른 에이전트가 대체할 것입니다.

할 일:
1. 타겟 물질에 대해 search_papers, search_sentences로 문헌 검색
2. get_route_performance로 각 공정별 성능 비교
3. get_intent_impact로 어떤 공정이 성능에 가장 큰 영향을 주는지 파악
4. 가장 낮은 overpotential을 달성한 논문들의 공정을 상세히 보고

보고 형식:
- Top 3 공정 (성능 순)
- 각 공정의 상세 단계 (논문 ID 포함)
- 달성된 overpotential
- 핵심 공정 조건 (온도, 시간, 농도 등)

한국어로 응답하세요. 반드시 도구를 사용해 실제 데이터를 검색하세요."""

INTENT_DECOMPOSER_PROMPT = """당신은 합성 공정 분해 전문가입니다.
문헌에서 보고된 합성 절차를 Chemical Intent 단위로 분해합니다.

사용 가능한 Chemical Intent (31종):
nucleation, crystallization, co_precipitation, electrodeposition, hydrothermal,
sol_gel, aging, drying, calcination, annealing, phase_transformation,
doping, etching, purification, washing, centrifugation, substrate_preparation,
deposition, surface_modification, ligand_exchange, intercalation,
exfoliation, phosphorization, sulfidation, nitridation, oxidation,
reduction, carbon_coating, mixing, grinding, characterization

할 일:
1. Paper Finder가 찾은 공정을 단계별로 분해
2. 각 단계를 위 31개 intent 중 하나로 매핑
3. 각 intent의 구체적 조건 기록:
   - 온도, 시간, 압력
   - 시약과 농도
   - 필요 장비
   - 화학적 목적 (이 단계가 왜 필요한가?)

출력 형식:
  Step 1: [intent_name] — 장비: xxx, 조건: xxx, 목적: xxx
  Step 2: [intent_name] — 장비: yyy, 조건: yyy, 목적: yyy
  ...

한국어로 응답하세요."""

GAP_ANALYZER_PROMPT = f"""당신은 실험실 장비 제약 분석가입니다.

사용 가능한 장비:
{json.dumps(LAB_EQUIPMENT["available"], ensure_ascii=False)}

사용 불가능한 장비:
{json.dumps(LAB_EQUIPMENT["unavailable"], ensure_ascii=False)}

할 일:
1. Intent Decomposer가 분해한 각 단계를 검사
2. check_equipment_compatibility 도구를 사용하여 검증
3. 각 단계를 다음으로 분류:
   ✅ 실행 가능 — 그대로 사용
   ❌ 실행 불가 — 장비 부족 (어떤 장비가 필요한지 명시)
   ⚠️ 수정 필요 — 장비는 있으나 조건 수정 필요

4. ❌ 표시된 단계에 대해 필요한 장비와 그 이유를 명확히 설명

이것이 번역의 핵심입니다 — 정확하게 분석하세요.
반드시 도구를 사용하세요. 한국어로 응답하세요."""

SUBSTITUTION_FINDER_PROMPT = """당신은 공정 대체 전문가입니다.
불가능한 합성 단계를 실험실에서 실행 가능한 방법으로 대체합니다.

할 일:
1. Gap Analyzer가 ❌ 표시한 각 단계에 대해 대체 방법 탐색
2. 도구를 사용하여 문헌에서 근거 검색:
   - search_sentences: 대체 방법 검색
   - get_process_transfer_evidence: 공정 전환 근거
   - get_synthesis_network: 다른 경로 탐색
   - get_intent_impact: 대체 공정의 성능 영향

3. 각 대체에 대해 보고:
   원래: [intent] (장비 X 필요) → 목적: ...
   대체: [대체 intent] (장비 Y 사용) → 근거: [논문 ID / 데이터]
   신뢰도: HIGH / MEDIUM / LOW
   예상 성능 변화: Δη = ... mV

핵심 원칙:
- 같은 화학적 목적(pH 변화, 결정화, 상변화 등)을 달성하는 다른 경로 찾기
- 예: hydrothermal(crystallization) → urea hydrolysis(같은 pH 변화, 90°C)
- 예: annealing(furnace, 300°C) → aging(hotplate, 80°C, 24h)
- 반드시 문헌 근거를 도구로 검색하여 제시

한국어로 응답하세요. 반드시 도구를 사용하세요."""

REASSEMBLER_PROMPT = """당신은 합성 프로토콜 재조립 전문가입니다.
대체된 공정 단계들을 하나의 완전한 프로토콜로 재조립합니다.

사용 가능한 장비: 비커, hotplate, 자석교반기, 시린지 펌프, pH 미터,
피펫, capping agent, 원심분리기, RDE, 전자저울, 건조 오븐

할 일:
1. 원래 공정의 각 단계 + 대체 공정을 하나의 흐름으로 결합
2. 각 단계 간 자연스러운 전환 확인
3. 상세 프로토콜 작성:
   - 정확한 시약량 (mmol, mL, 농도)
   - 온도, 시간, 교반 속도
   - pH 목표값
   - 장비 설정
   - 관찰 포인트 (색 변화, 침전 형성 등)

4. 원래 공정과 비교:
   - 어떤 단계가 대체되었는지
   - 예상 성능 차이
   - 주의사항

프로토콜 형식:
[준비] → [합성] → [세척/분리] → [건조] → [특성분석] → [전기화학 평가]

한국어로 응답하세요."""

REVIEWER_PROMPT = """당신은 최종 검토자입니다.
재조립된 프로토콜의 과학적 타당성을 최종 검증합니다.

할 일:
1. 도구를 사용하여 최종 확인:
   - get_process_transfer_evidence: 대체 근거 재확인
   - search_sentences: 추가 증거 검색
   - get_route_performance: 예상 성능과 문헌 비교
   - get_intent_impact: 각 단계의 성능 영향 확인

2. 최종 평가:
   - 과학적 타당성: HIGH / MEDIUM / LOW
   - 예상 overpotential: ... ± ... mV
   - 주요 리스크: ...
   - 권장 사항: ...

3. 전체 요약:
   원래 공정: ... (η = ... mV)
   번역된 공정: ... 
   대체된 단계: ...개
   신뢰도: ...

매우 중요: 검토가 끝나면 반드시 마지막 줄에 "PROTOCOL_TRANSLATED" 라고 적으세요.
이 단어가 없으면 시스템이 멈추지 않습니다.

한국어로 응답하세요. 반드시 도구를 사용하세요."""


# ============================================================
# Build Team
# ============================================================

def build_team():
    paper_finder = AssistantAgent(
        name="Paper_Finder",
        model_client=model_client,
        system_message=PAPER_FINDER_PROMPT,
        tools=paper_finder_tools,
    )

    intent_decomposer = AssistantAgent(
        name="Intent_Decomposer",
        model_client=model_client,
        system_message=INTENT_DECOMPOSER_PROMPT,
    )

    gap_analyzer = AssistantAgent(
        name="Gap_Analyzer",
        model_client=model_client,
        system_message=GAP_ANALYZER_PROMPT,
        tools=gap_tools,
    )

    substitution_finder = AssistantAgent(
        name="Substitution_Finder",
        model_client=model_client,
        system_message=SUBSTITUTION_FINDER_PROMPT,
        tools=substitution_tools,
    )

    reassembler = AssistantAgent(
        name="Reassembler",
        model_client=model_client,
        system_message=REASSEMBLER_PROMPT,
    )

    reviewer = AssistantAgent(
        name="Reviewer",
        model_client=model_client,
        system_message=REVIEWER_PROMPT,
        tools=review_tools,
    )

    termination = TextMentionTermination("PROTOCOL_TRANSLATED") | MaxMessageTermination(14)

    team = RoundRobinGroupChat(
        participants=[
            paper_finder,        # 1. 최적 문헌 탐색 (장비 무관)
            intent_decomposer,   # 2. Intent 단위 분해
            gap_analyzer,        # 3. 장비 갭 식별
            substitution_finder, # 4. 대체 수단 + 근거 탐색
            reassembler,         # 5. 프로토콜 재조립
            reviewer,            # 6. 최종 검증
        ],
        termination_condition=termination,
    )

    return team


# ============================================================
# Main
# ============================================================

async def run(task: str):
    team = build_team()

    print(f"\n{'='*60}")
    print(f"  OER Catalyst Protocol TRANSLATOR")
    print(f"  '불가능한 공정 → 가능한 공정' 번역 시스템")
    print(f"  Model: {MODEL}")
    print(f"  Flow:")
    print(f"    Paper_Finder → Intent_Decomposer → Gap_Analyzer")
    print(f"    → Substitution_Finder → Reassembler → Reviewer")
    print(f"{'='*60}\n")

    await Console(team.run_stream(task=task))


if __name__ == "__main__":
    import asyncio

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        print("OER Catalyst Protocol Translator")
        print("-" * 40)
        task = input("목표를 입력하세요: ").strip()
        if not task:
            task = "NiFe LDH 촉매를 합성하고 싶습니다. 문헌에서 가장 좋은 성능을 보고한 방법을 찾고, 우리 실험실에 없는 장비(autoclave, furnace, microwave)가 필요한 단계를 대체 가능한 방법으로 번역해주세요."

    asyncio.run(run(task))
