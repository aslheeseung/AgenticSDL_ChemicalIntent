"""
OER Catalyst Synthesis Protocol Generator
Multi-Agent System using AutoGen + OpenAI GPT-4o

Agents (execution order enforced):
  1. Literature_Analyst  — searches 905 papers, extracts relevant data
  2. Synthesis_Planner   — designs synthesis routes
  3. Lab_Checker         — validates equipment compatibility
  4. Evidence_Reviewer   — verifies scientific validity
  5. Protocol_Writer     — writes the final detailed protocol
  6. Orchestrator        — coordinates and summarizes

Usage:
  python run.py "NiFe LDH on Ni foam, target η < 260 mV"
  python run.py  # interactive mode
"""
import os
import sys
import json
from pathlib import Path

# Load .env — read directly to avoid redaction issues
_env_paths = [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
]
for _ep in _env_paths:
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
from data_tools import (
    TOOL_FUNCTIONS,
    TOOL_SCHEMAS,
    LAB_EQUIPMENT,
    UNAVAILABLE_INTENTS,
)

# ============================================================
# Model Client
# ============================================================
MODEL = "gpt-4o-mini"

_api_key = os.environ.get("OPENAI_API_KEY", "")
if not _api_key or len(_api_key) < 50:
    for _ep in _env_paths:
        if _ep.exists():
            for line in _ep.read_text().strip().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    _api_key = line.split("=", 1)[1].strip()
                    break

model_client = OpenAIChatCompletionClient(
    model=MODEL,
    api_key=_api_key,
)

# ============================================================
# Tool mapping
# ============================================================
tool_map = {fn.__name__: fn for fn in TOOL_FUNCTIONS}

analyst_tools = [tool_map[n] for n in [
    "search_sentences", "search_papers", "get_synthesis_network",
    "get_common_routes", "get_route_performance", "get_intent_impact",
] if n in tool_map]

checker_tools = [tool_map[n] for n in [
    "check_equipment_compatibility", "get_lab_compatible_routes",
] if n in tool_map]

evidence_tools = [tool_map[n] for n in [
    "get_process_transfer_evidence", "search_sentences", "get_nife_protocols",
] if n in tool_map]


# ============================================================
# Agent System Prompts
# ============================================================

ORCHESTRATOR_PROMPT = """당신은 OER 촉매 합성 설계 팀의 지휘자(Orchestrator)입니다.

사용자의 요청을 받아 아래 5명의 전문 에이전트가 순서대로 일하도록 조정합니다.
모든 에이전트의 결과를 종합하여 최종 결론을 내립니다.

작업 순서:
1. Literature_Analyst: 문헌 검색으로 관련 합성법과 성능 데이터 수집
2. Synthesis_Planner: 수집된 데이터로 합성 루트 설계
3. Lab_Checker: 설계된 루트의 장비 호환성 검증
4. Evidence_Reviewer: 공정 대체의 과학적 타당성 검토
5. Protocol_Writer: 최종 프로토콜 작성

마지막에 모든 결과를 요약하고 "PROTOCOL_COMPLETE"라고 말하세요.
한국어로 응답하세요."""

ANALYST_PROMPT = """당신은 OER 촉매 문헌 분석 전문가입니다.
905편의 연구 논문과 14,331개의 추출 문장 데이터베이스에 접근할 수 있습니다.

사용 가능한 도구:
- search_sentences: 키워드로 문장 검색
- search_papers: 논문 검색
- get_synthesis_network: 합성 네트워크 조회
- get_common_routes: 일반적인 합성 루트
- get_route_performance: 루트별 OER 성능 데이터
- get_intent_impact: 각 공정 단계가 성능에 미치는 영향

할 일:
1. 대상 물질에 대한 문헌을 검색하세요
2. 어떤 합성법이 존재하는지 보고하세요
3. 각 방법의 성능(overpotential)을 비교하세요
4. 가장 효과적이고 일반적인 방법을 식별하세요

반드시 논문 ID와 수치 데이터를 인용하세요.
한국어로 응답하세요."""

PLANNER_PROMPT = """당신은 OER 촉매 합성 루트 설계자입니다.
문헌 분석 결과를 바탕으로 합성 공정 시퀀스를 설계합니다.

사용 가능한 chemical intent (31종):
nucleation, crystallization, co_precipitation, electrodeposition, hydrothermal,
sol_gel, aging, drying, calcination, annealing, phase_transformation,
doping, etching, purification, washing, centrifugation, substrate_preparation,
deposition, surface_modification, ligand_exchange, intercalation,
exfoliation, phosphorization, sulfidation, nitridation, oxidation,
reduction, carbon_coating, mixing, grinding, characterization

할 일:
1. 문헌 데이터를 기반으로 2-3개의 대체 합성 루트를 제안하세요
2. 각 루트를 Step1 → Step2 → Step3 형식으로 표현하세요
3. 장단점을 비교하세요
4. 실험실에 autoclave, furnace, microwave가 없다는 점을 고려하세요

한국어로 응단세요."""

CHECKER_PROMPT = f"""당신은 실험실 장비 제약 검증자입니다.

사용 가능한 장비:
{json.dumps(LAB_EQUIPMENT["available"], ensure_ascii=False, indent=2)}

사용 불가능한 장비:
{json.dumps(LAB_EQUIPMENT["unavailable"], ensure_ascii=False, indent=2)}

사용 가능한 도구:
- check_equipment_compatibility: intent 목록의 장비 호환성 검사
- get_lab_compatible_routes: 실험실 가능 루트 조회

할 일:
1. 제안된 모든 합성 단계를 장비와 대조하여 검증하세요
2. 불가능한 단계를 표시하고 대안을 제시하세요
3. 모든 대체 공정이 실험실에서 실행 가능한지 확인하세요

일반적인 대체:
- hydrothermal (autoclave) → co-precipitation + 80-90°C aging
- annealing/calcination (furnace) → extended aging on hotplate
- phase_transformation (furnace) → electrodeposition 중 in-situ 변환

반드시 도구를 사용하여 검증하세요. 한국어로 응답하세요."""

REVIEWER_PROMPT = """당신은 과학적 근거 검토자입니다.
공정 대체가 과학적으로 타당한지 검증합니다.

사용 가능한 도구:
- get_process_transfer_evidence: 공정 대체 문헌 근거
- search_sentences: 구체적 증거 구절 검색
- get_nife_protocols: NiFe LDH 참조 프로토콜

할 일:
1. 대체된 각 공정에 대해 문헌 근거를 확인하세요
2. 구조적 동등성(XRD, SEM)을 확인하세요
3. 성능 동등성(overpotential)을 확인하세요
4. 신뢰도를 평가하세요: HIGH / MEDIUM / LOW
5. 근거가 부족하면 명확히 표시하세요

우리 데이터베이스의 핵심 근거:
- 2편의 논문이 hydrothermal vs co-precipitation을 직접 비교
- Co-precipitated NiFe LDH: η = 100-330 mV (평균 248 mV, n=11)
- 36개 구절이 aging-driven crystallization을 지지

반드시 도구를 사용하여 근거를 조회하세요. 한국어로 응답하세요."""

WRITER_PROMPT = """당신은 합성 프로토콜 작성자입니다.
검증된 합성 루트를 바탕으로 상세한 프로토콜을 작성합니다.

사용 가능한 장비: 비커, hotplate, 자석교반기, 시린지 펌프, pH 미터,
피펫, capping agent, 원심분리기, RDE, 전자저울, 건조 오븐

할 일:
1. 검증된 합성 루트를 받아 상세 프로토콜을 작성하세요
2. 포함해야 할 내용:
   - 정확한 시약량 (mmol, mL, 농도)
   - 각 단계별 온도와 시간
   - 사용 장비
   - pH 목표값
   - 세척/건조 조건
   - 특성분석 방법
3. 안전 주의사항을 포함하세요
4. 예상 결과를 포함하세요 (XRD 피크, SEM 형태, 예상 η)

프로토콜 형식:
[준비] → [합성] → [세척/분리] → [건조] → [특성분석] → [전기화학 평가]

프로토콜 작성이 완료되면 마지막에 "PROTOCOL_COMPLETE"라고 말하세요.
한국어로 응답하세요."""


# ============================================================
# Build Team (RoundRobin for ordered execution)
# ============================================================

def build_team():
    """Build the multi-agent team with RoundRobin execution order."""
    
    analyst = AssistantAgent(
        name="Literature_Analyst",
        model_client=model_client,
        system_message=ANALYST_PROMPT,
        tools=analyst_tools,
    )

    planner = AssistantAgent(
        name="Synthesis_Planner",
        model_client=model_client,
        system_message=PLANNER_PROMPT,
    )

    checker = AssistantAgent(
        name="Lab_Checker",
        model_client=model_client,
        system_message=CHECKER_PROMPT,
        tools=checker_tools,
    )

    reviewer = AssistantAgent(
        name="Evidence_Reviewer",
        model_client=model_client,
        system_message=REVIEWER_PROMPT,
        tools=evidence_tools,
    )

    writer = AssistantAgent(
        name="Protocol_Writer",
        model_client=model_client,
        system_message=WRITER_PROMPT,
    )

    orchestrator = AssistantAgent(
        name="Orchestrator",
        model_client=model_client,
        system_message=ORCHESTRATOR_PROMPT,
    )

    termination = TextMentionTermination("PROTOCOL_COMPLETE") | MaxMessageTermination(25)

    # RoundRobin: agents speak in order, then cycle
    team = RoundRobinGroupChat(
        participants=[
            analyst,     # 1st: search literature
            planner,     # 2nd: design route
            checker,     # 3rd: validate equipment
            reviewer,    # 4th: verify evidence
            writer,      # 5th: write protocol
            orchestrator, # 6th: summarize & close
        ],
        termination_condition=termination,
    )

    return team


# ============================================================
# Main
# ============================================================

async def run(task: str):
    """Run the multi-agent team on a task."""
    team = build_team()

    print(f"\n{'='*60}")
    print(f"  OER Catalyst Synthesis Protocol Generator")
    print(f"  Model: {MODEL}")
    print(f"  Team: Literature_Analyst → Synthesis_Planner → Lab_Checker")
    print(f"        → Evidence_Reviewer → Protocol_Writer → Orchestrator")
    print(f"{'='*60}\n")

    await Console(team.run_stream(task=task))


if __name__ == "__main__":
    import asyncio

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        print("OER Catalyst Protocol Generator")
        print("-" * 40)
        task = input("목표를 입력하세요 (예: 'NiFe LDH on Ni foam, η < 260 mV'): ").strip()
        if not task:
            task = "NiFe LDH 촉매를 Ni foam 위에 합성하고 싶습니다. 목표 overpotential은 260 mV 이하입니다. 실험실에 autoclave와 furnace가 없으므로 co-precipitation이나 electrodeposition 기반 공정으로 설계해주세요."

    asyncio.run(run(task))
