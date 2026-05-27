"""
Cross-Domain CID Agent
같은 Chemical Intent를 다른 도메인에서 달성한 Mechanism을 검색

Flow:
  1. Problem Analyzer    — 현재 합성의 문제를 Intent로 분해
  2. CrossDomain Searcher — 다른 재료/방법에서 같은 Intent의 Mechanism 검색
  3. Applicability Judge  — 실험실 제약 내에서 적용 가능한지 판단

Usage:
  python cross_domain_agent.py "NiFe LDH의 결정성이 낮다" 
"""
import os
import sys
import json
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data_tools import search_sentences, search_papers
from cid_agent import CORE_INTENTS, _load_api_key

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

MODEL = "gpt-4o-mini"
model_client = OpenAIChatCompletionClient(model=MODEL, api_key=_load_api_key())

# ============================================================
# Prompts
# ============================================================

PROBLEM_ANALYZER_PROMPT = """당신은 합성 문제 분석가입니다.
사용자가 설명하는 합성 문제를 CID 관점에서 분해합니다.

할 일:
1. 문제를 Chemical Intent 단위로 분해
2. 각 Intent에서 "무엇이 부족한지" 파악
3. 어떤 Intent의 Mechanism을 개선하면 문제가 해결되는지 식별
4. 개선이 필요한 Intent를 "search_keywords"로 정리
   - 예: Crystallization이 약하면 → "crystallization, template, aging, ostwald"
   - 예: Morphology가 안 좋으면 → "morphology, template, capping, surfactant"

출력:
- problem_intents: 개선이 필요한 Intent 목록
- search_keywords: 각 Intent에 대해 검색할 키워드
- goal: "이 Intent를 강화하기 위해 다른 도메인에서 어떤 Mechanism을 가져올 수 있는가?"

한국어로 응답하세요."""

CROSSDOMAIN_SEARCHER_PROMPT = f"""당신은 Cross-Domain 화학 지식 검색 전문가입니다.
한 도메인의 합성 문제를 다른 도메인에서 해결한 Mechanism을 찾습니다.

핵심 원칙: "같은 Chemical Intent를 달성하는 Mechanism은 재료와 무관하게 작동할 수 있다"

예시:
- MOF에서 "리간드 배위 → 열분해" Mechanism은 OER 촉매의 Morphology Control에 적용 가능
- Polymer에서 "soft template 자기조립"은 촉매의 중공 구조 형성에 적용 가능
- Zeolite에서 "hard template"는 메조포어 구조 제어에 적용 가능
- Colloid에서 "coagulation"은 복합 촉매의 균일 분산에 적용 가능

10개 Chemical Intent:
{chr(10).join(f"  {i+1}. {x}" for i, x in enumerate(CORE_INTENTS))}

할 일:
1. search_sentences로 **반드시 아래 키워드들을 검색**하세요:
   - "MOF", "ZIF", "metal-organic framework"
   - "template", "soft template", "hard template", "sacrificial"
   - "encapsulat", "core-shell", "yolk-shell"
   - "polymer", "block copolymer", "self-assembly"
   - "zeolite", "SBA", "mesoporous"
   - "citrate", "capping", "surfactant"
   - "urea hydrolysis", "epoxide"
   각 키워드를 개별적으로 search_sentences에 전달하세요! 한 번에 하나씩!
2. 찾은 문장에서 "다른 도메인의 Mechanism을 현재 문제에 적용"할 수 있는지 분석
3. 각 사례에 대해:
   - 원래 도메인 (어디서 왔는지)
   - 적용 도메인 (어디에 쓸 건지)
   - Mechanism (화학적 원리)
   - 필요한 장비/조건
   - 예상 효과

반드시 search_sentences를 여러 번 호출하세요!
한국어로 응답하세요."""

APPLICABILITY_PROMPT = """당신은 실험실 적용성 평가 전문가입니다.
Cross-Domain Mechanism이 우리 실험실에서 실제로 실행 가능한지 판단합니다.

사용 가능한 장비: 비커, hotplate, 자석교반기, 시린지 펌프, pH 미터,
피펫, capping agent, 원심분리기, RDE, 전자저울, 건조 오븐(80-100°C)
사용 불가: autoclave, furnace, microwave, vacuum oven, glove box, CVD

할 일:
1. 각 Cross-Domain 제안에 대해 ✅ 가능 / ❌ 불가 / ⚠️ 수정 필요 판정
2. 불가능한 제안은 같은 Intent를 달성하는 대체 방법 제안
3. 최종 추천: 가장 실효성 높은 Cross-Domain 적용 1~2개 선정
4. 실행 가능한 프로토콜 초안 작성

check_capabilities_feasibility 도구를 사용하세요.
마지막에 "CROSS_DOMAIN_COMPLETE" 라고 적으세요.
한국어로 응답하세요."""


# ============================================================
# Build Team
# ============================================================
from data_tools import TOOL_FUNCTIONS, check_capabilities_feasibility

tool_map = {fn.__name__: fn for fn in TOOL_FUNCTIONS}

def build_cross_domain_team():
    analyzer = AssistantAgent(
        name="Problem_Analyzer",
        model_client=model_client,
        system_message=PROBLEM_ANALYZER_PROMPT,
        tools=[tool_map["search_sentences"], tool_map["search_papers"]],
    )

    searcher = AssistantAgent(
        name="CrossDomain_Searcher",
        model_client=model_client,
        system_message=CROSSDOMAIN_SEARCHER_PROMPT,
        tools=[tool_map["search_sentences"], tool_map["search_papers"],
               tool_map["get_synthesis_network"], tool_map["get_intent_impact"]],
    )

    judge = AssistantAgent(
        name="Applicability_Judge",
        model_client=model_client,
        system_message=APPLICABILITY_PROMPT,
        tools=[tool_map["check_capabilities_feasibility"],
               tool_map["check_equipment_compatibility"],
               tool_map["search_sentences"]],
    )

    termination = TextMentionTermination("CROSS_DOMAIN_COMPLETE") | MaxMessageTermination(18)

    return RoundRobinGroupChat(
        participants=[analyzer, searcher, judge],
        termination_condition=termination,
    )


# ============================================================
# Main
# ============================================================
async def run(problem: str):
    print(f"\n{'='*60}")
    print(f"  Cross-Domain CID Agent")
    print(f"  Problem: {problem}")
    print(f"{'='*60}\n")

    # Step 1: Pre-search Cross-Domain keywords (code, not agent)
    cross_domain_keywords = [
        "MOF", "ZIF", "metal-organic framework",
        "template", "sacrificial template",
        "encapsulat", "core-shell",
        "polymer", "self-assembly",
        "citrate", "capping", "surfactant",
        "urea hydrolysis", "epoxide",
        "aging", "ostwald",
        "crystallization",
    ]

    print("  [Pre-search] Cross-Domain 키워드 검색 중...")
    search_context = ""
    for kw in cross_domain_keywords:
        result = search_sentences(kw, top_n=3)
        if "Found 0" not in result:
            search_context += f"\n--- Keywords: {kw} ---\n{result}\n"

    print(f"  [Pre-search] {len(search_context)} chars of context\n")

    # Step 2: Build team with context injected
    team = build_cross_domain_team()

    enriched_task = f"""{problem}

━━━ Cross-Domain 검색 결과 (905편 문헌에서 자동 검색) ━━━
{search_context}
━━━ 검색 결과 끝 ━━━

위 검색 결과에서 다른 도메인의 Mechanism을 찾아서 현재 문제에 적용하세요.
특히 MOF-derived, template, encapsulation, polymer 등의 Cross-Domain 사례에 주목하세요."""

    stream = team.run_stream(task=enriched_task)

    async for message in stream:
        msg_type = type(message).__name__

        if msg_type == "TextMessage":
            source = getattr(message, 'source', '')
            content = getattr(message, 'content', '')
            if source != 'user':
                print(f"\n{'─'*50}")
                print(f"  {source}")
                print(f"{'─'*50}")
                print(content[:3000])

        elif msg_type == "ToolCallSummaryMessage":
            content = getattr(message, 'content', '')
            print(f"\n  [Tool] {content[:500]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", help="Description of the synthesis problem")
    args = parser.parse_args()
    asyncio.run(run(args.problem))
