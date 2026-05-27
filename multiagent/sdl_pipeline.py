"""
CID SDL Pipeline — Orchestrator
조건부 Knowledge Agent 호출이 포함된 전체 SDL 파이프라인

Flow:
  Phase 1 (필수): Literature_Extractor → CID_Agent → Experimental_Agent
  Phase 2 (조건부): Knowledge_Agent ← 특정 조건에서만 호출
  Phase 3 (필수): Protocol_Writer → 최종 프로토콜

Knowledge Agent 트리거 조건:
  A. Experimental Agent가 ❌를 발견했는데 내부 대체가 불가능
  B. Literature 검색 결과가 5건 미만 (문헌 부족)
  C. CID의 Mechanism이 15자 미만 (품질 부족)
  D. 사용자가 명시적으로 "외부 검색" 요청

Usage:
  python sdl_pipeline.py "NiFe LDH"
  python sdl_pipeline.py "CsPbBr3 perovskite" --external
"""
import os
import sys
import json
import asyncio
import re
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from data_tools import search_sentences, TOOL_FUNCTIONS, check_capabilities_feasibility
from cid_agent import (
    build_team as build_cid_team,
    parse_cid_json,
    CORE_INTENTS,
    _load_api_key,
)

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

MODEL = "gpt-4o-mini"
model_client = OpenAIChatCompletionClient(model=MODEL, api_key=_load_api_key())

tool_map = {fn.__name__: fn for fn in TOOL_FUNCTIONS}


# ============================================================
# Phase 2: Knowledge Agent
# ============================================================
KNOWLEDGE_AGENT_PROMPT = f"""당신은 Cross-Domain Knowledge Agent입니다.
CID에서 해결하지 못한 문제를 외부 지식으로 보완합니다.

10개 Chemical Intent:
{chr(10).join(f"  {i+1}. {x}" for i, x in enumerate(CORE_INTENTS))}

할 일:
1. 전달받은 CID에서 ❌ 또는 약한 Mechanism을 식별
2. 문헌 검색(search_sentences)으로 관련 사례 찾기
3. **검색 결과가 없으면 LLM 화학 지식만으로 CID를 작성하세요!**
   - 일반적인 화학 원리를 바탕으로 Mechanism을 제안
   - 출처에 "LLM chemical knowledge"라고 표시
   - 신뢰도에 "LOW"라고 표시
4. 제안을 CID 6컬럼 형식으로 정리:
   - step_id, raw_step, chemical_intent, mechanism,
   - tunable_conditions, required_capabilities, output_form
5. 각 제안에 대해:
   - 출처 (문헌 / 일반 화학 지식 / 유추)
   - 신뢰도 (HIGH/MEDIUM/LOW)
   - 적용 가능성 근거

Mechanism 작성 기준:
- 라벨만 쓰지 않기 ("ostwald ripening" ✗)
- 논문 원문도 그대로 옮기지 않기
- ①원인 ②과정 ③결과 체인으로 작성

한국어로 응답하세요."""


PROTOCOL_WRITER_PROMPT = """당신은 최종 프로토콜 작성가입니다.
CID와 (있으면) Knowledge Agent의 제안을 종합하여 
실행 가능한 합성 프로토콜을 작성합니다.

할 일:
1. 모든 CID 행을 순서대로 나열
2. ❌인 행은 대체 Mechanism으로 교체
3. 각 step에 대해:
   - 시약 (정확한 양)
   - 장비 (우리 실험실 기준)
   - 조건 (온도, 시간, pH 등)
   - 주의사항
4. 최종 프로토콜 요약:
   - 전체 step 수
   - 대체된 step 수
   - 예상 성능 (근거와 함께)
   - 리스크/불확실성

마지막에 반드시 "PROTOCOL_FINAL" 이라고 적으세요.
한국어로 응답하세요."""


# ============================================================
# Trigger Conditions
# ============================================================
def should_trigger_knowledge(messages: list, cid_rows: list, user_external: bool = False) -> dict:
    """
    Knowledge Agent를 호출해야 하는지 판단.
    Returns: {"trigger": bool, "reason": str, "gaps": list}
    """
    if user_external:
        return {"trigger": True, "reason": "사용자 요청", "gaps": ["external_search_requested"]}

    gaps = []
    reasons = []

    # Condition A: ❌ in Experimental Agent with no internal alternative
    experimental_msgs = [m for m in messages if m.get("source") == "Experimental_Agent"]
    for msg in experimental_msgs:
        content = msg.get("content", "")
        # Check for ❌ without alternative
        if "❌" in content:
            # Check if alternative was proposed
            has_alt = any(kw in content.lower() for kw in [
                "대체", "alternative", "hotplate aging", "open beaker",
                "co-precipitation", "electrodeposition",
            ])
            if not has_alt:
                # Extract what failed
                failed = re.findall(r'❌\s*(.+?)(?:\n|$)', content)
                gaps.extend(failed)
                reasons.append(f"대체 불가능한 ❌: {', '.join(failed[:3])}")

    # Condition B: Insufficient literature (< 5 results)
    tool_msgs = [m for m in messages if m.get("type") == "tool_summary"]
    total_found = 0
    for msg in tool_msgs:
        content = msg.get("content", "")
        match = re.search(r"Found (\d+) sentences", content)
        if match:
            total_found += int(match.group(1))
    if total_found < 5:
        gaps.append("insufficient_literature")
        reasons.append(f"문헌 검색 결과 부족 ({total_found}건)")

    # Condition C: Weak Mechanism (< 15 chars)
    weak_steps = []
    for row in cid_rows:
        mech = row.get("mechanism", "")
        if len(mech) < 15:
            weak_steps.append(row.get("step_id", "?"))
    if weak_steps:
        gaps.append("weak_mechanism")
        reasons.append(f"Mechanism 품질 부족: step {weak_steps}")

    trigger = len(gaps) > 0
    reason = "; ".join(reasons) if reasons else "없음"

    return {"trigger": trigger, "reason": reason, "gaps": gaps}


# ============================================================
# Pipeline
# ============================================================
async def run_pipeline(target: str, external: bool = False):
    print(f"\n{'═'*60}")
    print(f"  SDL Pipeline — CID v3.2 + Conditional Knowledge Agent")
    print(f"  Target: {target}")
    print(f"  External search: {'ON' if external else 'AUTO'}")
    print(f"{'═'*60}\n")

    # ─── Phase 1: CID Agent (필수) ───────────────────────────
    print("┌─────────────────────────────────────────┐")
    print("│ Phase 1: CID Agent (Literature → CID → Exp) │")
    print("└─────────────────────────────────────────┘\n")

    # Phase 1 termination: also stop if NO_RESULTS_FOUND
    from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
    cid_termination = (
        TextMentionTermination("CID_COMPLETE") | 
        TextMentionTermination("NO_RESULTS_FOUND") |
        MaxMessageTermination(8)
    )
    cid_team = build_cid_team(termination=cid_termination)
    all_messages = []

    task = f"""{target}의 합성 절차를 CID로 번역해주세요.

단계:
1. Literature_Extractor: {target} 관련 합성 절차 문장을 원문 그대로 추출
   **중요**: {target}과 직접 관련된 문장만 찾으세요. 
   관련 문장이 없으면 "NO_RESULTS_FOUND: {target}에 대한 합성 문헌을 찾을 수 없습니다"라고 명시하세요.
   다른 물질의 합성법을 대신 검색하지 마세요!
2. CID_Agent: 추출된 문장을 CID 6컬럼(JSON)으로 번역
3. Experimental_Agent: CID를 읽고 우리 실험실에서 실행 가능한지 판단 + 대체 프로토콜 작성"""

    stream = cid_team.run_stream(task=task)
    async for message in stream:
        msg_type = type(message).__name__
        if msg_type == "TextMessage":
            source = getattr(message, 'source', '')
            content = getattr(message, 'content', '')
            if source != 'user':
                all_messages.append({
                    "type": "text", "source": source, "content": content
                })
                print(f"\n{'─'*50}")
                print(f"  {source}")
                print(f"{'─'*50}")
                print(content[:2000])
        elif msg_type == "ToolCallSummaryMessage":
            content = getattr(message, 'content', '')
            all_messages.append({"type": "tool_summary", "content": content})
            print(f"\n  [Tool] {content[:300]}")

    # Parse CID
    cid_rows = []
    for msg in all_messages:
        if msg.get("source") == "CID_Agent":
            parsed = parse_cid_json(msg.get("content", ""))
            if parsed:
                cid_rows = parsed

    print(f"\n\n  Phase 1 완료: CID {len(cid_rows)}행")

    # ─── Check Trigger ───────────────────────────────────────
    trigger = should_trigger_knowledge(all_messages, cid_rows, user_external=external)

    print(f"\n┌─────────────────────────────────────────┐")
    print(f"│ Knowledge Agent 트리거 확인               │")
    print(f"│ 호출 여부: {'YES' if trigger['trigger'] else 'NO':4s}                          │")
    print(f"│ 사유: {trigger['reason'][:35]:35s}│")
    print(f"└─────────────────────────────────────────┘\n")

    # ─── Phase 2: Knowledge Agent (조건부) ────────────────────
    knowledge_cid = []
    if trigger["trigger"]:
        print("┌─────────────────────────────────────────┐")
        print("│ Phase 2: Knowledge Agent (External)       │")
        print("└─────────────────────────────────────────┘\n")

        # Pre-search cross-domain keywords
        cross_kw = [
            "MOF", "template", "encapsulat", "core-shell",
            "polymer", "self-assembly", "citrate", "surfactant",
            "aging", "ostwald", "urea hydrolysis",
            "crystallization", "morphology",
        ]
        search_context = ""
        for kw in cross_kw:
            result = search_sentences(kw, top_n=2)
            if "Found 0" not in result:
                search_context += f"\n--- {kw} ---\n{result}\n"

        # Summarize gaps for Knowledge Agent
        gap_summary = ""
        if cid_rows:
            gap_summary += "\n현재 CID:\n"
            for row in cid_rows:
                gap_summary += (
                    f"  {row.get('step_id', '?')}: "
                    f"Intent={row.get('chemical_intent', '?')} | "
                    f"Mechanism={row.get('mechanism', '?')[:80]} | "
                    f"Capabilities={row.get('required_capabilities', '?')}\n"
                )
        else:
            gap_summary = f"""
⚠️ Phase 1에서 '{target}'에 대한 문헌을 찾지 못했습니다.
내부 DB(905편 OER 논문)에 해당 물질이 없습니다.

당신의 화학 지식으로 다음을 수행하세요:
1. {target}의 일반적인 합성 절차를 4~6단계로 작성
2. 각 단계를 CID 6컬럼으로 번역
3. 실험실 제약(autoclave/furnace/microwave 없음)을 고려하여
   필요하면 대체 Mechanism 제안
"""

        knowledge_agent = AssistantAgent(
            name="Knowledge_Agent",
            model_client=model_client,
            system_message=KNOWLEDGE_AGENT_PROMPT,
            tools=[tool_map["search_sentences"], tool_map["search_papers"]],
        )

        # Run knowledge agent directly
        from autogen_agentchat.messages import TextMessage

        knowledge_task = f"""다음 CID에서 해결되지 않은 문제를 외부 지식으로 보완해주세요.

트리거 사유: {trigger['reason']}

{gap_summary}

━━━ 문헌 검색 결과 ━━━
{search_context[:8000]}
━━━ 끝 ━━━

위 데이터와 당신의 화학 지식을 활용하여:
1. ❌인 step의 대체 Mechanism을 제안
2. 약한 Mechanism을 보강
3. 결과를 CID 6컬럼 JSON으로 출력

**매우 중요**: 문헌 검색 결과가 부족하더라도 반드시 CID JSON을 출력하세요!
당신의 일반적인 화학 지식으로 합성 절차를 추론할 수 있습니다.
JSON이 없으면 다음 단계로 넘어갈 수 없습니다.

출력 형식:
```json
[{{"step_id": "K01", "raw_step": "...", "chemical_intent": "...", ...}}]
```

KNOWLEDGE_DONE"""

        # Run knowledge agent directly (not RoundRobin — single agent doesn't work well)
        knowledge_messages = []

        from autogen_agentchat.messages import TextMessage as TMsg
        from autogen_core import CancellationToken

        cancel_token = CancellationToken()

        # Direct invoke — reliable for single agent
        response = await knowledge_agent.on_messages(
            [TMsg(content=knowledge_task, source="user")],
            cancellation_token=cancel_token,
        )

        # Collect response
        if hasattr(response, 'chat_message') and response.chat_message:
            content = getattr(response.chat_message, 'content', '')
            knowledge_messages.append({"source": "Knowledge_Agent", "content": content})
            print(f"\n{'─'*50}")
            print(f"  Knowledge_Agent")
            print(f"{'─'*50}")
            print(content[:3000])

        # Handle tool calls if any
        if hasattr(response, 'inner_messages') and response.inner_messages:
            for inner in response.inner_messages:
                inner_type = type(inner).__name__
                if inner_type == "ToolCallSummaryMessage":
                    ic = getattr(inner, 'content', '')
                    print(f"\n  [Tool] {ic[:300]}")

        # Parse knowledge CID
        for msg in knowledge_messages:
            parsed = parse_cid_json(msg.get("content", ""))
            if parsed:
                knowledge_cid = parsed

        print(f"\n  Phase 2 완료: Knowledge CID {len(knowledge_cid)}행")
    else:
        print("  Phase 2 스킵: Knowledge Agent 불필요\n")

    # ─── Phase 3: Protocol Writer (필수) ──────────────────────
    print("\n┌─────────────────────────────────────────┐")
    print("│ Phase 3: Protocol Writer                  │")
    print("└─────────────────────────────────────────┘\n")

    # Merge CID rows
    final_cid = cid_rows.copy()
    # Mark knowledge CID rows
    for row in knowledge_cid:
        row["source"] = "knowledge_agent"
        final_cid.append(row)

    # Build protocol context
    cid_context = json.dumps(final_cid, ensure_ascii=False, indent=2)

    writer = AssistantAgent(
        name="Protocol_Writer",
        model_client=model_client,
        system_message=PROTOCOL_WRITER_PROMPT,
    )

    writer_team = RoundRobinGroupChat(
        participants=[writer],
        termination_condition=TextMentionTermination("PROTOCOL_FINAL") | MaxMessageTermination(4),
    )

    protocol_task = f"""다음 CID를 바탕으로 최종 합성 프로토콜을 작성하세요.

Target: {target}

CID 데이터:
{cid_context[:6000]}

{'Knowledge Agent 제안이 포함되어 있습니다.' if knowledge_cid else '내부 문헌만으로 작성합니다.'}

우리 실험실 장비: 비커, hotplate, 교반기, 시린지 펌프, pH 미터, 피펫, 원심분리기, RDE
없음: autoclave, furnace, microwave, vacuum oven, glove box

프로토콜을 작성하고 PROTOCOL_FINAL 로 끝내세요."""

    protocol_stream = writer_team.run_stream(task=protocol_task)
    protocol_text = ""
    async for message in protocol_stream:
        msg_type = type(message).__name__
        if msg_type == "TextMessage":
            source = getattr(message, 'source', '')
            content = getattr(message, 'content', '')
            if source != 'user':
                protocol_text = content
                print(f"\n{'─'*50}")
                print(f"  {source}")
                print(f"{'─'*50}")
                print(content[:3000])

    # ─── Save ─────────────────────────────────────────────────
    result = {
        "target": target,
        "phase1_cid": cid_rows,
        "knowledge_triggered": trigger["trigger"],
        "knowledge_reason": trigger["reason"],
        "phase2_knowledge_cid": knowledge_cid,
        "final_cid": final_cid,
        "protocol": protocol_text,
    }

    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"sdl_{target.replace(' ', '_').lower()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n\n{'═'*60}")
    print(f"  완료!")
    print(f"  CID 행: {len(cid_rows)} + Knowledge: {len(knowledge_cid)}")
    print(f"  Knowledge Agent: {'호출됨' if trigger['trigger'] else '스킵됨'}")
    print(f"  저장: {out_path}")
    print(f"{'═'*60}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="Target material")
    parser.add_argument("--external", action="store_true", help="Force external search")
    args = parser.parse_args()
    asyncio.run(run_pipeline(args.target, external=args.external))
