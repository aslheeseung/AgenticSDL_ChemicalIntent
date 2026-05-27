"""
OER Catalyst Protocol Translator — Web UI
Gradio 기반 실시간 에이전트 대화 시각화
"""
import os
import sys
import json
import asyncio
from pathlib import Path

def _load_api_key():
    """Load OpenAI API key from .env file."""
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

# Load .env
for _ep in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
    if _ep.exists():
        for line in _ep.read_text().strip().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        break

import gradio as gr
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

sys.path.insert(0, str(Path(__file__).parent))
from data_tools import TOOL_FUNCTIONS, LAB_EQUIPMENT, UNAVAILABLE_INTENTS

# ============================================================
# Model
# ============================================================
MODEL = "gpt-4o-mini"
_api_key = _load_api_key()

model_client = OpenAIChatCompletionClient(model=MODEL, api_key=_api_key)

# ============================================================
# Agent colors & icons
# ============================================================
AGENT_STYLES = {
    "Paper_Finder": {"icon": "🔍", "color": "#4A90D9", "role": "문헌 탐색"},
    "Intent_Decomposer": {"icon": "⚗️", "color": "#7B68EE", "role": "공정 분해"},
    "Gap_Analyzer": {"icon": "🔧", "color": "#E67E22", "role": "장비 검증"},
    "Substitution_Finder": {"icon": "🔄", "color": "#2ECC71", "role": "대체 탐색"},
    "Reassembler": {"icon": "📝", "color": "#9B59B6", "role": "프로토콜 작성"},
    "Reviewer": {"icon": "✅", "color": "#E74C3C", "role": "최종 검증"},
    # CID Agent 에이전트 추가
    "Literature_Extractor": {"icon": "📚", "color": "#3498DB", "role": "문헌 추출"},
    "CID_Agent": {"icon": "🔬", "color": "#8E44AD", "role": "CID 번역"},
    "Experimental_Agent": {"icon": "🔧", "color": "#E67E22", "role": "실험 판단"},
}

TOOL_STYLES = {"icon": "🛠️", "color": "#95A5A6"}

# ============================================================
# Prompts (same as run_translator.py)
# ============================================================
PAPER_FINDER_PROMPT = """당신은 OER 촉매 문헌 탐색 전문가입니다.
성능이 가장 좋은 방법을 우선 찾으세요. autoclave, furnace 등 장비 제약은 신경 쓰지 마세요.

할 일:
1. 타겟 물질에 대해 search_papers, search_sentences로 문헌 검색
2. get_route_performance로 각 공정별 성능 비교
3. get_intent_impact로 어떤 공정이 성능에 가장 큰 영향을 주는지 파악
4. 가장 낮은 overpotential을 달성한 논문들의 공정을 상세히 보고

반드시 도구를 사용해 실제 데이터를 검색하세요. 한국어로 응답하세요."""

INTENT_DECOMPOSER_PROMPT = """당신은 합성 공정 분해 전문가입니다.
문헌에서 보고된 합성 절차를 Chemical Intent 단위로 분해합니다.

사용 가능한 intent: nucleation, crystallization, co_precipitation, electrodeposition, hydrothermal, sol_gel, aging, drying, calcination, annealing, phase_transformation, doping, etching, purification, washing, centrifugation, substrate_preparation, deposition, surface_modification, characterization

각 단계를 intent로 매핑하고: 온도, 시간, 시약, 필요 장비, 화학적 목적을 기록하세요.
한국어로 응답하세요."""

GAP_ANALYZER_PROMPT = f"""당신은 실험실 장비 제약 분석가입니다.

사용 가능: {json.dumps(LAB_EQUIPMENT["available"], ensure_ascii=False)}
사용 불가: {json.dumps(LAB_EQUIPMENT["unavailable"], ensure_ascii=False)}

각 단계를 분류: ✅ 실행 가능 / ❌ 실행 불가 (장비 부족) / ⚠️ 수정 필요
check_equipment_compatibility 도구를 반드시 사용하세요. 한국어로 응답하세요."""

SUBSTITUTION_FINDER_PROMPT = """당신은 공정 대체 전문가입니다.
불가능한 합성 단계를 실험실에서 실행 가능한 방법으로 대체합니다.

도구를 사용하여 문헌에서 근거를 검색하고, 각 대체에 대해 원래 공정, 대체 공정, 근거, 신뢴도를 보고하세요.

핵심: 같은 화학적 목적을 달성하는 다른 경로 찾기
- hydrothermal → urea hydrolysis / co-precipitation + aging
- annealing → extended aging on hotplate
반드시 도구를 사용하세요. 한국어로 응답하세요."""

REASSEMBLER_PROMPT = """당신은 합성 프로토콜 재조립 전문가입니다.
대체된 공정 단계들을 하나의 완전한 프로토콜로 재조립합니다.

사용 가능 장비: 비커, hotplate, 자석교반기, 시린지 펌프, pH 미터, 피펫, capping agent, 원심분리기, RDE, 전자저울, 건조 오븐

정확한 시약량, 온도, 시간, pH 목표값을 포함하세요.
한국어로 응답하세요."""

REVIEWER_PROMPT = """당신은 최종 검토자입니다.
과학적 타당성을 최종 검증합니다. 도구를 사용하여 근거를 확인하세요.

최종 평가: 과학적 타당성, 예상 overpotential, 주요 리스크, 권장 사항
전체 요약 후 마지막 줄에 반드시 "PROTOCOL_TRANSLATED" 라고 적으세요.
한국어로 응답하세요."""

# ============================================================
# Build Team
# ============================================================
tool_map = {fn.__name__: fn for fn in TOOL_FUNCTIONS}

def build_team():
    agents = []
    configs = [
        ("Paper_Finder", PAPER_FINDER_PROMPT, 
         [tool_map[n] for n in ["search_papers", "search_sentences", "get_route_performance", "get_common_routes", "get_intent_impact"] if n in tool_map]),
        ("Intent_Decomposer", INTENT_DECOMPOSER_PROMPT, []),
        ("Gap_Analyzer", GAP_ANALYZER_PROMPT,
         [tool_map[n] for n in ["check_equipment_compatibility", "get_lab_compatible_routes"] if n in tool_map]),
        ("Substitution_Finder", SUBSTITUTION_FINDER_PROMPT,
         [tool_map[n] for n in ["search_sentences", "search_papers", "get_process_transfer_evidence", "get_synthesis_network", "get_intent_impact"] if n in tool_map]),
        ("Reassembler", REASSEMBLER_PROMPT, []),
        ("Reviewer", REVIEWER_PROMPT,
         [tool_map[n] for n in ["get_process_transfer_evidence", "search_sentences", "get_nife_protocols", "get_route_performance", "get_intent_impact"] if n in tool_map]),
    ]
    
    for name, prompt, tools in configs:
        agents.append(AssistantAgent(
            name=name,
            model_client=model_client,
            system_message=prompt,
            tools=tools if tools else [],
        ))
    
    termination = TextMentionTermination("PROTOCOL_TRANSLATED") | MaxMessageTermination(14)
    
    return RoundRobinGroupChat(
        participants=agents,
        termination_condition=termination,
    )

# ============================================================
# Format message for display
# ============================================================
def format_agent_msg(name, content):
    style = AGENT_STYLES.get(name, {"icon": "🤖", "color": "#666", "role": name})
    return f"""
<div style="border-left: 4px solid {style['color']}; padding: 12px 16px; margin: 8px 0; 
            background: {style['color']}11; border-radius: 8px;">
  <div style="font-weight: bold; color: {style['color']}; margin-bottom: 6px;">
    {style['icon']} {name} <span style="font-weight: normal; font-size: 0.85em; opacity: 0.7;">
    ({style['role']})</span>
  </div>
  <div style="white-space: pre-wrap; line-height: 1.6; font-size: 0.95em;">
    {content}
  </div>
</div>"""

def format_tool_call(tool_name, args, result=""):
    args_str = json.dumps(args, ensure_ascii=False, indent=2) if isinstance(args, dict) else str(args)
    result_preview = result[:300] + "..." if len(result) > 300 else result
    return f"""
<div style="border-left: 4px solid #95A5A6; padding: 8px 12px; margin: 4px 0 4px 20px; 
            background: #f8f9fa; border-radius: 6px; font-size: 0.9em;">
  <div style="color: #7f8c8d; font-weight: bold;">🛠️ Tool: {tool_name}</div>
  <div style="color: #555; margin-top: 4px;"><b>Args:</b> {args_str[:200]}</div>
  {f'<div style="color: #27ae60; margin-top: 4px;"><b>Result:</b> {result_preview}</div>' if result else ''}
</div>"""

def format_user_msg(content):
    return f"""
<div style="border-left: 4px solid #3498DB; padding: 12px 16px; margin: 8px 0; 
            background: #EBF5FB; border-radius: 8px;">
  <div style="font-weight: bold; color: #2980B9; margin-bottom: 6px;">👤 사용자</div>
  <div style="white-space: pre-wrap;">{content}</div>
</div>"""

# ============================================================
# Run pipeline
# ============================================================
async def run_pipeline(task, progress_callback):
    team = build_team()
    conversation_html = format_user_msg(task)
    progress_callback(conversation_html, "에이전트 대화 시작 중...")
    
    agent_msgs = []
    
    try:
        stream = team.run_stream(task=task)
        
        current_tool_calls = []
        current_agent = None
        
        async for message in stream:
            msg_type = type(message).__name__
            
            # Agent text message
            if msg_type == "TextMessage":
                source = getattr(message, 'source', 'unknown')
                content = getattr(message, 'content', '')
                if source != "user":
                    agent_msgs.append({"type": "text", "source": source, "content": content})
                    conversation_html += format_agent_msg(source, content)
                    style = AGENT_STYLES.get(source, {"icon": "🤖", "role": source})
                    progress_callback(conversation_html, f"{style['icon']} {source} ({style['role']}) 응답 중...")
                    
            # Tool call request
            elif msg_type == "ToolCallRequestEvent":
                calls = getattr(message, 'content', [])
                source = getattr(message, 'source', 'unknown')
                if not isinstance(calls, list):
                    calls = [calls]
                for call in calls:
                    tool_name = getattr(call, 'name', str(call))
                    args = getattr(call, 'arguments', {})
                    agent_msgs.append({"type": "tool_call", "source": source, "tool": tool_name, "args": args})
                    conversation_html += format_tool_call(tool_name, args)
                    progress_callback(conversation_html, f"🛠️ {tool_name} 실행 중...")
                            
            # Tool execution result
            elif msg_type == "ToolCallExecutionEvent":
                results = getattr(message, 'content', [])
                if not isinstance(results, list):
                    results = [results]
                for r in results:
                    tool_name = getattr(r, 'name', '')
                    content = getattr(r, 'content', '')
                    agent_msgs.append({"type": "tool_result", "tool": tool_name, "content": content})
                    conversation_html += format_tool_call(tool_name, {}, content)
                    
            # Tool call summary
            elif msg_type == "ToolCallSummaryMessage":
                pass  # skip, we already showed individual calls
                
    except Exception as e:
        conversation_html += f"""
<div style="border-left: 4px solid #E74C3C; padding: 12px; margin: 8px 0; background: #FDEDEC; border-radius: 8px;">
  <b>❌ 오류:</b> {str(e)}
</div>"""
    
    # Final summary
    conversation_html += """
<div style="border: 2px solid #27AE60; padding: 16px; margin: 12px 0; background: #EAFAF1; border-radius: 10px;">
  <h3 style="color: #27AE60; margin: 0 0 8px 0;">🎯 프로토콜 번역 완료</h3>
  <p>모든 에이전트의 협업이 완료되었습니다.</p>
</div>"""
    
    progress_callback(conversation_html, "완료!")
    return conversation_html

# ============================================================
# Gradio UI
# ============================================================
EXAMPLES = [
    "NiFe LDH 촉매를 합성하고 싶습니다. 문헌에서 가장 좋은 성능을 보고한 방법을 찾고, autoclave/furnace가 필요한 단계를 대체 가능한 방법으로 번역해주세요. 목표 η < 260 mV.",
    "CoFe LDH 촉매를 Ni foam 위에 electrodeposition으로 합성하는 프로토콜을 설계해주세요.",
    "NiFe LDH의 overpotential을 200 mV 이하로 낮추고 싶습니다. doping이나 etching이 효과적인지 문헌에서 확인해주세요.",
]

def run_and_display(task, progress=gr.Progress()):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    result_html = [""]  # mutable container
    
    def update(html, status):
        result_html[0] = html
    
    result = loop.run_until_complete(run_pipeline(task, update))
    loop.close()
    return result

with gr.Blocks(
    title="OER Catalyst Protocol Translator",
    theme=gr.themes.Soft(),
    css="""
    .main-container { max-width: 1000px; margin: auto; }
    .chat-area { height: 70vh; overflow-y: auto; }
    """
) as demo:
    
    gr.Markdown("""
    # 🔬 OER Catalyst Protocol Translator
    ### 불가능한 공정 → 가능한 공정 번역 시스템
    6개 AI 에이전트가 협업하여 문헌의 최적 합성법을 당신의 실험실 장비에 맞게 번역합니다.
    
    **에이전트 흐름:** 🔍 문헌탐색 → ⚗️ 공정분해 → 🔧 장비검증 → 🔄 대체탐색 → 📝 프로토콜작성 → ✅ 검증
    """)
    
    with gr.Row():
        task_input = gr.Textbox(
            label="목표를 입력하세요",
            placeholder="예: NiFe LDH 촉매, η < 260 mV, autoclave/furnace 대체",
            lines=3,
            scale=4,
        )
        run_btn = gr.Button("🚀 실행", variant="primary", scale=1, min_width=100)
    
    gr.Examples(examples=EXAMPLES, inputs=task_input)
    
    status_output = gr.Textbox(label="상태", visible=True)
    chat_output = gr.HTML(label="에이전트 대화", elem_classes=["chat-area"])
    
    run_btn.click(
        fn=run_and_display,
        inputs=[task_input],
        outputs=[chat_output],
    ).then(
        fn=lambda: "완료!",
        outputs=[status_output],
    )

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  OER Catalyst Protocol Translator Web UI")
    print("  브라우저에서 아래 주소로 접속하세요")
    print("="*50 + "\n")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
