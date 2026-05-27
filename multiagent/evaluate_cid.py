"""
CID Quality Evaluator
CID Agent의 출력 품질을 자동 평가

평가 항목:
1. Intent Accuracy — 10개 core intent 안에서 선택
2. Mechanism Quality — ①원인 ②과정 ③결과 체인
3. Raw Step Preservation — 논문 원문 보존
4. Capability Detection — 장비 제약 정확히 감지
5. Substitution Quality — 같은 Intent, 다른 Mechanism

Usage:
  python evaluate_cid.py
"""
import os
import sys
import json
import asyncio
import re
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from cid_agent import build_team, parse_cid_json, CORE_INTENTS

# ============================================================
# Evaluation Criteria
# ============================================================

CORE_INTENT_NAMES = {ci.lower() for ci in CORE_INTENTS}

def score_intent_accuracy(cid_rows: list) -> dict:
    """Check if all intents are from the 10 core set."""
    results = {"score": 0, "max": 0, "details": []}
    for row in cid_rows:
        intent = row.get("chemical_intent", "")
        results["max"] += 1
        if intent.lower().strip() in CORE_INTENT_NAMES:
            results["score"] += 1
            results["details"].append(f"  ✅ {intent}")
        else:
            results["details"].append(f"  ❌ {intent} (not in core 10)")
    return results


def score_mechanism_quality(cid_rows: list) -> dict:
    """Check if mechanism follows ①cause ②process ③result chain."""
    results = {"score": 0, "max": 0, "details": []}
    
    for row in cid_rows:
        mechanism = row.get("mechanism", "")
        results["max"] += 3  # 3 sub-criteria
        
        score = 0
        issues = []
        
        # Check ①: Has causal language (→, 때문, 의해, 으로, by, through, via)
        causal_patterns = [r"→", r"때문", r"의해", r"으로", r"로 인", r"by", r"through", r"via", r"due to"]
        has_cause = any(re.search(p, mechanism) for p in causal_patterns)
        if has_cause:
            score += 1
        else:
            issues.append("인과 관계(→/의해) 없음")
        
        # Check ②: Mentions atoms/ions/molecules level
        chemistry_patterns = [
            r"이온", r"원자", r"분자", r"결정", r"격자", r"핵", r"층",
            r"ion", r"atom", r"molecule", r"crystal", r"lattice", r"nuclei",
            r"Ni", r"Fe", r"OH", r"H\+", r"pH",
        ]
        has_chemistry = any(re.search(p, mechanism) for p in chemistry_patterns)
        if has_chemistry:
            score += 1
        else:
            issues.append("화학적 수준(이온/원자/분자) 기술 없음")
        
        # Check ③: Describes result (structure/phase/composition)
        result_patterns = [
            r"형성", r"생성", r"구조", r"달성", r"제거", r"변화",
            r"formed", r"created", r"structure", r"achieved", r"removed",
        ]
        has_result = any(re.search(p, mechanism) for p in result_patterns)
        if has_result:
            score += 1
        else:
            issues.append("결과(형성/생성/구조) 기술 없음")
        
        results["score"] += score
        status = "✅" if score == 3 else "⚠️" if score >= 2 else "❌"
        results["details"].append(
            f"  {status} [{score}/3] {row.get('chemical_intent', '?')}: "
            f"{', '.join(issues) if issues else 'OK'}"
        )
    
    return results


def score_raw_step_preservation(cid_rows: list) -> dict:
    """Check if raw steps are actual paper quotes (not summaries)."""
    results = {"score": 0, "max": 0, "details": []}
    
    for row in cid_rows:
        raw = row.get("raw_step", "")
        results["max"] += 1
        
        # Indicators of actual paper text (English, contains specific values)
        has_english = bool(re.search(r'[a-zA-Z]{10,}', raw))
        has_specifics = bool(re.search(r'\d+\s*(°C|mL|mmol|h|min|mV|mM)', raw))
        
        # Indicators of summary (Korean, route labels)
        is_korean_summary = bool(re.search(r'[가-힣]{5,}', raw))
        is_route_label = bool(re.match(r'^(nucleation|crystallization|dissolution)\s*→', raw.lower()))
        
        if is_route_label:
            results["details"].append(f"  ❌ Route label: {raw[:60]}")
        elif is_korean_summary and not has_english:
            results["details"].append(f"  ⚠️ Korean summary (no English): {raw[:60]}")
        elif has_english and has_specifics:
            results["score"] += 1
            results["details"].append(f"  ✅ Paper quote with specifics: {raw[:60]}...")
        elif has_english:
            results["score"] += 1
            results["details"].append(f"  ✅ Paper quote: {raw[:60]}...")
        else:
            results["details"].append(f"  ⚠️ Unclear: {raw[:60]}")
    
    return results


def score_capability_detection(conversation: list) -> dict:
    """Check if check_capabilities_feasibility was called and caught issues."""
    results = {"score": 0, "max": 3, "details": []}
    
    tool_called = False
    caught_sealed = False
    suggested_alt = False
    
    for msg in conversation:
        content = str(msg.get("content", ""))
        if "Required Capabilities Feasibility Check" in content:
            tool_called = True
            results["score"] += 1
            results["details"].append("  ✅ check_capabilities_feasibility 호출됨")
        if "sealed vessel" in content.lower() and "❌" in content:
            caught_sealed = True
            results["score"] += 1
            results["details"].append("  ✅ 'sealed vessel' ❌로 감지됨")
        if "alternative" in content.lower() and ("beaker" in content.lower() or "hotplate" in content.lower()):
            suggested_alt = True
            results["score"] += 1
            results["details"].append("  ✅ 대체 방안 제안됨")
    
    if not tool_called:
        results["details"].append("  ❌ check_capabilities_feasibility 호출 안 됨")
    if not caught_sealed:
        results["details"].append("  ❌ 'sealed vessel' 감지 못함")
    if not suggested_alt:
        results["details"].append("  ❌ 대체 방안 없음")
    
    return results


def score_substitution_quality(conversation: list, cid_rows: list) -> dict:
    """Check if Experimental Agent proposed same-intent different-mechanism."""
    results = {"score": 0, "max": 2, "details": []}
    
    has_substitution = False
    has_evidence = False
    
    for msg in conversation:
        content = str(msg.get("content", ""))
        source = msg.get("source", "")
        if source == "Experimental_Agent" or source == "Literature_Extractor":
            # Look for substitution language
            sub_patterns = [
                r"대체", r"alternative", r"ostwald", r"aging",
                r"co-precipitation.*aging", r"open beaker",
            ]
            if any(re.search(p, content, re.IGNORECASE) for p in sub_patterns):
                has_substitution = True
            
            # Look for evidence reference
            if "evidence" in content.lower() or "증거" in content or "문헌" in content:
                has_evidence = True
    
    if has_substitution:
        results["score"] += 1
        results["details"].append("  ✅ 대체 메커니즘 제안됨")
    else:
        results["details"].append("  ❌ 대체 메커니즘 없음")
    
    if has_evidence:
        results["score"] += 1
        results["details"].append("  ✅ 문헌 근거 인용됨")
    else:
        results["details"].append("  ❌ 문헌 근거 없음")
    
    return results


# ============================================================
# Run evaluation
# ============================================================

async def evaluate():
    """Run CID Agent and evaluate output quality."""
    team = build_team()
    
    test_targets = [
        "NiFe LDH",
        "CoFe LDH",
        "NiFe LDH electrodeposition on Ni foam",
    ]
    
    all_scores = []
    
    for target in test_targets:
        print(f"\n{'='*60}")
        print(f"  평가 대상: {target}")
        print(f"{'='*60}")
        
        task = f"{target}의 합성 절차를 CID로 번역해주세요."
        
        all_messages = []
        stream = team.run_stream(task=task)
        
        async for message in stream:
            msg_type = type(message).__name__
            if msg_type == "TextMessage":
                source = getattr(message, 'source', '')
                content = getattr(message, 'content', '')
                if source != 'user':
                    all_messages.append({"source": source, "content": content})
            elif msg_type == "ToolCallSummaryMessage":
                content = getattr(message, 'content', '')
                all_messages.append({"source": "tool", "content": content})
        
        # Parse CID
        cid_rows = []
        for msg in all_messages:
            if msg.get("source") == "CID_Agent":
                parsed = parse_cid_json(msg.get("content", ""))
                if parsed:
                    cid_rows = parsed
        
        if not cid_rows:
            print(f"  ❌ CID 파싱 실패!")
            continue
        
        print(f"\n  CID 행 수: {len(cid_rows)}")
        
        # Evaluate
        scores = {}
        
        print(f"\n  ── 1. Intent Accuracy ──")
        r = score_intent_accuracy(cid_rows)
        scores["intent"] = r
        print(f"  Score: {r['score']}/{r['max']}")
        for d in r["details"]:
            print(d)
        
        print(f"\n  ── 2. Mechanism Quality ──")
        r = score_mechanism_quality(cid_rows)
        scores["mechanism"] = r
        print(f"  Score: {r['score']}/{r['max']}")
        for d in r["details"]:
            print(d)
        
        print(f"\n  ── 3. Raw Step Preservation ──")
        r = score_raw_step_preservation(cid_rows)
        scores["raw_step"] = r
        print(f"  Score: {r['score']}/{r['max']}")
        for d in r["details"]:
            print(d)
        
        print(f"\n  ── 4. Capability Detection ──")
        r = score_capability_detection(all_messages)
        scores["capability"] = r
        print(f"  Score: {r['score']}/{r['max']}")
        for d in r["details"]:
            print(d)
        
        print(f"\n  ── 5. Substitution Quality ──")
        r = score_substitution_quality(all_messages, cid_rows)
        scores["substitution"] = r
        print(f"  Score: {r['score']}/{r['max']}")
        for d in r["details"]:
            print(d)
        
        # Total
        total = sum(s["score"] for s in scores.values())
        max_total = sum(s["max"] for s in scores.values())
        pct = (total / max_total * 100) if max_total > 0 else 0
        print(f"\n  ══ TOTAL: {total}/{max_total} ({pct:.0f}%) ══")
        
        all_scores.append({
            "target": target,
            "cid_rows": len(cid_rows),
            "total": total,
            "max": max_total,
            "pct": pct,
            "breakdown": {k: f"{v['score']}/{v['max']}" for k, v in scores.items()},
        })
        
        # Save CID
        out_path = Path(__file__).parent.parent / "output" / f"eval_cid_{target.replace(' ', '_').lower()}.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"target": target, "cid_rows": cid_rows, "scores": scores}, f, ensure_ascii=False, indent=2)
    
    # Summary
    print(f"\n\n{'='*60}")
    print(f"  전체 평가 요약")
    print(f"{'='*60}")
    for s in all_scores:
        print(f"  {s['target']}: {s['total']}/{s['max']} ({s['pct']:.0f}%)")
        print(f"    CID행={s['cid_rows']} | {s['breakdown']}")
    
    avg_pct = sum(s["pct"] for s in all_scores) / len(all_scores) if all_scores else 0
    print(f"\n  평균: {avg_pct:.0f}%")


if __name__ == "__main__":
    asyncio.run(evaluate())
