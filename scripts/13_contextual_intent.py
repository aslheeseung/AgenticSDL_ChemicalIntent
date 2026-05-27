"""
Phase 8: Context-Aware Chemical Intent Tagging
같은 논문 + 섹션의 연속된 문장들을 묶어서 문맥 기반 intent 판별

핵심: 개별 문장의 operation을 보고, 그것이 속한 공정 단계의
     화학적 의미(nucleation, crystallization 등)를 추론
"""

import os, json, re
import pandas as pd
import numpy as np

BASE = "/home/hs/oer-catalyst-project/output"
SENTENCES_PATH = f"{BASE}/sentences_with_intent.csv"
OUTPUT_PATH = f"{BASE}/sentences_contextual_intent.csv"

# ============================================================
# 화학적 intent 추론 규칙
# ============================================================

# 패턴 정의
PATTERNS = {
    # --- Nucleation ---
    'nucleation': [
        # 환원제에 의한 금속 나노입자 핵형성
        r'(?i)(na[bh]4|sodium\s+borohydride|ascorbic\s+acid|citrate).{0,100}(reduc|nuclea)',
        r'(?i)(reduc|nuclea).{0,100}(na[bh]4|sodium\s+borohydride)',
        # 급격한 농도 변화에 의한 핵형성
        r'(?i)(rapidly|quickly|suddenly).{0,50}(add|inject|pour)',
        # 핵형성 언급
        r'(?i)nuclea(tion|ting|ted)',
        # 단시간 고온 (microwave 등)
        r'(?i)microwave.{0,100}(minute|min\.)',
    ],
    
    # --- Crystal Growth ---
    'crystal_growth': [
        r'(?i)(crystal|particle|grain|nanoparticle|nanosheet).{0,80}(grow|growth|ripening|aging|aged|ageing)',
        r'(?i)(grow|growth).{0,80}(crystal|particle|nanoparticle|nanosheet)',
        r'(?i)(aging|aged|ageing|aging\s+process)',
        # 장시간 열처리 (성장)
        r'(?i)(maintained|kept|held).{0,50}(hour|h\.)',
    ],
    
    # --- Crystallization (수열/용매열) ---
    'crystallization': [
        r'(?i)(hydrothermal|solvothermal|autoclave)',
        r'(?i)(teflon.{0,20}autoclave|autoclave.{0,20}teflon)',
        r'(?i)(autoclave|oven).{0,100}(heat|maintain|kept).{0,50}\d+\s*°?[cC]',
        r'(?i)crystalli[zr]',
    ],
    
    # --- Intercalation ---
    'intercalation': [
        r'(?i)intercalat',
        r'(?i)(insert|incorporat).{0,50}(layer|between|gallery|interlayer)',
    ],
    
    # --- Exfoliation ---
    'exfoliation': [
        r'(?i)exfoliat',
        r'(?i)delaminat',
        r'(?i)(separate|split).{0,50}(layer|sheet|flake)',
    ],
    
    # --- Phase Transformation ---
    'phase_transformation': [
        r'(?i)(phase\s+transformat|phase\s+change|phase\s+transit)',
        r'(?i)(anneal|calcine|sinter).{0,50}(furnace|tube|atmosphere|argon|nitrogen|n2|ar)',
        r'(?i)(furnace|tube).{0,50}(anneal|calcine|sinter|heat\s+treat)',
        r'(?i)(phosphori[sz]|sulfuri[sz]|nitrid|carboni[sz])',
        r'(?i)(na[h2]po2|na2s|thiourea|nh3).{0,80}(anneal|heat|furnace)',
    ],
    
    # --- Precipitation ---
    'precipitation': [
        r'(?i)(precipitat|coprecipitat)',
        r'(?i)(solid|precipitate|deposit).{0,50}(form|appear|obtain)',
        r'(?i)(naoh|koh|nh3|nh4oh|urea).{0,80}(precipitat|add|adjust)',
    ],
    
    # --- Reduction ---
    'reduction': [
        r'(?i)(na[bh]4|sodium\s+borohydride).{0,100}(add|dropwis|inject)',
        r'(?i)(h2|hydrogen).{0,80}(reduc|flow|atmosphere)',
        r'(?i)(reduc).{0,50}(metal|ion|species|nanoparticle)',
        r'(?i)(na[bh]4|h2|ascorbic|ethylene\s+glycol).{0,30}(reduc)',
    ],
    
    # --- Oxidation ---
    'oxidation': [
        r'(?i)(hummer|hummers).{0,50}(method|approach|procedure)',
        r'(?i)(kmno4|potassium\s+permanganate).{0,100}(graphite|add)',
        r'(?i)(h2so4|sulfuric).{0,100}(graphite|kmno4|oxid)',
    ],
    
    # --- Etching ---
    'etching': [
        r'(?i)(etch|etching)',
        r'(?i)(hf|hydrofluoric|fluoride).{0,80}(remove|etch|dissolve)',
        r'(?i)(hcl|acid).{0,80}(remove|etch|dissolve|selective)',
    ],
    
    # --- Ion Exchange ---
    'ion_exchange': [
        r'(?i)(ion\s+exchange|anion\s+exchange|cation\s+exchange)',
        r'(?i)(exchange).{0,50}(ion|anion|cation)',
    ],
    
    # --- Sol-Gel ---
    'sol_gel': [
        r'(?i)(sol[\s-]?gel)',
        r'(?i)(gel|gelation).{0,50}(form|aging|dry)',
    ],
    
    # --- Electrodeposition ---
    'deposition': [
        r'(?i)(electrodeposit|electroplating|cathodic\s+deposit)',
        r'(?i)(deposit|coat).{0,50}(electrod|substrate|surface)',
        r'(?i)(cv|cyclic\s+voltammetry).{0,50}(cycle|scan|deposit)',
    ],
    
    # --- Doping ---
    'doping': [
        r'(?i)(dop|doping).{0,50}(incorporat|substitut|introduc)',
        r'(?i)(fe|co|ni|mn|cu|zn|al).{0,30}(dop|substitut)',
    ],
    
    # --- Purification ---
    'purification': [
        r'(?i)(centrifug|wash|rinse|filter|dialy)',
        r'(?i)(purif|clean).{0,50}(impurit|by.product|residual)',
    ],
    
    # --- Drying ---
    'drying': [
        r'(?i)(dry|dried|drying).{0,50}(oven|vacuum|freeze|air|60|80|100|120)',
        r'(?i)(vacuum|freeze|oven|air).{0,50}(dry|dried)',
    ],
}


def classify_contextual_intent(text_block, operation_intents):
    """
    문장 묶음의 전체 텍스트와 operation-level intent를 보고
    화학적 intent를 추론
    """
    text = ' '.join(text_block)
    
    # 1. 패턴 매칭으로 후보 intent 찾기
    scores = {}
    for intent_name, patterns in PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = re.findall(pattern, text)
            score += len(matches)
        if score > 0:
            scores[intent_name] = score
    
    # 2. operation-level intent 조합으로 추론
    op_set = set(operation_intents)
    
    # 조합 규칙
    if 'dissolution' in op_set and 'crystallization' in op_set:
        # 용해 후 수열 = 결정화
        if scores.get('crystallization', 0) > 0:
            return 'crystallization'
    
    if 'dissolution' in op_set and 'mixing' in op_set:
        # 용해 + 교반만 = 전구체 준비 (nucleation 전단계)
        if 'reduction' in op_set:
            return 'nucleation'
        if 'precipitation' in op_set:
            return 'precipitation'
        if 'crystallization' in op_set:
            return 'crystallization'
    
    if 'reduction' in op_set:
        return 'nucleation'  # 환원에 의한 핵형성
    
    if 'phase_transformation' in op_set:
        # 구체적인 상변화 타입 구분
        if any(w in text.lower() for w in ['phosphori', 'nah2po2']):
            return 'phosphorization'
        if any(w in text.lower() for w in ['sulfuri', 'na2s', 'thiourea']):
            return 'sulfurization'
        if any(w in text.lower() for w in ['nitrid', 'nh3']):
            return 'nitridation'
        if any(w in text.lower() for w in ['carboni', 'carboniz']):
            return 'carbonization'
        if any(w in text.lower() for w in ['anneal', 'annealing']):
            return 'annealing'
        return 'phase_transformation'
    
    if 'etching' in op_set:
        return 'etching'
    
    if 'oxidation' in op_set:
        return 'oxidation'
    
    # 3. 패턴 매칭 결과가 있으면 그걸 사용
    if scores:
        return max(scores, key=scores.get)
    
    # 4. 폴백: operation intent 그대로
    return None  # None이면 원래 intent 유지


def group_into_process_blocks(group_df):
    """
    같은 섹션의 문장들을 공정 단계별로 그룹화
    규칙: 같은 operation intent가 연속되면 하나의 블록
    의미가 바뀌면 새 블록
    """
    blocks = []
    current_block = []
    current_ops = set()
    
    for idx, row in group_df.iterrows():
        op = row['chemical_intent']
        
        # 블록 분리 조건
        should_split = False
        
        # characterization/reagent_info는 독립 블록
        if op in ('characterization', 'reagent_info', 'reference_synthesis',
                   'ink_preparation', 'electrode_fabrication', 'measurement_condition'):
            if current_block:
                blocks.append({
                    'indices': [r['orig_idx'] for r in current_block],
                    'texts': [r['sentence'] for r in current_block],
                    'ops': set(current_ops),
                })
                current_block = []
                current_ops = set()
            blocks.append({
                'indices': [row['orig_idx']],
                'texts': [row['sentence']],
                'ops': {op},
            })
            continue
        
        # operation이 바뀌면 분리 고려
        if current_block and op != current_block[-1]['chemical_intent']:
            # dissolution → mixing은 같은 블록 (혼합 과정)
            if {current_block[-1]['chemical_intent'], op} <= {'dissolution', 'mixing'}:
                pass
            # crystallization → cooling은 같은 블록 (수열 과정)
            elif {current_block[-1]['chemical_intent'], op} <= {'crystallization', 'cooling'}:
                pass
            # dissolution → crystallization은 같은 블록 (수열 합성)
            elif {current_block[-1]['chemical_intent'], op} <= {'dissolution', 'crystallization'}:
                pass
            # dissolution → reduction 같은 블록 (환원 합성)
            elif {current_block[-1]['chemical_intent'], op} <= {'dissolution', 'reduction'}:
                pass
            # separation → drying 같은 블록
            elif {current_block[-1]['chemical_intent'], op} <= {'separation', 'drying'}:
                pass
            # precipitation → mixing 같은 블록
            elif {current_block[-1]['chemical_intent'], op} <= {'precipitation', 'mixing'}:
                pass
            else:
                should_split = True
        
        if should_split and current_block:
            blocks.append({
                'indices': [r['orig_idx'] for r in current_block],
                'texts': [r['sentence'] for r in current_block],
                'ops': set(current_ops),
            })
            current_block = []
            current_ops = set()
        
        current_block.append(row.to_dict())
        current_ops.add(op)
    
    # 마지막 블록
    if current_block:
        blocks.append({
            'indices': [r['orig_idx'] for r in current_block],
            'texts': [r['sentence'] for r in current_block],
            'ops': set(current_ops),
        })
    
    return blocks


def main():
    # 데이터 로드
    df = pd.read_csv(SENTENCES_PATH)
    df['orig_idx'] = df.index
    
    print(f"Total sentences: {len(df)}")
    print(f"Unique papers: {df['paper_id'].nunique()}")
    print(f"Unique (paper, section): {df.groupby(['paper_id', 'section']).ngroups}")
    
    # 컨텍스트 intent 컬럼 초기화
    df['contextual_intent'] = df['chemical_intent'].copy()
    df['process_block_id'] = -1
    
    # 논문 + 섹션별로 그룹화
    block_id = 0
    changes = 0
    
    for (paper_id, section), group in df.groupby(['paper_id', 'section']):
        if len(group) < 2:
            continue
        
        # 공정 블록으로 분리
        blocks = group_into_process_blocks(group)
        
        for block in blocks:
            # 문맥 기반 intent 추론
            new_intent = classify_contextual_intent(
                block['texts'], block['ops']
            )
            
            # 기존 intent와 다르면 업데이트
            for orig_idx in block['indices']:
                df.loc[orig_idx, 'process_block_id'] = block_id
                if new_intent and new_intent != df.loc[orig_idx, 'chemical_intent']:
                    df.loc[orig_idx, 'contextual_intent'] = new_intent
                    changes += 1
            
            block_id += 1
    
    print(f"\nProcess blocks: {block_id}")
    print(f"Sentences with changed intent: {changes}")
    
    # 결과 저장
    df.to_csv(OUTPUT_PATH, index=False)
    
    # 분포
    print(f"\n=== Contextual Intent Distribution ===\n")
    ctx_counts = df['contextual_intent'].value_counts()
    for intent, count in ctx_counts.items():
        pct = count / len(df) * 100
        print(f"  {intent:30s} {count:5d} ({pct:5.1f}%)")
    
    # 변경된 것 샘플
    changed = df[df['chemical_intent'] != df['contextual_intent']]
    print(f"\n=== Changed Samples (first 20) ===\n")
    for _, row in changed.head(20).iterrows():
        print(f"  [{row['chemical_intent']:25s} → {row['contextual_intent']:25s}]")
        print(f"    {row['sentence'][:120]}...")
        print()
    
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
