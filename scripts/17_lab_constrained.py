"""
Phase 11: Lab-Constrained Synthesis Route Generator
실험실 장비 제약에 맞는 합성 루트만 필터링 및 새로운 루트 생성

우리 실험실 장비:
  ✓ Beaker, hotplate, magnetic stirrer  → 용해, 교반, 가열(실온~끓는점)
  ✓ Syringe pump                        → 정밀 적가, controlled addition
  ✓ pH meter                             → pH 조절
  ✓ Pipettes                             → 정량 분취
  ✓ Capping agents (various)             → 표면 코팅, 나노입자 크기 제어
  ✓ Centrifuge                           → 원심분리, 세척
  ✓ RDE (Rotating Disk Electrode)        → 전기화학 측정 (OER 성능)
  ✗ Autoclave                            → 수열합성 불가
  ✗ Furnace                              → 소성, 어닐링, 인화 불가
  ✗ Microwave oven                       → 마이크로웨이브 합성 불가
  ✗ Vacuum oven                          → 진공건조 불가 (공기건조는 가능)
  ✗ Glove box                            → 불활성 분위기 불가 (공기 중 합성만)
"""

import os, json, re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import product

BASE = "/home/hs/oer-catalyst-project/output"
SEQUENCES_PATH = f"{BASE}/process_sequences.csv"
ROUTE_PERF_PATH = f"{BASE}/route_performance.csv"
NETWORK_PATH = f"{BASE}/synthesis_network.json"

# ============================================================
# 장비 제약 매핑
# ============================================================

EQUIPMENT_CONSTRAINTS = {
    # intent: (가능여부, 필요장비, 대체방안)
    'dissolution':            (True,  'beaker + stirrer', None),
    'mixing':                 (True,  'beaker + stirrer', None),
    'stirring':               (True,  'magnetic stirrer', None),
    'nucleation':             (True,  'beaker + syringe pump', None),
    'precipitation':          (True,  'beaker + pH meter', None),
    'reduction':              (True,  'beaker + syringe pump (NaBH4 dropwise)', None),
    'purification':           (True,  'centrifuge', None),
    'separation':             (True,  'centrifuge', None),
    'drying':                 (True,  'hotplate (air drying)', None),
    'cooling':                (True,  'ambient', None),
    'oxidation':              (True,  'beaker (Hummers method)', None),
    'etching':                (True,  'beaker + HF/HCl', None),
    'doping':                 (True,  'beaker (in-solution)', None),
    'deposition':             (True,  'RDE/electrochemical cell', 'electrodeposition'),
    'crystal_growth':         (True,  'beaker + hotplate (aging)', None),
    'sol_gel':                (True,  'beaker + stirrer', None),
    'ion_exchange':           (True,  'beaker + stirrer', None),
    'intercalation':          (True,  'beaker + stirrer', None),
    'exfoliation':            (True,  'beaker + sonication', None),
    
    # 불가능한 공정
    'crystallization':        (False, 'autoclave (hydrothermal)', 'co-precipitation + aging at 80°C'),
    'phase_transformation':   (False, 'furnace (high temp)', 'low-temp phase transformation (rare)'),
    'annealing':              (False, 'furnace', None),
    'phosphorization':        (False, 'furnace + NaH2PO2', 'solution-phase phosphorization (rare)'),
    'sulfurization':          (False, 'furnace + Na2S/thiourea', 'solution-phase with thioacetamide'),
    'nitridation':            (False, 'furnace + NH3', None),
    'carbonization':          (False, 'furnace', None),
}

# 합성 가능한 intent 목록
FEASIBLE_INTENTS = {k for k, v in EQUIPMENT_CONSTRAINTS.items() if v[0]}
INFEASIBLE_INTENTS = {k for k, v in EQUIPMENT_CONSTRAINTS.items() if not v[0]}

print("=== Lab Equipment Constraints ===")
print(f"\nFeasible processes ({len(FEASIBLE_INTENTS)}):")
for intent in sorted(FEASIBLE_INTENTS):
    equip = EQUIPMENT_CONSTRAINTS[intent][1]
    print(f"  ✓ {intent:25s} → {equip}")

print(f"\nInfeasible processes ({len(INFEASIBLE_INTENTS)}):")
for intent in sorted(INFEASIBLE_INTENTS):
    equip = EQUIPMENT_CONSTRAINTS[intent][1]
    alt = EQUIPMENT_CONSTRAINTS[intent][2]
    alt_str = f" → ALT: {alt}" if alt else ""
    print(f"  ✗ {intent:25s} → needs {equip}{alt_str}")


def filter_feasible_routes():
    """실험실에서 가능한 루트만 필터링"""
    seq_df = pd.read_csv(SEQUENCES_PATH)
    
    # 루트가 feasible한지 판별
    def is_feasible(sequence_str):
        try:
            seq = eval(sequence_str) if isinstance(sequence_str, str) else sequence_str
        except:
            return False
        return all(s in FEASIBLE_INTENTS for s in seq)
    
    seq_df['feasible'] = seq_df['sequence'].apply(is_feasible)
    
    feasible = seq_df[seq_df['feasible']]
    infeasible = seq_df[~seq_df['feasible']]
    
    print(f"\n{'='*60}")
    print("=== Route Feasibility Analysis ===")
    print(f"{'='*60}")
    print(f"Total sequences: {len(seq_df)}")
    print(f"Feasible (100% our lab): {len(feasible)} ({len(feasible)/len(seq_df)*100:.1f}%)")
    print(f"Infeasible: {len(infeasible)} ({len(infeasible)/len(seq_df)*100:.1f}%)")
    
    # 성능 데이터와 조인
    if os.path.exists(ROUTE_PERF_PATH):
        perf_df = pd.read_csv(ROUTE_PERF_PATH)
        
        # feasible 루트의 성능
        feasible_papers = set(feasible['paper_id'])
        feasible_perf = perf_df[perf_df['paper_id'].isin(feasible_papers)]
        infeasible_perf = perf_df[~perf_df['paper_id'].isin(feasible_papers)]
        
        if len(feasible_perf) > 0:
            print(f"\nFeasible routes with performance data: {len(feasible_perf)}")
            print(f"  Overpotential: {feasible_perf['overpotential_mV'].mean():.0f} ± {feasible_perf['overpotential_mV'].std():.0f} mV")
        
        if len(infeasible_perf) > 0:
            print(f"\nInfeasible routes with performance data: {len(infeasible_perf)}")
            print(f"  Overpotential: {infeasible_perf['overpotential_mV'].mean():.0f} ± {infeasible_perf['overpotential_mV'].std():.0f} mV")
    
    # Feasible 루트의 패턴 분석
    print(f"\n{'='*60}")
    print("=== Feasible Route Patterns ===")
    print(f"{'='*60}")
    
    feasible_routes = Counter()
    for _, row in feasible.iterrows():
        try:
            seq = eval(row['sequence']) if isinstance(row['sequence'], str) else row['sequence']
            feasible_routes[tuple(seq)] += 1
        except:
            pass
    
    print(f"\nTop 20 feasible routes:")
    for (route, count) in feasible_routes.most_common(20):
        route_str = ' → '.join(route)
        print(f"  ({count:3d}x) {route_str}")
    
    # Infeasible 이유 분석
    print(f"\n{'='*60}")
    print("=== Why Routes Are Infeasible ===")
    print(f"{'='*60}")
    
    blocker_counts = Counter()
    for _, row in infeasible.iterrows():
        try:
            seq = eval(row['sequence']) if isinstance(row['sequence'], str) else row['sequence']
        except:
            continue
        blocked = [s for s in seq if s in INFEASIBLE_INTENTS]
        for b in blocked:
            blocker_counts[b] += 1
    
    print(f"\nMost common blockers:")
    for blocker, count in blocker_counts.most_common():
        pct = count / len(infeasible) * 100
        alt = EQUIPEMENT_CONSTRAINTS.get(blocker, (None, None, None))[2]
        alt_str = f" | Alternative: {alt}" if alt else ""
        print(f"  {blocker:25s} blocks {count:4d} routes ({pct:5.1f}%){alt_str}")
    
    return feasible, infeasible


def generate_lab_routes():
    """실험실 장비 기반 합성 루트 생성"""
    print(f"\n{'='*60}")
    print("=== Generated Lab-Compatible Routes ===")
    print(f"{'='*60}")
    
    # 장비로 가능한 공정 단계들
    # 전형적인 순서: dissolution → mixing/addition → reaction → purification → drying
    
    building_blocks = {
        'step_1_preparation': [
            ['dissolution'],                                    # 전구체 용해
            ['dissolution', 'mixing'],                          # 용해 + 교반
            ['dissolution', 'mixing', 'precipitation'],         # 용해 + 교반 + pH 조절 침전
        ],
        'step_2_reaction': [
            ['nucleation'],                                     # 환원에 의한 핵형성 (NaBH4)
            ['nucleation', 'crystal_growth'],                   # 핵형성 + 에이징
            ['precipitation'],                                  # 침전
            ['precipitation', 'crystal_growth'],                # 침전 + 에이징
            ['reduction'],                                      # 환원
            ['etching'],                                        # 에칭 (MXene)
            ['deposition'],                                     # 전기증착
            ['doping'],                                         # 도핑
        ],
        'step_3_purification': [
            ['purification'],                                   # 원심분리 + 세척
            ['purification', 'drying'],                         # 세척 + 건조
        ],
        'step_4_optional': [
            [],                                                 # 없음
            ['doping'],                                         # 후처리 도핑
            ['etching'],                                        # 후처리 에칭
        ],
    }
    
    # 모든 조합 생성
    all_routes = []
    for prep in building_blocks['step_1_preparation']:
        for reaction in building_blocks['step_2_reaction']:
            for purification in building_blocks['step_3_purification']:
                for optional in building_blocks['step_4_optional']:
                    full_route = prep + reaction + purification + optional
                    # 중복 제거 (연속 같은 건 제외)
                    compressed = [full_route[0]]
                    for s in full_route[1:]:
                        if s != compressed[-1]:
                            compressed.append(s)
                    
                    if len(compressed) >= 2:
                        all_routes.append(compressed)
    
    # 중복 제거
    unique_routes = []
    seen = set()
    for route in all_routes:
        key = tuple(route)
        if key not in seen:
            seen.add(key)
            unique_routes.append(route)
    
    print(f"\nGenerated {len(unique_routes)} unique lab-compatible routes")
    print(f"\nSample routes:")
    for i, route in enumerate(unique_routes[:30]):
        route_str = ' → '.join(route)
        print(f"  {i+1:3d}. {route_str}")
    
    # 문헌에서 실제 존재하는 루트와 비교
    seq_df = pd.read_csv(SEQUENCES_PATH)
    existing_routes = set()
    for _, row in seq_df.iterrows():
        try:
            seq = tuple(eval(row['sequence']) if isinstance(row['sequence'], str) else row['sequence'])
            existing_routes.add(seq)
        except:
            pass
    
    novel_routes = []
    verified_routes = []
    for route in unique_routes:
        key = tuple(route)
        if key in existing_routes:
            verified_routes.append(route)
        else:
            novel_routes.append(route)
    
    print(f"\n{'='*60}")
    print("=== Route Verification ===")
    print(f"{'='*60}")
    print(f"Verified (exists in literature): {len(verified_routes)}")
    print(f"Novel (not in our literature): {len(novel_routes)}")
    
    if verified_routes:
        print(f"\nVerified routes:")
        for route in verified_routes:
            route_str = ' → '.join(route)
            print(f"  ✓ {route_str}")
    
    if novel_routes:
        print(f"\nNovel routes (first 20):")
        for route in novel_routes[:20]:
            route_str = ' → '.join(route)
            print(f"  ★ {route_str}")
    
    # 결과 저장
    output = []
    for route in unique_routes:
        is_verified = tuple(route) in existing_routes
        output.append({
            'route': ' → '.join(route),
            'sequence': route,
            'verified_in_literature': is_verified,
            'feasible': True,
        })
    
    out_df = pd.DataFrame(output)
    out_df.to_csv(f"{BASE}/lab_compatible_routes.csv", index=False)
    
    with open(f"{BASE}/lab_compatible_routes.json", 'w') as f:
        json.dump([{
            'route': r['route'],
            'verified': r['verified_in_literature'],
        } for r in output], f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved: {BASE}/lab_compatible_routes.csv")
    
    return unique_routes, verified_routes, novel_routes


# 오타 수정
EQUIPEMENT_CONSTRAINTS = EQUIPMENT_CONSTRAINTS


if __name__ == '__main__':
    print("LAB EQUIPMENT INVENTORY")
    print("=" * 40)
    print("✓ Beaker, hotplate, magnetic stirrer")
    print("✓ Syringe pump")
    print("✓ pH meter")
    print("✓ Pipettes")
    print("✓ Capping agents (various)")
    print("✓ Centrifuge")
    print("✓ RDE (Rotating Disk Electrode)")
    print("✗ Autoclave")
    print("✗ Furnace")
    print("✗ Microwave oven")
    print("✗ Vacuum oven")
    print("✗ Glove box")
    print()
    
    feasible, infeasible = filter_feasible_routes()
    unique_routes, verified, novel = generate_lab_routes()
