"""
Phase 9: Synthesis Route Network
논문별 합성 공정 순서를 추출하여 디렉션 그래프 구축

각 논문의 합성 섹션에서 contextual_intent 순서를 시퀀스로 추출
→ 전이 행렬(transition matrix) + 네트워크 그래프 생성
"""

import os, json, re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

BASE = "/home/hs/oer-catalyst-project/output"
INPUT_PATH = f"{BASE}/sentences_contextual_intent.csv"

# 네트워크 분석에서 제외할 intent (합성 공정이 아님)
EXCLUDE_INTENTS = {
    'characterization', 'reagent_info', 'reference_synthesis',
    'measurement_condition', 'ink_preparation', 'electrode_fabrication',
}

# 합성 관련 intent만 유지
SYNTHESIS_INTENTS = {
    'dissolution', 'mixing', 'nucleation', 'crystallization',
    'crystal_growth', 'precipitation', 'reduction', 'oxidation',
    'phase_transformation', 'annealing', 'phosphorization',
    'sulfurization', 'nitridation', 'carbonization',
    'etching', 'intercalation', 'exfoliation', 'ion_exchange',
    'sol_gel', 'deposition', 'doping', 'purification',
    'drying', 'cooling', 'separation', 'substrate_preparation',
}


def extract_process_sequences(df):
    """논문별 합성 공정 시퀀스 추출"""
    # 합성 관련 문장만 필터
    syn_df = df[df['contextual_intent'].isin(SYNTHESIS_INTENTS)].copy()
    print(f"Synthesis-related sentences: {len(syn_df)} / {len(df)}")
    
    # 논문 + 섹션별로 그룹화
    sequences = []
    
    for (paper_id, section), group in syn_df.groupby(['paper_id', 'section']):
        # 원본 순서대로 정렬 (CSV에서의 행 번호)
        group = group.sort_index()
        
        # 연속된 같은 intent는 하나로 압축
        intents = group['contextual_intent'].tolist()
        compressed = [intents[0]]
        for i in range(1, len(intents)):
            if intents[i] != compressed[-1]:
                compressed.append(intents[i])
        
        sequences.append({
            'paper_id': paper_id,
            'section': section,
            'sequence': compressed,
            'length': len(compressed),
            'raw_intents': intents,
        })
    
    print(f"Process sequences extracted: {len(sequences)}")
    return sequences


def build_transition_matrix(sequences):
    """전이 행렬 (어떤 intent 다음에 어떤 intent가 오는가)"""
    transitions = Counter()
    intent_counts = Counter()
    
    for seq in sequences:
        s = seq['sequence']
        for i in range(len(s) - 1):
            transitions[(s[i], s[i+1])] += 1
            intent_counts[s[i]] += 1
        if s:
            intent_counts[s[-1]] += 1
    
    # 전이 확률 계산
    all_intents = sorted(intent_counts.keys())
    n = len(all_intents)
    intent_idx = {intent: i for i, intent in enumerate(all_intents)}
    
    matrix = np.zeros((n, n))
    for (src, dst), count in transitions.items():
        if src in intent_idx and dst in intent_idx:
            matrix[intent_idx[src]][intent_idx[dst]] = count
    
    # 행 정규화 (확률로)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    prob_matrix = matrix / row_sums
    
    return {
        'intents': all_intents,
        'intent_idx': intent_idx,
        'count_matrix': matrix,
        'prob_matrix': prob_matrix,
        'transitions': transitions,
        'intent_counts': intent_counts,
    }


def build_network_data(transition_data, min_count=5):
    """네트워크 시각화용 데이터 생성"""
    transitions = transition_data['transitions']
    intent_counts = transition_data['intent_counts']
    
    # 노드
    nodes = []
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        nodes.append({
            'id': intent,
            'count': count,
            'label': intent,
        })
    
    # 엣지 (최소 빈도 이상)
    edges = []
    for (src, dst), count in sorted(transitions.items(), key=lambda x: -x[1]):
        if count >= min_count:
            src_total = intent_counts[src]
            prob = count / src_total if src_total > 0 else 0
            edges.append({
                'source': src,
                'target': dst,
                'count': count,
                'probability': round(prob, 3),
            })
    
    return {'nodes': nodes, 'edges': edges}


def find_common_routes(sequences, min_length=3, top_n=20):
    """가장 흔한 합성 루트 찾기"""
    route_counter = Counter()
    
    for seq in sequences:
        s = tuple(seq['sequence'])
        # 길이 3 이상의 서브루트 추출
        for length in range(min_length, len(s) + 1):
            for start in range(len(s) - length + 1):
                subroute = s[start:start + length]
                route_counter[subroute] += 1
    
    return route_counter.most_common(top_n)


def find_frequent_pairs(sequences):
    """자주 등장하는 연속 쌍 (bigram)"""
    pair_counter = Counter()
    for seq in sequences:
        s = seq['sequence']
        for i in range(len(s) - 1):
            pair_counter[(s[i], s[i+1])] += 1
    return pair_counter.most_common(30)


def main():
    # 데이터 로드
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded: {len(df)} sentences")
    print(f"Unique intents: {df['contextual_intent'].nunique()}")
    
    # 1. 공정 시퀀스 추출
    sequences = extract_process_sequences(df)
    
    # 2. 전이 행렬
    trans_data = build_transition_matrix(sequences)
    print(f"\nIntents in network: {len(trans_data['intents'])}")
    
    # 3. 네트워크 데이터
    network = build_network_data(trans_data, min_count=5)
    print(f"Nodes: {len(network['nodes'])}, Edges: {len(network['edges'])}")
    
    # 4. 빈도 높은 루트
    print(f"\n{'='*60}")
    print("=== Top 20 Most Common Synthesis Routes ===")
    print(f"{'='*60}")
    routes = find_common_routes(sequences, min_length=3, top_n=20)
    for i, (route, count) in enumerate(routes):
        route_str = ' → '.join(route)
        print(f"  {i+1:2d}. ({count:3d}x) {route_str}")
    
    # 5. 빈도 높은 전이 쌍
    print(f"\n{'='*60}")
    print("=== Top 30 Transition Pairs ===")
    print(f"{'='*60}")
    pairs = find_frequent_pairs(sequences)
    for i, ((src, dst), count) in enumerate(pairs):
        prob = count / trans_data['intent_counts'][src] * 100
        print(f"  {i+1:2d}. {src:25s} → {dst:25s} ({count:4d}x, {prob:5.1f}%)")
    
    # 6. 통계
    print(f"\n{'='*60}")
    print("=== Synthesis Process Statistics ===")
    print(f"{'='*60}")
    seq_lengths = [s['length'] for s in sequences]
    print(f"  Total sequences: {len(sequences)}")
    print(f"  Avg sequence length: {np.mean(seq_lengths):.1f}")
    print(f"  Median: {np.median(seq_lengths):.0f}")
    print(f"  Max: {max(seq_lengths)}")
    
    # Starting intents (첫 번째 공정)
    starters = Counter(s['sequence'][0] for s in sequences if s['sequence'])
    print(f"\n  Most common first step:")
    for intent, count in starters.most_common(10):
        print(f"    {intent:25s} {count:4d} ({count/len(sequences)*100:.1f}%)")
    
    # Ending intents (마지막 공정)
    enders = Counter(s['sequence'][-1] for s in sequences if s['sequence'])
    print(f"\n  Most common last step:")
    for intent, count in enders.most_common(10):
        print(f"    {intent:25s} {count:4d} ({count/len(sequences)*100:.1f}%)")
    
    # 7. 전체 결과 저장
    # 네트워크 JSON
    with open(f"{BASE}/synthesis_network.json", 'w') as f:
        json.dump(network, f, indent=2, ensure_ascii=False)
    
    # 전이 행렬 CSV
    intents = trans_data['intents']
    trans_df = pd.DataFrame(
        trans_data['count_matrix'],
        index=intents, columns=intents
    )
    trans_df.to_csv(f"{BASE}/transition_matrix.csv")
    
    # 루트 데이터
    routes_data = [{'route': ' → '.join(r), 'count': c} for r, c in routes]
    with open(f"{BASE}/common_routes.json", 'w') as f:
        json.dump(routes_data, f, indent=2, ensure_ascii=False)
    
    # 시퀀스 데이터
    seq_data = [{'paper_id': s['paper_id'], 'section': s['section'],
                 'sequence': s['sequence'], 'length': s['length']}
                for s in sequences]
    pd.DataFrame(seq_data).to_csv(f"{BASE}/process_sequences.csv", index=False)
    
    print(f"\nSaved:")
    print(f"  {BASE}/synthesis_network.json")
    print(f"  {BASE}/transition_matrix.csv")
    print(f"  {BASE}/common_routes.json")
    print(f"  {BASE}/process_sequences.csv")


if __name__ == '__main__':
    main()
