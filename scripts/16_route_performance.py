"""
Phase 10: OER Performance Extraction & Route-Performance Analysis
논문에서 OER 성능 수치를 추출하고, 합성 루트와 연결하여 통계 분석
"""

import os, json, re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

BASE = "/home/hs/oer-catalyst-project/output"
SEQUENCES_PATH = f"{BASE}/process_sequences.csv"
SENTENCES_PATH = f"{BASE}/sentences_contextual_intent.csv"
DATA_DIR = "/home/hs/oer-catalyst-project/data/raw"

# ============================================================
# OER 성능 추출 패턴
# ============================================================

# Overpotential at X mA/cm²
OVERPOTENTIAL_PATTERNS = [
    # "overpotential of 280 mV at 10 mA cm-2"
    r'(?i)overpotential\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*m[vV]\s*(?:at|to\s+reach|for)\s+(\d+(?:\.\d+)?)\s*m[aA]\s*c?m[−\-]?2',
    # "280 mV at 10 mA cm-2"
    r'(\d+(?:\.\d+)?)\s*m[vV]\s*(?:at|to\s+reach|for)\s+(\d+(?:\.\d+)?)\s*m[ aA]+\s*c?m[−\-]?2',
    # "η = 280 mV", "η10 = 280 mV"
    r'[ηϵ]\s*(?:\d{0,3}\s*=)?\s*(\d+(?:\.\d+)?)\s*m[vV]',
    # "requires an overpotential of 280 mV"
    r'(?i)(?:require|need|exhibit|show|achieve|deliver|display).{0,30}overpotential.{0,20}(\d+(?:\.\d+)?)\s*m[vV]',
    # "10 mA cm-2 at an overpotential of 280 mV"
    r'(?i)(\d+(?:\.\d+)?)\s*m[ aA]+\s*c?m[−\-]?2.{0,40}?overpotential.{0,20}?(\d+(?:\.\d+)?)\s*m[vV]',
]

# Tafel slope
TAFEL_PATTERNS = [
    r'(?i)tafel\s+slope\s+(?:of\s+|is\s+|value.{0,5}\s+)?(\d+(?:\.\d+)?)\s*m[vV]\s*[dD][eE][cC][−\-]?1',
    r'(?i)tafel\s+slope.{0,30}?(\d+(?:\.\d+)?)\s*m[vV]\s*/\s*[dD][eE][cC]',
]

# Electrolyte
ELECTROLYTE_PATTERNS = [
    r'(?i)(\d+(?:\.\d+)?)\s*[mM]\s*(?:[mM]ol\s*)?[kK][oO][hH]',
    r'(?i)(\d+(?:\.\d+)?)\s*[mM]\s*(?:[mM]ol\s*)?[nN][aA][oO][hH]',
    r'(?i)(\d+(?:\.\d+)?)\s*[mM]\s*(?:[mM]ol\s*)?[hH]2[sS][oO]4',
]


def extract_performance_from_paper(paper_id, json_path):
    """개별 논문 JSON에서 OER 성능 수치 추출"""
    import glob
    
    # JSON 파일 찾기
    candidates = glob.glob(os.path.join(DATA_DIR, "**", f"{paper_id}.json"), recursive=True)
    if not candidates:
        return None
    
    with open(candidates[0]) as f:
        data = json.load(f)
    
    doc = data.get('document', {})
    
    # 전체 텍스트 합치기
    all_text = []
    for section_name, section_content in doc.items():
        if isinstance(section_content, str):
            all_text.append(section_content)
        elif isinstance(section_content, dict):
            for sub_name, sub_content in section_content.items():
                if isinstance(sub_content, str):
                    all_text.append(sub_content)
                elif isinstance(sub_content, list):
                    all_text.extend(str(x) for x in sub_content)
    
    full_text = '\n'.join(all_text)
    
    results = {
        'paper_id': paper_id,
        'overpotentials': [],     # [(value_mV, current_density)]
        'tafel_slopes': [],       # [value_mV/dec]
        'electrolyte': None,
    }
    
    # Overpotential 추출
    for pattern in OVERPOTENTIAL_PATTERNS:
        matches = re.finditer(pattern, full_text)
        for m in matches:
            groups = m.groups()
            if len(groups) == 2:
                val = float(groups[0])
                current = float(groups[1])
                if 50 <= val <= 800 and current <= 1000:
                    results['overpotentials'].append((val, current))
            elif len(groups) == 1:
                val = float(groups[0])
                if 50 <= val <= 800:
                    results['overpotentials'].append((val, 10))  # assume 10 mA/cm2
    
    # Tafel slope 추출
    for pattern in TAFEL_PATTERNS:
        for m in re.finditer(pattern, full_text):
            val = float(m.group(1))
            if 10 <= val <= 500:
                results['tafel_slopes'].append(val)
    
    # Electrolyte 추출
    for pattern in ELECTROLYTE_PATTERNS:
        m = re.search(pattern, full_text)
        if m:
            results['electrolyte'] = f"{m.group(1)}M KOH" if 'koh' in pattern.lower() else f"{m.group(1)}M NaOH"
            break
    
    return results


def main():
    # 시퀀스 데이터 로드
    seq_df = pd.read_csv(SEQUENCES_PATH)
    print(f"Process sequences: {len(seq_df)}")
    print(f"Unique papers: {seq_df['paper_id'].nunique()}")
    
    # 각 논문에서 성능 추출
    paper_ids = seq_df['paper_id'].unique()
    print(f"\nExtracting OER performance from {len(paper_ids)} papers...")
    
    all_results = []
    for i, pid in enumerate(paper_ids):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{len(paper_ids)}...")
        
        perf = extract_performance_from_paper(pid, None)
        if perf and perf['overpotentials']:
            # 가장 낮은 overpotential 선택 (best performance)
            best_overpotential = min(perf['overpotentials'], key=lambda x: x[0])
            best_tafel = min(perf['tafel_slopes']) if perf['tafel_slopes'] else None
            
            all_results.append({
                'paper_id': pid,
                'overpotential_mV': best_overpotential[0],
                'current_density': best_overpotential[1],
                'tafel_slope': best_tafel,
                'electrolyte': perf['electrolyte'],
            })
    
    perf_df = pd.DataFrame(all_results)
    print(f"\nPapers with extractable overpotential: {len(perf_df)}")
    
    if len(perf_df) == 0:
        print("WARNING: No overpotential data extracted!")
        print("Trying broader patterns...")
        return
    
    # 시퀀스와 조인
    # 각 paper_id의 대표 루트 (가장 긴 시퀀스)
    main_routes = seq_df.sort_values('length', ascending=False).drop_duplicates('paper_id')
    merged = perf_df.merge(main_routes[['paper_id', 'sequence']], on='paper_id')
    merged['route'] = merged['sequence'].apply(lambda x: str(x))
    
    print(f"Merged: {len(merged)} papers with route + performance")
    
    # === 분석 ===
    
    # 1. 전체 성능 분포
    print(f"\n{'='*60}")
    print("=== OER Performance Distribution ===")
    print(f"{'='*60}")
    print(f"  Overpotential: {merged['overpotential_mV'].mean():.0f} ± {merged['overpotential_mV'].std():.0f} mV")
    print(f"  Range: {merged['overpotential_mV'].min():.0f} - {merged['overpotential_mV'].max():.0f} mV")
    print(f"  Median: {merged['overpotential_mV'].median():.0f} mV")
    
    if merged['tafel_slope'].notna().sum() > 0:
        tafel_valid = merged['tafel_slope'].dropna()
        print(f"  Tafel slope: {tafel_valid.mean():.0f} ± {tafel_valid.std():.0f} mV/dec")
    
    # 2. 합성 루트별 성능 비교
    print(f"\n{'='*60}")
    print("=== Route → Performance (Top routes with ≥5 papers) ===")
    print(f"{'='*60}")
    
    route_stats = merged.groupby('route').agg(
        count=('overpotential_mV', 'count'),
        mean_eta=('overpotential_mV', 'mean'),
        std_eta=('overpotential_mV', 'std'),
        min_eta=('overpotential_mV', 'min'),
        median_eta=('overpotential_mV', 'median'),
    ).reset_index()
    
    route_stats = route_stats[route_stats['count'] >= 3].sort_values('mean_eta')
    
    print(f"\nTop 15 best-performing routes:")
    for _, row in route_stats.head(15).iterrows():
        route_str = row['route'].replace("'", "").replace("[", "").replace("]", "")
        print(f"  η={row['mean_eta']:6.0f}±{row['std_eta']:4.0f} mV (min={row['min_eta']:.0f}, n={row['count']:.0f})")
        print(f"    {route_str}")
    
    print(f"\nTop 15 worst-performing routes:")
    for _, row in route_stats.tail(15).iterrows():
        route_str = row['route'].replace("'", "").replace("[", "").replace("]", "")
        print(f"  η={row['mean_eta']:6.0f}±{row['std_eta']:4.0f} mV (min={row['min_eta']:.0f}, n={row['count']:.0f})")
        print(f"    {route_str}")
    
    # 3. 개별 공정 단계가 성능에 미치는 영향
    print(f"\n{'='*60}")
    print("=== Single Process Step Impact on Overpotential ===")
    print(f"{'='*60}")
    
    # 각 intent가 포함된 루트 vs 포함되지 않은 루트 비교
    from itertools import chain
    all_intents = set(chain.from_iterable(
        eval(s) if isinstance(s, str) else s for s in merged['sequence']
    ))
    
    intent_impact = []
    for intent in all_intents:
        has_intent = merged['sequence'].apply(lambda s: intent in (eval(s) if isinstance(s, str) else s))
        with_it = merged[has_intent]['overpotential_mV']
        without_it = merged[~has_intent]['overpotential_mV']
        
        if len(with_it) >= 5 and len(without_it) >= 5:
            diff = without_it.mean() - with_it.mean()  # positive = intent helps
            intent_impact.append({
                'intent': intent,
                'with_mean': with_it.mean(),
                'with_n': len(with_it),
                'without_mean': without_it.mean(),
                'without_n': len(without_it),
                'difference_mV': diff,
            })
    
    impact_df = pd.DataFrame(intent_impact).sort_values('difference_mV', ascending=False)
    
    print(f"\nIntents that LOWER overpotential (better performance):")
    for _, row in impact_df.head(10).iterrows():
        print(f"  {row['intent']:25s} WITH={row['with_mean']:.0f}mV (n={row['with_n']:.0f}) "
              f"WITHOUT={row['without_mean']:.0f}mV (n={row['without_n']:.0f}) "
              f"Δ={row['difference_mV']:+.0f}mV")
    
    print(f"\nIntents that HIGHER overpotential (worse performance):")
    for _, row in impact_df.tail(10).iterrows():
        print(f"  {row['intent']:25s} WITH={row['with_mean']:.0f}mV (n={row['with_n']:.0f}) "
              f"WITHOUT={row['without_mean']:.0f}mV (n={row['without_n']:.0f}) "
              f"Δ={row['difference_mV']:+.0f}mV")
    
    # 4. 루트 재조립 제안
    print(f"\n{'='*60}")
    print("=== Route Reassembly Suggestions ===")
    print(f"{'='*60}")
    
    # 성능 좋은 루트의 패턴 추출
    top_25 = merged.nsmallest(int(len(merged)*0.25), 'overpotential_mV')
    bottom_25 = merged.nlargest(int(len(merged)*0.25), 'overpotential_mV')
    
    # Top 25%에 자주 등장하는 intent
    top_intents = Counter()
    for s in top_25['sequence']:
        seq = eval(s) if isinstance(s, str) else s
        for intent in set(seq):
            top_intents[intent] += 1
    
    bot_intents = Counter()
    for s in bottom_25['sequence']:
        seq = eval(s) if isinstance(s, str) else s
        for intent in set(seq):
            bot_intents[intent] += 1
    
    print(f"\nIntents enriched in TOP 25% (low η):")
    for intent, count in top_intents.most_common(15):
        pct = count / len(top_25) * 100
        bot_pct = bot_intents.get(intent, 0) / max(len(bottom_25), 1) * 100
        marker = " ★" if pct > bot_pct + 5 else ""
        print(f"  {intent:25s} {pct:5.1f}% (bottom 25%: {bot_pct:5.1f}%){marker}")
    
    # Top 25%에서만 쓰이는 패턴
    print(f"\nPatterns UNIQUE to top performers:")
    top_pairs = Counter()
    for s in top_25['sequence']:
        seq = eval(s) if isinstance(s, str) else s
        for i in range(len(seq)-1):
            top_pairs[(seq[i], seq[i+1])] += 1
    
    bot_pairs = Counter()
    for s in bottom_25['sequence']:
        seq = eval(s) if isinstance(s, str) else s
        for i in range(len(seq)-1):
            bot_pairs[(seq[i], seq[i+1])] += 1
    
    unique_top = [(pair, count) for pair, count in top_pairs.items()
                  if pair not in bot_pairs and count >= 3]
    unique_top.sort(key=lambda x: -x[1])
    
    for (src, dst), count in unique_top[:15]:
        print(f"  {src:25s} → {dst:25s} ({count}x in top 25%)")
    
    # 결과 저장
    merged.to_csv(f"{BASE}/route_performance.csv", index=False)
    impact_df.to_csv(f"{BASE}/intent_impact.csv", index=False)
    
    print(f"\nSaved:")
    print(f"  {BASE}/route_performance.csv")
    print(f"  {BASE}/intent_impact.csv")


if __name__ == '__main__':
    main()
