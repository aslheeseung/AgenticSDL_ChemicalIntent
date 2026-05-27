"""
Phase 7: LLM-based Chemical Intent Tagging
각 문장의 진짜 화학적 의미를 LLM이 판별

전략:
1. 37개 클러스터별 대표 문장 3개씩 = ~111문장을 먼저 LLM에 태깅
2. LLM이 도출한 intent 카테고리 확인
3. 전체 14,331문장 배치 태깅
"""

import os, json, pandas as pd, numpy as np

BASE = "/home/hs/oer-catalyst-project/output"
SENTENCES_PATH = f"{BASE}/sentences.csv"
SUMMARY_PATH = f"{BASE}/clusters_summary_v3a.json"
INTENT_RESULTS_PATH = f"{BASE}/llm_intent_sample.json"

# 재료공학 합성 공정 Chemical Intent 카테고리 초안
CHEMICAL_INTENT_CATEGORIES = """
You are a materials science expert analyzing synthesis procedures for OER (Oxygen Evolution Reaction) catalysts.

For each sentence, identify its CHEMICAL INTENT - the underlying physicochemical meaning, not just the operational action.

Categories (but you may suggest new ones if needed):

Nucleation & Growth:
- nucleation: formation of initial crystal nuclei from solution/melt
- crystal_growth: growth of existing crystals, Ostwald ripening
- crystallization: formation of crystalline phase (general)
- amorphization: formation of amorphous/disordered phase

Phase Transformation:
- phase_transformation: conversion between crystal phases
- calcination: thermal decomposition / phase formation by heat
- annealing: thermal treatment to modify crystal structure
- sintering: particle coalescence at high temperature
- carbonization: conversion to carbonaceous material
- phosphorization: reaction with phosphorus source
- sulfurization: reaction with sulfur source
- nitridation: reaction with nitrogen source

Chemical Reaction:
- reduction: chemical reduction (by NaBH4, H2, etc.)
- oxidation: oxidation reaction
- precipitation: formation of solid from solution
- coprecipitation: simultaneous precipitation of multiple species
- intercalation: insertion of species between layers
- exfoliation: separation of layered materials into sheets
- etching: selective removal of material
- ion_exchange: exchange of ions in structure
- hydrolysis: reaction with water
- decomposition: thermal or chemical breakdown

Solution Chemistry:
- dissolution: dissolving precursors in solvent
- complexation: formation of coordination complexes
- sol_gel: sol-gel process (gel formation)
- condensation: condensation reaction forming network

Surface/Interface:
- adsorption: adsorption onto surface
- deposition: material deposition (electrodeposition, CVD, etc.)
- coating: applying surface layer
- doping: incorporation of dopant elements
- functionalization: surface functional group modification

Separation & Processing:
- washing: removal of impurities
- drying: removal of solvent
- centrifugation: separation by density
- grinding: mechanical size reduction
- filtration: solid-liquid separation

Electrode Fabrication:
- ink_preparation: catalyst ink formulation
- electrode_fabrication: making the working electrode

Meta/Information:
- reagent_info: information about chemicals used (purchased, grade)
- substrate_preparation: preparing substrate (nickel foam, carbon cloth)
- characterization: analytical measurement description
- measurement_condition: electrochemical test conditions

If the sentence describes multiple intents, pick the PRIMARY one.
If none fits, use "other" and suggest what it should be.

Output format: JSON array of objects with keys:
- "sentence_id": int
- "chemical_intent": string (category name)
- "confidence": float (0-1)
- "reasoning": string (brief explanation, 1 sentence)
"""

def build_prompt_for_sample():
    """클러스터별 대표 문장으로 LLM 프롬프트 구성"""
    df = pd.read_csv(SENTENCES_PATH)
    with open(SUMMARY_PATH) as f:
        summaries = json.load(f)

    # 클러스터별 대표 문장 3개 + 랜덤 2개 = 5개씩
    sample_sentences = []
    sentence_id = 0

    for s in summaries:
        cid = s['cluster_id']
        reps = s['representative_sentences'][:3]
        
        for sent in reps:
            sample_sentences.append({
                'sentence_id': sentence_id,
                'cluster_id': cid,
                'sentence': sent,
            })
            sentence_id += 1

        # 랜덤 2개 추가
        cluster_mask = df['section'].notna()  # placeholder
        # 실제로는 클러스터 라벨 기반으로
    
    print(f"Total sample sentences: {len(sample_sentences)}")
    
    # 프롬프트 구성
    prompt = CHEMICAL_INTENT_CATEGORIES + "\n\nSentences to classify:\n\n"
    for s in sample_sentences:
        prompt += f"[{s['sentence_id']}] (Cluster {s['cluster_id']}): {s['sentence']}\n"
    
    prompt += "\n\nClassify each sentence. Return ONLY a JSON array."
    
    return prompt, sample_sentences


if __name__ == '__main__':
    prompt, samples = build_prompt_for_sample()
    
    # 프롬프트를 파일로 저장 (LLM에 직접 입력용)
    with open(f"{BASE}/llm_prompt_sample.txt", 'w') as f:
        f.write(prompt)
    
    print(f"\nPrompt saved to {BASE}/llm_prompt_sample.txt")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Sample sentences: {len(samples)}")
    
    # 첫 5개 미리보기
    print("\nFirst 5 samples:")
    for s in samples[:5]:
        print(f"  [{s['sentence_id']}] C{s['cluster_id']}: {s['sentence'][:80]}...")
