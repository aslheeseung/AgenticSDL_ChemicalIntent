"""
Phase 7b: LLM-based Chemical Intent Tagging (Full Pipeline)
14,331문장을 배치 단위로 LLM에 태깅 요청

배치 전략:
- 클러스터별로 묶어서 처리 (동일 클러스터는 문맥 비슷)
- 각 배치: ~50문장씩
- LLM 응답 파싱 → 전체 결과 저장
"""

import os, json, pandas as pd, numpy as np, time, re

BASE = "/home/hs/oer-catalyst-project/output"
SENTENCES_PATH = f"{BASE}/sentences.csv"
LABELS_PATH = f"{BASE}/clustering_labels_v3a.npy"
RESULTS_PATH = f"{BASE}/chemical_intent_results.json"

SYSTEM_PROMPT = """You are a materials science expert specializing in electrocatalyst synthesis for OER (Oxygen Evolution Reaction).

Your task: classify each sentence by its CHEMICAL INTENT — the underlying physicochemical meaning, not just the operational action.

INTENT CATEGORIES:

1. nucleation - formation of initial crystal nuclei
2. crystal_growth - growth of existing crystals, Ostwald ripening  
3. crystallization - formation of crystalline phase (general)
4. amorphization - formation of amorphous/disordered phase
5. phase_transformation - conversion between crystal phases
6. calcination - thermal decomposition or phase formation by heat
7. annealing - thermal treatment to modify structure/stress
8. sintering - particle coalescence at high temperature
9. carbonization - conversion to carbonaceous material
10. phosphorization - reaction with phosphorus source (NaH2PO2, etc.)
11. sulfurization - reaction with sulfur source
12. nitridation - reaction with nitrogen source (NH3, etc.)
13. reduction - chemical reduction (NaBH4, H2, ascorbic acid, etc.)
14. oxidation - oxidation reaction
15. precipitation - formation of solid from solution
16. coprecipitation - simultaneous precipitation of multiple species
17. intercalation - insertion of species between layers
18. exfoliation - separation of layered materials into sheets
19. etching - selective removal of material (HF, acid etching)
20. ion_exchange - exchange of ions in structure
21. hydrolysis - reaction with water breaking bonds
22. decomposition - thermal or chemical breakdown
23. dissolution - dissolving precursors in solvent
24. complexation - formation of coordination complexes
25. sol_gel - sol-gel process
26. condensation - condensation reaction
27. adsorption - adsorption onto surface
28. deposition - electrodeposition, CVD, sputtering
29. coating - applying surface layer
30. doping - incorporation of dopant elements
31. functionalization - surface functional group modification
32. washing - removal of impurities
33. drying - removal of solvent
34. centrifugation - separation by density
35. grinding - mechanical size reduction
36. filtration - solid-liquid separation
37. ink_preparation - catalyst ink formulation
38. electrode_fabrication - making the working electrode
39. reagent_info - info about chemicals purchased/used
40. substrate_preparation - preparing substrate
41. characterization - analytical measurement (XRD, SEM, TEM, XPS, etc.)
42. measurement_condition - electrochemical test setup
43. mixing - general mixing of solutions/components
44. stirring - stirring operation
45. sonication - ultrasonication treatment
46. heating - general heating (when specific intent unclear)
47. cooling - cooling to room temperature
48. pH_adjustment - adjusting pH of solution
49. other - if none fits

RULES:
- Pick the SINGLE most important chemical intent
- If a sentence is purely informational (e.g., "Deionized water was used"), classify as reagent_info
- If a sentence describes characterization (XRD, SEM, etc.), use characterization
- Be specific: prefer "calcination" over "heating" when heating creates a new phase
- Prefer "precipitation" over "mixing" when a solid product forms

OUTPUT FORMAT: Return ONLY a valid JSON array. Each element:
{"id": <number>, "intent": "<category>", "confidence": <0.0-1.0>}"""


def call_llm(prompt, max_retries=3):
    """LLM API 호출 (현재 모델 사용)"""
    # Hermes 환경에서는 직접 LLM을 호출할 수 없으므로
    # 결과를 파일로 출력하여 상위 에이전트가 처리하도록 함
    pass


def build_batch_prompt(batch_sentences):
    """배치 문장으로 프롬프트 구성"""
    lines = []
    for i, row in enumerate(batch_sentences):
        lines.append(f"[{i}] {row['sentence']}")
    
    user_prompt = "Classify these sentences:\n\n" + "\n".join(lines)
    user_prompt += "\n\nReturn ONLY the JSON array. No other text."
    return user_prompt


def parse_llm_response(response_text, batch_size):
    """LLM 응답 파싱"""
    # JSON 배열 추출 시도
    try:
        # ```json ... ``` 블록 제거
        text = response_text.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        
        results = json.loads(text.strip())
        if isinstance(results, list):
            return results
    except:
        pass
    
    # 폴백: 한 줄씩 파싱
    results = []
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if line.startswith('{'):
            try:
                results.append(json.loads(line.rstrip(',')))
            except:
                pass
    
    return results if results else None


def main():
    # 데이터 로드
    df = pd.read_csv(SENTENCES_PATH)
    labels = np.load(LABELS_PATH)
    df['cluster'] = labels
    
    print(f"Total sentences: {len(df)}")
    print(f"Unique clusters: {df['cluster'].nunique()}")
    
    # 배치 준비 (클러스터 순서대로, 50개씩)
    BATCH_SIZE = 50
    all_results = []
    
    # 클러스터별로 정렬
    df_sorted = df.sort_values('cluster').reset_index(drop=True)
    
    n_batches = (len(df_sorted) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Batches: {n_batches}")
    
    # 프롬프트 파일들 저장 (LLM 처리용)
    os.makedirs(f"{BASE}/llm_batches", exist_ok=True)
    
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(df_sorted))
        batch = df_sorted.iloc[start:end]
        
        prompt = build_batch_prompt(batch.to_dict('records'))
        
        batch_file = f"{BASE}/llm_batches/batch_{batch_idx:03d}_prompt.txt"
        with open(batch_file, 'w') as f:
            f.write(f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{prompt}")
    
    print(f"\nSaved {n_batches} batch prompts to {BASE}/llm_batches/")
    print("Ready for LLM processing.")


if __name__ == '__main__':
    main()
