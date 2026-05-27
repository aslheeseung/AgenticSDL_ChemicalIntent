"""
Phase 7c: Chemical Intent 태깅 - 클러스터별 Intent 확정 후 전체 문장에 적용
"""
import json, os
import numpy as np
import pandas as pd

BASE = "/home/hs/oer-catalyst-project/output"
SENTENCES_PATH = f"{BASE}/sentences.csv"
LABELS_PATH = f"{BASE}/clustering_labels_v3a.npy"
OUTPUT_PATH = f"{BASE}/sentences_with_intent.csv"

# ============================================================
# 클러스터별 Chemical Intent 매핑
# 37개 클러스터를 재료공학 관점에서 분류
# ============================================================
CLUSTER_INTENT_MAP = {
    # C0: 시약 구매 정보 (Co(NO3)2, Fe(NO3)3, KOH 등 리스트)
    0: {
        "intent": "reagent_info",
        "sub_intent": "precursor_listing",
        "reasoning": "Lists chemical reagents with grades and suppliers. Not a process step, just chemical inventory.",
    },
    # C1: 증류수 사용
    1: {
        "intent": "reagent_info",
        "sub_intent": "solvent_info",
        "reasoning": "States deionized water was used throughout. Solvent specification, not a process action.",
    },
    # C2: chemicals used as received
    2: {
        "intent": "reagent_info",
        "sub_intent": "purity_statement",
        "reasoning": "Statement about chemical purity and usage without further purification.",
    },
    # C3: analytical grade, without purification
    3: {
        "intent": "reagent_info",
        "sub_intent": "purity_statement",
        "reasoning": "Statement about analytical grade chemicals without further purification.",
    },
    # C4: used as received without purification
    4: {
        "intent": "reagent_info",
        "sub_intent": "purity_statement",
        "reasoning": "Statement about chemicals used as received without further purification.",
    },
    # C5: nickel foam cleaning (HCl, acetone, ethanol)
    5: {
        "intent": "substrate_preparation",
        "sub_intent": "surface_cleaning",
        "reasoning": "Cleaning nickel foam substrate with HCl, acetone, ethanol to remove impurities and oxides. Pretreatment before catalyst deposition.",
    },
    # C6: centrifuged, washed, dried
    6: {
        "intent": "separation",
        "sub_intent": "washing_drying",
        "reasoning": "Post-synthesis separation: centrifugation to collect product, washing with water/ethanol, drying. Standard workup procedure.",
    },
    # C7: elemental analysis (ICP, EDS)
    7: {
        "intent": "characterization",
        "sub_intent": "elemental_analysis",
        "reasoning": "ICP-AES, EDS elemental composition analysis. Characterization, not synthesis.",
    },
    # C8: graphene oxide Hummers method
    8: {
        "intent": "oxidation",
        "sub_intent": "graphene_oxide_synthesis",
        "reasoning": "Hummers method is a strong oxidation of graphite to graphene oxide using KMnO4/H2SO4. This is a chemical oxidation process.",
    },
    # C9: MXene etching
    9: {
        "intent": "etching",
        "sub_intent": "mxene_synthesis",
        "reasoning": "Selective etching of Al from Ti3AlC2 MAX phase using HF or HCl/LiF to produce Ti3C2Tx MXene. Intercalation/exfoliation follows.",
    },
    # C10: catalyst ink preparation (Nafion, solvent)
    10: {
        "intent": "ink_preparation",
        "sub_intent": "catalyst_ink",
        "reasoning": "Dispersing catalyst in Nafion/solvent mixture to prepare ink for electrode coating. Precursor to electrode fabrication.",
    },
    # C11: drop-casting onto electrode
    11: {
        "intent": "electrode_fabrication",
        "sub_intent": "drop_casting",
        "reasoning": "Depositing catalyst ink onto glassy carbon electrode by drop casting. Final electrode preparation step.",
    },
    # C12: boride SSM reactions
    12: {
        "intent": "phase_transformation",
        "sub_intent": "solid_state_reaction",
        "reasoning": "Solid-state metathesis (SSM) reactions between metal chlorides and MgB2 to form metal borides. High-temperature phase formation.",
    },
    # C13: ZIF-67, ZIF-8 synthesis
    13: {
        "intent": "crystallization",
        "sub_intent": "mof_crystallization",
        "reasoning": "Formation of ZIF (zeolitic imidazolate framework) crystals by reacting metal ions with 2-methylimidazole. Nucleation and crystallization of MOF structure.",
    },
    # C14: RuO2 prepared (reference method)
    14: {
        "intent": "reference_synthesis",
        "sub_intent": "benchmark_preparation",
        "reasoning": "Brief mention of RuO2 preparation by reported method. Reference/benchmark catalyst, no detailed process.",
    },
    # C15: Ir-based catalyst synthesis
    15: {
        "intent": "crystallization",
        "sub_intent": "noble_metal_oxide_formation",
        "reasoning": "Synthesis of Ir-based oxides (IrO2, IrxRu1-xO2) via hydrothermal, sol-gel, or template methods. Formation of noble metal oxide crystal phase.",
    },
    # C16: "synthesized as described", general
    16: {
        "intent": "reference_synthesis",
        "sub_intent": "general_preparation",
        "reasoning": "Generic statements about synthesis/preparation without specific process details. Cross-references to other methods.",
    },
    # C17: autoclave, Teflon-lined, heated
    17: {
        "intent": "crystallization",
        "sub_intent": "hydrothermal_crystallization",
        "reasoning": "Transferring solution to autoclave and heating at high temperature (120-200°C) for extended time. This is hydrothermal treatment where nucleation and crystal growth occur under pressure.",
    },
    # C18: color change observation
    18: {
        "intent": "characterization",
        "sub_intent": "visual_observation",
        "reasoning": "Describing color changes during reaction - visual observation of phase/composition change. In-situ characterization.",
    },
    # C19: electrochemical measurement conditions
    19: {
        "intent": "characterization",
        "sub_intent": "electrochemical_measurement",
        "reasoning": "Electrochemical test setup (three-electrode, KOH electrolyte, OER measurement). Not synthesis, performance evaluation.",
    },
    # C20: MoSe2, Mo2C characterization
    20: {
        "intent": "characterization",
        "sub_intent": "morphology_analysis",
        "reasoning": "Description of MoSe2/Mo2C flake thickness, crystal structure analysis via XRD/Raman. Material characterization.",
    },
    # C21: Ir, IrO2 structure discussion
    21: {
        "intent": "characterization",
        "sub_intent": "structure_analysis",
        "reasoning": "Analysis of Ir-based nanoparticle structure, composition, crystal phase via TEM-EDX, XRD. Characterization.",
    },
    # C22: SEM, TEM morphology
    22: {
        "intent": "characterization",
        "sub_intent": "morphology_analysis",
        "reasoning": "SEM and TEM imaging for morphology analysis. Standard characterization technique.",
    },
    # C23: XRD patterns
    23: {
        "intent": "characterization",
        "sub_intent": "crystal_structure_analysis",
        "reasoning": "X-ray diffraction analysis of crystal structure and phase identification. Characterization.",
    },
    # C24: dispersed in solution (dissolution + mixing)
    24: {
        "intent": "dissolution",
        "sub_intent": "dispersion_mixing",
        "reasoning": "Dispersing or dissolving precursors in solvent with ultrasonication. Initial step where reactants are brought into solution for subsequent reaction.",
    },
    # C25: sonicated for X min
    25: {
        "intent": "mixing",
        "sub_intent": "sonication",
        "reasoning": "Ultrasonication to ensure homogeneous mixing/dispersion. Physical process aiding dissolution and mixing.",
    },
    # C26: XPS measurement
    26: {
        "intent": "characterization",
        "sub_intent": "surface_composition_analysis",
        "reasoning": "X-ray photoelectron spectroscopy for surface elemental composition and chemical state analysis.",
    },
    # C27: XPS peak fitting (binding energy)
    27: {
        "intent": "characterization",
        "sub_intent": "chemical_state_analysis",
        "reasoning": "Detailed XPS peak deconvolution and binding energy assignment for chemical state identification.",
    },
    # C28: Raman, FT-IR spectroscopy
    28: {
        "intent": "characterization",
        "sub_intent": "vibrational_spectroscopy",
        "reasoning": "Raman and FT-IR spectroscopy analysis. Bond vibration analysis for structural identification.",
    },
    # C29: Hummers method (graphite + KMnO4 + H2SO4)
    29: {
        "intent": "oxidation",
        "sub_intent": "graphite_oxidation",
        "reasoning": "Hummers method: strong oxidation of graphite using KMnO4 in concentrated H2SO4. Aggressive chemical oxidation to produce graphene oxide.",
    },
    # C30: tube furnace, annealed, N2/Ar atmosphere
    30: {
        "intent": "phase_transformation",
        "sub_intent": "thermal_annealing_phosphorization",
        "reasoning": "Tube furnace heat treatment under inert atmosphere. Often used for annealing, phosphorization (NaH2PO2), carbonization, or phase transformation. High-temperature crystal structure modification.",
    },
    # C31: cooled to room temperature
    31: {
        "intent": "cooling",
        "sub_intent": "natural_cooling",
        "reasoning": "Cooling reaction product to room temperature. Post-synthesis thermal management. Can affect crystallinity and phase.",
    },
    # C32: dissolved precursors (mmol, g, solution)
    32: {
        "intent": "dissolution",
        "sub_intent": "precursor_dissolution",
        "reasoning": "Dissolving metal precursors (nitrates, chlorides) in solvent with specified amounts. First step of most solution-based synthesis routes. Creates the reaction medium.",
    },
    # C33: ZIF synthesis (2-methylimidazole + metal salt in methanol)
    33: {
        "intent": "crystallization",
        "sub_intent": "mof_crystallization",
        "reasoning": "Mixing 2-methylimidazole with metal salts in methanol for ZIF crystallization. Nucleation begins upon mixing ligand and metal source.",
    },
    # C34: solution added dropwise (NaBH4, NaOH, etc.)
    34: {
        "intent": "reduction",
        "sub_intent": "controlled_addition",
        "reasoning": "Dropwise addition of reagents (NaBH4 is a reducing agent, NaOH for pH). Controlled addition triggers nucleation or precipitation. NaBH4 specifically reduces metal ions to nanoparticles.",
    },
    # C35: stirred at room temperature
    35: {
        "intent": "mixing",
        "sub_intent": "stirring",
        "reasoning": "Stirring solution at room temperature for extended period. Promotes homogeneity, can allow slow nucleation or aging.",
    },
    # C36: pH adjustment (NaOH, NH3·H2O)
    36: {
        "intent": "precipitation",
        "sub_intent": "ph_adjustment",
        "reasoning": "Adjusting pH with NaOH or NH3·H2O. pH control is critical for controlling precipitation, nucleation rate, and product morphology.",
    },
}


def main():
    # 데이터 로드
    df = pd.read_csv(SENTENCES_PATH)
    labels = np.load(LABELS_PATH)
    df['cluster'] = labels

    # Intent 매핑 적용
    df['chemical_intent'] = df['cluster'].map(
        lambda c: CLUSTER_INTENT_MAP.get(c, {}).get('intent', 'unknown')
    )
    df['sub_intent'] = df['cluster'].map(
        lambda c: CLUSTER_INTENT_MAP.get(c, {}).get('sub_intent', 'unknown')
    )
    df['intent_reasoning'] = df['cluster'].map(
        lambda c: CLUSTER_INTENT_MAP.get(c, {}).get('reasoning', '')
    )

    # 결과 저장
    df.to_csv(OUTPUT_PATH, index=False)

    # 통계
    print("=== Chemical Intent Distribution ===\n")
    intent_counts = df['chemical_intent'].value_counts()
    for intent, count in intent_counts.items():
        pct = count / len(df) * 100
        print(f"  {intent:30s} {count:5d} ({pct:5.1f}%)")

    print(f"\n=== Sub-Intent Distribution ===\n")
    sub_counts = df['sub_intent'].value_counts()
    for sub, count in sub_counts.items():
        pct = count / len(df) * 100
        print(f"  {sub:35s} {count:5d} ({pct:5.1f}%)")

    # Intent별 클러스터 매핑 요약
    print(f"\n=== Cluster → Intent Mapping ===\n")
    for cid in sorted(CLUSTER_INTENT_MAP.keys()):
        info = CLUSTER_INTENT_MAP[cid]
        size = (df['cluster'] == cid).sum()
        print(f"  C{cid:2d} ({size:4d}) → {info['intent']:25s} / {info['sub_intent']}")

    # unknown 확인
    unknown = df[df['chemical_intent'] == 'unknown']
    if len(unknown) > 0:
        print(f"\nWARNING: {len(unknown)} sentences with 'unknown' intent")
        print(f"  Cluster IDs: {unknown['cluster'].unique()}")

    # JSON으로도 저장
    mapping_output = []
    for cid, info in sorted(CLUSTER_INTENT_MAP.items()):
        size = (df['cluster'] == cid).sum()
        mapping_output.append({
            'cluster_id': cid,
            'size': size,
            'chemical_intent': info['intent'],
            'sub_intent': info['sub_intent'],
            'reasoning': info['reasoning'],
        })

    with open(f"{BASE}/cluster_intent_mapping.json", 'w') as f:
        json.dump(mapping_output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Saved: {BASE}/cluster_intent_mapping.json")


if __name__ == '__main__':
    main()
