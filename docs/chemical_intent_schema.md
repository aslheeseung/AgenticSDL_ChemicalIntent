# OER Catalyst Synthesis - Chemical Intent Schema v0.1

## Intent Categories

### 1. Precursor Preparation (전구체 준비)
| Intent | Description | Example |
|--------|-------------|---------|
| `dissolution` | 용해 - 시약을 용매에 녹임 | "Ni(NO3)2·6H2O was dissolved in 35 mL DI water" |
| `mixing` | 혼합 - 두 용액/물질을 섞음 | "solution A was mixed with solution B" |
| `weighing` | 무게 측정 | "0.6 g NaNO3 and 0.4 g KNO3" |
| `solution_preparation` | 용액 제조 | "0.1 M nickel nitrate solution was prepared" |

### 2. Reaction/Process (반응/공정)
| Intent | Description | Example |
|--------|-------------|---------|
| `addition` | 첨가/적가 - 한 물질을 다른 곳에 넣음 | "NaOH solution was added dropwise" |
| `stirring` | 교반 | "under vigorous stirring for 30 min" |
| `heating` | 가열 (일반) | "heated at 80°C" |
| `hydrothermal` | 수열합성 | "transferred to autoclave and heated at 180°C" |
| `solvothermal` | 용매열합성 | |
| `calcination` | 소성/하소 | "calcined at 500°C for 2 h" |
| `annealing` | 어닐링/열처리 | "annealed at 300°C in Ar atmosphere" |
| `pyrolysis` | 열분해 | "pyrolyzed at 800°C" |
| `microwave` | 마이크로웨이브 처리 | "irradiated at 300 W" |
| `electrodeposition` | 전기증착 | "electrodeposited at -1.0 V" |
| `coprecipitation` | 공침 | "co-precipitated by adding KOH" |
| `reduction` | 환원 | "reduced by NaBH4" |
| `oxidation` | 산화 | |
| `carbonization` | 탄화 | |
| `freeze_drying` | 동결건조 | "freeze-dried for 24 h" |
| `evaporation` | 증발/건조(액체) | "evaporated to dryness" |

### 3. Separation/Purification (분리/정제)
| Intent | Description | Example |
|--------|-------------|---------|
| `centrifugation` | 원심분리 | "centrifuged at 8000 rpm for 5 min" |
| `filtration` | 여과 | "filtered and washed" |
| `washing` | 세척 | "washed with ethanol and DI water" |
| `drying` | 건조 | "dried at 80°C for 12 h" |
| `drying_vacuum` | 진공건조 | "vacuum dried at 60°C" |
| `grinding` | 분쇄/막자사발 | "ground in a mortar with pestle" |

### 4. Modification (표면처리/변형)
| Intent | Description | Example |
|--------|-------------|---------|
| `etching` | 에칭 | "etched with HF solution" |
| `doping` | 도핑 | "N-doped by annealing in NH3" |
| `coating` | 코팅 | "coated with a thin layer of..." |
| `functionalization` | 표면기능화 | |

### 5. Electrode Fabrication (전극 제작)
| Intent | Description | Example |
|--------|-------------|---------|
| `ink_preparation` | 촉매 잉크 제조 | "dispersed in Nafion/water/ethanol" |
| `drop_casting` | 드롭캐스팅 | "drop-casted onto glassy carbon" |
| `electrode_fabrication` | 전극 제작 (일반) | |

### 6. Characterization (특성분석) - reference only
| Intent | Description |
|--------|-------------|
| `characterization` | 분석 장비 측정 (XRD, SEM, TEM 등) |

---

## Tagging Format (per sentence)

```json
{
  "paper_id": "Elsevier_OER_02946",
  "section": "2.2. Synthesis of RuO2-250 nanosheet",
  "sentence": "0.6 g NaNO3 and 0.4 g KNO3 were dissolved in 5 mL water at 60 ℃",
  "chemical_intent": "dissolution",
  "entities": {
    "precursors": ["NaNO3", "KNO3"],
    "solvent": "water",
    "quantities": ["0.6 g", "0.4 g", "5 mL"],
    "conditions": {"temperature": "60 ℃"}
  }
}
```
