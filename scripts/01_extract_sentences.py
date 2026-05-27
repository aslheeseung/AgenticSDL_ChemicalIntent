"""
Phase 1: OER Catalyst Synthesis Sentence Extractor
905개 JSON에서 합성 공정 문장을 추출하여 CSV로 저장
"""

import json
import os
import glob
import re
import csv
import pandas as pd

# ============================================================
# 설정
# ============================================================
DATA_DIR = "/home/hs/oer-catalyst-project/data/raw"
OUTPUT_PATH = "/home/hs/oer-catalyst-project/output/sentences.csv"

# 합성 관련 섹션 키워드 (포함)
SECTION_INCLUDE = [
    'synthesis', 'synth', 'prepar', 'fabricat',
    'material', 'method', 'chemical', 'reagent',
]

# 제외할 서브섹션 키워드 (characterization/전기화학 측정 등)
SUBSECTION_EXCLUDE = [
    'characteriz', 'instrument', 'electrochem', 'measure',
    'computational', 'dft', 'theoretical', 'calculation',
    'xrd', 'xps', 'sem', 'tem', 'ftir', 'raman',
    'xicorr', 'eis', 'cv', 'lsv', 'polarograph',
]

# 문장 분리 시 예외 처리할 약어들
ABBREVIATIONS = [
    r'et\s+al\.', r'e\.g\.', r'i\.e\.', r'Fig\.', r'eq\.',
    r'vs\.', r'approx\.', r'ca\.', r'No\.', r'vol\.',
    r'pH', r'\d+\.\d+', r'wt\.', r'at\.', r'ref\.',
]

# ============================================================
# 문장 분리기
# ============================================================
def split_sentences(text):
    """텍스트를 문장 단위로 분리"""
    if not isinstance(text, str) or len(text.strip()) < 10:
        return []

    # 약어 보호 (임시 치환)
    protected = text
    placeholders = {}
    for i, abbr in enumerate(ABBREVIATIONS):
        matches = re.findall(abbr, protected)
        for m in matches:
            key = f"__ABBR{i}_{len(placeholders)}__"
            protected = protected.replace(m, key, 1)
            placeholders[key] = m

    # °C 보호
    protected = protected.replace('°C', '__DEGC__')
    protected = protected.replace('℃', '__DEGC__')

    # 문장 분리: 마침표 + 공백 + 대문자
    # 또는 마침표 + 줄바꿈
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\(])', protected)

    # 복원
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        for key, val in placeholders.items():
            s = s.replace(key, val)
        s = s.replace('__DEGC__', '°C')
        if len(s) > 0:
            sentences.append(s)

    return sentences


def is_valid_sentence(s):
    """노이즈 문장 필터링"""
    word_count = len(s.split())
    if word_count < 5:
        return False
    if len(s) > 500:
        return False
    # 참고문헌만 있는 문장 제거
    if re.match(r'^\[\d+\]', s) and word_count < 15:
        return False
    # 숫자만 있는 문장 제거
    if re.match(r'^[\d\s\-.,]+$', s):
        return False
    return True


# ============================================================
# 합성 관련 서브섹션 판별
# ============================================================
def is_synthesis_section(section_name):
    """섹션이 합성 관련인지 판별"""
    name_lower = section_name.lower()
    # 포함 키워드 체크
    has_include = any(kw in name_lower for kw in SECTION_INCLUDE)
    if not has_include:
        return False
    # 제외 키워드 체크
    has_exclude = any(kw in name_lower for kw in SUBSECTION_EXCLUDE)
    return not has_exclude


# ============================================================
# 메인 추출
# ============================================================
def extract_all_sentences():
    all_json = glob.glob(os.path.join(DATA_DIR, "**/*.json"), recursive=True)
    print(f"Found {len(all_json)} JSON files")

    all_sentences = []
    errors = []

    for i, fpath in enumerate(all_json):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{len(all_json)}...")

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            errors.append((fpath, str(e)))
            continue

        paper_id = os.path.basename(fpath).replace('.json', '')
        metadata = data.get('metadata', {})
        title = metadata.get('title', '')
        journal = metadata.get('journal', '')
        doi = metadata.get('doi', '')
        doc = data.get('document', {})

        # 모든 최상위 섹션에서 합성 관련 찾기
        for section_name, section_content in doc.items():
            if isinstance(section_content, dict):
                # 서브섹션 순회
                for sub_name, sub_content in section_content.items():
                    if is_synthesis_section(sub_name):
                        text = sub_content if isinstance(sub_content, str) else ''
                        if isinstance(sub_content, list):
                            text = ' '.join(str(x) for x in sub_content)
                        if not text:
                            continue

                        sentences = split_sentences(text)
                        for sent in sentences:
                            if is_valid_sentence(sent):
                                all_sentences.append({
                                    'paper_id': paper_id,
                                    'title': title,
                                    'journal': journal,
                                    'doi': doi,
                                    'section': sub_name,
                                    'sentence': sent,
                                })

    print(f"\nExtracted {len(all_sentences)} valid sentences from {len(all_json) - len(errors)} papers")
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:5]:
            print(f"  {e[0]}: {e[1]}")

    return pd.DataFrame(all_sentences)


if __name__ == '__main__':
    df = extract_all_sentences()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")

    # 통계
    print(f"\nUnique papers: {df['paper_id'].nunique()}")
    print(f"Unique sections: {df['section'].nunique()}")
    print(f"Unique journals: {df['journal'].nunique()}")
    print(f"\nJournal distribution (top 10):")
    print(df['journal'].value_counts().head(10))
