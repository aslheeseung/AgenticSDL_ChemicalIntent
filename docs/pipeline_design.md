# OER Catalyst Chemical Intent Clustering Pipeline
# 전체 설계서 v0.1

================================================================================
PHASE 1: DATA EXTRACTION (문장 추출)
================================================================================

Input:  905개 JSON (Elsevier/ACS/RSC/Wiley/Nature)
Output: sentences.csv

Steps:
  1. 모든 JSON에서 합성 관련 서브섹션 추출
     - 키워드 필터: synthesis, prepar, fabricat, synth, method, material
     - 단 characterization, electrochem, measure, instrument는 제외
  2. 텍스트를 문장 단위로 분리
     - 마침표, 느낌표, 물음표 기준
     - 단 "et al.", "e.g.", "i.e.", "Fig.", "°C" 등은 분리 안 되게 예외 처리
     - "2 h", "0.5 g" 같은 숫자+단위 보존
  3. 각 문장에 메타데이터 태깅
     - paper_id, title, journal, doi, section_name
  4. 노이즈 필터링
     - 5단어 미만 문장 제거 (fragment)
     - 500자 초과 문장 제거 (잘못 분리된 것)
     - 참고문헌 인용만 있는 문장 제거 (예: "[12] Smith et al.")

  예상 문장 수: 논문당 ~20-50문장 × 905 = ~18,000-45,000문장

================================================================================
PHASE 2: EMBEDDING (문장 임베딩)
================================================================================

모델 선택 (GPU 없음, CPU 환경):

  옵션 A: sentence-transformers/all-MiniLM-L6-v2
    - 384차원, 경량, 빠름, 일반 텍스트에 강함
    - 단점: 화학 도메인 특화 아님

  옵션 B: allenai/specter2
    - 768차원, 논문 abstract 학습, 학술 텍스트에 특화
    - 화학 용어 이해도 높음

  옵션 C: BAAI/bge-small-en-v1.5
    - 384차원, 경량, MTEB 벤치마크 상위권

  추천: B (specter2) > C (bge-small) > A (MiniLM)
  이유: 화학 실험 공정 문장이니 학술 특화가 중요
  fallback: B가 너무 무거우면 C

  처리:
  - 배치 단위 임베딩 (batch_size=64)
  - CPU 환경에서 905개 문장 ~1-2시간 예상
  - 결과: numpy 배열로 저장 (sentences_embeddings.npy)

================================================================================
PHASE 3: DIMENSIONALITY REDUCTION (차원 축소)
================================================================================

목적: 클러스터링 전 노이즈 제거 + UMAP 시각화

  Step 1: UMAP으로 384/768차원 → 15차원 (클러스터링용)
    - n_neighbors=15 (로컬 구조 강조)
    - min_dist=0.0 (클러스터링에 유리)
    - metric=cosine (텍스트 임베딩에 적합)

  Step 2: UMAP으로 15차원 → 2차원 (시각화용)
    - n_neighbors=15
    - min_dist=0.1
    - 결과: 2D scatter plot

================================================================================
PHASE 4: CLUSTERING (클러스터링)
================================================================================

알고리즘: HDBSCAN
  이유:
    - 클러스터 수 자동 결정 (K-means처럼 K 안 정해도 됨)
    - 노이즈 포인트 처리 ( outlier = -1 라벨)
    - 밀도 기반 → 불균형 클러스터 처리 가능
    - 하이퍼파라미터 직관적

  파라미터:
    - min_cluster_size: 30~50 (전체 문장 수에 따라 조정)
    - min_samples: 5~10
    - metric: euclidean (UMAP 축소 후)

  대안 (비교용):
    - K-means: baseline 비교용
    - Agglomerative: 덴드로그램으로 계층 구조 확인

================================================================================
PHASE 5: CLUSTER INTERPRETATION (클러스터 해석)
================================================================================

각 클러스터에 대해:

  1. 키워드 추출
     - TF-IDF로 클러스터 내 고빈도/특징 단어 추출
     - 각 클러스터 top 10 키워드

  2. 대표 문장 선정
     - 클러스터 중심점에서 가장 가까운 문장 5개
     - 사람이 읽고 intent 판별 가능하게

  3. LLM 어시스트 (선택)
     - 각 클러스터의 대표 문장을 LLM에 넣고
     - "이 문장들의 공통 chemical intent가 뭐야?" 질문
     - intent label 자동 제안

  4. 수동 검증
     - 클러스터별 대표 문장 + 키워드 출력
     - 사람이 최종 intent label 확정

  산출물: clusters_summary.json
    {
      "cluster_id": 0,
      "size": 342,
      "top_keywords": ["dissolved", "solution", "water", "mL"],
      "representative_sentences": [...],
      "suggested_intent": "dissolution",
      "confirmed_intent": null  // 사용자가 나중에 채움
    }

================================================================================
PHASE 6: VISUALIZATION (시각화)
================================================================================

  1. 2D UMAP scatter plot (클러스터별 색상)
     - interactive HTML (plotly)로 저장
     - hover시 문장 내용 보이게
     - 노이즈 포인트는 회색

  2. 클러스터 크기 바 차트

  3. 키워드 워드클라우드 (클러스터별)

  4. 덴드로그램 (Agglomerative 결과)

================================================================================
PHASE 7: SCHEMA FINALIZATION (스키마 확정)
================================================================================

  클러스터링 결과를 바탕으로:
  - 자동 도출된 intent 카테고리 정리
  - 기존 초안 스키마와 비교/병합
  - 최종 Chemical Intent Schema 확정

================================================================================
TECH STACK
================================================================================

  Python 3.10
  ├── sentence-transformers  (임베딩)
  ├── umap-learn             (차원 축소)
  ├── hdbscan                (클러스터링)
  ├── scikit-learn           (TF-IDF, K-means baseline)
  ├── pandas                 (데이터 처리)
  ├── numpy                  (배열 연산)
  ├── plotly                 (인터랙티브 시각화)
  └── matplotlib             (정적 시각화)

  설치 예상 시간: ~5분
  전체 실행 예상 시간: ~1-3시간 (CPU 기준)

================================================================================
OUTPUT FILES
================================================================================

  output/
  ├── sentences.csv                    # 추출된 전체 문장
  ├── sentences_embeddings.npy         # 임베딩 벡터
  ├── umap_2d.npy                      # 2D UMAP 좌표
  ├── umap_15d.npy                     # 15D UMAP (클러스터링용)
  ├── clustering_labels.npy            # HDBSCAN 클러스터 라벨
  ├── clusters_summary.json            # 클러스터별 요약
  ├── interactive_plot.html            # 인터랙티브 시각화
  └── cluster_visualization.png        # 정적 시각화
