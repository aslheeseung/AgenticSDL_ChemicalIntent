#!/bin/bash
# ============================================================
# OER Catalyst SDL — Setup & Reproduce
# 다른 컴퓨터에서 처음부터 전체 파이프라인을 재현하는 스크립트
#
# 사용법:
#   git clone git@github.com:aslheeseung/AgenticSDL_ChemicalIntent.git
#   cd AgenticSDL_ChemicalIntent
#   bash setup.sh
#
# 필요:
#   - Python 3.10+
#   - 인터넷 연결 (Semantic Scholar API)
#   - 디스크 1GB+
#   - .env에 OPENAI_API_KEY (멀티에이전트용)
# ============================================================

set -e

echo "============================================================"
echo "  OER Catalyst SDL — Setup"
echo "============================================================"

# ----- 1. Python 패키지 -----
echo ""
echo "[1/5] 패키지 설치..."
pip install --user \
    sentence-transformers \
    umap-learn \
    hdbscan \
    scikit-learn \
    pandas \
    numpy \
    plotly \
    matplotlib \
    pyautogen \
    "autogen-ext[openai]" \
    python-dotenv \
    requests \
    2>&1 | tail -5

# ----- 2. 원본 데이터 다운로드 -----
echo ""
echo "[2/6] 원본 데이터 (905편 OER 논문 JSON)..."
if [ ! -d "data/raw" ] || [ -z "$(ls -A data/raw/ 2>/dev/null)" ]; then
    if [ -f "data/oer_raw_data.tar.gz" ]; then
        echo "  압축 해제 중..."
        tar -xzf data/oer_raw_data.tar.gz -C data/
        mv data/hsdata data/raw 2>/dev/null || true
        echo "  ✓ 905편 JSON 압축 해제 완료"
    else
        echo "  Hugging Face에서 다운로드 중..."
        pip install --user huggingface_hub 2>/dev/null
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Heeseung123/oer-catalyst-lit', local_dir='hf_data', allow_patterns=['raw/*'])
" 2>/dev/null
        if [ -f "hf_data/raw/oer_raw_data.tar.gz" ]; then
            mkdir -p data/raw
            tar -xzf hf_data/raw/oer_raw_data.tar.gz -C data/
            mv data/hsdata data/raw 2>/dev/null || true
            rm -rf hf_data
            echo "  ✓ 905편 JSON 다운로드 + 압축 해제 완료"
        else
            echo "  ⚠️  다운로드 실패. Semantic Scholar API로 새 논문만 수집합니다."
        fi
    fi
fi

# ----- 3. .env 확인 -----
echo ""
echo "[3/6] .env 확인..."
if [ ! -f .env ]; then
    echo "OPENAI_API_KEY=your_key_here" > .env
    echo "  ⚠️  .env 생성됨. OPENAI_API_KEY를 입력하세요:"
    echo "      nano .env"
fi

# ----- 4. 논문 수집 (Semantic Scholar) -----
echo ""
echo "[4/6] 논문 수집 (Semantic Scholar API)..."
echo "  최대 500편, 약 5-10분 소요"
python3 scripts/paper_harvester.py --max 500

# ----- 5. 논문 분석 (클러스터링) -----
echo ""
echo "[5/6] 논문 분석 (임베딩 + 클러스터링)..."
echo "  약 3-5분 소요"
python3 scripts/analyze_papers.py

# ----- 6. 완료 -----
echo ""
echo "[6/6] 완료!"
echo "  Setup 완료!"
echo ""
echo "  생성된 파일:"
echo "    output/papers/oer_papers_db.json    (논문 DB)"
echo "    output/papers/sentences_clustered.csv (클러스터링)"
echo "    output/papers/cluster_scatter.html   (시각화)"
echo "    output/papers/pdfs/                  (PDF)"
echo ""
echo "  다음 단계:"
echo "    python3 multiagent/sdl_pipeline.py 'NiFe LDH'"
echo "    python3 multiagent/sdl_pipeline.py 'CsPbBr3' --external"
echo "============================================================"
