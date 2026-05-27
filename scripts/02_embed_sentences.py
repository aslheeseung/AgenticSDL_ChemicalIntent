"""
Phase 2: Sentence Embedding with SPECTER2
문장을 벡터로 변환하여 numpy 배열로 저장
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ============================================================
# 설정
# ============================================================
SENTENCES_PATH = "/home/hs/oer-catalyst-project/output/sentences.csv"
EMBEDDINGS_PATH = "/home/hs/oer-catalyst-project/output/sentences_embeddings.npy"

# BAAI/bge-small-en-v1.5 → 경량, 벤치마크 상위권, 호환성 좋음
# SPECTER2는 PEFT 이슈로 보류
MODEL_NAME = "BAAI/bge-small-en-v1.5"

BATCH_SIZE = 32

# ============================================================
# 메인
# ============================================================
def embed_sentences():
    # 데이터 로드
    print("Loading sentences...")
    df = pd.read_csv(SENTENCES_PATH)
    sentences = df['sentence'].tolist()
    print(f"Total sentences: {len(sentences)}")

    # 모델 로드
    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded. Dimension: {model.get_sentence_embedding_dimension()}")

    # 임베딩
    print(f"\nEncoding {len(sentences)} sentences (batch_size={BATCH_SIZE})...")
    print("This will take a while on CPU...")

    embeddings = model.encode(
        sentences,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity 위해 정규화
    )

    # 저장
    embeddings = np.array(embeddings)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"\nSaved embeddings: {embeddings.shape} to {EMBEDDINGS_PATH}")
    print(f"File size: {os.path.getsize(EMBEDDINGS_PATH) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    embed_sentences()
