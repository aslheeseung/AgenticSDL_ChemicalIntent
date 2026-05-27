
import numpy as np, pandas as pd, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

embeddings = np.load('/home/hs/oer-catalyst-project/output/sentences_embeddings.npy')
labels = np.load('/home/hs/oer-catalyst-project/output/clustering_labels_v2.npy')
df = pd.read_csv('/home/hs/oer-catalyst-project/output/sentences.csv')
sentences = df['sentence'].tolist()

noise_mask = labels == -1
print(f'Before reassignment: {(labels==-1).sum()} noise points')

cluster_ids = sorted(set(labels) - {-1})
centroids = {c: embeddings[labels==c].mean(axis=0) for c in cluster_ids}

for idx in np.where(noise_mask)[0]:
    emb = embeddings[idx].reshape(1, -1)
    best_c, best_sim = -1, -1
    for c, cent in centroids.items():
        sim = cosine_similarity(emb, cent.reshape(1, -1))[0, 0]
        if sim > best_sim:
            best_sim = sim
            best_c = c
    labels[idx] = best_c

print(f'After reassignment: {(labels==-1).sum()} noise points')
np.save('/home/hs/oer-catalyst-project/output/clustering_labels_merged.npy', labels)

summaries = []
for c in cluster_ids:
    mask = labels == c
    cluster_sents = [s for s, m in zip(sentences, mask) if m]
    cluster_embs = embeddings[mask]
    
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1,2), token_pattern=r'[A-Za-z0-9]+[A-Za-z0-9\.\-]*')
    tfidf = vectorizer.fit_transform(cluster_sents)
    features = vectorizer.get_feature_names_out()
    mean_tfidf = tfidf.mean(axis=0).A1
    top_kw = [features[i] for i in mean_tfidf.argsort()[-15:][::-1]]
    
    centroid = cluster_embs.mean(axis=0, keepdims=True)
    sims = cosine_similarity(cluster_embs, centroid).flatten()
    top_idx = sims.argsort()[-5:][::-1]
    rep = [cluster_sents[i] for i in top_idx]
    
    summaries.append({
        'cluster_id': int(c),
        'size': int(mask.sum()),
        'top_keywords': top_kw,
        'representative_sentences': rep,
    })
    
    print(f'\n{"="*60}')
    print(f'Cluster {c} ({mask.sum()} sentences)')
    print(f'  Keywords: {", ".join(top_kw[:10])}')
    for j, s in enumerate(rep[:3]):
        print(f'  Rep {j+1}: {s[:140]}...')

with open('/home/hs/oer-catalyst-project/output/clusters_summary_v2.json', 'w') as f:
    json.dump(summaries, f, indent=2, ensure_ascii=False)
print(f'\nSaved: output/clusters_summary_v2.json')
