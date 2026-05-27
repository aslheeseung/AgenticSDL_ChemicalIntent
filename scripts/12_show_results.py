import pandas as pd
import json

df = pd.read_csv('output/sentences_with_intent.csv')

# Intent별 샘플 문장 출력
for intent in sorted(df['chemical_intent'].unique()):
    subset = df[df['chemical_intent'] == intent]
    print(f'\n=== {intent} ({len(subset)} sentences) ===')
    for _, row in subset.head(3).iterrows():
        sub = row['sub_intent']
        sent = row['sentence'][:120]
        print(f'  [{sub}] {sent}...')

# JSON 매핑도 저장 (int 변환)
import numpy as np
labels = np.load('output/clustering_labels_v3a.npy')
df['cluster'] = labels

mapping = []
seen = set()
for _, row in df.drop_duplicates('cluster').iterrows():
    c = int(row['cluster'])
    if c in seen:
        continue
    seen.add(c)
    mapping.append({
        'cluster_id': c,
        'size': int((df['cluster'] == c).sum()),
        'chemical_intent': row['chemical_intent'],
        'sub_intent': row['sub_intent'],
    })

with open('output/cluster_intent_mapping.json', 'w') as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)

print(f'\n\nSaved mapping JSON with {len(mapping)} clusters')
