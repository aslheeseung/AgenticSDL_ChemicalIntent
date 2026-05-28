"""
Upload all output files to Hugging Face dataset
Usage: python3 upload_to_hf.py --token hf_YOUR_TOKEN
"""
import os
import sys
from huggingface_hub import HfApi

REPO_ID = "Heeseung123/oer-catalyst-lit"
BASE = "/home/hs/oer-catalyst-project"

token = None
if "--token" in sys.argv:
    token = sys.argv[sys.argv.index("--token") + 1]

api = HfApi(token=token)
print(f"Logged in as: {api.whoami()['name']}")

files_to_upload = [
    # Original 905-paper analysis
    ("output/clusters_summary.json", "output/clusters_summary.json"),
    ("output/clusters_summary_v3a.json", "output/clusters_summary_v3a.json"),
    ("output/interactive_plot_v3a.html", "output/interactive_plot_v3a.html"),
    ("output/interactive_plot_bars.html", "output/interactive_plot_bars.html"),
    ("output/lab_compatible_routes.json", "output/lab_compatible_routes.json"),
    ("output/intent_impact.csv", "output/intent_impact.csv"),
    ("output/transition_matrix.csv", "output/transition_matrix.csv"),
    ("output/network_heatmap.html", "output/network_heatmap.html"),
    ("output/nife_ldh_protocols.json", "output/nife_ldh_protocols.json"),
    ("output/process_sequences.csv", "output/process_sequences.csv"),
    ("output/cluster_intent_mapping.json", "output/cluster_intent_mapping.json"),
    ("output/sentences_with_intent.csv", "output/sentences_with_intent.csv"),
    ("output/sdl_cspbbr3_perovskite_quantum_dots.json", "output/sdl_cspbbr3_perovskite_quantum_dots.json"),
    ("output/process_transfer_evidence.json", "output/process_transfer_evidence.json"),
    ("output/network_interactive.html", "output/network_interactive.html"),
    ("output/common_routes.json", "output/common_routes.json"),
    ("output/sentences_contextual_intent.csv", "output/sentences_contextual_intent.csv"),
    
    # Papers analysis
    ("output/papers/embeddings.npy", "output/papers/embeddings.npy"),
    ("output/papers/cluster_scatter.html", "output/papers/cluster_scatter.html"),
    ("output/papers/cluster_sizes.html", "output/papers/cluster_sizes.html"),
    ("output/papers/sentences_extracted.csv", "output/papers/sentences_extracted.csv"),
    ("output/papers/sentences_clustered.csv", "output/papers/sentences_clustered.csv"),
    ("output/papers/cluster_interpretation.json", "output/papers/cluster_interpretation.json"),
    
    # Docs
    ("docs/CID_v3_research_briefing.md", "docs/CID_v3_research_briefing.md"),
    ("docs/cross_domain_cid.md", "docs/cross_domain_cid.md"),
]

uploaded = 0
skipped = 0
for local_path, hf_path in files_to_upload:
    full_path = os.path.join(BASE, local_path)
    if not os.path.exists(full_path):
        print(f"  SKIP (not found): {local_path}")
        skipped += 1
        continue
    
    size_mb = os.path.getsize(full_path) / 1024 / 1024
    print(f"  [{uploaded+1}/{len(files_to_upload)}] {local_path} ({size_mb:.1f}MB)...", end=" ", flush=True)
    
    try:
        api.upload_file(
            path_or_fileobj=full_path,
            path_in_repo=hf_path,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        print("OK")
        uploaded += 1
    except Exception as e:
        print(f"ERROR: {e}")
        skipped += 1

print(f"\nDone! Uploaded: {uploaded}, Skipped: {skipped}")
print(f"URL: https://huggingface.co/datasets/{REPO_ID}")
