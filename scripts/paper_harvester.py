"""
Paper Harvester — Semantic Scholar API로 OER 촉매 논문 수집
JSON + PDF(오픈액세스) 다운로드
"""

import requests
import json
import time
import os
import sys
from datetime import datetime

BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,title,abstract,year,authors,journal,externalIds,url,citationCount,fieldsOfStudy,openAccessPdf,isOpenAccess"

QUERIES = [
    "oxygen evolution reaction catalyst",
    "OER electrocatalyst water splitting",
    "OER overpotential transition metal",
    "nickel iron LDH oxygen evolution",
    "cobalt oxide OER electrocatalyst",
    "perovskite oxide oxygen evolution reaction",
    "metal organic framework OER catalyst",
    "single atom catalyst oxygen evolution",
    "OER alkaline water electrolysis",
    "bifunctional catalyst OER ORR",
    "OER acidic water splitting",
    "ruthenium iridium oxide OER",
    "layered double hydroxide OER",
    "OER catalyst stability degradation",
    "OER mechanism operando spectroscopy",
    "spinney oxide OER",
    "oxyhydroxide OER electrocatalyst",
    "OER catalyst support carbon nickel foam",
    "electrodeposition OER catalyst",
    "OER catalyst doping heteroatom",
]

OUTPUT_DIR = "/home/hs/oer-catalyst-project/output/papers"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdfs")


def search_papers(query, limit=100, offset=0):
    params = {
        "query": query,
        "limit": limit,
        "offset": offset,
        "fields": FIELDS,
        "year": "2018-2026",
    }
    try:
        r = requests.get(f"{BASE}/paper/search", params=params, timeout=30)
        if r.status_code == 429:
            print(f"  Rate limited, waiting 60s...")
            time.sleep(60)
            return search_papers(query, limit, offset)
        if r.status_code != 200:
            print(f"  API error: {r.status_code} {r.text[:200]}")
            return []
        return r.json().get("data", [])
    except Exception as e:
        print(f"  Error: {e}")
        return []


def paper_to_record(p):
    authors_list = p.get("authors") or []
    authors = ", ".join(a.get("name", "") for a in authors_list[:5])
    if len(authors_list) > 5:
        authors += " et al."

    journal = ""
    if p.get("journal"):
        j = p["journal"]
        journal = j.get("name", "")
        if j.get("volume"):
            journal += f" {j['volume']}"

    doi = (p.get("externalIds") or {}).get("DOI", "")
    oa_pdf = (p.get("openAccessPdf") or {}).get("url", "")

    return {
        "paper_id": p.get("paperId", ""),
        "title": p.get("title", ""),
        "abstract": p.get("abstract", "") or "",
        "year": p.get("year"),
        "authors": authors,
        "journal": journal,
        "doi": doi,
        "url": p.get("url", ""),
        "citations": p.get("citationCount", 0) or 0,
        "fields": ", ".join(p.get("fieldsOfStudy") or []),
        "is_open_access": p.get("isOpenAccess", False),
        "pdf_url": oa_pdf,
    }


def download_pdf(paper, pdf_dir):
    """Download PDF if open access."""
    pdf_url = paper.get("pdf_url", "")
    if not pdf_url:
        return None

    pid = paper["paper_id"]
    safe_name = pid.replace("/", "_").replace("\\", "_")
    pdf_path = os.path.join(pdf_dir, f"{safe_name}.pdf")

    if os.path.exists(pdf_path):
        return pdf_path

    try:
        r = requests.get(pdf_url, timeout=30, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and len(r.content) > 10000:
            with open(pdf_path, "wb") as f:
                f.write(r.content)
            return pdf_path
    except Exception:
        pass
    return None


def harvest(max_total=500, download_pdfs=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if download_pdfs:
        os.makedirs(PDF_DIR, exist_ok=True)

    all_papers = {}
    total_queries = len(QUERIES)

    for i, query in enumerate(QUERIES):
        if len(all_papers) >= max_total:
            break

        print(f"\n[{i+1}/{total_queries}] \"{query}\"")
        results = search_papers(query, limit=50)

        new = 0
        for p in results:
            pid = p.get("paperId", "")
            if pid and pid not in all_papers:
                all_papers[pid] = paper_to_record(p)
                new += 1

        print(f"  Found {len(results)} -> {new} new (total: {len(all_papers)})")
        time.sleep(1.5)

        if len(all_papers) >= max_total:
            break

    papers = list(all_papers.values())
    papers.sort(key=lambda x: x.get("citations", 0) or 0, reverse=True)

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "oer_papers_db.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(papers)} papers to {json_path}")

    # Save CSV summary
    csv_path = os.path.join(OUTPUT_DIR, "oer_papers_db.csv")
    import csv
    if papers:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=papers[0].keys())
            writer.writeheader()
            writer.writerows(papers)
    print(f"Saved CSV to {csv_path}")

    # Download PDFs
    if download_pdfs:
        oa_papers = [p for p in papers if p.get("is_open_access")]
        print(f"\nDownloading PDFs: {len(oa_papers)} open access papers...")
        pdf_count = 0
        for j, paper in enumerate(oa_papers):
            path = download_pdf(paper, PDF_DIR)
            if path:
                pdf_count += 1
            if (j + 1) % 20 == 0:
                print(f"  PDF progress: {j+1}/{len(oa_papers)} ({pdf_count} downloaded)")
            time.sleep(0.5)

        print(f"\nPDFs: {pdf_count}/{len(oa_papers)} downloaded to {PDF_DIR}")

    # Stats
    print(f"\n{'='*60}")
    print(f"  HARVEST COMPLETE")
    print(f"  Total papers: {len(papers)}")
    print(f"  Open access: {sum(1 for p in papers if p.get('is_open_access'))}")
    print(f"  Year range: {min(p.get('year',2026) for p in papers if p.get('year'))} - {max(p.get('year',2018) for p in papers if p.get('year'))}")
    print(f"  Top cited: {papers[0]['title'][:60]} ({papers[0]['citations']} cites)")
    print(f"{'='*60}")

    return papers


if __name__ == "__main__":
    max_total = 500
    if "--max" in sys.argv:
        idx = sys.argv.index("--max")
        if idx + 1 < len(sys.argv):
            max_total = int(sys.argv[idx + 1])

    print(f"{'='*60}")
    print(f"  OER Paper Harvester")
    print(f"  Target: {max_total} papers")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    harvest(max_total=max_total, download_pdfs=True)
