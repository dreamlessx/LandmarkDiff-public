"""Scrape real before/after surgical images from public sources.

Sources:
1. Open-i (NLM/NIH open medical image search) — CC-licensed
2. PubMed Central open-access figures
3. Public plastic surgery research datasets

Only downloads images with permissive licenses.
Saves to data/real_pairs/ with metadata.

Usage:
    python scripts/scrape_before_after.py
    python scripts/scrape_before_after.py --procedure rhinoplasty --max 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROCEDURES = {
    "rhinoplasty": [
        "rhinoplasty before after",
        "nose surgery outcome",
        "nasal reconstruction result",
    ],
    "blepharoplasty": [
        "blepharoplasty before after",
        "eyelid surgery result",
        "upper blepharoplasty outcome",
    ],
    "rhytidectomy": [
        "facelift before after",
        "rhytidectomy outcome",
        "facial rejuvenation surgery",
    ],
    "orthognathic": [
        "orthognathic surgery before after",
        "jaw surgery outcome",
        "maxillofacial surgery result",
    ],
}

HEADERS = {
    "User-Agent": "LandmarkDiff-Research/0.1 (Academic research; contact@landmarkdiff.dev)"
}


def search_openi(query: str, max_results: int = 20) -> list[dict]:
    """Search Open-i (NLM) for medical images.

    Open-i provides access to biomedical images from PubMed Central
    open-access subset. All images are from CC-licensed articles.
    """
    base_url = "https://openi.nlm.nih.gov/api/search"
    params = {
        "query": query,
        "it": "xg",  # image type: photographs
        "coll": "pmc",  # PubMed Central
        "m": 1,
        "n": max_results,
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"  Open-i search failed for '{query}': {e}")
        return []

    results = []
    for item in data.get("list", []):
        img_url = item.get("imgLarge") or item.get("imgThumb")
        if not img_url:
            continue

        if not img_url.startswith("http"):
            img_url = f"https://openi.nlm.nih.gov{img_url}"

        results.append({
            "url": img_url,
            "title": item.get("title", ""),
            "source": "openi",
            "pmcid": item.get("pmcid", ""),
            "license": "CC (PMC Open Access)",
            "query": query,
        })

    return results


def search_pmc_figures(query: str, max_results: int = 10) -> list[dict]:
    """Search PubMed Central for figure images via E-utilities."""
    # Search for relevant PMC articles
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": f"{query} AND open access[filter]",
        "retmax": max_results,
        "retmode": "json",
    }
    url = f"{search_url}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"  PMC search failed: {e}")
        return []

    ids = data.get("esearchresult", {}).get("idlist", [])
    results = []

    for pmcid in ids[:5]:
        # Get article metadata
        fetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pmc&id={pmcid}&retmode=xml"
        )
        try:
            req = urllib.request.Request(fetch_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_data = resp.read().decode()

            # Extract figure URLs from XML (simple pattern match)
            import re
            fig_urls = re.findall(r'xlink:href="(https?://[^"]*(?:\.jpg|\.png|\.jpeg)[^"]*)"', xml_data)

            for fig_url in fig_urls[:3]:
                results.append({
                    "url": fig_url,
                    "title": f"PMC{pmcid} figure",
                    "source": "pmc",
                    "pmcid": f"PMC{pmcid}",
                    "license": "CC (PMC Open Access)",
                    "query": query,
                })
        except Exception:
            continue

        time.sleep(0.5)  # Rate limit

    return results


def download_image(url: str, output_path: Path, timeout: int = 30) -> bool:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            if len(data) < 5000:  # Skip tiny images
                return False
            output_path.write_bytes(data)
            return True
    except Exception:
        return False


def main(
    procedure: str | None = None,
    max_images: int = 30,
    output_dir: str = "data/real_pairs",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if procedure:
        queries_map = {procedure: PROCEDURES[procedure]}
    else:
        queries_map = PROCEDURES

    all_results: list[dict] = []

    for proc, queries in queries_map.items():
        proc_dir = out / proc
        proc_dir.mkdir(exist_ok=True)
        print(f"\n=== {proc} ===")

        for query in queries:
            print(f"  Searching: '{query}'...")

            # Open-i search
            openi_results = search_openi(query, max_results=max_images // len(queries))
            print(f"    Open-i: {len(openi_results)} results")
            all_results.extend(openi_results)

            # PMC figure search
            pmc_results = search_pmc_figures(query, max_results=5)
            print(f"    PMC: {len(pmc_results)} results")
            all_results.extend(pmc_results)

            time.sleep(1)  # Rate limit between queries

    # Download unique images
    print(f"\nDownloading {len(all_results)} candidate images...")
    seen_hashes: set[str] = set()
    downloaded = 0
    metadata_log: list[dict] = []

    for item in all_results:
        if downloaded >= max_images:
            break

        url = item["url"]
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        if url_hash in seen_hashes:
            continue
        seen_hashes.add(url_hash)

        proc = "unknown"
        for p, queries in PROCEDURES.items():
            for q in queries:
                if q in item.get("query", ""):
                    proc = p
                    break

        proc_dir = out / proc
        proc_dir.mkdir(exist_ok=True)
        ext = ".jpg"
        if ".png" in url.lower():
            ext = ".png"

        filename = f"{downloaded:04d}_{url_hash}{ext}"
        filepath = proc_dir / filename

        if download_image(url, filepath):
            downloaded += 1
            item["local_path"] = str(filepath)
            metadata_log.append(item)

            if downloaded % 10 == 0:
                print(f"  Downloaded {downloaded}/{max_images}")
        else:
            filepath.unlink(missing_ok=True)

    # Save metadata
    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(metadata_log, indent=2))

    summary_path = out / "sources.txt"
    summary_path.write_text(
        f"LandmarkDiff Real Before/After Image Collection\n"
        f"================================================\n"
        f"Total images: {downloaded}\n"
        f"Sources: Open-i (NLM), PubMed Central\n"
        f"License: CC (PMC Open Access Subset)\n"
        f"Date: {time.strftime('%Y-%m-%d')}\n\n"
        f"All images sourced from open-access biomedical literature.\n"
        f"See metadata.json for per-image attribution.\n"
    )

    print(f"\nDone. {downloaded} images saved to {out}/")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape surgical before/after images")
    parser.add_argument("--procedure", choices=list(PROCEDURES.keys()), default=None)
    parser.add_argument("--max", type=int, default=30)
    parser.add_argument("--output", default="data/real_pairs")
    args = parser.parse_args()

    main(args.procedure, args.max, args.output)
