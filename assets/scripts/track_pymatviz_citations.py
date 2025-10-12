"""Script to track papers that use pymatviz by fetching citations from Google Scholar
and Zotero.

Invoke with 64-character SERPAPI_KEY from https://serpapi.com/dashboard (login with
GitHub account):

SERPAPI_KEY=ccd7f7ea8... python assets/scripts/track_pymatviz_citations.py
"""

# /// script
# dependencies = [
#     "google-search-results>=2.4.2",
#     "pyyaml>=6.0.2",
#     "python-dotenv>=1.1",
# ]
# ///

from __future__ import annotations

import difflib
import gzip
import os
import re
import shutil
import sys
from datetime import UTC, datetime, timedelta

import yaml
from dotenv import load_dotenv

from pymatviz import ROOT


if os.getenv("CI"):
    print(f"Skip {__file__.split('/')[-1]} in CI")
    raise SystemExit(0)

# NotRequired can't be imported below Python 3.11
from typing import NotRequired, TypedDict


load_dotenv(f"{ROOT}/site/.env")


class ScholarPaper(TypedDict):
    """Type for a paper fetched from Google Scholar."""

    title: str
    link: str
    authors: list[str]
    year: int | None
    citations: int
    # Additional metadata fields
    result_id: NotRequired[str]
    summary: NotRequired[str | None]
    fetch_date: NotRequired[str]
    snippet: NotRequired[str]  # Paper abstract/description
    resources: NotRequired[list[dict[str, str]]]  # Additional links (PDF, HTML, etc.)
    publication_info: NotRequired[dict[str, str]]  # Full publication metadata
    inline_links: NotRequired[dict[str, str]]  # Related links (citations, versions, ..)
    list_index: NotRequired[int]  # list index in search results
    source: NotRequired[str]  # Track if paper came from Zotero or Scholar


def remove_duplicates(
    papers: list[ScholarPaper], similarity_threshold: float = 0.85
) -> list[ScholarPaper]:
    """Remove duplicate papers based on title similarity.

    Args:
        papers (list[ScholarPaper]): List of papers to deduplicate
        similarity_threshold (float): Min similarity to consider papers as duplicates.

    Returns:
        list[ScholarPaper]: Deduplicated list of papers
    """
    # Sort papers by source (Zotero first) and citations (higher first)
    papers.sort(key=lambda x: (x.get("source") != "zotero", -x.get("citations", 0)))
    papers_to_remove: set[int] = set()

    for ii, paper1 in enumerate(papers):
        if ii in papers_to_remove:
            continue
        for jj, paper2 in enumerate(papers[ii + 1 :], start=ii + 1):
            if jj in papers_to_remove:
                continue

            norm1 = " ".join(paper1["title"].lower().split())
            norm2 = " ".join(paper2["title"].lower().split())
            similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()

            if similarity >= similarity_threshold:
                if paper1.get("source") == "zotero":
                    papers_to_remove.add(jj)
                elif paper2.get("source") == "zotero":
                    papers_to_remove.add(ii)
                elif paper1.get("citations", 0) >= paper2.get("citations", 0):
                    papers_to_remove.add(jj)
                else:
                    papers_to_remove.add(ii)

    return [paper for idx, paper in enumerate(papers) if idx not in papers_to_remove]


def should_update(filename: str, update_freq_days: int = 7) -> bool:
    """Check if file should be updated (missing or older than update_freq_days)."""
    try:
        mtime = os.path.getmtime(filename)
        last_modified = datetime.fromtimestamp(mtime, tz=UTC)
        delta = datetime.now(tz=UTC) - last_modified
        return delta > timedelta(days=update_freq_days)
    except FileNotFoundError:
        return True


def create_backup(filename: str) -> str | None:
    """Backup file with timestamp. Returns backup path or None if file doesn't exist."""
    if not os.path.isfile(filename):
        return None

    mtime = datetime.fromtimestamp(os.path.getmtime(filename), tz=UTC)
    timestamp = mtime.strftime("%Y%m%d-%H%M%S")
    base = filename.removesuffix(".yaml.gz")
    backup_path = f"{base}-{timestamp}.yaml.gz"
    shutil.copy2(filename, backup_path)
    return backup_path


def fetch_scholar_papers(
    api_key: str | None = None, query: str = "pymatviz", num_pages: int = 3
) -> list[ScholarPaper]:
    """Fetch papers from Google Scholar that mention pymatviz.

    Args:
        api_key (str | None): SerpAPI key. If None, will try to read from SERPAPI_KEY
            env var.
        query (str): Search query. Defaults to "pymatviz".
        num_pages (int): Number of pages to fetch. Defaults to 3. Increase this number
            as more mentions of pymatviz in literature are found.

    Returns:
        list[ScholarPaper]: List of papers with their metadata including title, authors,
            publication info, year, and citation count.
    """
    from serpapi import GoogleSearch

    api_key = api_key or os.getenv("SERPAPI_KEY")
    if not api_key:
        raise ValueError(
            "No API key provided. Either pass as argument or set SERPAPI_KEY env var."
        )

    papers: list[ScholarPaper] = []
    today = f"{datetime.now(tz=UTC):%Y-%m-%d}"

    for page in range(num_pages):
        params = {
            "api_key": api_key,
            "engine": "google_scholar",
            "q": query,
            "hl": "en",  # language
            "start": page * 10,  # Google Scholar uses 10 results per page
        }

        results = GoogleSearch(params).get_dict()

        if "error" in results:
            print(f"Error on page {page + 1}: {results['error']}", file=sys.stderr)
            continue
        if "organic_results" not in results:
            print(f"No results found on page {page + 1}", file=sys.stderr)
            break

        for idx, result in enumerate(results["organic_results"], start=1):
            if not result.get("title") or not result.get("link"):
                continue  # Skip if no title or link

            # Extract year from publication info if available
            year = None
            pub_info = result.get("publication_info", {})
            if "summary" in pub_info and (
                year_match := re.search(r"\b(19|20)\d{2}\b", pub_info["summary"])
            ):
                year = int(year_match.group())

            authors = []
            if isinstance(pub_info, dict) and "authors" in pub_info:
                authors = [
                    author["name"]
                    for author in pub_info.pop("authors")
                    if isinstance(author, dict) and "name" in author
                ]

            # Store all metadata from the result, overwrite only processed fields
            paper: ScholarPaper = {  # type:ignore[typeddict-item]
                **result,  # Keep all original fields
                "authors": authors
                or result.get("authors", []),  # Use processed authors
                "year": year,  # Use extracted year
                "fetch_date": today,  # Add fetch date
                # Add pagination-unwrapped index in search result
                "list_index": idx + page * 10,
                "citations": result.get("inline_links", {})
                .get("cited_by", {})
                .get("total", 0),
                "summary": pub_info.get("summary", ""),
            }
            if paper.get("authors"):
                papers.append(paper)

    return papers


def save_papers(papers: list[ScholarPaper], filename: str) -> None:
    """Save papers to gzipped YAML file with backup and diff."""
    old_papers: list[ScholarPaper] = []
    if os.path.isfile(filename):
        with gzip.open(filename, mode="rt", encoding="utf-8") as file:
            old_papers = yaml.safe_load(file) or []

    if old_papers == papers:
        print(f"No need to update {filename}, already contains {len(papers)} papers")
        return

    if backup_path := create_backup(filename):
        print(f"\nCreated backup at {backup_path}")
        # Print diff if we have old data
        if old_papers and len(old_papers) != len(papers):
            print(f"\nPaper count in {filename}: {len(old_papers)} → {len(papers)}")

    with gzip.open(filename, mode="wt", encoding="utf-8") as file:
        yaml.dump(papers, file, default_flow_style=False, allow_unicode=True)


def update_readme(
    papers: list[ScholarPaper], readme_path: str = f"{ROOT}/readme.md"
) -> None:
    """Update readme with papers sorted by citations and year."""
    sorted_papers = sorted(
        papers,
        key=lambda p: (-p["citations"], -p.get("year") or float("inf")),  # type: ignore[operator]
    )

    with open(readme_path, encoding="utf-8") as file:
        content = file.read()
    # Update paper count in the "how to cite" section
    content = re.sub(
        r"Check out \[(\d+) existing papers",
        f"Check out [{len(sorted_papers)} existing papers",
        content,
    )

    if "## Papers using" in content:
        content = re.sub(r"## Papers using.*?$", "", content, flags=re.DOTALL).rstrip()

    today = f"{datetime.now(tz=UTC):%Y-%m-%d}"
    papers_section = "\n\n## Papers using `pymatviz`\n\n"
    scholar_url = "https://scholar.google.com/scholar?q=pymatviz"
    edit_url = "https://github.com/janosh/pymatviz/edit/main/readme.md"
    papers_section += (
        f"Sorted by number of citations, then year. Last updated {today}. "
        f"Auto-generated [from Google Scholar]({scholar_url}). Manual additions "
        f"[via PR]({edit_url}) welcome.\n\n"
    )

    for paper in sorted_papers:
        if not paper.get("authors"):
            print(f"Paper {paper['title']} has no authors, skipping")
            continue
        authors_str = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors_str += " et al."

        year_str = f" ({paper['year']})" if paper["year"] else ""
        cite_str = f" (cited by {paper['citations']})" if paper["citations"] else ""
        papers_section += (
            f"1. {authors_str}{year_str}. [{paper['title']}]({paper['link']})"
            f"{cite_str}\n"
        )

    with open(readme_path, mode="w", encoding="utf-8") as file:
        file.write(content.rstrip() + papers_section)

    print(f"Wrote {len(sorted_papers)} papers to {readme_path}")


def convert_zotero_paper(zotero_paper: dict) -> ScholarPaper | None:
    """Convert Zotero paper to ScholarPaper format."""
    try:
        if not (title := zotero_paper.get("title", "")):
            print(f"Skipping paper with no title: {zotero_paper.get('id', 'unknown')}")
            return None

        # Extract year (handles both list and dict formats)
        issued = zotero_paper.get("issued")
        year = (issued[0] if isinstance(issued, list) and issued else issued or {}).get(
            "year"
        )

        # Build link from URL or DOI
        link = zotero_paper.get("URL")
        if not link and (doi := zotero_paper.get("DOI")):
            link = doi if doi.startswith("http") else f"https://doi.org/{doi}"
        if not link:
            print(f"Skipping paper without URL or DOI: {title}")
            return None

        authors = [
            f"{auth.get('given', '')} {auth.get('family', '')}".strip()
            for auth in zotero_paper.get("author", [])
            if auth.get("given") or auth.get("family")
        ]

        return {  # noqa: TRY300
            "title": title,
            "authors": authors,
            "year": year,
            "link": link,
            "citations": 0,
            "source": "zotero",
        }
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Error converting paper {zotero_paper.get('title', 'Unknown')}: {exc}")
        return None


def main(update_freq_days: int = 7) -> None:
    """Fetch papers from Scholar and Zotero, merge, and update readme."""
    module_dir = os.path.dirname(__file__)
    data_file = f"{module_dir}/pmv-used-by-list-google-scholar.yaml.gz"

    # Load existing papers
    if os.path.isfile(data_file):
        with gzip.open(data_file, mode="rt", encoding="utf-8") as file:
            scholar_papers = yaml.safe_load(file)
    else:
        scholar_papers = []

    # Check if we need to update
    if should_update(data_file, update_freq_days):
        new_papers = fetch_scholar_papers()  # Fetch new papers

        # Merge papers, keeping the most recent citation counts
        paper_dict: dict[str, ScholarPaper] = {
            paper["title"]: paper for paper in scholar_papers
        } | {paper["title"]: paper for paper in new_papers}

        # Save updated papers
        scholar_papers = list(paper_dict.values())
        save_papers(scholar_papers, filename=data_file)
    else:
        print(
            f"{data_file=} is less than {update_freq_days} days old, skipping update."
        )

    with open(f"{module_dir}/pmv-used-by-list-zotero.yaml", encoding="utf-8") as file:
        zotero_papers = yaml.safe_load(file)["references"]

    converted_zotero_papers = [
        converted
        for paper in zotero_papers
        if (converted := convert_zotero_paper(paper)) is not None
    ]

    all_papers = remove_duplicates(scholar_papers + converted_zotero_papers)
    update_readme(all_papers)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch papers citing pymatviz and update readme."
    )
    parser.add_argument(
        "--update-freq",
        type=int,
        default=7,
        help="Number of days to wait between updates (default: 7)",
    )
    args, _unknown = parser.parse_known_args()

    main(args.update_freq)
