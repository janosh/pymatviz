# /// script
# dependencies = [
#     "google-search-results>=2.4.2",
#     "pyyaml>=6.0.2",
# ]
# ///
"""Script to fetch papers that cite pymatviz from Google Scholar and update readme.

Invoke with 64-character SERPAPI_KEY:

SERPAPI_KEY=ccd7f7ea8... python assets/fetch_citations.py
"""
# ruff: noqa: T201

from __future__ import annotations

import gzip
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone

import yaml
from serpapi import GoogleSearch

from pymatviz import ROOT


if os.getenv("CI"):
    raise SystemExit("Skip scraping Google Scholar in CI")

# NotRequired can't be imported below Python 3.11
from typing import NotRequired, TypedDict


class ScholarPaper(TypedDict):
    """Type for a paper fetched from Google Scholar."""

    title: str
    link: str
    result_id: str
    authors: list[str]
    summary: str | None
    year: int | None
    citations: int
    fetch_date: str
    # Additional metadata fields
    snippet: NotRequired[str]  # Paper abstract/description
    resources: NotRequired[list[dict[str, str]]]  # Additional links (PDF, HTML, etc.)
    publication_info: NotRequired[dict[str, str]]  # Full publication metadata
    inline_links: NotRequired[dict[str, str]]  # Related links (citations, versions, ..)
    list_index: NotRequired[int]  # list index in search results


def should_update(filename: str, update_freq_days: int = 7) -> bool:
    """Check if the file should be updated based on its last modified time.

    Args:
        filename (str): Path to the file to check.
        update_freq_days (int): Number of days to wait between updates.

    Returns:
        bool: True if file doesn't exist or is older than update_freq_days.
    """
    try:
        mtime = os.path.getmtime(filename)
        last_modified = datetime.fromtimestamp(mtime, tz=timezone.utc)
        return (datetime.now(tz=timezone.utc) - last_modified) > timedelta(
            days=update_freq_days
        )
    except FileNotFoundError:
        return True


def create_backup(filename: str) -> str | None:
    """Backup the specified file with timestamp in new name.

    Args:
        filename (str): Path to the file to backup.

    Returns:
        str | None: Path to the backup file if created, None if source doesn't exist.
    """
    if not os.path.isfile(filename):
        return None

    # Get last modified time and format for filename
    mtime = datetime.fromtimestamp(os.path.getmtime(filename), tz=timezone.utc)
    timestamp = mtime.strftime("%Y%m%d-%H%M%S")

    # Create backup filename with timestamp
    base = filename.removesuffix(".yml.gz")
    backup_path = os.path.join(os.path.dirname(filename), f"{base}-{timestamp}.yml.gz")
    shutil.copy2(filename, backup_path)  # copy2 preserves metadata
    return str(backup_path)


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
    if api_key is None:
        api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise ValueError(
            "No API key provided. Either pass as argument or set SERPAPI_KEY env var."
        )

    papers: list[ScholarPaper] = []
    today = f"{datetime.now(tz=timezone.utc):%Y-%m-%d}"

    for page in range(num_pages):
        params = {
            "api_key": api_key,
            "engine": "google_scholar",
            "q": query,
            "hl": "en",  # language
            "start": page * 10,  # Google Scholar uses 10 results per page
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        if "error" in results:
            print(f"Error on page {page + 1}: {results['error']}", file=sys.stderr)
            continue

        if "organic_results" not in results:
            print(f"No results found on page {page + 1}", file=sys.stderr)
            break

        for idx, result in enumerate(results["organic_results"], start=1):
            # Skip if no title or link
            if not result.get("title") or not result.get("link"):
                continue

            # Extract year from publication info if available
            year = None
            pub_info = result.get("publication_info", {})
            if "summary" in pub_info and (
                year_match := re.search(r"\b(19|20)\d{2}\b", pub_info["summary"])
            ):
                year = int(year_match.group())

            # Extract authors from publication info
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
            if not paper.get("authors"):
                continue  # don't add papers without authors to YAML file
            papers.append(paper)

    return papers


def save_papers(
    papers: list[ScholarPaper], filename: str = "scholar-papers.yml.gz"
) -> None:
    """Save papers to a gzipped YAML file.

    Args:
        papers (list[ScholarPaper]): List of papers to save.
        filename (str): Name of the output file.
    """
    # Load existing papers for diff if file exists
    old_papers: list[ScholarPaper] = []
    if os.path.isfile(filename):
        with gzip.open(filename, mode="rt", encoding="utf-8") as file:
            old_papers = yaml.safe_load(file) or []

    # Create backup of existing file
    if backup_path := create_backup(filename):
        print(f"\nCreated backup at {backup_path}")
        # Print diff if we have old data
        if old_papers:
            print(f"\nPaper count: {len(old_papers)} → {len(papers)}")

    with gzip.open(filename, mode="wt", encoding="utf-8") as file:
        yaml.dump(papers, file, default_flow_style=False, allow_unicode=True)


def update_readme(
    papers: list[ScholarPaper], readme_path: str = f"{ROOT}/readme.md"
) -> None:
    """Update the readme with a list of papers sorted by citations.

    Args:
        papers (list[ScholarPaper]): List of papers to add to readme.
        readme_path (str): Path to the readme file.
    """
    # Sort papers by citations
    sorted_papers = sorted(papers, key=lambda x: x["citations"], reverse=True)

    # Read current readme
    with open(readme_path, encoding="utf-8") as file:
        content = file.read()

    # Remove existing papers section if it exists
    if "## Papers using" in content:
        pattern = r"## Papers using.*?$"
        content = re.sub(pattern, "", content, flags=re.DOTALL).rstrip()

    # Prepare the new section
    today = f"{datetime.now(tz=timezone.utc):%Y-%m-%d}"
    papers_section = "\n\n## Papers using `pymatviz`\n\n"
    papers_section += (
        f"Sorted by number of citations. Last updated {today}. "
        "Auto-generated from Google Scholar. Manual additions via PR welcome.\n\n"
    )

    for paper in sorted_papers:
        if not paper["authors"]:
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

    # Add papers section at the very end of the readme
    content = content.rstrip() + papers_section

    # Write updated content
    with open(readme_path, mode="w", encoding="utf-8") as file:
        file.write(content)


def main(update_freq_days: int = 7) -> None:
    """Main function to fetch papers and update readme.

    Args:
        update_freq_days (int): Number of days to wait between updates.
    """
    data_file = f"{ROOT}/assets/scholar-papers.yml.gz"

    # Load existing papers
    if os.path.isfile(data_file):
        with gzip.open(data_file, mode="rt", encoding="utf-8") as file:
            existing_papers = yaml.safe_load(file)
    else:
        existing_papers = []

    # Check if we need to update
    if not should_update(data_file, update_freq_days):
        print(
            f"{data_file=} is less than {update_freq_days} days old, skipping update."
        )
        # Still update readme with existing data
        update_readme(existing_papers)
        return

    # Fetch new papers
    new_papers = fetch_scholar_papers()

    # Merge papers, keeping the most recent citation counts
    paper_dict: dict[str, ScholarPaper] = {
        paper["title"]: paper for paper in existing_papers
    } | {paper["title"]: paper for paper in new_papers}

    # Convert back to list
    all_papers = list(paper_dict.values())

    # Save updated papers
    save_papers(all_papers, data_file)

    # Update readme
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
    args = parser.parse_args()

    main(args.update_freq)
