"""
paper_fetcher.py

Fetches AI safety papers from ArXiv using their free API.
Searches for papers about Constitutional AI, adversarial LLM evaluation,
AI safety benchmarks, and failure modes.
"""

import requests
import feedparser
import json
import os
from datetime import datetime
from tqdm import tqdm


# Search queries targeting AI safety and Constitutional AI research
SEARCH_QUERIES = [
    "constitutional AI safety LLM",
    "adversarial evaluation large language model",
    "LLM failure modes benchmark",
    "AI safety red teaming evaluation",
    "jailbreak prompt injection language model",
    "hallucination detection LLM benchmark",
]

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def fetch_papers(query: str, max_results: int = 10) -> list[dict]:
    """
    Fetch papers from ArXiv for a given search query.

    Args:
        query: Search string
        max_results: How many papers to fetch

    Returns:
        List of paper dicts with title, abstract, authors, url, id
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    response = requests.get(ARXIV_API_URL, params=params)
    response.raise_for_status()

    feed = feedparser.parse(response.text)

    papers = []
    for entry in feed.entries:
        paper = {
            "id": entry.get("id", ""),
            "title": entry.get("title", "").strip().replace("\n", " "),
            "abstract": entry.get("summary", "").strip().replace("\n", " "),
            "authors": [a.get("name", "") for a in entry.get("authors", [])],
            "published": entry.get("published", ""),
            "url": entry.get("link", ""),
            "query": query,
            "fetched_at": datetime.now().isoformat(),
        }
        papers.append(paper)

    return papers


def fetch_all_papers(max_per_query: int = 5) -> list[dict]:
    """
    Fetch papers across all search queries and deduplicate.

    Args:
        max_per_query: Papers to fetch per query

    Returns:
        Deduplicated list of all papers
    """
    all_papers = []
    seen_ids = set()

    print(f"Fetching papers from ArXiv across {len(SEARCH_QUERIES)} queries...")

    for query in tqdm(SEARCH_QUERIES):
        try:
            papers = fetch_papers(query, max_results=max_per_query)
            for paper in papers:
                if paper["id"] not in seen_ids:
                    seen_ids.add(paper["id"])
                    all_papers.append(paper)
        except Exception as e:
            print(f"Error fetching query '{query}': {e}")

    print(f"Fetched {len(all_papers)} unique papers")
    return all_papers


def save_papers(papers: list[dict], output_path: str) -> None:
    """Save papers to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(papers, f, indent=2)
    print(f"Saved {len(papers)} papers to {output_path}")


def load_papers(input_path: str) -> list[dict]:
    """Load papers from a JSON file."""
    with open(input_path) as f:
        return json.load(f)


if __name__ == "__main__":
    papers = fetch_all_papers(max_per_query=5)
    save_papers(papers, "data/papers.json")

    # Show a sample
    if papers:
        print("\nSample paper:")
        print(f"Title: {papers[0]['title']}")
        print(f"Abstract: {papers[0]['abstract'][:200]}...")
