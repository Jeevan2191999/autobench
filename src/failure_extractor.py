"""
failure_extractor.py

Uses Claude to extract failure modes and adversarial scenarios
from AI safety research papers.

For each paper, Claude reads the abstract and extracts:
- Specific failure modes described in the paper
- Attack patterns and techniques identified
- Categories of vulnerability
- Concrete examples of how models fail
"""

import json
import os
import anthropic
from tqdm import tqdm


client = anthropic.Anthropic()

EXTRACTION_PROMPT = """You are an AI safety researcher. Read this research paper abstract and extract the specific failure modes, attack patterns, and vulnerability categories it describes.

Paper Title: {title}

Abstract: {abstract}

Extract the following in JSON format:

{{
  "failure_modes": [
    {{
      "name": "short name for this failure mode",
      "description": "what the failure is and how it manifests",
      "attack_pattern": "the specific technique used to trigger this failure",
      "severity": "high/medium/low",
      "category": "one of: prompt_injection, jailbreak, hallucination, deception, manipulation, harmful_content, policy_violation"
    }}
  ],
  "key_vulnerabilities": ["list of 2-4 key vulnerability types found"],
  "paper_contribution": "one sentence on what this paper contributes to AI safety"
}}

Extract only what is explicitly described in the abstract. Do not invent failure modes not mentioned. Return only valid JSON."""


def extract_failure_modes(paper: dict) -> dict:
    """
    Extract failure modes from a single paper using Claude.

    Args:
        paper: Paper dict with title and abstract

    Returns:
        Dict with extracted failure modes and vulnerabilities
    """
    prompt = EXTRACTION_PROMPT.format(
        title=paper["title"],
        abstract=paper["abstract"]
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    # Clean up any markdown code blocks if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    if raw.endswith("```"):
        raw = raw[:-3]

    try:
        extracted = json.loads(raw.strip())
    except json.JSONDecodeError:
        # If JSON parsing fails, return a structured error
        extracted = {
            "failure_modes": [],
            "key_vulnerabilities": [],
            "paper_contribution": "Extraction failed",
            "raw_response": raw
        }

    # Add paper metadata
    extracted["paper_id"] = paper["id"]
    extracted["paper_title"] = paper["title"]
    extracted["paper_url"] = paper["url"]

    return extracted


def extract_all_papers(papers: list[dict]) -> list[dict]:
    """
    Extract failure modes from all papers.

    Args:
        papers: List of paper dicts

    Returns:
        List of extraction results
    """
    results = []
    print(f"Extracting failure modes from {len(papers)} papers...")

    for paper in tqdm(papers):
        try:
            result = extract_failure_modes(paper)
            results.append(result)
            print(f"  Extracted {len(result.get('failure_modes', []))} failure modes from: {paper['title'][:60]}...")
        except Exception as e:
            print(f"  Error on paper '{paper['title'][:60]}': {e}")
            results.append({
                "paper_id": paper["id"],
                "paper_title": paper["title"],
                "failure_modes": [],
                "error": str(e)
            })

    return results


def save_extractions(extractions: list[dict], output_path: str) -> None:
    """Save extraction results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(extractions, f, indent=2)
    total_modes = sum(len(e.get("failure_modes", [])) for e in extractions)
    print(f"Saved {len(extractions)} extractions with {total_modes} total failure modes to {output_path}")


def load_extractions(input_path: str) -> list[dict]:
    """Load extraction results from JSON."""
    with open(input_path) as f:
        return json.load(f)


if __name__ == "__main__":
    from paper_fetcher import load_papers

    papers = load_papers("data/papers.json")
    extractions = extract_all_papers(papers)
    save_extractions(extractions, "data/extractions.json")

    # Show sample
    if extractions and extractions[0].get("failure_modes"):
        print("\nSample extraction:")
        print(f"Paper: {extractions[0]['paper_title'][:60]}")
        for fm in extractions[0]["failure_modes"][:2]:
            print(f"  Failure mode: {fm['name']}")
            print(f"  Category: {fm['category']}")
            print(f"  Attack: {fm['attack_pattern'][:100]}")
