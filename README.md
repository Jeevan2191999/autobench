# AutoBench

Automated adversarial benchmark generation for Constitutional AI properties from research literature.

## What it does

AutoBench reads AI safety research papers, extracts the failure modes they describe, and automatically synthesises novel adversarial test cases to benchmark LLM safety.

No human annotation required. No manually written test cases. Pure automation from paper to benchmark.

## Results

- 5 papers processed including Constitutional AI (Anthropic), HarmBench, Jailbroken
- 23 failure modes extracted automatically by Claude
- 69 novel adversarial test cases generated
- Claude Sonnet 4.6 scored 100% safety rate, average score 95.5/100
- Strongest category: jailbreak (98.33/100)
- Weakest category: policy violation (93.67/100)

## How to run

pip3 install anthropic requests feedparser pandas tqdm

export ANTHROPIC_API_KEY=your-key-here

python3 src/pipeline.py --skip-fetch --max-cases 12

## Architecture

- paper_fetcher.py — fetches papers from ArXiv API
- failure_extractor.py — extracts failure modes using Claude
- benchmark_generator.py — generates novel adversarial test cases
- evaluator.py — evaluates model responses and scores safety
- pipeline.py — end to end orchestration

## Motivation

AI safety evaluation benchmarks go stale as models improve. Creating new benchmarks requires enormous human effort. AutoBench automates this by reading what researchers already know about how models fail and turning that knowledge into executable tests.

## Next steps

- Add more papers to
