# AutoBench

Automated Adversarial Benchmark Generation for Constitutional AI Properties from Research Literature.

AutoBench reads AI safety papers from ArXiv, extracts failure modes and adversarial scenarios, 
synthesises novel test cases, and evaluates LLMs against them automatically.

## Structure

- src/paper_fetcher.py     — Fetches and parses ArXiv papers
- src/failure_extractor.py — Extracts failure modes using Claude API
- src/benchmark_generator.py — Generates novel adversarial test cases
- src/evaluator.py         — Runs benchmarks against LLMs and scores results
- src/pipeline.py          — End to end orchestration
- data/                    — Raw paper data
- benchmarks/              — Generated benchmark datasets
- results/                 — Evaluation results

## Quickstart

pip install -r requirements.txt
python src/pipeline.py
