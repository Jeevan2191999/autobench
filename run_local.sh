#!/bin/bash
# Run AutoBench locally with your Anthropic API key
# Usage: ANTHROPIC_API_KEY=sk-xxx bash run_local.sh

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Error: Set your ANTHROPIC_API_KEY environment variable first"
  echo "Usage: ANTHROPIC_API_KEY=sk-xxx bash run_local.sh"
  exit 1
fi

echo "Installing dependencies..."
pip install anthropic requests feedparser python-dotenv pandas tqdm xmltodict

echo "Running AutoBench pipeline..."
cd src && python pipeline.py --max-cases 10
