#!/usr/bin/env bash
# Deploy the demo to a private Hugging Face Space.
# Usage: scripts/deploy_space.sh <username>/<space-name>      (log in first: uv run hf auth login)
# Only the app, the package, the fitted transform and spec, the model, made-up examples and the model card are uploaded: no data of any kind.
set -euo pipefail
SPACE="${1:?usage: scripts/deploy_space.sh <username>/<space-name>}"
uv run python scripts/build_space.py
uv run hf repos create "$SPACE" --repo-type space --space-sdk gradio --private --exist-ok
uv run hf upload "$SPACE" build/space . --repo-type space --commit-message "Deploy the probability-of-default demo"
echo "Space: https://huggingface.co/spaces/$SPACE"
