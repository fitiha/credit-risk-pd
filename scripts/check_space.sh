#!/usr/bin/env bash
# Is the Space running? Free Spaces go to sleep when idle; opening the page wakes them.
set -euo pipefail
SPACE="${1:?usage: scripts/check_space.sh <username>/<space-name>}"
uv run hf spaces info "$SPACE" | grep -iE "runtime|stage|sdk|private|hardware" || uv run hf spaces info "$SPACE"
