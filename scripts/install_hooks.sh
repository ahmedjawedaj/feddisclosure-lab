#!/usr/bin/env bash
# Installs the repository hooks for this clone: bash scripts/install_hooks.sh
set -e
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath scripts/hooks
chmod +x scripts/hooks/commit-msg scripts/hooks/pre-commit
echo "hooks installed from scripts/hooks"
