#!/usr/bin/env bash
# Fails if any commit in the given range has an AI assistant as author, committer
# or co-author. Usage: scripts/check_commit_attribution.sh <base> <head>
set -euo pipefail
base="${1:?base}"; head="${2:?head}"
if [[ "$base" =~ ^0+$ ]]; then range="$head"; else range="$base..$head"; fi
pattern='co-authored-by:.*(claude|anthropic|copilot|chatgpt|openai|gemini|cursor)|claude-session:|noreply@anthropic\.com'
bad=0
while read -r sha; do
  meta="$(git log -1 --format='%an <%ae>%n%cn <%ce>%n%B' "$sha")"
  if grep -Eiq "$pattern" <<<"$meta" || grep -Eiq '^(claude|anthropic)\b' <<<"$(git log -1 --format='%an%n%cn' "$sha")"; then
    echo "Disallowed AI attribution in commit $sha" >&2
    bad=1
  fi
done < <(git rev-list "$range")
exit "$bad"
