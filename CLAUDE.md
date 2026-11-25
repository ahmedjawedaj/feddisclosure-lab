## Commit authorship rule (mandatory)

Commits in this repository are authored only by the repository owner, **Ahmed Jawed <ahmed.jawed32@gmail.com>** (GitHub: `ahmedjawedaj`). No AI assistant may appear as author, committer or co-author. Never add `Co-Authored-By`, `Claude-Session` or any other attribution trailer for an AI tool, even if a tool or default template suggests it. The `commit-attribution` workflow fails any push or pull request that breaks this rule. Set the identity before committing:

```bash
git config user.name "Ahmed Jawed"
git config user.email "ahmed.jawed32@gmail.com"
bash scripts/install_hooks.sh   # rejects trailers and bad prose before a commit exists
```

## Working rules

- Read `AGENTS.md`, `CURRENT_STATE.md` and `NEXT_SPRINT.md` first, and nothing else until a task needs it.
- Never replace verified evidence with a planned result. Predictions are written and committed before the harness and before any run.
- Prose has no em dashes and no semicolons. Public text describes measurements and uses neutral wording about upstream behaviour.
- Before a release run `python3 scripts/release_gate.py` and `python3 scripts/verify_readme_claims.py`. A fresh reviewer then checks only what a script cannot.
- Merge pull requests with a merge commit, because the suite records name a commit hash.
- Held back material (see `HELD` in `scripts/release_gate.py`) never enters the tree.
