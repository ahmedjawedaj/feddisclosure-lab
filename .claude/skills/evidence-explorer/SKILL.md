---
name: evidence-explorer
description: Audit and publish read-only FedDisclosure research visualizations from reviewed scalar results, without leaking raw records or overstating privacy guarantees.
---

Read `docs/v0.9-results.md`, `CURRENT_STATE.md`, `dashboard/evidence.py`.

1. Verify source hashes using the research verifiers before publication.
2. Permit only reviewed aggregate scalar JSON; never embed NPZ, `.pt`, client SQLite data, evaluator labels or recovered images.
3. Regenerate with `python -m dashboard.build_dashboard`. The static export is offline; Streamlit is optional.
4. Label empirical MSE and attack privileges accurately; never claim native DP or SecAgg+ from idealized arithmetic aggregation.
5. Run dashboard regression tests. Publish source hashes, limitations and negative results.
