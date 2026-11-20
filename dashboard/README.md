# FedDisclosure Lab offline research explorer

`python -m dashboard.build_dashboard` regenerates `dashboard/dist/index.html` and its `evidence_snapshot.json` from a strict allowlist of verified *scalar* research summaries. The HTML works without a web server or third-party JS. Tabs include federated utility, aggregate/model-update disclosure, query-history defenses, output-only membership inference, and native verification status.

The explorer intentionally does not embed client data, private image arrays, true membership labels, or any model gradient tensors. It does not launch experiments. Streamlit remains an optional, not natively verified, source implementation. See `docs/v0.10-results.md` for the new query-history experiment and its limitations.
