# Roadmap, dependencies and gates

Sprint 0 (foundation): config/schema, threat models, test infrastructure, canonical handoff, reproducibility contract. Done for initial core.

Sprint 1 (analytics): three separated site processes/SQLite stores, gateway with typed queries, naive disclosure and history heuristic, cross-site error handling, metrics. Core built. Remaining: actual Docker validation, broader query-family regression, policy heterogeneity, audit statistics, stronger gateway request accounting.

Sprint 2 (FL): PUBLIC digit PyTorch reference on three OS worker processes, IID/label-skew non-IID, per-client held-out validation and public-data local/pooled comparators: **executed and artifact-backed**. Native Flower ServerApp/ClientApp written but cannot be verified without dependencies; actual MNIST data and longer/5-seed ablations remain pending. **Gate:** verified Flower three-client execution and one-round numerical cross-check. Do not relabel the offline reference as Flower.

Sprint 3 (attacks): exact gradient inversion reference using controlled public MNIST, batch sizes and budgets; output-only membership inference; observe-only attacks, explicit unsuccessful results. **Gate:** scored genuine attack results and artifact images, no truth path into attack.

Sprint 4 (defenses): OpenDP query release with sensitivity+adjacency+accounting; Opacus with epsilon/delta and clipping; Flower SecAgg+ threshold and dropout assumptions. **Gate:** formal claim parameters and matched, tested baselines.

Sprint 5 (benchmarks): multiple clients, seeds, heterogeneity, gradient observability variants, DP budgets, performance. **Gate:** all plots regenerate from versioned results, negative results included.

Sprint 6 (release): Streamlit read-only experiment explorer, paper-style report, reproducibility package and faculty presentation. **Gate:** fresh-environment replication and no unsupported thesis novelty or real-digital-twin claim.

Track C simulation is optional and cannot block the two core tracks.
