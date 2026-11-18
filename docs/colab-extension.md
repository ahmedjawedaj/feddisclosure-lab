# Future remote Colab design — NOT implemented

The original project owner intends to connect multiple Google Colab accounts later. Do not block the initial research release on this.

Each remote notebook should act as one **client worker**, loading only its own synthetic/public partition and pinned dependency versions. The coordinator should run on a durable, access-controlled host and be reachable through a secure, authenticated connection. The notebook should only receive its own partition and model parameters, and return only the permitted model updates. Do not place research API keys inside public notebook cells.

Colab runtimes expire, disconnect and have variable GPU availability. Reproducible rounds require idempotent client IDs, retry logic, timeout/dropout handling, checkpoints, and a manifest tracking which clients participated. Simulated distinct runtimes are **not** equivalent to independently governed institutions.

Before launching remote workers: validate Flower version compatibility; measure network roundtrips; decide permitted gradient/update visibility; document secure transport and participant threshold; verify that evaluator-only truth is absent from every client/attacker container.
