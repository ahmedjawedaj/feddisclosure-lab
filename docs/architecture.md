# Architecture and trust boundaries

## Running in Sprint 1

```text
Analytics user (published aggregate observations only)
                    |
                    v
        Gateway :8080 (localhost only)
           /        |        \
          v         v         v
     Site A      Site B      Site C
     FastAPI     FastAPI     FastAPI
     SQLite A    SQLite B    SQLite C
          \         |         /
           aggregate counts only
                    |
          Research evaluator
     synthetic truth + observed outcomes
```

Gateway fans out typed count requests and sums site responses, returning no raw rows or per-site counts to the analytics user. Each site enforces its own statistical policy and maintains a persisted per-experiment history. A single site rejection fails the whole federation request, rather than leaking a partial sum.

Data ownership: client service A alone mounts A's SQLite volume, and similarly B/C. Gateway does not mount any site volume. In Docker all site services are on an internal network and only the gateway binds a host port on loopback. In the verified local process harness the services have distinct DB paths and localhost sockets, but do **not** have container or OS-level filesystem isolation. The latter is strictly weaker than the Docker design.

Authentication: internal shared token for gateway -> institution, separate research key for user -> gateway. These are educational controls, not production-grade identity/authz. Policy selection is an explicit experiment control and must not be offered to arbitrary users in production.

## Planned in Sprints 2–4

- Flower SuperLink/ServerApp coordination and independent ClientApps/SuperNodes for actual distributed training.
- PyTorch MNIST/model parameters under versioned configs, IID and non-IID partition manifests.
- Separate attack subprocess/container with **only** approved gradient, update or model-output observation artifacts.
- Privacy evaluation with strictly segregated private synthetic ground truth.
- Opacus, OpenDP and Flower SecAgg+ in the relevant, separately documented threat models.

The existence of future directories or YAML plans does not mean these components are implemented.
