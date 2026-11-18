# FedDisclosure execution loops

## Loop 1: Engineering — every change

```
read current state -> choose one acceptance criterion -> write test -> implement
  -> run test suite -> run real integration -> save output -> update handoff
```

Exit condition: merged code has a passing reproducible command and no unsupported documentation claims. A planned directory without working code is not a completed task.

## Loop 2: Research — every experiment

```
question -> observer/threat model -> hypothesis -> fixed config -> reference
  -> attack/defense run -> evaluator scores -> ablations -> limitations -> report
```

Exit condition: actual machine-readable result artifacts, exact observations, matched baselines, negative results, correct error bars where applicable.

## Loop 3: Security/privacy review — every disclosure or defense change

```
what attacker sees -> what is forbidden -> what defense can change
  -> what formal claim is justified -> tests for boundary violations
  -> explicit caveats
```

Exit condition: no cross-role ground truth imports, no raw row transfer, no unaccounted privacy claims, site failures do not leak partial results.

## Loop 4: Scientific release — end of sprint

```
clean install -> tests -> network/Docker proof -> benchmark rerun
  -> compare to recorded baseline -> figures -> report -> commit -> state update
```

If an environment cannot run a gate, mark **NOT VERIFIED** with the reason and do not silently clear that gate.

## Loop 5: Faculty feedback — after meaningful benchmark

```
formulate one specific research limitation -> show evidence and negative cases
  -> ask for expert critique -> narrow research question -> update proposed thesis scope
```

Avoid selecting a novelty claim or digital twin without validating new contribution and collaboration feasibility.

## Compute plan

- Phase 1: single machine, three private data volumes, isolated Docker containers when a host is available.
- Phase 2: single host Flower clients, CPU first, optional local GPU or one Colab for costly attack reproduction.
- Phase 3: multiple remote Colab runtimes **only if** there is a safe reachable coordinator, a versioned experiment artifact channel and explicit connectivity/expiry handling. Separate clients should not assume persistent Colab runtimes.
