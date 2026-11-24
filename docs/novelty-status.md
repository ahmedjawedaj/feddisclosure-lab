# Novelty status

Date: 2026-11-24. Covers the v0.15 line and v0.16.0.

**Status: novelty not established.** This is a literature and tracker check, not proof that nothing similar exists. It is repeated at the start of every phase and before any paper draft.

## What this file is for

Each phase of this repository produces measurements. This file records which of our claims are new, which are not, and what was and was not checked, so that no document in the repository claims more than this file allows.

## Closest prior work

The last column states what the work does not establish. That column is the gap we are allowed to claim, and nothing more.

| Work | Status and how checked | Direct overlap | Does not establish |
|---|---|---|---|
| Cebere et al., audits of 12 DP libraries (arXiv 2602.17454) | 2026 preprint, full text read in the v0.16 planning session | Audits stated guarantees of DP libraries by measurement | Covers no federated learning framework. FL is not discussed |
| Flower optional privacy accountant, `flwr` 1.44.0 | Source read this phase, release that introduced it unknown | Exact accounting for the Message API fixed clipping strategies. Rejects unequal weights and withholds the aggregate on a short reply count | Not in 1.27.0. Not in the older strategy API. Does not cover adaptive clipping or `LocalDpMod` |
| Flower issue 7522 and PR 7570 | Public, read in the v0.15 phase | Seed reduction in the secure aggregation random generator | Unrelated to DP calibration |
| Apple pfl-research 0.6.0 | Package source and PyPI metadata read this phase, tracker searched by title | Privacy mechanisms calibrated through accountants, joint accountant | Not yet measured by us. Tracker shows open issue 86 on bisection speed only |
| FedML 0.9.6 | Package source and PyPI metadata read this phase, last release Feb 2025, tracker searched by title | Gaussian, Laplace, clipping and RDP accountant modules | Open issue 1586 asks about sensitivity calculation. Not yet measured by us |
| NVIDIA FLARE 2.9.1 | Tracker searched by title this phase | Sparse vector and percentile privacy filters, with accounting changes in 2026 pull requests 4470 and 4498 and an older question in issue 343 | Filters are not Gaussian mechanisms, so our harness does not apply. Issue 5680 adds memorization audit components for fine-tuning, which is adjacent to our attack work |
| Dishonest-server membership inference (arXiv 2203.16463) and model inconsistency (arXiv 2111.07380) | Title and abstract level only | Active server attacks on federated learning | Not tied to DP or secure aggregation measurements here |
| Integration of an L-RDP accountant into Flower (arXiv 2510.12908) | Title and abstract level only | Accounting for Flower | Does not audit the executed mechanism |
| Balle and Wang 2018 | Formula used in the harness and checked numerically | Exact Gaussian mechanism delta and the limits of the classical calibration | Not a federated learning result |

## Claims disposition

| Claim | Verdict | Reason |
|---|---|---|
| The classical Gaussian calibration is only proved for epsilon below 1 and falls short at large epsilon | **Not novel** | Known since Balle and Wang 2018. We only measured where `LocalDpMod` sits on it |
| Noise calibrated for equal client weights is weaker when weights are unequal or results are fewer | **Not novel as a concept, partly resolved upstream** | Follows from the code. `flwr` 1.44.0 rejects both in accounted mode. The default mode and the older API are unchanged |
| Measured numbers for these Flower configurations, with exact epsilon and an empirical lower bound | **Established as a measurement, not as a discovery** | Pre-registered, reproduced on two versions, reviewed. Value is the harness and the numbers |
| Adaptive clipping updates the norm before noising, which lowers noise when clients are clipped | **Unverified against the source paper** | Direction depends on the clipped-bit convention, which was not compared with the paper |
| A comparison of DP mechanisms across federated learning frameworks | **Not established** | Only Flower measured. Frameworks listed above are not yet measured and their trackers were searched by title only |
| v0.15 results on SecAgg+ parity, LiRA and update inversion | **Not novel** | Reproductions at course scale. Verdicts record two failures and several weak passes |
| Active-server and dropout channels leak information that DP noise does not cover | **Hypothesis only** | Not tested. See `docs/v0.16-p2-server-view-inventory.md` |

## Validated and not validated

**Validated this phase.** Source of Flower 1.27.0 and 1.44.0, pfl 0.6.0 and FedML 0.9.6 and FLARE 2.9.1 package contents were read. PyPI release dates for pfl and FedML were read. Tracker search results were read by title and state for FedML, pfl-research and FLARE. All v0.16.0 numbers were re-derived by a separate review.

**Not validated.** Tracker threads were not read beyond titles. Flower's release history was not searched to find when the accountant arrived. Papers listed at title and abstract level were not read in full. No framework other than Flower has been run. The Andrew et al. adaptive clipping paper was not compared with the Flower implementation because the fetch needed an approval that was not given.

## Research decision

A phase may be released as a measurement with this status line. A paper draft may not be started until at least three frameworks are measured with prior-art tables filled from threads and release notes, and until the adaptive clipping row is checked against its paper.
