# Novelty status

Date: 2026-11-26 (first written 2026-11-24). Covers the v0.15 line and releases 0.16.0 to 0.16.11.

**Status: novelty not established.** This is a literature and tracker check, not proof that nothing similar exists. It is repeated at the start of every phase and before any paper draft. No statement here says that no prior work exists. Each row says what a check found and what it could not read.

## What this file is for

Each phase of this repository produces measurements. This file records which of our claims are new, which are not, and what was and was not checked, so that no document in the repository claims more than this file allows.

## How the checks were made

1. Public web pages were read through a summarising fetch tool. It returns a short answer about a page and not the page text. Every item below is a summary of a page unless stated otherwise.
2. The GitHub API was not available. Single issue and pull request pages loaded in a logged-out view, one at a time. Logged-out views hide some comments, so "no maintainer reply visible" means visible in that view and nothing stronger.
3. Search and list pages on GitHub were not available to automated clients. Titles were found with a web search limited to github.com, and each page was then fetched alone.
4. Release notes were not read for Flower or TFF. The Flower changelog pages did not render their entries, and TFF release notes were not searched.

## Closest prior work

The last column states what the work does not establish. That column is the gap we are allowed to claim, and nothing more.

| Work | Status and how checked | Direct overlap | Does not establish |
|---|---|---|---|
| Cebere et al., audits of 12 DP libraries (arXiv 2602.17454) | 2026 preprint. The first 100,000 of 109,716 characters were read through the summarising fetch in this refresh. The earlier note said full text was read, and the two statements differ | Audits stated guarantees of DP libraries by measurement | The part read names none of the four frameworks and does not discuss federated learning. The remainder was not read |
| Flower issues 8295 and 8297 (flwrlabs/flower), opened 2026-10-06, open | Pages read in the logged-out view on 2026-11-26. Comments not visible | Report that the adaptive clipping wrappers set the clipped-update bit opposite to the source paper, and that noise is scaled to the clip already updated for the next round | The reproduction scripts were not run by us in this check. Our measurement is separate. A linked pull request (8296) could not be read |
| Flower issue 8298 and pull request 8381, open | Pages read on 2026-11-26 | Integer arrays in the DP clipping helpers inflate the update norm. The pull request proposes excluding non-floating arrays | A related area. Our integer array row measured noise and found no privacy mismatch, which is a different question. Diffs not read |
| Flower optional privacy accountant, `flwr` 1.44.0 | Source read in the v0.16 phase. The release that introduced it is unknown, and release notes were not read | Exact accounting for the Message API fixed clipping strategies. Rejects unequal weights and withholds the aggregate on a short reply count | Not in 1.27.0. Not in the older strategy API. Does not cover adaptive clipping or `LocalDpMod` |
| Flower issue 7522 and PR 7570 | Public, read in the v0.15 phase | Seed reduction in the secure aggregation random generator | Unrelated to DP calibration |
| Wang, "Checking the Output Does Not Check the Aggregator" (NeurIPS 2026 listing) | Abstract only | Says re-performing an aggregation step is not a sound audit of robust aggregation rules, and reports a Flower 1.33.0 behaviour on client submissions | Full text and code not read. It concerns robust aggregation and not DP noise. It shows that a framework level audit of Flower exists in that sense, so we do not claim Flower was never audited |
| FedML 0.9.6 | Package source and PyPI metadata read, last release Feb 2025 | Gaussian, Laplace, clipping and RDP accountant modules | Measured in v0.16.2 for the DP classes called directly |
| FedML issue 1586, "DP Sensitivity Calculation", open | Page and its one reply read on 2026-11-26 | A reply from the assignee says sensitivity depends on the application, and that an API to define the Laplace mechanism by its scale was coming | Whether that API was released. It measures nothing. It shows the single sensitivity argument was a known question |
| NVIDIA FLARE 2.9.1 | Tracker pages read on 2026-11-26 | Sparse vector and percentile privacy filters. Pull request 4470 added a filter level accountant and 4498 clarified the documentation. Issue 343 was an older, fixed question on the same filter | Diffs not read. None of these tests the executed mechanism. Issue 5680 adds memorization audit components for fine-tuning, which is adjacent to our attack work |
| TensorFlow Federated 0.87.0 | Newest on PyPI, uploaded 2024-09-17. Documentation read through the summarising fetch. Release notes not searched | Differentially private aggregator factories with a stated DP event and no stated adjacency relation | A web search limited to github.com showed the title of a closed tensorflow/privacy thread on adaptive clipping noise (not opened here, and no number or date was kept). Its title is consistent with our budget split finding. No tracker item about configured versus delivered noise was found. That is a search result and not proof of absence |
| Distributed DP for TFF (`DistributedDpSumFactory`) and the mechanism papers on the Skellam and distributed discrete Gaussian sums (arXiv 2110.04995 and 2102.06387) | Search titles only, none opened | The papers define the mechanisms | They are not audits of an implementation. No hit measured what the shipped factory delivers against its configuration. That is a search result and not proof of absence |
| Dishonest-server membership inference (arXiv 2203.16463) and model inconsistency (arXiv 2111.07380) | Title and abstract level only | Active server attacks on federated learning | Not tied to DP or secure aggregation measurements here |
| Andrew et al., adaptive clipping (arXiv 1905.03871) | Read through summarising fetches in the v0.16.3 phase and again in this refresh | Source of the clipped-bit meaning, the geometric update, the noise order and the budget split | The paper does not state in words why the budget relation has a factor 2. Our centred reading of the bit is an inference |
| Andrew et al., one-shot empirical privacy estimation (arXiv 2302.03098) | PDF title and authors confirmed | Estimates empirical privacy inside one federated training run | Does not measure shipped framework classes against configured parameters |
| Behnia et al., local DP with fixed memory (arXiv 2510.12908) | PDF title, authors and the Flower mention confirmed | Adds an accountant for local DP and integrates it into Flower | Does not audit the mechanism Flower already runs, as far as the part read shows |
| Valadi et al., practical feasibility of gradient inversion (arXiv 2508.19819) | Text fetched through the summarising fetch, not read line by line | Finds reconstruction mostly fails in realistic settings | No label-only or class-average floor, and no audit of the four frameworks |
| Balle and Wang 2018 | Formula used in the harnesses and checked numerically | Exact Gaussian mechanism delta and the limits of the classical calibration | Not a federated learning result |

## Claims disposition

| Claim | Verdict | Reason |
|---|---|---|
| The classical Gaussian calibration is only proved for epsilon below 1 and falls short at large epsilon | **Not novel** | Known since Balle and Wang 2018. We only measured where `LocalDpMod` sits on it |
| Noise calibrated for equal client weights is weaker when weights are unequal or results are fewer | **Not novel as a concept, partly resolved upstream** | Follows from the code. `flwr` 1.44.0 rejects both in accounted mode. The default mode and the older API are unchanged |
| Measured numbers for these Flower configurations, with exact epsilon and an empirical lower bound | **Established as a measurement, not as a discovery** | Pre-registered, reproduced on two versions, reviewed. Value is the harness and the numbers |
| The adaptive clipping bit direction in Flower | **Reported before our measurement** | Issues 8295 and 8297 predate it. What remains ours is the pre-registered measurement across four classes and two versions and the composed multiplier. We do not call it a first report |
| Adaptive clipping updates the norm before noising, which lowers noise when clients are clipped | **Measured against the source paper in v0.16.3** | The comparison used summarising fetches and not a full read of the paper |
| A pre-registered measurement of shipped DP classes in four framework versions against their configured parameters, including the TFF distributed DP sum | **Established as a measurement. Novelty not established** | We did not find a prior measurement of this kind in the pages opened. We do not claim none exists |
| v0.15 results on SecAgg+ parity, LiRA and update inversion | **Not novel** | Reproductions at course scale. Verdicts record two failures and several weak passes |
| Active-server and dropout channels leak information that DP noise does not cover | **Hypothesis only** | Only the sum of clipped updates, the release rule and the weight effect were measured, in v0.16.0, and the v0.15 tests cover the array content. The pre-registered predictions S4, S6 and S6b in `docs/v0.16-p2-server-view-inventory.md` were not run |
| A label-only class-average floor for federated gradient inversion | **Not claimed as new** | We did not find it used this way in the pages opened |
| Possible repeated noise between clients in the TFF distributed sum | **Open observation, not a finding** | First item in `NEXT_SPRINT.md`. No tracker thread was found, which is a search result and not proof of absence |

## Validated and not validated

**Validated.** Source of Flower 1.27.0 and 1.44.0, FedML 0.9.6, FLARE 2.9.1 and TFF 0.87.0 package contents were read and the DP classes were run. All v0.16.0 numbers were re-derived by a separate review. Reviewer notes for some later releases are in their own documents, and not every release had one. Single tracker threads for the items above were read in a logged-out view.

**Not validated.** Comments hidden in the logged-out view. Release notes for Flower and TFF. The release that introduced the Flower accountant. Pull request 8296 and the diffs of 8381, 4470 and 4498. The remainder of the Cebere et al. text. Papers listed at title and abstract level. No paper named here was compared with a framework beyond the rows above.

## Research decision

A phase may be released as a measurement with this status line. A paper draft needs three things.

1. At least three frameworks measured. **Met.** Four are measured.
2. Prior-art tables filled from threads and release notes. **Only partly met.** Threads were read for the Flower, FedML, FLARE and one TFF item. Release notes were not read for Flower or TFF. This stays open.
3. The adaptive clipping row checked against its paper. **Met** in v0.16.3, through summarising fetches and not a line by line read.

A draft may be written while item 2 is open if it states that fact. The draft makes no claim that depends on release notes.
