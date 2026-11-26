# Measuring the noise that shipped differential privacy code in four federated learning frameworks delivers against its configured parameters

Ahmed Jawed. Draft for review by the supervisor. Affiliation and venue to be added.

Status: draft P5, not peer reviewed. All measurements concern named versions and configurations. Nothing in this paper is a statement that a framework is or is not private.

## Abstract

Federated learning frameworks ship differential privacy (DP) code whose parameters imply a privacy level. We ask whether the mechanism that runs matches that level. We ran the shipped DP classes of Flower (1.27.0 and 1.44.0), FedML 0.9.6, NVIDIA FLARE 2.9.1 and TensorFlow Federated (TFF) 0.87.0 in process on synthetic updates, with predictions committed before the harness was committed and a stated power for most null results (void, vacuous and missing power statements are listed). We find that the configured noise is delivered when client weights are equal and the expected number of clients reports: with equal client weights and a full cohort, the central Gaussian classes of Flower and TFF deliver the configured noise within 0.04% and 0.21%. Two Flower configurations differ from the configured reading. For one round at delta 1e-5 with zero-out adjacency, a client holding 91% of the weight leaves an effective noise multiplier of 0.110 and an exact epsilon of 79.4 where the configured reading gives 4.38. With `num_sampled_clients` above the number of aggregated results the multiplier is 0.200 and the exact epsilon 33.1. Flower 1.44.0 has an optional accountant that rejects these inputs, and it is off by default. As we read the paper, and as public Flower issues also report, Flower's adaptive clipping sets the clip bit opposite to the paper reading and scales noise with the clip after the update, and TFF follows the paper on both. FedML's local DP class applies no clipping, and for the events and input pairs tested no empirical lower bound exceeded the claimed epsilon of the FLARE sparse vector filter (each is a lower bound and not a certificate). TFF's distributed DP sum delivers the configured noise at the expected client count, a multiplier of 0.708 and 1.407 when 5 and 20 clients report against 10, and more noise than configured for the discrete Gaussian at low bit widths. Two inversion studies on CIFAR-10 found that with 8 records per client the attack did not beat a label-only floor in PSNR in the plaintext view (SSIM disagreed), and with 2 records it did. For Flower 1.27.0, three further pre-registered predictions about what the server process observes all held. In two Docker runs with SecAgg+, the metric values each client returned reached the server in 8 of 8 replies. In process, with the fixed clipping DP strategy called directly and without SecAgg+, under a participation dependence that we assumed and did not observe, the release bit and the result count reflected participation. We list every failed prediction, void power statement and limit, including one open observation we could not resolve.

## 1 Introduction

A framework user who configures a noise multiplier, a clipping norm and a number of sampled clients computes a privacy level from those numbers. The level holds only if the code adds the noise those parameters imply, for the sensitivity the code actually has. That gap is cheap to measure for one class and rarely measured for a framework.

This paper reports such measurements for four open source frameworks. For each shipped DP class we measured the noise standard deviation and the sensitivity of the output to one client, and from them the Gaussian parameter mu and the exact (epsilon, delta), with an empirical lower bound on epsilon where a check has power. The attack side asks what a gradient inversion attack recovers from the views a server can see, against a label-only floor.

The contributions are pre-registered measurement tables for Flower, FedML, NVIDIA FLARE and TFF (including its Gaussian and its distributed DP aggregators) for named versions, a reporting style in which each null result carries a detection power and each failed prediction stays in the record, and, as a secondary result, two pre-registered inversion studies with a label-only floor and gates that decide which rows may be read. The inversion studies run an attack library and no framework code.

We do not claim that any framework is or is not private, or what its authors intended. Each "claimed" value is defined per row from the parameters, class documentation or the paper the class cites.

## 2 Related work and prior art

We checked public pages in November 2026, through a summarising fetch tool, in a logged-out view that hides some comments. Release notes of Flower and TFF were not read, so the prior-art check is incomplete. Cebere et al. audit 12 DP libraries and report 13 violations [3]. The part we opened names none of the four frameworks, and we did not read the remainder. Andrew et al. estimate empirical privacy inside one federated training run [6], and Behnia et al. add an accountant for local DP to Flower [7] without auditing the mechanism Flower already runs. We did not find a measurement of the DP classes of the four frameworks against their configured parameters, within the searches we made.

Three Flower issues opened on 2026-10-06 by one third party report observations in the same code as some of our rows: issue 8295 the clip moving away from the target quantile, issue 8297 noise scaled with the next round's clip, and issue 8298 integer arrays disturbing the clipping norm [12]. Our adaptive clipping measurement is therefore not the first report of the bit direction or the noise order. It adds a pre-registered measurement on two versions and four strategy classes, and the composed multiplier. Likewise the effect of unequal weights and short result counts follows from the code, and is not new as a concept. What we add is the measurement and the exact epsilon. FedML issue 1586 (2023, open) asks how to compute `sensitivity`, NVIDIA FLARE issue 343 (2022) reported that the filter could leave values neither noised nor zeroed, and pull requests 4470 and 4498 (April 2026) refined its accounting and documentation [12].

Pasquini et al. [8] and Pichler et al. [9] study attacks by a dishonest server. Wang reports, in an abstract we read, that re-performing an aggregation step is not a sound audit of robust aggregation rules and that Flower 1.33.0 deletes a round's client submissions when the server collects replies [15], which concerns robust aggregation and not DP noise. Geiping et al. [4] introduced the inversion attack that `breaching` implements [13]. Valadi et al. report that reconstruction mostly fails in realistic settings [14], with no label-only or class-average floor in the text we read. Hayes et al. [5] study reconstruction under DP-SGD against an adversary with a prior. We did not find a class-average floor used this way for federated gradient inversion in the pages we opened, and we do not claim that none exists.

## 3 Methods

### 3.1 Subjects

| Framework | Version | Code measured |
|---|---|---|
| Flower | `flwr` 1.27.0 (pinned by the repository) and 1.44.0 | fixed and adaptive clipping strategies (legacy and Message API) and `LocalDpMod`, run in process on fake client replies |
| FedML | 0.9.6 (2025-02-24, newest on PyPI as of November 2026) | classes under `fedml/core/dp`, called directly, FedML's own `__init__` never executed |
| NVIDIA FLARE | 2.9.1 | `SVTPrivacy` and `PercentilePrivacy`, `process_dxo` called in process |
| TFF | 0.87.0 (newest on PyPI as of November 2026, uploaded 2024-09-17) | `DifferentiallyPrivateFactory`, `MeanFactory`, `tff.learning.dp_aggregator` and `DistributedDpSumFactory`, through TFF's C++ runtime |

The FedML DP files are byte-identical in seven earlier releases and the two FLARE filter files in 2.8.3, 2.9.0 and 2.9.1, but only the versions in the table were run.

### 3.2 Pre-registration

Each phase has a markdown file with a question, what was already known, numeric predictions with tolerances, a power section and a deviation log. It is committed before the harness and any run, nothing above the deviation log is edited afterwards, and a release gate (run on pull requests since release 0.16.7) checks this against the commit that added it. Two points limit this. The harness was drafted in the same session as its pre-registration, with predictions from code reading, and commit order is the only evidence of order, so history cannot prove that no mechanism ran earlier. Some rows are partly confirmatory because a quantity had been computed before the file was written.

### 3.3 Measurement of a DP mechanism

For each row the real class runs. Sensitivity is measured with the noise multiplier set to 0, on two neighbouring inputs. The noise scale is the per-coordinate standard deviation over repeated runs, 20,000 runs in the Flower harness. The Gaussian parameter is mu, the sensitivity divided by the measured sigma. The exact (epsilon, delta) follows from the Balle and Wang characterisation of the Gaussian mechanism [2], computed from the measured sigma unless a row says it used the formula sigma. The empirical lower bound on epsilon is a 95% Clopper-Pearson bound from a threshold test with 200,000 trials per world. It cannot certify more than about 10.9 at that trial count.

Adjacency is stated per row: zero-out for Flower central rows (replace-one as a second reading), replace-one for local DP, and add or remove for TFF, whose documentation states none.

The verdict words in Table 1 are ours and summarise each audit document. "Matches" means the measured value is inside the pre-registered tolerance for that row. "Differs" means the measured value is outside the tolerance around the configured reading, whether or not the pre-registration predicted it. "Meets" means the exact epsilon at the configured delta is not above the configured epsilon. "Holds" means every empirical lower bound is at or below the claim. Meets is under the stated sensitivity. Tolerances are in the audit documents.

### 3.4 Power statements

Every null result is paired with the smallest effect the check would find with 80% power, with the run count it assumes. Where a check could not fail at its claim, we say so and treat the null as uninformative. Positive controls are used where possible. In the FLARE events study 500,000 runs per input give a lower bound ceiling of 11.64, far above every claim.

### 3.5 Inversion studies

Both studies use `breaching` 0.1.2 with 4,000 iterations against a small CNN with 25,578 parameters (held-out accuracy 0.5802) on real CIFAR-10, with 4 clients, one local step, known labels, a known initial model and 20 evaluation seeds. The views are plaintext (the victim's update), SecAgg+ aggregate (simulated with the upstream quantiser arithmetic) and n-1 collusion. The floor rebuilds each victim record as the class-average image of a public pool. Gain is the paired difference in PSNR (dB) or SSIM between attack and floor on the same image, averaged per seed, with a 95% percentile bootstrap over seeds (10,000 resamples). In P3b a row behind a failed plaintext gate is declared NOT INTERPRETABLE in advance.

## 4 Results

### 4.1 Overview

Table 1 gives, per mechanism and framework, the claimed reading, the measured result and a verdict. "n/a" means the code measured has no such mechanism. "not measured" means the row was not run.

**Table 1.** Mechanisms by frameworks. Claimed reading in the first clause, measured result next, verdict last. Verdicts are under the stated sensitivity and the verdict definitions of Section 3.3. Row labels refer to the audit documents. Local DP epsilons come from the formula sigma, and measured sigma gives values within 0.3%.

| Mechanism | Flower 1.27.0 and 1.44.0 | FedML 0.9.6 | NVIDIA FLARE 2.9.1 | TFF 0.87.0 |
|---|---|---|---|---|
| Central Gaussian, fixed clip, equal weights, all results arrive | Claimed multiplier 1 (epsilon 4.38). Within 0.04% (C1). Matches | `GlobalDP` noise path only. Sigma ratio 1.0004 (CD1). Matches | n/a | Ratios 1.0021, 1.0003, 0.9983 (F1). Matches |
| Central Gaussian, fewer results than configured count | Multiplier 0.200, exact epsilon 33.1, lower bound 10.49 (C4). Differs | not measured | n/a | Delivered multiplier 0.9997 and 1.0012 (F3). Matches. The distributed DP sum differs, see its row |
| Central Gaussian, unequal client weights | Multiplier 0.110, exact epsilon 79.4, lower bound 10.90 (C3). Differs | not measured | n/a | Factory unweighted. `MeanFactory` ratios 0.9993 and 0.9996 (F5). Matches |
| Adaptive clipping | Bit opposite to the paper reading, noise scaled with the clip after the update, composed multiplier 0.8952 in the default cell (A3, A6, G3). Differs from the paper reading | not measured | n/a | Direction as in the paper, noise uses the clip the clients used (0.9988, 0.9989), composed multiplier 0.9989 and 1.0049 (A3, A4, A7). Matches |
| Local Gaussian, classical calibration | `LocalDpMod` meets up to epsilon 5 and not at 10 and 20 (L1). With `sensitivity` equal to the clip it meets at none (L2) | `LocalDP` at epsilon 1 meets for norm 1 (0.751) and not for norms 5 and 25 (4.54, 34.5). No clipping (LD1, LD2) | n/a | not measured |
| Laplace | n/a | Meets for an L1 bound. An L2 bound of 1 passed as `sensitivity`, for an update spread evenly over 16 coordinates, executes pure epsilon 3.999 against 1.00001 implied (LP2). Configuration note | n/a | n/a |
| Sparse vector filter | n/a | n/a | Ledger and Laplace scales match their formulas (N1, N2). No lower bound above its claim (S1 to S3, E1 to E6). Holds | n/a |
| Distributed DP sum (Skellam, discrete Gaussian) | n/a | n/a | n/a | Within 0.2% at the expected count, tolerance 0.7% (A1). Multiplier 0.708 and 1.407 at 5 and 20 of 10 (C). Discrete Gaussian 2.41687 times configured at 4 bits (A2). Sensitivity 1.45129 times the clip at 5 bits (D). Differs outside the expected count and at low bits |
| Accounting that the framework reports | None in 1.27.0. Optional accountant in 1.44.0 raises on C3 and releases nothing on C4 | RDP conversion 4.729 against exact 4.377. `get_epsilon` raises after the first step (RA1, RA2, CD2) | Filter-level ledger. One call at defaults records epsilon 255.9 (N3) | Per-round `DpEvent`, no epsilon in the Gaussian aggregators (F4). `UnsupportedDpEvent` with no fields in the distributed sum (G) |


### 4.2 Flower: central DP with fixed clipping

The setup is 16 float32 parameters, clipping norm 1, noise multiplier 1, `num_sampled_clients` 10 and delta 1e-5. 

**Table 2.** Flower central DP, fixed clipping (claimed multiplier 1, epsilon 4.38 at delta 1e-5).

| Row | Configuration | Sensitivity | Sigma | Effective multiplier | Exact epsilon | Empirical lower bound |
|---|---|---|---|---|---|---|
| C1 | 10 results, equal weights | 0.100 | 0.09996 | 1.000 | 4.38 | not run |
| C3 | target reports 91 examples, nine others report 1 | 0.910 | 0.09996 | 0.110 | 79.4 | 10.90 (at the ceiling) |
| C4 | `num_sampled_clients` 10, 2 results aggregated, no failure reported | 0.500 | 0.09996 | 0.200 | 33.1 | 10.49 |

The numbers hold for both strategy APIs on both versions. A Kolmogorov-Smirnov check is consistent with Gaussian noise (p = 0.72), a weak check. Noise is scaled as the multiplier times the clipping norm over `num_sampled_clients`, while the wrapped FedAvg weights clients by reported `num_examples`, so a client with weight share s moves the aggregate by s times the clip and the effective noise multiplier for the client with the large share falls. Under replace-one adjacency the C1 sensitivity is 0.200 and the exact epsilon 10.0. A reported client failure does not lead to C4, because the wrapper then releases no aggregate. C4 is reached when `num_sampled_clients` differs from the number of aggregated results, and Flower logs a warning. Flower's how-to page labels DP a preview [10] and states no adjacency relation.

On 1.44.0 an optional accountant exists on the Message API fixed clipping strategies. With it, C1 reports epsilon 4.73 by RDP against the exact 4.38, the C3 input raises `ValueError: Accounted aggregation requires equal client weights.`, and the C4 input releases nothing. Without it, and in the legacy API, C3 and C4 are unchanged. The introducing release is unknown. Whether replies that are absent at a timeout also lead to C4 in the Message API was not verified. Further rows: a client returning the global model unchanged raises `ZeroDivisionError` in the clipping helper (C0), the client-side variant relies on clients to clip (sensitivity 0.5 for an update of norm 5 times the clip, C5), and integer arrays gave no privacy mismatch (C6).

### 4.3 Flower: adaptive clipping against Andrew et al.

The four adaptive strategy classes ran on both versions, 82 rows per version, all of which held, with the same numbers on both versions and all four variants. The reference is Andrew et al. [1]. The noise budget split, default count noise and noisy fraction follow the paper (A1, A2, A5). The bit feeding the clip update is set when an update is clipped, where the paper's bit is 1 when it is not. With all updates clipped, the next clip from 0.1 is 0.0904837 where the paper reading gives 0.1105171 (A3), and after 23 such rounds 0.010025884 against 0.9974182 (A10). Against fixed norms from 0.5 to 1.5, 200 rounds take a clip starting at 0.1 to 2.0611536e-10 and one starting at 10 to 4851651954.1, so the clip moves away from the target quantile (A11). Model noise is scaled with the clip after the round's update, not the clip used in the round. Measured noise divided by the multiplier times the new clip is 1.0014, and times the old clip 0.6092, where the paper reading expects 1.0 for the old clip (A6). With fewer replies than `num_sampled_clients`, the noise scale uses the configured count and the fraction the reply count, and Flower logs a warning (A7).

The whole-mechanism study simulated 10 trajectories of 30 rounds in each of 24 cells per variant with 100 clients, and all 640 rows per version held. No epsilon was computed. With all updates clipped, count noise 5.0, multiplier 1.0 and learning rate 0.2, the effective multiplier of the model sum is 0.910 of configured. Flower releases a plain sum of bits in {0, 1}, so count sensitivity is 1 under add or remove, where the paper's centred bit has magnitude 1/2 (the centred reading of the paper's factor 2 is an inference). Under replacement the range is 1 for both, and the comparison then depends on a model sum sensitivity that was not measured. Both readings are reported and we make no claim about which Flower intends. The composed multiplier is 0.8952 of configured with Flower's count and 0.9062 with the paper's count sensitivity. At learning rates 0.05 and 1.0 the same quantity is 0.980 and 0.609 of configured (0.975 and 0.606 of the stored multiplier). Fixed clipping classes have exactly the configured multiplier in every round. These assume Gaussian releases, 100 equal-weight clients, a joint release of sum and count, and composition by adding precisions.

### 4.4 Flower: local DP

`LocalDpMod` adds Gaussian noise with sigma = sensitivity times sqrt(2 ln(1.25 / delta)) over epsilon. Measured sigma matches within 0.2% in every row, and the replace-one sensitivity of the clipped update is 2C. The classical calibration is proved only for epsilon below 1. At delta 1e-5 and `sensitivity` 2C, configured epsilon 0.5, 1, 2 and 5 gives exact epsilon 0.35, 0.75, 1.61 and 4.54 (formula sigma) and meets. Configured 10 and 20 gives 10.39 and 25.44 and does not meet, and at 20 the exact delta is 153 times the configured one. With `sensitivity` set equal to C, configured 0.5 to 20 gives 0.75, 1.61, 3.51, 10.39, 25.44 and 68.49, so no tested epsilon meets. No Flower text was found that recommends that setting. The Clopper-Pearson test has no power at these levels, as the pre-registration states.

### 4.5 FedML

The harness calls the DP classes of `fedml` 0.9.6 directly. 130 checks were built from the pre-registered rows, 18 added after the first quick run, and both failed checks were in the original set. 128 held. FedML ships no documentation text for these parameters in the wheel, so "claimed" is the reading implied by class docstrings and cited papers. FedML's call sites and online documentation were not exercised or read.

The Gaussian and Laplace mechanisms deliver their formula's scale within 0.5% in all 18 cells, and the Gaussian class raises `ValueError` for epsilon above 1. `sensitivity` is one argument for both, while the Gaussian class needs an L2 bound and the Laplace class an L1 bound (LP2, Table 1). `LocalDP` does no clipping and `max_grad_norm` has no effect (LD2). With `sensitivity` 2, epsilon 1 and update norms 1, 5 and 25, exact epsilon is 0.751, 4.54 and 34.5, and the empirical bound certifies 0.17, 1.17 and 10.57. `GlobalDP` adds the configured noise when its accountant is off (sigma ratio 1.0004), and with `enable_rdp_accountant` true its construction raises `TypeError` (CD2). `DP_Clip.compute_noise` divides its stored scale by `qw` on every call. For z 1, clip 1 and `qw` 100 the sigma is 1e-2, 1e-4 and 1e-6 for calls 1, 2 and 3, and exactly zero from call 23 (K2). `DP_Clip.add_global_noise` returns `None` (K3). Reading the source suggests no call site selects this class, which was read and not run. In `RDP_Accountant`, `get_epsilon` raises after the first step in all three configurations tried, and the conversion underneath gives 4.729 at noise multiplier 1 where the exact value is 4.377, 8.0% above and a valid upper bound.

### 4.6 NVIDIA FLARE

The `SVTPrivacy` and `PercentilePrivacy` filters were called in process. 18 pre-registered rows became 63 checks, and all held. The documentation scopes the guarantee to the filter and not to end-to-end accounting [11]. The ledger split is epsilon over 1 plus (2n) to the 2/3 for the threshold phase, with no relative difference in 16 cells (N1). With defaults on 25,578 parameters one call records epsilon 255.9, of which 255.8 is the release phase, and ten calls sum to 2559.0 with delta 0 (N3). The acceptance step runs a whole pass, so with epsilon 100 and 100 entries wanted, 656 of 1,000 entries were accepted in one pass (N4). There is no pass limit. At epsilon 1e4 a call returned after 30 passes, and at 1e6 it had not returned after 10 seconds (N5). With the default `replace` True, fewer than `n_upload` entries are released in every one of 200 runs (mean 92.89 of 100, N7). `PercentilePrivacy` is deterministic and its documentation makes no DP claim.

The first selection check (N6b) had a claim of 20 and could not fail, because 200,000 runs cannot give a lower bound above about 10.9. Later studies replaced it. Three small configurations of 1,000,000 calls gave lower bounds 0.418, 0.357 and 0.015 against claims 1, 1 and 0.1, with 80% power values 1.02, 1.03 and 0.126. Six larger configurations (default `replace` True, values between 0 and gamma, release values, a joint event) gave lower bounds 0.414, 0.206, 0.224 and 0.280 against 1, 0.272 against 0.3 for the release values and 0.502 against 1.2 for the joint event, with 80% power values 1.083, 1.063, 1.079, 1.064, 0.316 and 1.259. None exceeded its claim. The release row is closest, with an event epsilon of 0.976 of `epsilon_release` by a closed form and 0.962 measured. Each bound covers one event and one pair of inputs only.

### 4.7 TFF Gaussian aggregators

15 of 16 rows and 126 of 127 checks held. The one failed check is a pre-registered number that was an arithmetic slip (Section 4.11). Fixed clipping noise equals the multiplier times the clip over `clients_per_round` within 0.21%, and one client's effect is exactly that clip over `clients_per_round`. With 5 or 20 clients against 10, the noise ratios are 0.9997 and 1.0012. A division by the reporting count would have given 0.5 and 2.0, and the check saw 0.4998 and 2.0024. Adaptive clipping lowers the clip when more clients are below it, as in the paper. The delivered noise matches the old clip within 0.12% and would be 10% higher than the new clip's value (mean ratio 1.1003 and 1.1005), and the composed delivered multiplier is 0.9989 for n 100 and 1.0049 for n 12. The counted fraction uses the configured client number, so with 50 of 100 clients all below the clip it is 0.75 and not 1. The default zeroing step of `dp_aggregator` replaces a client with a largest coordinate above 10 by zeros, which then counts as below the clip. Two fresh processes that each set the Python and TensorFlow seed gave different outputs. These aggregators report a per-round `DpEvent` and no epsilon, and none was computed.

### 4.8 TFF distributed DP sum

The script runs the real `DistributedDpSumFactory` (Skellam and discrete Gaussian mechanisms, discrete Fourier and Hadamard rotations, modular clipping, adaptive clip) through `process.next`. A call of `next` failed in TFF's default local C++ context with the gRPC error "Cannot embed a federated value", so every row used the test C++ context. Each row uses one float32 tensor of 2048 coordinates, noise rows send zero vectors, and the secure sum is a plain sum in this runtime. The pre-registration (commit `b32e31c`) was committed alone before the script. 8 of 12 rows and 194 of 199 checks held, and five failed (below). The run took 375.1 seconds. Adjacency is read as add or remove.

**Table 3.** TFF distributed DP sum against its configuration. Noise is the root mean square of delivered noise over `noise_multiplier * l2_clip`. Row labels refer to the audit document.

| Question | Claimed reading | Measured | Verdict |
|---|---|---|---|
| Noise at the expected count (A1) | multiplier 1 at 10 clients | 1.00121 and 0.99871 (Skellam, two rotations), 1.00104 and 1.00196 (discrete Gaussian), each within 0.2% of prediction (tolerance 0.7%) | Matches |
| Docstring against code (B) | local noise is the central one divided by `expected_clients_per_round` | the code divides by its square root. Noise 0.99641 for 4 clients and 1.00015 for 25, which is 1.99 and 5.00 times the 0.5 and 0.2 the sentence implies | Documentation and code differ. The configured multiplier is delivered |
| Reporting count differs from expected (C) | multiplier stays 1 | noise 0.70807 and 1.40720 at 5 and 20 reporting against 10 (discrete Gaussian 0.70547 and 1.40360), sensitivity stays 1.0 times the clip, so the delivered multiplier is 0.708 and 1.407 | Differs. Holds only at the expected count |
| Bit width (A2) | configured level | Skellam 0.99714, 1.00494, 0.99952 and 1.00175 at 4, 5, 8 and 12 bits. Discrete Gaussian 2.41687, 1.67157, 1.08342 and 1.00292. Dense input rounding adds 1.02155 at 5 bits and 1.06457 at 4 | Matches for Skellam with zero inputs. Above configured for the discrete Gaussian at low bits |
| Sensitivity above the clip (D) | the clip | mean change 1.00000, 1.00708 and 1.45129 times the clip at 16, 8 and 5 bits, above the clip in every call at 8 and 5 bits (80 calls) | Differs at 8 and 5 bits |
| Modular range (E) | not stated | `modclip_prob` 0.3 and 0.7 gives noise 0.60842 and 0.22789, equal to the wrap model (0.60814 and 0.22783) and not saturation (0.74456 and 0.35147). True sum norms 16 and 64 came out as 8.96 and 9.00 with no error | Wraps, does not saturate |
| Adaptive clip (F) | not stated | noise follows the clip the clients used (ratios 0.9948 to 1.0039), and the Python seed does not control noise (H) | Order as in the paper reading |
| Accounting (G) | an epsilon or Gaussian event | `UnsupportedDpEvent` with no fields. No name contains "epsilon" in eight source files. A docstring TODO mentions exact accounting | No epsilon reported |
| `ddp_secure_aggregator` (J) | RMS 0.0057735 and 0.0081650 | constructs, and the call of `next` did not run | Not measured |

No epsilon guarantee is computed or implied. The audit document labels one table of implied epsilons as a Gaussian approximation conditional on the measured numbers, with add or remove adjacency, ignoring discrete noise, wrap, composition and sampling. It gives 4.371 for Skellam at 10 of 10 and 6.562 and 2.960 at 5 and 20 reporting. These are not guarantees.

Failed predictions that stand. The noise-free norm at 20 clients was 9.97287 against 10, an exact check with no replication. A scratch run outside the evidence file fits a wrap of one coordinate in about 18.5% of calls when 20 aligned clients reach the design bound of the modular range, which is a reading and was not tested. Row J did not run. Three statistical checks failed in the first run and held in the single replication the rule allows, so no mismatch is called: the spread of Skellam per-call RMS (0.0218 against a window of 0.0116 to 0.0197, then 0.0172), the 16-bit rounding value (1.00812, then 1.00084) and the beta difference at 5 bits (0.00484, then 0.00013).

Open observation, possibly repeated noise between clients. This is not a finding and is unresolved. The first run had one Skellam call at RMS 1.123. A scratch run after the evidence made 500 zero-input calls per mechanism. Seven Skellam calls had an RMS between 1.094 and 1.128, against none for the discrete Gaussian (largest 1.046), and two clients drawing the same noise would give about 1.095. Skellam seeds come from a microsecond clock, and the cause was not isolated. An independent probe by a separate reviewer instance (see Section 6) with its own script (1,500 calls per mechanism at 10 clients, zero inputs, 16 bits) found 10 Skellam calls above 1.06 times the mean (0.67%, largest 1.236) and none for the discrete Gaussian (largest 1.044). That rate is below the 1.4% of the scratch run, the intervals overlap, and the largest value is higher than one colliding pair would give. Neither run is in the evidence file, and the A1 check could find a share of about 2.3% with 80% power.

Review notes on power. The beta check has a standard error of the difference of about 0.0035 at 5 bits, so it passes about 23% of the time at a true difference of zero, and "held on replication" says nothing about beta. The sensitivity check of C (norm 3 against zero, 1.0 within 1e-5, one call) is stochastic at about 1e-5 with no replication and no power statement. 3 of 73 statistical checks failed in the first run against a pre-registered estimate of under 1% for any false failure, and the 16-bit value was 4.65 standard errors from 1 where the design allowed 4. Some of the 199 checks do not test TFF, and the H threshold in the script (0.01398) was stricter than the pre-registered 0.0198.

### 4.9 What the server sees: an inventory and three measured predictions

The server-view inventory for Flower 1.27.0 lists 8 channels (among them the weight folded into the masked vector, metrics returned with the fit result, which clients sent masked vectors, and whether a round released an aggregate) and states for each whether SecAgg+ or DP covers it or it was not examined. Three predictions about it (S4, S6, S6b) were pre-registered before the harness and before the full run, and all held. None failed and no row was re-drawn (seed 0 needed one automatic retry of the CLI call for a start order problem). The inventory does not say that any channel leaks anything.

S4 used real Docker runs of the SecAgg+ stack, two runs of 4 clients. The three metrics each client returned (two floats and an integer, added by the study's own probe client) were present in the collect-stage reply that the ServerApp received, with the values the client wrote to its own receipt, and in the `FitRes` objects that reached `aggregate_fit` (8 of 8 replies in both checks). The array records of those replies held zero arrays (8 of 8). The application strategy did not aggregate the metrics, so they were received and not used. This shows that metric values reach the server process. It says nothing about what anyone can infer from them, and whether a deployment keeps or discards them was not examined. This was one Docker host, one round, two seeds and the legacy strategy API.

S6 and S6b drove the Flower fixed clipping DP strategy in process, through `configure_fit` and `aggregate_fit`, with 16 parameters and equal weights. Client 0 was given a participation probability that depends on its clipped norm (0.85 for a high norm and 0.15 for a low norm, in the registered model). Over 2,000 rounds the release bit predicted the norm class with accuracy 0.8425 (Wilson 95% interval 0.8259 to 0.8578) and the participant set predicted it with accuracy 0.8515 (0.8352 to 0.8664). Controls with equal participation gave 0.5045 and 0.5035. The dependence is an assumption of this study and was not observed in any client. The observer was told the direction of the dependence, and the nine other clients always responded with equal norm and weight. The value 0.85 is what the registered probabilities give for an observer that reads participation exactly. So this result shows that the real strategy passes participation through to the release bit and the result list without extra noise. It does not show that any client behaves this way, and we did not observe such a client. With 9 results at `num_sampled_clients` 10 (S6b), the effective noise multiplier went from 1.0001 to 0.9001, a ratio of 0.9000, and the exact epsilon for one round at delta 1e-5 from 4.377 to 4.947. The multiplier ratio of 0.9 is the factor of row C4. The control arms would have found a participation difference of 0.062 with 80% power, so they make a dependence of that size or more unlikely for this threshold and observer (80% power, one seed per arm), and say nothing about a smaller one.

Not covered: the Message API strategies, other releases, more than one round, message timing and size, an active server, and the relay streams that were saved and not analysed. S6 and S6b called the strategy class directly and not the SecAgg+ workflow, in process, with one seed per arm.

### 4.10 Gradient inversion on CIFAR-10

**Table 4.** Gain over the label-only floor, with 95% bootstrap interval over 20 seeds. P3 has 8 records per client. P3b has 2 and 4 records, and a control with 1 record.

| Study and row | Gain PSNR (dB) | Interval | Gain SSIM | Reading |
|---|---|---|---|---|
| P3 control, batch of 1, plaintext | +5.50 | +4.71 to +6.31 | +0.444 | K0 passed |
| P3 plaintext, 8 records | -0.12 | -0.72 to +0.42 | +0.116 | R1 failed |
| P3 collusion, 8 records | -0.01 | -0.64 to +0.56 | +0.124 | R3 failed |
| P3 aggregate, 8 records | -1.47 | -1.83 to -1.13 | -0.007 | R2 passed, vacuous |
| P3b control, 1 record, plaintext | +4.71 | +2.41 to +7.11 | +0.354 | K0 passed, power void |
| P3b plaintext, 2 records | +2.79 | +1.47 to +4.12 | +0.294 | both gates passed |
| P3b collusion, 2 records | +2.76 | +1.51 to +4.04 | +0.302 | C2P and C2S passed |
| P3b aggregate, 2 records (8 mixed images) | +0.02 | -0.83 to +1.01 | +0.110 | A2P failed, underpowered. A2S failed |
| P3b plaintext, 4 records | +0.74 | -0.03 to +1.57 | +0.182 | both gates failed |
| P3b collusion and aggregate, 4 records | not read | not read | not read | NOT INTERPRETABLE |

In P3 the label-only floor has mean PSNR 13.07 dB. The control passed, so the harness works when the problem is easy. With 8 records the plaintext view did not beat the floor in PSNR, and PSNR and SSIM disagreed in sign. The aggregate view sat 1.47 dB below the floor, and its pass is vacuous by the pre-registered rule because the upper limit is below -1.0. Because plaintext did not clear the floor, a gain over the floor cannot say what the aggregate view leaks. The repository's own attack was weaker than `breaching` in every view (mean squared error ratio 2.59 in plaintext).

In P3b a plaintext view had to clear the floor by +1.0 dB in PSNR or +0.20 in SSIM, per arm and metric, before collusion and aggregate rows of that arm were read. At 4 records both plaintext gates failed, so every collusion, aggregate and difference row of that arm is NOT INTERPRETABLE and we do not quote them. At 2 records both gates passed. Collusion was close to plaintext (Q2P, Q2S passed, plaintext minus collusion +0.03 dB with interval -0.33 to +0.39), although rule Q has limited power at this size, so a small cost would not have been detected. The aggregate view cost the attacker (D2P and D2S passed, plaintext minus aggregate +2.77 dB with interval +2.00 to +3.51). Mixing and quantisation cannot be separated here. In SSIM the aggregate attack stayed above the floor (+0.110, interval +0.052 to +0.176). Fewer records helped the attacker (T1, +2.08 dB, interval +0.96 to +3.22). The SSIM margins and the margins and expected values of rules D, A and Q for both metrics were set after the P3 numbers were read, and P3b was designed after P3 failed, so P3b is not a blind replication. A P3 row with the library default of 24,000 iterations gave +0.46 dB with interval -0.26 to +1.14 and is not in Table 4.

These are statements about the attacks that ran, for one model, one local step and the stated batch sizes, and say nothing about privacy.

### 4.11 Failed predictions and void power statements

We keep every failed prediction as written. Table 5 lists them.

**Table 5.** Failed or partly held predictions.

| Study | Prediction | What was measured |
|---|---|---|
| FedML, RA1 | `TypeError` after one step at sampling rate 0.1 with default orders | `NameError`. The pre-registration named no sampling rate |
| FedML, RA5 | raises only after a step | raises before a step as well |
| TFF, F5 | weight 1 client output norm 0.0035714 | 0.005. The weights sent sum to 10, not 14. The slip was in the pre-registration and shows nothing about TFF |
| Flower, C6 | no noise on an integer array, by one mechanism | outcome as predicted, mechanism wrong. No privacy mismatch measured |
| Flower, R1 | no accountant ships | held on 1.27.0, not on 1.44.0 |
| FLARE selection, prediction 2 | log ratio 0.40 to 0.45 of the claim in S1 and S2 and 0.44 in S3 | numeric values 0.451, 0.400 and 0.441. Measured 0.441, 0.386 and 0.425. Held partly |
| P3 | R1, R3, R1r, R3r, B1c, O1 | all failed. R1 and O1 had confidence 0.6 |
| P3b | G4P, G4S, A2P, A2S | all failed. A2P and A2S had confidence 0.8 each |
| TFF distributed DP, C | noise-free output norm 10 at 20 clients | 9.97287. Exact check, no replication. A scratch run outside the evidence fits a wrap, not tested |
| TFF distributed DP, J | RMS 0.0057735 and 0.0081650 for `ddp_secure_aggregator` | the call did not run |
| TFF distributed DP, A1, A3, D | spread 0.0116 to 0.0197, value within 0.008 of 1, beta difference below 0.001 | failed in the first run (0.0218, 1.00812, 0.00484) and held in the one replication (0.0172, 1.00084, 0.00013) |

Earlier work (v0.15) recorded five failed predictions, E2, D3, B1, P2 and X1, which we do not rely on here.

Power statements that are void, vacuous or weak:

1. P3 rows R2 and R2r passed with upper limits below -1.0 and are labelled vacuous.
2. The P3b control K0 passed, but its observed seed-level standard deviation was 5.57 dB against 3.3 dB assumed, a ratio of 1.69, so its power statement is void.
3. P3b A2P failed with an upper limit of +1.008 dB against a margin of +1.0 dB. The pre-registered pass chance for a true gain of 0 was 81 in 100 at an assumed standard deviation of 1.6 dB, and about 55 in 100 at the observed 2.19 dB.
4. FLARE row N6b had zero power at its claim of 20, because the ceiling is about 10.9. The block agreement test in the later studies has power about 0.3 against a shift of 3 binomial standard errors.
5. In Flower and FedML the empirical Clopper-Pearson bound has no power at the local DP levels tested and has a ceiling of about 10.9.
6. TFF distributed DP: the beta check passes about 23% of the time at a true difference of zero, so its replication says nothing about beta. One sensitivity check is stochastic at about 1e-5 with no replication and no power statement. The A1 dispersion check could not have seen a repeated-noise share below about 2.3%.

## 5 Discussion

Most differences we found sit between the configured parameters and the assumptions of the mechanism. Some sit in the noise arithmetic itself: FedML `DP_Clip`, the order of noise and clip update in Flower adaptive clipping, the TFF distributed sum documentation sentence and the discrete Gaussian at low bit widths. With an equal-weight average over the configured cohort, Flower and TFF deliver the configured noise. Flower differs when weights are unequal or the result count departs from `num_sampled_clients`. TFF's Gaussian aggregators keep the configured count in noise and sensitivity, while its distributed sum follows the reporting count in its noise only. In FedML one `sensitivity` argument serves an L1 and an L2 mechanism and the local class does not clip. In FLARE the ledger arithmetic matches its formulas and the default release budget makes the recorded epsilon large. Flower 1.44.0 has an optional accountant that raises on unequal weights and releases nothing on a short result count. It is off by default. The adaptive clipping rows replicate observations that public Flower issues describe. Beyond the noise arithmetic, the server of Flower 1.27.0 received client metric values in our SecAgg+ Docker runs, and in the in-process test of its fixed clipping strategy participation reached the release bit and the result list. By the inventory's source reading the guarantees do not cover these channels. That is a statement about what is observable and not about what leaks.

On the attack side, with 8 records per client the attack was at the floor in plaintext, so no reading of the aggregate was possible. Plaintext gain was +0.74 dB at 4 records and +2.79 dB at 2. We have not shown, and do not claim, that any framework is not private. The measurements cover noise and lower bounds under stated configurations, on synthetic updates, for named versions, and not deployment practice, client sampling, composition in a trained run or other versions.

## 6 Threats to validity

Construct. "Claimed" is defined per row from parameters, class text or the cited paper, and is not always what a framework documents. FedML's online documentation was not consulted. TFF documentation states no adjacency and the add or remove reading is ours. Flower's count sensitivity convention is not established. The paper readings (bit direction, order, budget split) came from summarising fetches and not a line by line comparison with the PDF. The implied epsilons for the distributed sum are a Gaussian approximation of a discrete mechanism and not a guarantee.

Internal. Predictions came from code reading, several rows are partly confirmatory, and the harness was drafted in the same session as its pre-registration, with quick runs listed in deviation logs. Pre-registration order rests on commit history and cannot prove that no mechanism ran earlier. In the FedML, P3 and Flower adaptive clipping studies the commits carry committer dates later than author dates, so only author dates show the sequence of pre-registration and harness. The SSIM margins of P3b, and the margins and expected values of rules D, A and Q, were set after P3, and P3b was designed after P3 failed. Flower variants and versions share seeds, so their agreement is not independent. No peer review has taken place.

External. One version each of FedML, FLARE and TFF, and two of Flower. Synthetic updates on small vectors (16 coordinates for Flower and FedML, 2048 for the distributed sum). One round for the central rows, no training run, no sampling amplification, no composition. The distributed sum ran in the test C++ context because the default context failed, with a plain sum and not a secure aggregation protocol, and `ddp_secure_aggregator` did not run. The server view study (Section 4.9) used one Flower version, two Docker runs of 4 clients with one round each, and a participation dependence that was assumed in the model. The inversion studies use one small CNN with 58% held-out accuracy, one local step, known labels and a simulated SecAgg+ quantiser. The aggregate rows speak about 8 and 16 mixed images in P3b and 32 in P3. The 4,000 iteration attack is shorter than the library default. P3 tuning chose the largest total variation weight in its grid for all three views, which may be under-tuning. The 4-record tuning winner was reused for the 2-record arm and the control without a test. The aggregate attacker is a joint oracle with all labels. FedML ran on numpy 2.5.3 although it declares `numpy<2`. Release notes of Flower and TFF were not read, and the release that introduced the Flower accountant is unknown.

Statistical. No multiplicity correction was applied across inversion rows or the six FLARE event rows. Several power numbers rest on normal approximations. The DP measurements ran on one local machine each, and the inversion studies on Google Colab T4 runtimes (two runtimes for P3). TFF noise cannot be seeded, so the saved evidence is one run with one replication where needed. The repeated-noise observation rests on scratch runs and a probe outside the evidence file, and its rate and cause were not measured under a pre-registration.

Provenance. The P3 run was split across two Colab runtimes (seeds 0 to 14 and 15 to 19, from run notes) and the seed files do not record which wrote which. In P3b a resumed invocation after a disconnect rewrote `summary.json` and overwrote the live `run.log`, which is lost (deviation entry 9). The seed and tuning files were not touched. The model checkpoint and image files are on the author's Google Drive and not in the repository.

Disclosure. The pre-registrations of release 0.15, P3 and P3b state that findings about third-party software are reported privately to the maintainers before any public description. This rule was not followed. The repository is public and every result was released within days of measurement (23 to 26 November 2026). No finding about a DP mechanism or an attack was sent to any maintainer, before or after release. The one contact is a public Flower issue (8601) and a pull request (8603) opened and closed by the author on 23 November 2026. They concern SecAgg+ documentation, namely that per-client example counts and fit metrics reach the server unmasked, which relates to channels 3 and 4 of Section 4.9. No maintainer reply is visible. The repository documents do not mention either item.

AI assistance. The code, pre-registrations, analysis scripts and drafts were produced with AI coding assistants (Claude), working under the author's instructions. Results were checked by scripts and by separate assistant instances acting as reviewers, and a reviewer in this paper means such an instance. The repository records that commits are authored only by the author and carry no assistant trailers. The author is responsible for the content.

## 7 Reproducibility

Everything is in the public repository `ahmedjawedaj/feddisclosure-lab` at release 0.16.13 (Apache-2.0). Each study has a script `scripts/audit_*.py` (flower_dp, flower_adaptive_clipping, flower_adaptive_guarantee, fedml_dp, nvflare_dp, nvflare_selection, nvflare_events, tff_dp, tff_ddp, p2_server_view), a test file `tests/privacy/test_audit_*.py` and evidence under `reports/results/privacy/v0160/`. The inversion studies use `scripts/p3_breaching_cifar10.py`, `scripts/p3b_breaching_cifar10.py` and `scripts/p3b_power_table.py`. Pre-registrations are in `docs/preregistration/`. Run times range from 3 seconds (FedML) to about 35 minutes (Flower central), 176 seconds for the server view study, and about 2.7 hours for P3 and 72 minutes for P3b.

`scripts/release_gate.py` runs eight mechanical checks (commit identity, pre-registration text unchanged above its deviation log, prose rules, held back material, version agreement, the README claim check, gate integrity with pinned hashes, power section). It runs on every pull request since release 0.16.7, pushes to main are not gated, and the pins file and workflow file are not pinned, so a reviewer has to read them.

`scripts/verify_readme_claims.py` compares every headline README number with the saved evidence file. Run on a fresh copy of main at commit e32b61c, it printed "142 of 142 README claims verified". That checks that prose matches saved files and does not validate the science. The fast suite records 753 passed and 32 skipped in the full environment and 536 passed and 74 skipped in the CI environment, and tests that call TFF skip where it is not installed. Not reproducible from the repository alone: the inversion image files and model checkpoint (the author's Google Drive, hashes in every seed file), the TFF install route (`jaxlib` 0.4.14 from Google's release index, Python 3.11), and bitwise GPU results.

## 8 Conclusion

For the versions and configurations measured, the central Gaussian classes of Flower and TFF deliver the configured noise when weights are equal and the cohort is full. Flower differs from the configured reading when weights are unequal, when the result count departs from `num_sampled_clients`, and in adaptive clipping against the paper reading. TFF's distributed sum delivers the configured noise only at the expected client count. FedML's Gaussian and Laplace classes and the `GlobalDP` noise path match their formulas. Its `DP_Clip` class and accountant paths differ or raise when called directly, its local class does no clipping and one `sensitivity` argument serves both an L1 and an L2 mechanism. The FLARE filter's lower bounds stay at or below its claims. In these inversion studies the plaintext gain over the floor was larger at smaller batch sizes (8, 4 and 2 records per client), for one model and one local step. For Flower 1.27.0, client metric values reach the server in two SecAgg+ Docker runs. With the fixed clipping strategy called in process and without SecAgg+, under an assumed dependence, the release bit and result count reflect participation. The limits in Section 6 apply to every sentence above.

## References

No reference was read in full unless stated. Most were read at abstract level or through a summarising fetch.

[1] G. Andrew, O. Thakkar, H. B. McMahan, S. Ramaswamy. Differentially Private Learning with Adaptive Clipping. arXiv 1905.03871.

[2] B. Balle, Y.-X. Wang. Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising. ICML 2018.

[3] T. Cebere, D. Erb, D. Desfontaines, A. Bellet, J. Fitzsimons. Grey-box auditing of differential privacy libraries (title shortened). arXiv 2602.17454, 2026.

[4] J. Geiping, H. Bauermeister, H. Droege, M. Moeller. Inverting Gradients: How easy is it to break privacy in federated learning? NeurIPS 2020.

[5] J. Hayes, S. Mahloujifar, B. Balle. Bounding Training Data Reconstruction in DP-SGD. NeurIPS 2023, arXiv 2302.07225.

[6] G. Andrew, P. Kairouz, S. Oh, A. Oprea, H. B. McMahan, V. M. Suriyakumar. One-Shot Empirical Privacy Estimation for Federated Learning. arXiv 2302.03098.

[7] R. Behnia, J. Birrell, A. Riasi, R. Ebrahimi, K. Dutta, T. Hoang. Local Differential Privacy for Federated Learning with Fixed Memory Usage and Per-Client Privacy. arXiv 2510.12908.

[8] D. Pasquini, D. Francati, G. Ateniese. Eluding Secure Aggregation in Federated Learning via Model Inconsistency. arXiv 2111.07380.

[9] G. Pichler, M. Romanelli, L. Rey Vega, P. Piantanida. Perfectly Accurate Membership Inference by a Dishonest Central Server in Federated Learning. arXiv 2203.16463.

[10] Flower documentation, DP how-to and API page of the fixed clipping strategy. Summarising fetch.

[11] NVIDIA FLARE documentation, Differential Privacy in FLARE. Summarising fetch.

[12] Public tracker pages (summarising fetch, logged out): Flower issues 8295, 8297 and 8298, FedML issue 1586, NVIDIA FLARE issue 343 and pull requests 4470 and 4498, and the author's own Flower issue 8601 and pull request 8603 (Section 6). Issue numbers and dates come from the web. Comments needing a sign-in were not readable.

[13] `breaching` library, version 0.1.2, `github.com/JonasGeiping/breaching`. Repository page, summarising fetch.

[14] V. Valadi, L. Beerens, M. Akesson, J. Ostman, F. Hoseini, S. Toor, A. Hellander. Practical Feasibility of Gradient Inversion Attacks in Federated Learning. arXiv 2508.19819. Summarising fetch of the full text.

[15] R. Wang. Checking the Output Does Not Check the Aggregator: A Bit-Exact Counterexample in Federated Learning. NeurIPS 2026 listing.
