# Multi-round note: SecAgg+ quantisation error over T rounds

Status: written for v0.15 work package 6, prediction M1 in `docs/preregistration/v0.15.md`. Scope is quantisation only, as in `docs/proofs/one-round-secagg-error.md`. That note gives the one-round per coordinate bound B used here.

Labels: **textbook** for the convex theorem, **measured** for the executed trials, **counterexample** for the non-convex case. Nothing in this note is proved for non-convex losses.

## 1. Setting

Let G be the exact FedAvg round map. For client i with local map F_i and true weight p_i = m_i / sum m, G(x) = sum_i p_i F_i(x). The SecAgg+ trajectory is x_t = A_t where A_t is the upstream aggregate of the client states F_i(x_{t-1}). The exact trajectory is y_t = G(y_{t-1}), with x_0 = y_0. Define

* e_t = A_t - G(x_{t-1}), the one-round error at the state the SecAgg+ run actually reached. By the one-round note, |e_t,j| <= B_t,j for every coordinate j, where B_t,j is the full one-round bound (worst case term, float64 server term and weight rounding term) evaluated on the client states of round t. It needs no clipping, 2 <= n <= 1023 and sizes at most max_weight in every round.
* B = the largest B_t,j over all rounds and coordinates.
* Delta_t = x_t - y_t.

Norms. |.|_inf is the largest absolute coordinate. ||.|| is the Euclidean (L2) norm. The one-round bound is in |.|_inf. For a vector in R^d, ||e|| <= sqrt(d) |e|_inf. The theorem is stated in L2 and converts once, in that one place.

## 2. Theorem (textbook, convex case)

Assume every client loss L_i is convex and beta-smooth (its gradient is beta-Lipschitz in L2), and every local update is K steps of full batch gradient descent with step eta <= 2/beta. If clients use mini-batches, the same mini-batches are used in the exact and the SecAgg+ trajectory. Then for every T,

||x_T - y_T|| <= sum over t of ||e_t|| <= T * sqrt(d) * B.

## 3. Proof

*Step 1, a gradient step is non-expansive.* For convex beta-smooth f the gradient is co-coercive (Baillon and Haddad): <grad f(x) - grad f(y), x - y> >= (1/beta) ||grad f(x) - grad f(y)||^2. Let g = grad f(x) - grad f(y) and u = x - y. Then

||(x - eta grad f(x)) - (y - eta grad f(y))||^2 = ||u||^2 - 2 eta <g, u> + eta^2 ||g||^2 <= ||u||^2 - eta (2/beta - eta) ||g||^2.

This is at most ||u||^2 exactly when eta <= 2/beta. The boundary eta = 2/beta is included. K steps are a composition of non-expansive maps, so F_i is non-expansive.

*Step 2, weighted averaging is non-expansive.* ||G(x) - G(y)|| = ||sum_i p_i (F_i(x) - F_i(y))|| <= sum_i p_i ||F_i(x) - F_i(y)|| <= ||x - y|| by the triangle inequality and p_i >= 0 with sum 1.

*Step 3, telescoping.* Delta_t = A_t - G(y_{t-1}) = (A_t - G(x_{t-1})) + (G(x_{t-1}) - G(y_{t-1})) = e_t + (G(x_{t-1}) - G(y_{t-1})). So ||Delta_t|| <= ||e_t|| + ||Delta_{t-1}|| with Delta_0 = 0. Summing gives ||Delta_T|| <= sum ||e_t||.

*Step 4, one conversion of norms.* ||e_t|| <= sqrt(d) |e_t|_inf <= sqrt(d) B. Therefore ||Delta_T|| <= T sqrt(d) B. A sharper form is the sum over t of ||B_t|| where B_t is the vector of per coordinate bounds. End of proof.

Remarks.

* The weights in G are the true sizes, the weight rounding error of the one-round note is inside e_t.
* If the surviving set of clients differs between rounds, G in round t uses that set. The proof is unchanged.
* If the model is stored in float32 between rounds, each trajectory is rounded each round. That adds a term of at most 2 * 2^-24 sqrt(d) max|theta| per round to the right side. This follows the same telescoping but is a remark and was not executed.
* The bound uses B evaluated on the realised states. It is a bound given the run, not an a priori constant. A uniform a priori B needs a bound on the weighted state values, which for convex problems with small step sizes can come from a bound on the iterates. This is not done here.

## 4. Executed check of the theorem (measured)

`tests/training/test_multiround_note.py` and `scripts/run_secagg_bound_search.py` run 1,000 random trials. Each has 2 to 8 clients, 2 to 24 parameters, 2 to 6 rounds and 1 to 3 local full batch steps. Half are least squares and half are ridge logistic regression (509 and 491). The step is kappa times 2/beta_max where beta_max is the largest client smoothness. In 391 of the 1,000 trials kappa is exactly 1, so the step is exactly 2/beta for the client with the largest smoothness. The SecAgg+ arithmetic is the real upstream quantiser through `secagg_aggregate` with float64 client states.

* Violations of ||Delta_T|| <= T sqrt(d) B: **0 of 1,000**.
* Violations of the sharper ||Delta_T|| <= sum ||e_t||: 0 of 1,000.
* Violations of the one-round per coordinate bound inside the trajectories: 0.
* The largest ratio of ||Delta_T|| to the stated bound was 0.234 and the median was 0.056, so the bound is safe but not tight.
* Negative control: with step 1.3 times 2/beta, identical client data and 6 rounds of 2 local steps, the stated bound is violated in 34 of 100 trials. So the check can fail when the theorem's assumption is removed.

Result for M1 (pre-registered rule: counterexample reproduced and zero violations in at least 1,000 convex trials): **PASS**.

## 5. Non-convex counterexample (two rounds, ReLU)

The convex proof fails without Step 1 because a gradient step through a ReLU is not Lipschitz. A tiny change in the weights can switch a unit on or off and change the gradient by an amount that does not shrink with the change.

**Saved evidence from v0.14** (not re-run here, cited):

* `reports/results/secagg/v014_dropout_matrix/error_propagation.json`, key `relu_flip_study`. Noise inside the one-round bound added to the exact round-1 digits MLP, then an exact round 2. Over 200 trials, 39.5% exceed 2 times the one-round bound, median 3.6e-5, 95th percentile 2.0e-4, maximum 2.08e-4.
* `reports/results/secagg/v014_dropout_matrix/multiround_acceptance_rule.json`. 300 Monte Carlo two-round runs with the upstream quantiser. 38.7% exceed 2 times the one-round bound, median 4.2e-5, 99th percentile 2.09e-4 and maximum 2.19e-4. The original Docker baseline (1.89e-4 against a naive tolerance of 7.9e-5) sits in the upper mode.
* `docs/v0.14-privacy-secagg-results.md` lines 84 to 88 describe the same finding.

**Reproduced here, repository data.** `scripts/probe_secagg_error_propagation.py` function `relu_flip_study` with seed 7 re-run with 60 trials (the test uses 20): 36.7% above 2 times the bound (saved 39.5%), maximum 2.05e-4, median 3.6e-5, one-round bound 3.97e-5.

**Reproduced here, minimal explicit case.** One hidden ReLU unit, model u * relu(w x), two clients with sizes 300 and 600, one sample each (x = 1, y = -1 and x = -1, y = -1), local step size 1, initial (w, u) = (0.5, 1). Exact round 1 gives (w, u) = (0, 0.75) exactly, which is the ReLU kink. SecAgg+ round 1 through the real quantiser gives w = e with |e| inside the one-round bound 8.9e-6. In 67 of 200 seeds e is not exactly zero. In every such seed the round 2 difference between the SecAgg+ and the exact trajectory is at least 0.2499, and at most 0.5 over all seeds. That is at least 28,000 times the one-round bound and far above the old tolerance of 2 times the bound. The reason is that w moves to one side of the kink, so the other client becomes active in round 2. In the other 133 seeds the round 1 error left w exactly at zero and the difference was small. This is a deterministic construction. It does not depend on the data set. The test is `test_two_round_relu_counterexample_is_reproduced`.

So for non-convex losses no bound of the form (number of rounds) times (one-round bound) holds in general. This does not rule out a statistical statement for a fixed setting, which is the next section.

## 6. The old acceptance rule as a tolerance limit

The v0.14 rule accepted a Docker run when its two-round difference was at or below the empirical 99th percentile of 300 Monte Carlo runs (2.09e-4). Restated as a statistical tolerance limit:

* Let X_1, ..., X_300 be independent draws of the difference from one distribution with a continuous distribution function, and let q be its true 99th percentile. Then P(max X_i < q) = 0.99^300 = 0.0490, so with probability **0.951** the maximum of the 300 runs is at least q. The sample maximum (2.19e-4) is therefore an upper 95% confidence limit for the 99th percentile. A further run from the same distribution exceeds the maximum with probability 1/301, about 0.3%.
* The old rule used the empirical 99th percentile (2.09e-4) and not the maximum. A sample 99th percentile of 300 runs lies between the 297th and 298th smallest values. It is at least q with probability only between 0.35 and 0.58, so it is not a 95% limit. The recommended rule is to accept at or below the maximum of the Monte Carlo runs (2.19e-4). All four saved Docker runs (1.89e-4, 4.1e-5, 3.7e-5, 4.1e-5) pass either way, so no earlier acceptance changes.

Caveat. The 300 trials share the data, the initial model, the local training seeds and the client sample sizes. Only the rounding draws vary. The limit therefore describes the spread caused by rounding for this one configuration. It is not a limit over data or initialisations, and it is not a bound in the sense of section 2. A different data set needs its own Monte Carlo run. Tolerance limits also assume the Docker run behaves like one more exchangeable draw. A systematic difference between the Docker path and the simulation (for example a float32 client path that the simulation did not use) would break that. The repository wrapper feeds float64 states and casts the output to float32 afterwards. The one-round note shows that the float32 client path does not change the worst case bound, only adds a small deterministic offset.

## 7. What is and is not claimed

* Proved (textbook): the L2 bound of section 2 for convex beta-smooth losses with step at most 2/beta, given the one-round bound at the realised states.
* Executed: 1,000 trials with zero violations and a negative control that fails as it should.
* Counterexample: non-convex two-round case, saved evidence plus a minimal explicit construction.
* Not proved: anything for non-convex losses, any a priori constant B, the float32 model storage remark, and anything for training with data dependent clipping or momentum.
