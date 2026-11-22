# One-round SecAgg+ quantisation error: statement and proof

Status: written for v0.15 work package 6. Checked against the installed upstream `flwr` 1.27.0 and against the executed searches in `reports/results/privacy/v0150/secagg_bound_search.json`. The pre-registered predictions E1, E2 and E3 are in `docs/preregistration/v0.15.md`. Their results are in `reports/results/privacy/v0150/prereg_check_proofs.json` and are summarised in section 8.

Scope. This note only analyses quantisation. It takes the integer sum of the clients' quantised vectors to be recovered exactly modulo 2^32, and it analyses what the quantiser, the weight ratio and the final division do to the result. Nothing here is a statement about privacy.

Labels used below.

* **Textbook** means a standard fact that is cited and not re-proved at length.
* **Upstream-specific** means it depends on how Flower 1.27.0 writes the arithmetic.
* **Measured** means it comes from an executed search and is not a proof.
* **Conjecture** means it is not proved here.

## 1. The arithmetic as upstream writes it

Files read: `flwr/common/secure_aggregation/quantization.py`, `flwr/client/mod/secure_aggregation/secaggplus_mod.py` (lines 450 to 470, the fit stage), `flwr/server/workflow/secure_aggregation/secaggplus_workflow.py` (the final stage, from `factor_extract` on) and `flwr/common/secure_aggregation/ndarrays_arithmetic.py`. The repository wrapper `secagg_aggregate` in `src/feddisclosure/attacks/view_study.py` repeats these operations in the same order.

Defaults: clipping range c = 8, quantisation range Q = 2^22, modulus 2^32, max weight M = 1000.

Client i holds a vector v_i and a sample count m_i.

1. `q_i = round((m_i / M) * Q)` with Python `round`. The weight used is `w_i = q_i / Q`. When 0 < q_i <= Q this is an exact binary fraction with at most 22 fractional bits, so it is exact in float32 and float64.
2. `a_i = v_i * w_i` in the dtype of v_i. Flower clients send float32 arrays, and a float32 array times a Python float stays float32.
3. `quantize` computes `x_i = (clip(a_i, -c, c) + c) * (Q / (2c))`. The factor `Q / (2c) = 2^18` is a power of two, so the multiplication is exact. The intermediate arrays are in the dtype of v_i.
4. Stochastic rounding: `k = ceil(x)` as int32, and then `k` is lowered by one wherever `rand < k - x`, where `rand` is a float64 uniform array. For a float32 `x` the difference `k - x` is computed in float64 (int32 and float32 promote to float64), so it is exact. Rounding down therefore has probability `ceil(x) - x`, up to the 2^-53 resolution of the uniform draw.
5. The client sends `k_i` and the integer `q_i`. The server sums both over the surviving clients (modulo 2^32). Call the sums S and T.
6. The server computes `dequantize(S) = S * (2c/Q) - c`, adds `-(n-1) c`, and multiplies by `Q / T`, all in float64. The output is A.

Upstream-specific facts used later.

* **F1.** The quantised values lie in {0, 1, ..., Q} inclusive. The value `+c` maps to Q exactly. The workflow docstring says the range is `[0, Q-1]`, which is not what the code does. This matters for overflow in F2.
* **F2. Limit on n.** The sum S is reduced modulo 2^32. Each term is at most Q = 2^22, so no wrap is possible when n * 2^22 < 2^32, that is n <= 1023. At n = 1024 with every client at `+c` the sum is exactly 2^32 and wraps to zero. The upstream constructor warns only for `num_shares` above 1024. This is checked in `overflow_limit_check` (executed, n = 1023 does not wrap, n = 1024 wraps to 0). The sum T of weight ratios obeys the same limit when every m_i <= M.
* **F3.** Only the survivors' vectors are summed, and the offset uses the number of active clients. So n below means the number of clients whose vectors were summed.
* **F4.** Weights are applied before quantisation. The final division by `T / Q` happens in float64 after dequantisation. This is why the error is amplified by `Q / T`, which is about `M / N` where N is the total sample count.

## 2. Notation and assumptions

Quantisation unit: one step of the integer grid. In weighted-value space it equals `2c/Q = 2^-18`.

* n clients, indexed by i. Coordinates indexed by j, d coordinates.
* **A1.** 2 <= n <= 1023, from F2.
* **A2.** Each m_i is a positive integer with m_i <= M, so w_i <= 1, and each q_i >= 1. Write T = sum of q_i.
* **A3. No clipping.** |fl(w_i v_ij)| <= c for every i and j, where fl is the product in the dtype of v. A sufficient condition independent of the weights is |v_ij| <= c for all entries, because w_i <= 1. To check it exactly, compute `np.abs(v * (q_i / Q)).max() <= c` in the dtype of v. The repository function `no_clipping` in `scripts/run_secagg_bound_search.py` does this and is called for every search case. If clipping is active, the output contains the clipped values and the bound below does not apply (a test shows an error above 100 times the bound for inputs of 9 with c = 8).
* **A4.** Default parameters c = 8 and Q = 2^22 for the float32 lemma in section 4.2. With another c and Q the bounds (a) and (b) still hold when `Q/(2c)` is a power of two. Otherwise add the rounding of that multiplication, which this note does not do.
* **A5 (for part b only).** The rounding draws of different clients are independent of each other and of the data. In upstream the draws come from numpy's global random state of the process that quantises, so independence means different clients have different, independently seeded states.

Targets. Define two means of the client vectors.

* `mu_j = (sum_i q_i v_ij) / T`. This is the mean with the integer weights that the quantiser actually uses.
* `mu_true_j = (sum_i m_i v_ij) / (sum_i m_i)`. This is the FedAvg mean the experiment wants.

Per client, per coordinate, write `x_ij = (w_i v_ij + c) Q/(2c)` for the exact real number before rounding, and `Y_ij = k_ij - x_ij` for the signed rounding error in units.

## 3. Theorem

Under A1 to A3, for every coordinate j, with A the output of the upstream path:

**(a) Worst case.** In exact arithmetic for the client side, |A_j - mu_j| < 2cn / T. With the float64 server stage included, |A_j - mu_j| <= 2cn/T + 2^-47 n Q / T. The second term is at most 2^-29 of the first. This holds for float32 and for float64 inputs, for any rounding draws.

**(b) Unbiasedness and Hoeffding.** In exact arithmetic, and under A5, E[A_j] = mu_j. For every delta in (0,1), with probability at least 1 - delta, simultaneously for all d coordinates,

|A_j - mu_j| <= (2c / T) * sqrt( (n/2) * ln(2d/delta) ) + (float terms of part c).

The range of each client's rounding error given its input is exactly 1 (the error is either -f or 1-f, where f is the fractional part of x), so the Hoeffding denominator is n, not 4n.

**(c) Separate terms.**

1. *Weight rounding.* |mu_j - mu_true_j| <= n (1 + 2^-30) R_j / (2T), where R_j is the largest minus the smallest of the v_ij over i. It is zero when all sizes are equal.
2. *Float32 client path.* The path adds a deterministic offset b_ij = x32_ij - x_ij with |b_ij| <= 3/16 unit, and the offset is not random. It does not break part (a). It adds a term (2c/T) |sum_i b_ij| <= (2c/T) (3/16) n to part (b). For float64 inputs |b_ij| <= 2^-32 (1 + 1/16) unit.
3. *Final cast.* The output A is float64. If the receiving client loads it into a float32 model, each coordinate moves by at most 2^-24 |A_j|. This cast is not in `secagg_aggregate`. It is in the repository client at `src/feddisclosure/training/secagg_client.py` line 44 (`load_state_dict` from the received arrays) and in the repository Monte Carlo script, which casts with `.float()`.

**(d) Shared rounding draws.** If all clients use the same uniform draw for coordinate j, each marginal stays unbiased, but Var(sum_i Y_ij) = sum over i and k of (min(f_ij, f_kj) - f_ij f_kj). With equal fractional parts f this is n^2 f (1-f). The error of the average then does not shrink with n, and the part (b) bound fails for large n.

## 4. Proofs

### 4.1 The identity (textbook algebra, upstream-specific constants)

Let a_ij = w_i v_ij and suppose A3 so that nothing is clipped. Then sum_i x_ij * (2c/Q) = sum_i a_ij + n c. The server computes

A_j = (Q/T) * ( S_j (2c/Q) - c - (n-1) c ) = (Q/T) * ( S_j (2c/Q) - n c ).

Subtract (Q/T) * sum_i a_ij = (Q/T) * ( (2c/Q) sum_i x_ij - n c ). The difference is

A_j - (Q/T) sum_i a_ij = (2c / T) * ( S_j - sum_i x_ij ) = (2c / T) * sum_i Y_ij.

In exact arithmetic (Q/T) sum_i a_ij = sum_i q_i v_ij / T = mu_j. So A_j - mu_j = (2c/T) sum_i Y_ij. This is the whole connection between the quantiser and the error. The factor 2c/T is the quantisation step amplified by the final division.

### 4.2 Part (a)

*Per client error in exact arithmetic.* If x is an integer then k = x and Y = 0. Otherwise let f = x - floor(x) in (0,1). The rounding returns floor(x) with probability f and ceil(x) with probability 1 - f, up to the draw resolution. So Y is -f or 1 - f, and |Y| < 1 in both cases. Hence |sum_i Y_ij| < n and |A_j - mu_j| < 2cn/T.

*Float32 and float64 clients.* The client rounds the computed value x' (not x). Then Y' = k - x' is -f' or 1-f' and |Y'| < 1 as above, and the error against the exact x is Y' + b with b = x' - x. The cell lemma below shows that x' and x lie in the same open unit cell (floor(x'), ceil(x')), or x' is an integer and |b| <= 3/16. So |k - x| < 1 in every case, and part (a) holds for float32 inputs.

*Cell lemma (new, upstream-specific, float32 and default c and Q).* Let a = fl(v w) with |a| <= 8, s = fl(a + 8), and x' = 2^18 s. Then 2^18 s is a multiple of 1/8, and either x' is an integer with |x' - x| <= 3/16, or the distance from x' to the nearest integer is strictly larger than |x' - x|, so x lies in the same open unit cell as x'.

Proof. Write ulp for the float32 spacing and j = floor(log2 |a|), so j <= 2 (and |a| = 8 only for a = -8 or 8, which are exact). Measure everything in units, that is multiply real values by 2^18. The product error is at most half an ulp of a, which is 2^(j-24), so at most 2^(j-6) units. The sum error is at most half an ulp of s. Cases.

* *a >= 0.* Then s lies in [8, 16]. For s < 16 the float32 spacing is 2^-20, so x' is a multiple of 1/4 units. The sum error is at most 2^-21, that is 1/8 unit. The product error is at most 2^(j-6) <= 1/16 unit. The total is at most 3/16 < 1/4. If s = 16 then x' = 2^22 is an integer and a = 8 is exact.
* *a < 0 and j = 2*, so a is in [-8, -4]. Then s = 8 - |a| lies in [0, 4]. The value a is a multiple of 2^-21 and s is a multiple of 2^-21 below 8, so s is exactly representable and the sum error is zero. Then x' is a multiple of 1/8 unit and the total error is the product error, at most 1/16 < 1/8.
* *a < 0 and j = 1*, so |a| is in [2, 4). Then s lies in (4, 6] with spacing 2^-21. The sum error is at most 2^-22, which is 1/16 unit, and the product error is at most 2^-5. Total at most 3/32 < 1/8, and x' is a multiple of 1/8.
* *a < 0 and j <= 0*, so |a| < 2 and s lies in (6, 8). The spacing of s is 2^-21 and the sum error is at most 1/16 unit. The product error is at most 2^(j-6) <= 1/64. Total at most 5/64 < 1/8, and x' is a multiple of 1/8.
* *a = 0* gives s = 8 and x' = 2^21, an integer.

In every case the total error |x' - x| is below the smallest nonzero distance from x' to an integer (1/4 when a >= 0, 1/8 otherwise), or x' is an integer with error at most 3/16. This proves the lemma. It was also checked numerically on 24,000,000 float32 elements through the real upstream `quantize` with zero violations (section 6). The bound 3/16 is a supremum. The measured maximum was 0.18749994.

For float64 inputs the same argument holds with every ulp smaller by 2^29, so the grid is 2^-31 unit and the error is at most 2^-32 (1 + 1/16) unit, which is below the grid. This case is argued by the same table and is covered by the executed float64 control (2,000 cases, no violations). The grid claim for float64 is not separately enumerated and is marked in section 9.

### 4.3 The float64 server stage

The server computes r = S 2^-18 exactly (S < 2^32 is an integer, scaling by a power of two is exact), then r1 = fl(r - 8), then r2 = fl(r1 - 8(n-1)), then the product with fl(Q/T). We have |r1| <= 16n, |r2| <= 8n. Each float64 operation has relative error at most 2^-53. So the absolute error before the last product is at most 16n 2^-53 + 8n 2^-53 = 3n 2^-50 (which is 1.5 n 2^-49). The last product and the division add at most 8n * 2^-52 (relative error 2^-53 twice). Multiplying by Q/T, the total is at most 2.5 n 2^-49 (1 + tiny) Q/T, which is below 2^-47 n Q/T. Relative to the main term 2cn/T = 16n/T this is 2^-47 Q / 16 = 2^-29. This is textbook floating-point error analysis. The Flower strategy then re-averages n copies of the same vector with float64 scale factors. That adds a relative error of order n 2^-53 and is not included in the theorem.

### 4.4 Part (b): unbiasedness and Hoeffding

*Unbiasedness* (textbook). For a non-integer x, With the upstream rule the probability of rounding down is ceil(x) - x = 1 - f, so E[k] = floor(x) (1-f) + ceil(x) f = floor(x) + f = x. Integer x gives k = x. So E[Y_ij] = 0 in exact arithmetic, and by 4.1 E[A_j] = mu_j. For float32 inputs the same holds around x', so E[Y'] = 0 and E[A_j] - mu_j = (2c/T) sum_i b_ij, which is the deterministic offset of part (c.2). The rounding draw has 53 bits, so the probability of rounding is exact up to 2^-53 per element, which is negligible and ignored.

*Hoeffding* (textbook, Hoeffding 1963, Theorem 2). For independent X_i with X_i in [alpha_i, beta_i], P(|sum (X_i - E X_i)| >= t) <= 2 exp(-2 t^2 / sum (beta_i - alpha_i)^2). Here X_i = Y'_ij lies in [-f_i, 1 - f_i], an interval of length exactly 1 (upstream-specific: the rounding rule gives a two point distribution with these end points). So the denominator is n and P(|sum_i Y'_ij| >= t) <= 2 exp(-2 t^2 / n). Set t = sqrt((n/2) ln(2d/delta)). Each coordinate fails with probability at most delta/d. A union bound over j (no independence across coordinates is needed) gives the statement. Multiply by 2c/T and add the deterministic terms from 4.2, 4.3 and the offset.

*Remark on the range.* A common mistake is to take the range of the error as [-1, 1], length 2, which gives a bound twice as wide as needed. The earlier throwaway probe used sqrt(2 n ln(2d/delta)), which has this factor of two.

*Remark, the bound is weak at small n.* The Hoeffding term exceeds the worst case 2cn/T when ln(2d/delta) > 2n. For d = 64 and delta = 0.01 this happens at n <= 4. The usable bound is the smaller of the two.

### 4.5 Part (c.1): weight rounding

Let omega_i = m_i / M and W = sum omega_i, and let w_i = q_i / Q, W~ = T / Q = sum w_i. Write p_i = omega_i / W (the true FedAvg weights) and p~_i = w_i / W~. Let e_i = w_i - omega_i and E = sum e_i. The rounding of q_i gives |q_i - m_i Q / M| <= 1/2 + 2^-31, where the second term is the float64 error in `(m/M) * Q`. So |e_i| <= (1/2 + 2^-31) / Q.

p~_i - p_i = (e_i W - omega_i E) / (W~ W). Summing absolute values, sum_i |p~_i - p_i| <= (sum |e_i| + |E|) / W~ <= 2 sum |e_i| / W~ <= n (1 + 2^-30) / T.

Both vectors sum to one, so for any constant m0, mu_j - mu_true_j = sum_i (p~_i - p_i)(v_ij - m0). Take m0 as the midpoint of the v_ij. Then |v_ij - m0| <= R_j / 2 and |mu_j - mu_true_j| <= n (1 + 2^-30) R_j / (2T). This is elementary. Equal sizes give e_i = 0 and p~ = p. The term compares to the main term 2cn/T by the factor R_j / (4c).

### 4.6 Part (c.2): size of the float32 offset

The offset b_ij is deterministic. By the cell lemma proof its supremum for a >= 0 is 1/8 + 2^(j-6) with j the binade of |a|, so 0.1328 for |a| in [0.5, 1), 0.1406 for [1, 2), 0.1563 for [2, 4) and 0.1875 for [4, 8]. For a < 0 it is smaller. Over random arrays the offset is close to zero mean (it changes sign with the rounding direction), but for one value it is fixed. So with identical clients it adds linearly in n (section 6 example).

### 4.7 Part (d): shared draws

Given fractional parts f_i and one shared uniform u, the upstream rule gives Y_i = 1{u' <= f_i} - f_i where u' = 1 - u is uniform (round down exactly when the draw is below 1 - f_i). Then E[Y_i Y_k] = P(u' <= min(f_i, f_k)) - f_i f_k = min(f_i, f_k) - f_i f_k. The variance formula follows by summing. For equal f this equals n^2 f(1-f). In general Var(sum Y) <= (sum sd(Y_i))^2 by Minkowski, and shared draws raise the variance from at most n/4 (independent) to as much as n^2/4. If the f_ij are independent uniform across clients, the cross covariance is E[min(f,f') - f f'] = 1/3 - 1/4 = 1/12, so Var is about n^2/12 and the error grows linearly in n even with different data. Each marginal stays unbiased because every Y_i has mean zero. The formula is verified by exact quadrature in `test_shared_stream_variance_formula`.

## 5. What is textbook and what is upstream-specific

| Item | Status |
|---|---|
| Unbiased stochastic rounding, Hoeffding inequality, union bound, floating-point model with relative error 2^-53 | Textbook |
| Minkowski inequality for the standard deviation of a sum | Textbook |
| Identity A - mu = (2c/T) sum Y | Upstream-specific (offset n c, division by T/Q, ratio carried as an integer) |
| Quantised levels 0 to Q inclusive, limit n <= 1023 | Upstream-specific, executed |
| Per client error range exactly 1 | Upstream-specific (the rounding rule) |
| Cell lemma, offsets up to 3/16, grid of 1/8 | Upstream-specific, proved by cases and checked on 24 million elements |
| Weight rounding term | Elementary, specific to `round(ratio * Q)` |

## 6. Measurements

All numbers are from `reports/results/privacy/v0150/secagg_bound_search.json` produced by `PYTHONPATH=src:. python scripts/run_secagg_bound_search.py` (runtime about 40 seconds).

**E1 search.** 20,000 float32 cases through `secagg_aggregate` (n from 2 to 64, d up to 48, unequal sizes, five value families including exact boundary values and neighbours of the clipping limit, identical clients). Zero violations of the stated bound, zero violations against the true FedAvg mean with the weight term added, and zero cases above the float64-only bound 2cn/T. The largest observed error over the stated bound was 0.906. A mutation check with the bound cut to 0.85 of its value finds 3 violations in 3,000 cases, so the search can fail. A float64 control of 2,000 cases also had zero violations (largest ratio 0.855).

**Float32 offset.** On 24,000,000 float32 elements through the real `quantize`, the offset b had mean -9.5e-6 units (no systematic sign), mean absolute value 0.047, root mean square 0.063 and maximum 0.18749994. By size of |w v| the maximum was 0.1328 for |wv| below 1, 0.1562 for 1 to 4 and 0.1875 for 4 to 8. The reviewer's figure of about 0.137 is therefore right for small weighted values (between the 0.1328 and 0.1406 steps) but is not a general bound. The general supremum is 3/16 = 0.1875 units. The largest single client error under either rounding direction was 0.9375 units, always below 1.

**Identical clients example (float32).** n = 256 equal sizes, value 0.999 * 2^-21, weight 1. The offset is -0.1249 units for every client and x' is an integer, so no randomness is left. The aggregate error is 31.97 units. The pure Hoeffding term for one coordinate at delta 0.01 is 26.04 units, so the pure Hoeffding statement fails by a deterministic float32 effect. The worst case bound is 256 units and holds. The statement of part (b) with the offset term (3/16) * 256 = 48 units is also satisfied.

**Tightness of (a).** With the rounding draws forced to the worst case and float64 inputs, the error reaches 0.999999 of 2cn/T, so the bound cannot be improved by a constant for float64. For float32 inputs the grid keeps each client error below 0.9375 in the search (Conjecture C2 below).

**E3 (d = 64 coordinates, 300 trials per cell, equal sizes 450, M = 1000, float64 inputs, delta = 0.01).**

| n | Hoeffding bound (units) | independent p99 over bound | shared draws, same data | shared draws, different data |
|---|---|---|---|---|
| 4 | 4.35 | 0.670 | 0.911 | 0.804 |
| 16 | 8.70 | 0.688 | 1.803 | 1.247 |
| 64 | 17.4 | 0.706 | 3.626 | 2.134 |
| 256 | 34.8 | 0.727 | 7.206 | 3.950 |

For independent draws the 99th percentile of the maximum error in units divided by sqrt(n) is between 1.46 and 1.58. For shared draws on identical data it divided by n is between 0.98 and 0.99. At n = 4 the Hoeffding bound is above the worst case 4 units, so that cell is a weak check.

## 7. Link to the repository numbers

For the Docker 4 client run (sizes 96 each, M = 1000), T = 1,610,612 and 2cn/T = 3.9736e-5, matching the saved bound 3.9736e-5 and the saved observed 3.2e-5.

## 8. Pre-registered predictions

* **E1 PASS.** 20,000 float32 cases, zero counterexamples.
* **E2 FAIL.** Neither clause held. The float32 path does add a deterministic offset of up to 3/16 unit per client (so the second clause, a term below 1e-6 of the bound, is false). But no counterexample to the float64-only bound exists (no case, no element and no adversarial rounding choice reaches an error of 1 unit per client), so the first clause is false. The prediction that the float64 bound misses the float32 bias is wrong for part (a). It is right for part (b), where an extra term is needed.
* **E3 PASS.** Independent draws are below the bound at n = 4, 16, 64 and 256, and shared draws on identical data violate it at n = 256 (7.2 times the bound).

## 9. Not proved, and conjectures

* **C1.** The float64 case of the cell lemma. The argument is the float32 table with every grid divided by 2^29. It is not enumerated and rests on the executed float64 control.
* **C2.** For float32 inputs the per client error is at most 31/32 unit. The case table gives 1 - 1/8 + (error) for each region and the largest is in the |a| in [2,4) region. The proved statement is only < 1. The observed maximum was 0.9375.
* **C3.** Parameters other than c = 8 and Q = 2^22. The proof uses that Q/(2c) = 2^18 is a power of two and uses the grid constants.
* **C4.** The effect of the Flower strategy re-averaging (float64 scale factors summing to 1 up to order n 2^-53) is not bounded here.
* The bounds are worst case or, for Hoeffding, sub-Gaussian. They are not tight for typical data. The measured maximum was 0.906 of the stated bound over 20,000 cases but typical errors are far smaller.
* Clients that drop out after their vectors were summed are outside this note, because n is defined as the number of summed vectors.
* Multi-round statements are in `docs/proofs/multi-round-note.md`.
