# Wire-level validation (WP3)

Seeds analysed: 5 of 5. Attack iterations: 300.

| ID | Prediction | Pass rule | Result | Observed |
|---|---|---|---|---|
| W1 | Inversion on one captured masked vector is at the label floor | Recognisable rate at most floor plus 5 points | PASS | seeds 5, attack_rate 0, floor_rate 0.725, margin 0.05, attack_mean_mse 0.3348, seeds_with_result 5, seeds_requested 5 |
| W2 | The dequantised captured aggregate matches the exact weighted mean inside the one-round bound | Maximum coordinate error below the proven bound for all coordinates in all runs | PASS | runs 5, worst_max_error 0.0003737, bound 0.0004768, worst_error_over_bound 0.7837, runs_all_below_bound 5, coordinates_per_run 2410, T 134216, n 4, no_clipping_in_all_runs True, seeds_with_result 5, seeds_requested 5 |
| W3 | Attack results on the captured aggregate agree with the simulated results | Difference in recognisable rate within 10 points and in mean MSE within 0.01, over at least 5 seeds | PASS | seeds 5, min_seeds 5, captured_rate 0.6, simulated_rate 0.575, captured_mean_mse 0.05565, simulated_mean_mse 0.05282, abs_rate_difference 0.025, abs_mse_difference 0.00283, seeds_with_result 5, seeds_requested 5 |
| W4 | In plaintext mode a passive observer on the relay recovers individual updates | Inversion recognisable rate at least 85% | PASS | seeds 5, recognisable_rate 0.975, threshold 0.85, relay_updates_equal_proxy 20, relay_updates_checked 20, seeds_with_result 5, seeds_requested 5 |
| W5 | Per-client sample counts and metrics are visible to the server in clear | Observed in 100% of recorded SecAgg+ replies | PASS | replies 20, visible_in_clear 20, fraction 1, seeds_with_result 5, seeds_requested 5 |
| W6 | The relay stream and the proxy record agree | Every proxied object hash appears in the relay stream in plaintext mode | PASS | objects 100, found_in_relay 100, relay_logs_consistent True, seeds_with_result 5, seeds_requested 5 |

## Per seed

| Seed | Errors | Selection matches simulation | Update receipts match | Max recompute diff |
|---|---|---|---|---|
| 0 | none | True | True | 1.4901161193847656e-08 |
| 1 | none | True | True | 1.4901161193847656e-08 |
| 2 | none | True | True | 1.4901161193847656e-08 |
| 3 | none | True | True | 1.4901161193847656e-08 |
| 4 | none | True | True | 1.4901161193847656e-08 |

## Limitations

- Four clients, eight public digits each, one local full batch SGD step, labels known to the attacker
- One Docker host and an insecure internal network. This is not a multi-host deployment
- The relay sees the SuperNode to SuperLink Fleet API only. The ServerApp to SuperLink link is seen by the proxy
- Sample counts and metrics are visible by protocol design (W5 documents this, it is not a finding about a flaw)
- The simulated reference uses the repository SecAgg+ quantiser arithmetic with its own rounding draws
- A failed reconstruction is evidence about this attack only, not a privacy guarantee
