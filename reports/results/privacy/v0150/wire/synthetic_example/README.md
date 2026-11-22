# Synthetic example, not a result

These three files are the output of `scripts/analyze_wire_capture.py` on synthetic capture directories
built by `build_synthetic_run` in `tests/training/test_wire_capture.py`. No Docker, Flower deployment or
relay produced the inputs. The SecAgg+ aggregate was computed with the repository quantiser arithmetic and
the masked vectors are random numbers. Passing here only shows that the analysis code works on a capture
whose answers are known. It says nothing about a real deployment.

The real results come from the `wire-capture` workflow and will be written under
`reports/results/privacy/v0150/wire/live/`.
