# Dataset provenance (v0.14, updated for v0.15)

Only public data are used. Nothing under the top-level `data/` directory is tracked by Git (`/data/` is ignored). One small exception exists for sklearn digits and is described under the licence decision below. No MNIST or Fashion-MNIST file is committed.

## sklearn digits (8x8)
`sklearn.datasets.load_digits`, 1,797 samples, bundled with scikit-learn (UCI optical recognition of handwritten digits).

## MNIST (28x28, full training split)
- Creators: Y. LeCun, C. Cortes, C. J. C. Burges. 60,000 training images, 10 classes, 784 features.
- Obtained 2026-11-21 from the Google CVDF mirror (`storage.googleapis.com/cvdf-datasets/mnist/`).
  The original host (yann.lecun.com) was not reachable from the execution sandbox.
- MD5 values equal the widely published ones (train images `f68b3c2d...8873`, labels `d53e105e...9432`).
- SHA-256 pinned in `src/feddisclosure/training/datasets.py` and `data/source/mnist/SHA256SUMS`:
  - `train-images-idx3-ubyte.gz` 440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609
  - `train-labels-idx1-ubyte.gz` 3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c
- Licence: the original page states no formal open licence. Secondary sources commonly cite CC BY-SA 3.0.
  Treat redistribution terms as unverified. Cite the creators and do not redistribute the raw files from this repo.
- Use here: the 60,000-image train split is partitioned across three institutions plus a disjoint evaluator test
  slice (20 percent of records). The official 10,000-image t10k set is not used.

## Fashion-MNIST (28x28, full training split, optional second dataset)
- Creators: Han Xiao, Kashif Rasul and Roland Vollgraf, Zalando Research. 60,000 training images, 10 clothing classes, 784 features.
- Obtained 2026-11-23 from the canonical host named in the dataset README (`fashion-mnist.s3-website.eu-central-1.amazonaws.com`) through the normal Python download path (`urllib`).
- MD5 values equal the ones published in the dataset README (train images `8d4fb7e6...bf0d`, labels `25c81989...ffbe`).
- SHA-256 pinned in `src/feddisclosure/training/fashion_mnist.py`. The loader refuses any file that does not match:
  - `train-images-idx3-ubyte.gz` 3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84
  - `train-labels-idx1-ubyte.gz` a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845
- Licence: MIT, Copyright (c) 2017 Zalando SE. Verified on 2026-11-23 against the LICENSE file and the README of `github.com/zalandoresearch/fashion-mnist`. The MIT notice must travel with any redistributed copy.
- Use here: the 60,000-image train split only, through `feddisclosure.evaluation.fashion_mnist_study`. The official 10,000-image test set is not used. Raw files live under `data/source/fashion_mnist/` and are never committed.
- `synthetic_standin` in the loader is a deterministic fake for unit tests. No reported number comes from it.

## Licence decision (v0.15)
- **Defaults:** sklearn digits and Fashion-MNIST. Every default experiment, test and workflow in this repository runs on these two.
  - sklearn digits is the UCI "Optical Recognition of Handwritten Digits" test set (E. Alpaydin, C. Kaynak), shipped inside scikit-learn. The UCI dataset page states CC BY 4.0 (checked 2026-11-23, page `archive.ics.uci.edu/dataset/80`). Experiments read it from the installed package. A small subset is committed as wire-capture evidence, see the next bullet.
  - Fashion-MNIST is MIT licensed, see above.
- **Optional:** MNIST. It stays available for the native Docker runs and the 28x28 experiments that already use it, but no default path needs it. Its licence is not clear. The original page (`yann.lecun.com/exdb/mnist/`) shows no licence statement, and on 2026-11-23 the URL served only a bare directory listing. Secondary catalogues often cite CC BY-SA 3.0, which we could not verify at the source. Treat the terms as unverified.
- **Committed subset (sklearn digits only):** 45 `.npz` files under `reports/results/privacy/v0150/wire/live/data/seed0/` to `seed4/` hold raw 8x8 digits images, 260 rows in total (247 distinct images of the 1,797, with the same values as the scikit-learn copy). They are the partitions the live wire-capture runs trained on, kept so the captured traffic can be checked against the data. The source is the UCI "Optical Recognition of Handwritten Digits" test set (E. Alpaydin, C. Kaynak), and the licence is the one recorded above (CC BY 4.0 per the UCI page, checked 2026-11-23). That check was not repeated when this paragraph was written. Cite the creators if you reuse these files.
- **Not committed:** no MNIST or Fashion-MNIST file is committed and nothing under the top-level `data/` directory is tracked. The SHA-256 pins let anyone re-fetch and verify the exact files. Result files for the Fashion-MNIST and 8-client studies contain only derived numbers and per-image error values, not images.
- Attribution for the numbers: cite the dataset creators in any report that uses them.

## Redaction of host paths in v0.14 evidence (v0.15.2)

In release 0.15.2 the home directory of the execution host was replaced by `/home/fedlab` in 156 saved v0.14 evidence files under `reports/results/native_flower/v014`, `reports/results/secagg/v014*` and `reports/results/privacy/v014` (the `env_file`, `run_dir` and `sites[].path` fields of `deployment_preparation.json`, the `flower_command` entries of `flower_run.json`, and the same path in stdout and stderr logs, `05_prepare_only.log`, `00_privacy_tests.log` and one `docker_inspect_independent.txt`). One line of `reports/results/native_flower/v014/00_environment.log` that listed proxy host names was replaced by a redaction marker. No number, verdict, hash of a model or of a dataset, or any other field changed. Three `flower_run.json` files have their SHA-256 recorded in `dashboard/dist/evidence_snapshot.json`, and those three hashes were regenerated with `python -m dashboard.build_dashboard`. `reports/results/privacy/v0150/fault_matrix/gates/gate_results.json` also records two tree digests of these folders (`saved_evidence_digest_before` and `saved_evidence_digest_after`, keys `secagg` and `native_flower_v014`). They were computed before the redaction and match the files as of v0.15.1, not the redacted files. They were left as recorded. The original files remain in the git history before 0.15.2.
