# Dataset provenance (v0.14, updated for v0.15)

Only public data are used. Raw arrays and partitions are never committed to Git (`/data/` is ignored).

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
  Treat redistribution terms as unverified; cite the creators and do not redistribute the raw files from this repo.
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
  - sklearn digits is the UCI "Optical Recognition of Handwritten Digits" test set (E. Alpaydin, C. Kaynak), shipped inside scikit-learn. The UCI dataset page states CC BY 4.0 (checked 2026-11-23, page `archive.ics.uci.edu/dataset/80`). We do not copy it into the repo, we read it from the installed package and cite the creators.
  - Fashion-MNIST is MIT licensed, see above.
- **Optional:** MNIST. It stays available for the native Docker runs and the 28x28 experiments that already use it, but no default path needs it. Its licence is not clear. The original page (`yann.lecun.com/exdb/mnist/`) shows no licence statement, and on 2026-11-23 the URL served only a bare directory listing. Secondary catalogues often cite CC BY-SA 3.0, which we could not verify at the source. Treat the terms as unverified.
- **Never redistributed:** no dataset file of any kind is committed. `/data/` is ignored by Git, the SHA-256 pins let anyone re-fetch and verify the exact files, and result files contain only derived numbers and per-image error values, not images.
- Attribution for the numbers: cite the dataset creators in any report that uses them.
