# Dataset provenance (v0.14)

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
