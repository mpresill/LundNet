[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4443146.svg)](https://doi.org/10.5281/zenodo.4443146)

LundNet
=======

This repository contains the code and results presented in
[arXiv:2012.08526](https://arxiv.org/abs/2012.08526 "LundNet paper").

## About

LundNet is a jet tagging framework to train graph-based jet tagging strategies.

## Install LundNet

### Linux (64-bit)

LundNet was originally developed and tested on 64-bit Linux systems.

Install LundNet with Python's pip package manager:
```
git clone https://github.com/fdreyer/lundnet.git
cd lundnet
pip install -e .
```
To install the package in a specific location, use
the "--target=PREFIX_PATH" flag.

This process will copy the `lundnet` program to your environment python path.

We recommend the installation of the LundNet package using a `miniconda3`
environment with the
[configuration specified here](https://github.com/fdreyer/LundNet/blob/master/environment.yml).

### macOS (Apple Silicon M1/M2/M3, miniforge)

LundNet can also be run on Apple Silicon Macs using
[miniforge](https://github.com/conda-forge/miniforge) (recommended over
miniconda on arm64).

**Prerequisites:** make sure you have the Xcode Command Line Tools installed,
as `fastjet` needs to be compiled from source:

```
xcode-select --install
```

**1. Create the conda environment**

```
conda env create -f environment_mac_arm64.yml
conda activate lundnet_mac
```

**2. Install LundNet**

```
pip install -e .
```

**3. Run a quick test**

```
lundnet --demo --save test --device cpu --num-epochs 1
```

**Device options on Apple Silicon:**

- `--device cpu` — safe default; uses all CPU cores via PyTorch's ARM-optimised
  backend with Apple's Accelerate framework (BLAS).
- `--device mps` — uses Apple's Metal GPU (MPS backend, requires PyTorch ≥ 2.0).
  DGL's MPS support is still experimental; `--device cpu` is recommended for
  stability.

**Performance note:** Training on an M2 CPU is roughly 5–10× slower than a
mid-range NVIDIA GPU.  Using `--device mps` can reduce that gap to roughly
2–4×.  For exploratory work, demos, and inference on pre-trained models, the
M2 is very capable and noticeably faster than most x86 laptop CPUs.

LundNet requires the following python 3 packages:
- torch
- dgl (≥ 0.9, installed via pip)
- numpy
- [fastjet](http://fastjet.fr/) (installed via pip; compiled from source)
- pandas
- json
- gzip
- argparse
- tqdm
- networkx
- uproot3-methods
- scipy
- sklearn

## Pre-trained models

The final models presented in
[arXiv:2012.08526](https://arxiv.org/abs/2012.08526 "LundNet paper")
are stored in:
- models/LundNet3: contains the LundNet-3 models for each benchmark.
- models/LundNet5: contains the LundNet-5 models for each benchmark.

## Input data

All data used for the final models can be downloaded from the git-lfs repository
at https://github.com/JetsGame/data.

## Running the code

To launch a test of the code, use
```
lundnet --demo --save test --device cpu --num-epochs 1
```

This will run the LundNet code on a sample of 5000 signal and background events and train a model on the CPU for one epoch, saving the results in a new test/ directory.

To train a full model, you can type:
```
lundnet --model lundnet5 --train-sig TRAIN_SIG --train-bkg TRAIN_BKG
        --val-sig VAL_SIG --val-bkg VAL_BKG --test-sig TEST_SIG --test-bkg TEST_BKG
        --save OUTPUT
```
where the first six filenames are the locations of the signal and background training, validation and testing samples, and the model is saved to an OUTPUT folder.

To apply an existing LundNet model to a new data set, you can use
```
lundnet --model lundnet5 --load PATH/TO/model_state.pt --test-sig TEST_SIG --test-bkg TEST_BKG --test-output OUTPUT
```
which loads the model given as input, before applying it to the TEST_SIG and TEST_BKG samples, with the results then saved to OUTPUT.pickle

To find more options on how to run full models, use
```
lundnet --help
```

## References

* F. A. Dreyer and H. Qu, "Jet tagging in the Lund plane with graph networks,"
  [arXiv:2012.08526](https://arxiv.org/abs/2012.08526 "LundNet paper")
