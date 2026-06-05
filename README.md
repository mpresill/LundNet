[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4443146.svg)](https://doi.org/10.5281/zenodo.4443146)

LundNet
=======

This repository contains the code and results presented in
[arXiv:2012.08526](https://arxiv.org/abs/2012.08526 "LundNet paper").

## About

LundNet is a jet tagging framework to train graph-based jet tagging strategies.

## Paper Summary

**"Jet tagging in the Lund plane with graph networks"**
F. A. Dreyer and H. Qu — [arXiv:2012.08526](https://arxiv.org/abs/2012.08526)

### Motivation

Identifying the origin of a high-energy jet (a collimated spray of particles
produced in a particle collider) is a central challenge in experimental
high-energy physics.  Previous deep-learning taggers either treat jets as
images or as unordered sets of particles, discarding the rich hierarchical
structure of the jet's internal radiation pattern.

### Approach

The paper proposes representing a jet as a **graph built from its Cambridge/Aachen
declustering tree** — the Lund tree — and applying **graph neural networks
(GNNs)** directly on that tree.

Key ideas:

1. **Lund plane representation.**  A jet is recursively declustered into pairs
   of subjets.  Each splitting is described by five Lund coordinates:
   `ln(1/z)`, `ln(1/Δ)`, `ln(kT)`, the azimuthal angle `ψ`, and `ln(m)`.
   These variables have well-understood physical interpretations and are
   directly calculable in perturbative QCD.

2. **Graph construction.**  Each branching in the declustering tree becomes a
   node in the graph, and edges connect parent–child splittings, preserving
   the hierarchical structure of the shower.  Both the primary (harder) and
   secondary (softer) branches are included.

3. **EdgeConv layers.**  The GNN uses EdgeConv blocks (from Dynamic Graph CNN),
   which aggregate messages from neighbouring nodes by computing a learned
   function of `(h_i − h_j, h_j)` for each edge `(i, j)`.  Multiple stacked
   EdgeConv layers with a feature-fusion mechanism allow the network to capture
   both local and global radiation patterns.

4. **Model variants.**  Several LundNet variants are studied by varying the
   number of Lund input features (LundNet-2 through LundNet-5) to probe which
   physical variables carry the most discriminating information.

### Results

The models are evaluated on two standard jet-tagging benchmarks:

- **W-jet tagging** (boosted W boson vs. QCD background): LundNet-5 matches
  or surpasses the state-of-the-art ParticleNet, while using a physically
  motivated, interpretable graph structure instead of a dense particle cloud.
- **Top-jet tagging** (boosted top quark vs. QCD background): similar
  competitive performance, demonstrating the approach generalises across
  topologies.

Beyond raw performance, the Lund-plane graph structure makes the network
**more interpretable**: by inspecting which nodes and edges receive high
attention, one can trace the classification decision back to specific soft
and collinear splittings in the jet shower — something that is much harder
with image- or particle-cloud-based approaches.

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

## Code Structure

The source code lives under `src/lundnet/`.  Below is a description of each
module, what it does, and why it is needed.

```
src/lundnet/
├── JetTree.py          # jet declustering & Lund coordinates
├── read_data.py        # input data reader (JSON/gzip)
├── dgl_dataset.py      # PyTorch Dataset wrappers & graph builders
├── dgl_utils.py        # low-level graph utilities (k-NN graph)
├── EdgeConv.py         # EdgeConv building block
├── LundNet.py          # LundNet model
├── ParticleNet.py      # ParticleNet baseline model
└── scripts/
    └── lundnet.py      # command-line entry point
```

### `JetTree.py` — Jet declustering and Lund coordinates

This module converts a raw FastJet `PseudoJet` (a jet clustered with the
Cambridge/Aachen algorithm) into a binary tree that records every successive
splitting.

- **`LundCoordinates`** computes and stores the five Lund-plane variables for a
  single splitting: `lnz` (log of the momentum fraction), `lnDelta` (log of
  the angular separation), `lnKt` (log of the transverse momentum), `psi`
  (azimuthal angle between the two prongs), and `lnm` (log of the invariant
  mass).  These are the node features fed into the GNN.  The static
  `change_dimension` method lets the caller restrict to a subset of features,
  which is how LundNet-2/3/4/5 variants are controlled.
- **`JetTree`** walks the declustering history recursively, building a
  `harder`/`softer` binary tree and storing a `LundCoordinates` object at
  each internal node.  Leaf nodes (single particles with no further splitting)
  carry momentum four-vectors but no Lund coordinates.  Optional `ktmin` and
  `deltamin` cuts prune very soft or very collinear splittings.
- **`LundImage`**, **`RSD`** are helper classes for Lund-image generation and
  Recursive Soft Drop grooming, kept here for completeness but not used by the
  default LundNet training pipeline.

### `read_data.py` — Input data reader

Jets are stored on disk as gzip-compressed JSON files (one event per line).
Each event is a list of particle four-vectors `{px, py, pz, E}`.

- **`Reader`** handles streaming, gzip decompression, JSON parsing, and
  string-valued header lines.
- **`Jets`** inherits from the abstract `Image` class.  It re-clusters each
  event's particles with FastJet's Cambridge/Aachen algorithm (large-radius
  `R = 1000`, effectively capturing all particles into one jet), then returns
  either the leading `PseudoJet` (for tree-based models) or the raw list of
  constituent four-vectors (for particle-cloud models).  The optional
  `groomer` hook allows soft-drop pre-processing before passing the jet to the
  dataset builder.

### `dgl_dataset.py` — PyTorch Dataset wrappers and graph builders

This module bridges the physics data and the deep-learning framework.

- **`DGLGraphDatasetLund`** is the dataset used by LundNet.  For each jet it
  calls `JetTree` to build the declustering tree, then converts that tree into
  a DGL graph with `dgl.from_networkx`.  Each graph node stores two tensors:
  `features` (the Lund coordinates — the actual GNN inputs) and `coordinates`
  (η, φ position of the corresponding subjet, used only as spatial coordinates
  for k-NN graph construction in the particle-cloud path but dropped for the
  tree path).  The `_build_tree` method traverses the `JetTree` recursively
  and adds one graph node per internal splitting, with edges connecting each
  node to its parent.
- **`DGLGraphDatasetParticle`** is the dataset used by ParticleNet.  Instead
  of a tree it builds a flat graph with one node per jet constituent, storing
  η, φ, log(pT) and log(E) as features.  The graph edges are added dynamically
  during training (k-NN in feature space).
- **`_LundTreeBatch`** and **`collate_wrapper_tree`** are the collate functions
  that batch multiple Lund graphs together for a single forward pass.  They
  call `dgl.batch` to merge graphs, and pop the `coordinates` node attribute
  (no longer needed after batching).
- **`_SimpleCustomBatch`** and **`collate_wrapper`** do the same for the
  particle-cloud graphs, but additionally build a k-NN graph from the spatial
  coordinates on the fly (needed because ParticleNet updates its graph
  dynamically).

### `dgl_utils.py` — Low-level graph utilities

Contains manual implementations of **`knn_graph`** and
**`segmented_knn_graph`** that were adapted from an early DGL version to fix a
bug in the original upstream code.  These compute pairwise squared distances
between node features and connect each node to its *k* nearest neighbours,
returning a DGL graph.  `segmented_knn_graph` handles the batched case where
multiple independent point clouds are concatenated along the first axis and
separated by the `segs` array.  `reversed_graph` returns the edge-reversed
version of a graph (a thin wrapper around `dgl.reverse`).

### `EdgeConv.py` — EdgeConv building block

Implements the **EdgeConv** message-passing layer from *Dynamic Graph CNN for
Learning on Point Clouds* (Wang et al., 2019).  For each directed edge
`(j → i)` the layer computes:

```
e_{ij} = MLP( h_i − h_j,  h_j )
```

where `h` are node features, and then aggregates incoming messages by mean
pooling.  A residual (shortcut) connection is added, and BatchNorm + ReLU are
applied after each linear layer.  This block is shared by both LundNet and
ParticleNet.

### `LundNet.py` — LundNet model

Stacks six EdgeConv blocks with progressively wider hidden dimensions
(32 → 32 → 64 → 64 → 128 → 128 channels).  A **fusion layer** concatenates
the output of every EdgeConv block before the final classifier head; this
multi-scale aggregation is crucial for performance because different layers
capture different scales of the jet shower.  The classifier head is a small
MLP with dropout.  The input graph is the pre-built Lund tree; the graph
topology is fixed and does not change between layers.

### `ParticleNet.py` — ParticleNet baseline

Implements the **ParticleNet** architecture as a baseline for comparison.
Uses three EdgeConv blocks and, unlike LundNet, **reconstructs a new k-NN
graph at every layer** from the current node embeddings (dynamic graph
convolution).  The graph is built in feature space rather than physical space,
so the neighbourhood structure evolves as the network learns.

### `scripts/lundnet.py` — Command-line entry point

The main driver for training, validation, and inference.  It:
1. Parses command-line arguments (model variant, data paths, device, learning
   rate schedule, etc.).
2. Selects the correct dataset class and collate function.
3. Instantiates the chosen model and moves it to the target device.
4. Runs the training loop with Adam + multi-step LR decay, saving the best
   checkpoint by validation accuracy.
5. After training (or by loading a saved checkpoint) evaluates the model on
   the test set, computes the ROC curve, AUC, and background rejection at 50%
   and 30% signal efficiency, and writes results to disk.

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
