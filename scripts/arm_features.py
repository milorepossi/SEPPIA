#!/usr/bin/env python3
"""Input features for the arms of the ladder. Dependencies: numpy.

Every arm feeds the same MLP block; only the input changes.

    A0  onehot  one-hot of the 43 kept positions, no PLM   (in-memory, no cache)
    A1  L0      ESM-2 layer 0, context-free control        concat_L0.npy
    A2  L15     ESM-2 layer 15, mid-stack                  concat_L15.npy
    A3  L33     ESM-2 layer 33, final                      concat_L33.npy

A cached arm is a (rows, 43, D) array whose row order is the dataset's, joined
to a split by source_row through index.json. source_row is the spreadsheet-row
convention of split_dataset.py, so cache position = source_row -
SOURCE_ROW_OFFSET; this module never assumes that and always goes through the
index, then verifies the join.

Two transforms sit between the cache and the MLP, and both matter for the
comparison to mean anything:

POOLING turns (43, D) into a vector. Flattening is the most faithful reading of
"only the input features differ", but it makes the first layer's width scale
with D, so a flattened layer 33 gets 64x the parameters of A0 and the arms no
longer share a budget. The options:

    flatten   43*D      faithful; A0 860, PLM arms 55040
    mean      2*D       mean over the peptide block and the HLA block
                        separately; cheap, but discards position within each
                        block, which matters for a 9-mer whose P2/P9 anchors
                        carry most of the signal
    pca:K     43*K      one D->K projection fitted on the training rows'
                        position vectors and applied at every position, then
                        flattened

pca:K is the dimension- and parameter-matched comparison: with K=20 every arm
has exactly A0's 860 inputs and an identical first layer, so only the feature
content differs. It is also principled for A1, whose layer-0 embedding matrix
has rank exactly 20 on this alphabet, making pca:20 lossless there. It is lossy
for layers 15 and 33, which is the honest question the ladder asks: given one
budget, which representation carries more signal?

STANDARDISATION is fitted on the training rows only. Layer 33 is recorded after
emb_layer_norm_after while layer 15 is not (mean abs 0.145 vs 2.344, a 16x scale
gap), so without it A2 vs A3 partly measures input scaling rather than
representation quality. One-hot is already unit-scale, so the default leaves it
alone to keep A0's published numbers reproducible; pass always to put every arm
through an identical pipeline.
"""
import json
from pathlib import Path

import numpy as np

from extract_embeddings import SOURCE_ROW_OFFSET

CACHED_ARMS = {"L0": "concat_L0.npy", "L15": "concat_L15.npy", "L33": "concat_L33.npy"}
ONEHOT_ARM = "onehot"
ARMS = (ONEHOT_ARM, *CACHED_ARMS)

POOLINGS = ("flatten", "mean")  # plus "pca:K" for any positive integer K


# --- Pooling ------------------------------------------------------------------
def parse_pooling(pooling):
    """Return (mode, k). k is None unless the mode is pca:K."""
    if pooling in POOLINGS:
        return pooling, None
    if pooling.startswith("pca:"):
        try:
            k = int(pooling.split(":", 1)[1])
        except ValueError:
            raise ValueError(f"pooling {pooling!r} must be pca:<integer>") from None
        if k < 1:
            raise ValueError(f"pooling {pooling!r} needs a positive number of components")
        return "pca", k
    raise ValueError(f"unknown pooling {pooling!r}; use one of {POOLINGS} or pca:K")


def pooled_width(pooling, n_slots, dim, peptide_slots, pseudoseq_slots):
    """Feature count a pooling produces, without materialising anything."""
    mode, k = parse_pooling(pooling)
    if mode == "flatten":
        return n_slots*dim
    if mode == "mean":
        return 2*dim
    return n_slots*k


def apply_pooling(block, mode, peptide_slots, pseudoseq_slots, projection=None):
    """Pool a (rows, slots, dim) block into (rows, features) float32."""
    if mode == "flatten":
        return block.reshape(len(block), -1)
    if mode == "mean":
        return np.hstack((block[:, peptide_slots].mean(axis=1),
                          block[:, pseudoseq_slots].mean(axis=1)))
    if mode == "pca":
        if projection is None:
            raise ValueError("pca pooling needs a fitted projection")
        center, components = projection
        return ((block - center) @ components.T).reshape(len(block), -1)
    raise ValueError(f"unknown pooling mode {mode!r}")


def fit_projection(block, k):
    """Fit one dim->k PCA on every position vector of these rows.

    One projection shared across positions rather than 43 separate ones: the
    positions live in the same representation space, so pooling the position
    vectors uses 43x the samples and keeps the arms' features comparable
    position to position, mirroring one-hot's identical 20-d code everywhere.
    """
    flat = block.reshape(-1, block.shape[-1]).astype(np.float64)
    if k > flat.shape[1]:
        raise ValueError(f"pca:{k} exceeds the representation width {flat.shape[1]}")
    center = flat.mean(axis=0)
    # Economy SVD of the centred position vectors; rows >> dim here.
    _, singular, vt = np.linalg.svd(flat-center, full_matrices=False)
    explained = float((singular[:k]**2).sum()/(singular**2).sum())
    return (center.astype(np.float32), vt[:k].astype(np.float32)), explained


# --- Cache access -------------------------------------------------------------
def load_index(embeddings_dir):
    path = Path(embeddings_dir)/"index.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run scripts/extract_embeddings.py to build the "
            "embedding cache before training a PLM arm.")
    return json.loads(path.read_text())


def open_arm(embeddings_dir, arm):
    """Memory-map one arm's cache. Returns (array, index)."""
    if arm not in CACHED_ARMS:
        raise ValueError(f"{arm!r} is not a cached arm; expected one of {sorted(CACHED_ARMS)}")
    index = load_index(embeddings_dir)
    path = Path(embeddings_dir)/CACHED_ARMS[arm]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Arm {arm} needs it; extract with "
            f"scripts/extract_embeddings.py --layers {arm[1:]}")
    array = np.load(path, mmap_mode="r")
    if array.shape[0] != index["n_rows"]:
        raise ValueError(f"{path} has {array.shape[0]} rows but index.json says "
                         f"{index['n_rows']}")
    return array, index


def row_positions(index, source_rows):
    """Map a split's source_row values onto cache positions, and verify them.

    Goes through index.json rather than assuming the offset, then checks the
    arithmetic shortcut agrees. A silent offset error here would misalign
    almost every row while leaving all shapes intact, so it is worth asserting.
    """
    cached = np.asarray(index["source_row"], dtype=np.int64)
    # Check the convention before the per-row lookup, so a cache written with
    # the old 0-based numbering gets told what is wrong and how to fix it
    # instead of a misleading "rows are missing".
    if len(cached) and cached[0] != SOURCE_ROW_OFFSET:
        raise ValueError(
            f"index.json numbers source_row from {cached[0]}, but split_dataset.py "
            f"numbers from {SOURCE_ROW_OFFSET} (the spreadsheet row). This cache's "
            "index.json predates that fix.\n"
            "  The .npy arrays are fine: row order never changed, only the labels. "
            "Repair the index in place, without re-extracting:\n"
            "    python scripts/features.py <dataset.xlsx> "
            "--out-dir <embeddings-dir> --repair-index")

    lookup = {int(s): i for i, s in enumerate(cached)}
    missing = [int(s) for s in source_rows if int(s) not in lookup]
    if missing:
        raise KeyError(
            f"{len(missing)} source_row values are absent from the cache "
            f"(first: {missing[:5]}). The cache must cover every row the split "
            f"uses; a --limit extraction covers only the first rows.")
    positions = np.array([lookup[int(s)] for s in source_rows], dtype=np.int64)
    shortcut = np.asarray(source_rows, dtype=np.int64)-SOURCE_ROW_OFFSET
    if not np.array_equal(positions, shortcut):
        raise ValueError(
            "index.json source_row is not SOURCE_ROW_OFFSET plus the row position; "
            "the cache and split_dataset.py disagree about the convention")
    return positions


def verify_join(index, positions, dataset, hla_column="hla_seq", peptide_column="peptide"):
    """The joined rows must carry the split's own sequences."""
    pairs = index["pairs"]
    for offset, position in enumerate(positions):
        hla, peptide = pairs[position]
        if peptide != dataset[peptide_column][offset] or hla != dataset[hla_column][offset]:
            raise ValueError(
                f"join mismatch at split row {offset} (source_row "
                f"{dataset['source_row'][offset]}): cache holds {peptide!r} but the "
                f"split says {dataset[peptide_column][offset]!r}")
    return len(positions)


# --- Encoder ------------------------------------------------------------------
class CachedArmEncoder:
    """Turn a split's rows into pooled features from one layer's cache.

    Stateless unless the pooling is pca, in which case fit() must be called on
    the training rows first so nothing is fitted on test data.
    """

    def __init__(self, embeddings_dir, arm, pooling="flatten", verify=True):
        self.arm = arm
        self.pooling = pooling
        self.mode, self.components = parse_pooling(pooling)
        self.array, self.index = open_arm(embeddings_dir, arm)
        self.peptide_slots = list(self.index["peptide_slots"])
        self.pseudoseq_slots = list(self.index["pseudoseq_slots"])
        self.n_slots = self.array.shape[1]
        self.dim = self.array.shape[2]
        self.projection = None
        self.explained_variance = None
        self.verify = verify

    @property
    def width(self):
        return pooled_width(self.pooling, self.n_slots, self.dim,
                            self.peptide_slots, self.pseudoseq_slots)

    def block(self, dataset):
        """The (rows, slots, dim) float32 slice this split needs."""
        positions = row_positions(self.index, dataset["source_row"])
        if self.verify:
            verify_join(self.index, positions, dataset)
        return np.asarray(self.array[positions], dtype=np.float32)

    def fit(self, dataset):
        """Fit the pca projection on these rows only. No-op otherwise."""
        if self.mode != "pca":
            return self
        self.projection, self.explained_variance = fit_projection(
            self.block(dataset), self.components)
        return self

    def __call__(self, dataset):
        if self.mode == "pca" and self.projection is None:
            raise RuntimeError("fit() the pca pooling on the training rows first")
        return apply_pooling(self.block(dataset), self.mode, self.peptide_slots,
                             self.pseudoseq_slots, self.projection)


# --- Feature standardisation --------------------------------------------------
class Standardiser:
    """Per-column centring and scaling, fitted on the training rows only."""

    def __init__(self, center=None, scale=None):
        self.center = center
        self.scale = scale

    def fit(self, features):
        self.center = features.mean(axis=0)
        scale = features.std(axis=0)
        # Constant columns carry no information; leave them centred at zero
        # rather than dividing by ~0 and amplifying noise.
        self.scale = np.where(scale > 1e-8, scale, 1.0).astype(np.float32)
        self.center = self.center.astype(np.float32)
        return self

    def __call__(self, features):
        if self.center is None:
            raise RuntimeError("fit() the standardiser on the training rows first")
        return ((features-self.center)/self.scale).astype(np.float32)


def fitted_state(encoder, standardiser):
    """The fitted feature transforms, as torch tensors for a checkpoint.

    Tensors rather than numpy arrays so the checkpoint still loads under
    torch.load(weights_only=True), which rejects arbitrary numpy objects.
    Without this a reloaded PLM model cannot reproduce its own features.
    """
    import torch

    state = {}
    if encoder is not None and encoder.projection is not None:
        center, components = encoder.projection
        state["projection"] = dict(center=torch.from_numpy(np.asarray(center)),
                                   components=torch.from_numpy(np.asarray(components)))
    if standardiser is not None:
        state["standardiser"] = dict(center=torch.from_numpy(np.asarray(standardiser.center)),
                                     scale=torch.from_numpy(np.asarray(standardiser.scale)))
    return state


def from_checkpoint(checkpoint, embeddings_dir=None):
    """Rebuild (encoder, standardiser) so a saved model sees its own features.

    A checkpoint with no arm predates the ladder and is arm A0, whose features
    are rebuilt in-memory, so both are None. For a PLM arm this refuses rather
    than guessing: scoring a pca:20 model on one-hot features would not even
    raise, since both are 860 wide, and the numbers would be silently wrong.
    """
    arm = checkpoint.get("arm", ONEHOT_ARM)
    if arm == ONEHOT_ARM:
        return None, None

    pooling = checkpoint.get("pooling", "flatten")
    directory = embeddings_dir or checkpoint.get("embeddings_dir")
    if directory is None:
        raise ValueError(
            f"checkpoint is arm {arm} but records no embeddings_dir; pass one explicitly")

    encoder = CachedArmEncoder(directory, arm, pooling)
    mode, _ = parse_pooling(pooling)
    if mode == "pca":
        saved = checkpoint.get("projection")
        if saved is None:
            raise ValueError(
                f"checkpoint is arm {arm} with pooling {pooling} but stores no fitted "
                "projection, so its features cannot be reproduced. Retrain with the "
                "current train_mlp.py, which saves it.")
        encoder.projection = (np.asarray(saved["center"]), np.asarray(saved["components"]))

    standardiser = None
    saved = checkpoint.get("standardiser")
    if saved is not None:
        standardiser = Standardiser(np.asarray(saved["center"]), np.asarray(saved["scale"]))
    elif should_standardise(arm, checkpoint.get("standardise", "auto")):
        raise ValueError(
            f"checkpoint is arm {arm} and was standardised, but stores no standardiser. "
            "Retrain with the current train_mlp.py.")
    return encoder, standardiser


def should_standardise(arm, policy):
    """Resolve the auto/always/never policy for one arm."""
    if policy == "always":
        return True
    if policy == "never":
        return False
    if policy != "auto":
        raise ValueError(f"unknown standardise policy {policy!r}")
    return arm != ONEHOT_ARM
