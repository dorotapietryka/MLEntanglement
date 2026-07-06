#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entanglement_ml_pipeline.py

A reusable pipeline for studying whether SVMs and other machine-learning models
can learn the separability boundary for bipartite quantum states.

Supported systems
-----------------
1. 2x2 qubit-qubit states:
   * Entangled metric states are generated with Bures or Hilbert-Schmidt sampling.
   * Labels are assigned with the PPT criterion, which is necessary and sufficient
     for 2x2 separability.
   * Separable examples are generated as known separable convex mixtures of
     random product pure states.

2. 3x3 qutrit-qutrit states:
   * Entangled metric states are generated with Bures or Hilbert-Schmidt sampling.
   * NPT states are accepted as entangled immediately.
   * PPT states are delegated to the attached 3x3_svm.py labelling routine, which
     contains the DPS/Gilbert logic. Inconclusive labels are rejected.
   * Separable examples are generated as known separable convex mixtures of
     random product pure states.

Labels
------
y = -1  entangled
y = +1  separable

The saved datasets never contain raw density matrices. They contain only feature
columns and y. The optional NPZ bundle stores exactly:
    SU_features, Moment_features, RMInvariant_features, y

Example CLI usage
-----------------
# Method 1: existing logic. Known separable mixtures plus metric entangled search.
python entanglement_ml_pipeline.py generate \
    --system 2x2 --metric bures --generation-method 1 \
    --n-entangled 1000 --n-separable 1000 \
    --out data/qubit_bures.csv

# Method 2: metric-consistent rejection sampling for both labels.
python entanglement_ml_pipeline.py generate \
    --system 3x3 --metric hs --generation-method 2 \
    --n-entangled 500 --n-separable 500 \
    --qutrit-script /path/to/3x3_svm.py \
    --out data/qutrit_hs.csv

# Method 3: metric-local separable mixtures; entangled rows fall back to Method 1.
python entanglement_ml_pipeline.py generate \
    --system 2x2 --metric bures --generation-method 3 \
    --sep-mixture-terms 16 --n-entangled 1000 --n-separable 1000 \
    --out data/qubit_method3_bures.csv

# Method 4: controlled depolarization after a rejection threshold.
python entanglement_ml_pipeline.py generate \
    --system 3x3 --metric bures --generation-method 4 \
    --method4-depolarize-after 250 --method4-depolarize-step 0.01 \
    --n-entangled 500 --n-separable 500 \
    --qutrit-script /path/to/3x3_svm.py \
    --out data/qutrit_method4_bures.csv

# Evaluate feature groups and models.
python entanglement_ml_pipeline.py evaluate \
    --dataset data/qubit_bures.csv --out-dir results/qubit_bures

# Plot t-SNE for a chosen feature scenario.
python entanglement_ml_pipeline.py tsne \
    --dataset data/qubit_bures.csv --feature-set ALL \
    --out results/qubit_bures/tsne_ALL.png

# Plot held-out accuracy degradation after removing top-ranked RFE features.
python entanglement_ml_pipeline.py ts_rfe_ablation \
    --dataset data/qubit_bures.csv --feature-set SU \
    --model LinearSVC --cv-folds 5 --repeats 5 \
    --out results/rfe_ablation_SU.png \
    --out-csv results/rfe_ablation_SU.csv

# Compare unrestricted training with purity-constrained training while testing
# on the same unrestricted test distribution.
python entanglement_ml_pipeline.py purity_experiment \
    --system 2x2 --metric bures --eta 0.02 \
    --n-train 1000 --n-test 1000 --features Moments \
    --out-dir results/purity_experiment
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import os
import sys
import tempfile
import types
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import sklearn.metrics as sk_metrics

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    get_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

LOGGER = logging.getLogger("entanglement_ml_pipeline")

ENTANGLED_LABEL = -1
SEPARABLE_LABEL = +1

FeatureColumns = Dict[str, List[str]]


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for dataset generation.

    Parameters
    ----------
    system:
        Either "2x2" or "3x3".
    metric:
        Either "bures" or "hs"/"hilbert-schmidt". This controls the random
        full-state distribution used to draw candidate entangled states.
    generation_method:
        Selects one of the state-generation methods exposed by the generate
        CLI command:
            1 = existing logic,
            2 = metric-consistent rejection sampling,
            3 = metric-distributed separable mixtures plus Method 1 entangled,
            4 = controlled depolarization of metric candidates.
    n_entangled:
        Number of entangled examples to accept.
    n_separable:
        Number of known separable examples to construct.
    sep_mixture_terms:
        Number of random product pure states in each separable convex mixture.
        Use 1 to reproduce the pure product-state logic from the beginner
        2x2 notebook. Larger values sample interior points of the separable set.
    max_draws:
        Safety cap on candidate full-state draws while searching for entangled
        metric states.
    method4_depolarize_after:
        Method 4 starts controlled depolarization only after this many rejected
        or unneeded candidates.
    method4_depolarize_step:
        Method 4 decreases p by this step in rho(p)=p*rho+(1-p)I/D while
        searching monotonically for a PPT state.
    purity_filter:
        If True, accept generated states only in the two extreme-purity bands
        [1/D, 1/D + eta] or [1 - eta, 1], where D is the total Hilbert-space
        dimension.
    eta:
        Width of the low- and high-purity acceptance windows used when
        purity_filter is enabled.
    purity_sampling_mode:
        "targeted" samples directly inside the purity windows for efficient
        small-eta experiments. "rejection" keeps the original candidate
        samplers and rejects candidates outside the windows.
    store_purity:
        If True, add a diagnostic ``purity`` column to generated CSV datasets.
        The column is never inferred as a classifier feature.
    ppt_tol:
        Numerical tolerance for declaring a partial transpose non-positive.
    qutrit_script:
        Path to the attached 3x3_svm.py script. Needed only when a qutrit-qutrit
        candidate is PPT and must be tested by the DPS/Gilbert logic.
    reject_ppt_qutrit_without_script:
        If True, PPT qutrit candidates are rejected when no external script is
        supplied. If False, an error is raised.
    random_state:
        Seed for reproducible sampling.
    """

    system: str = "2x2"
    metric: str = "bures"
    generation_method: int = 1
    n_entangled: int = 1000
    n_separable: int = 1000
    sep_mixture_terms: int = 1
    max_draws: int = 10_000_000
    method4_depolarize_after: int = 1000
    method4_depolarize_step: float = 0.01
    purity_filter: bool = False
    eta: float = 0.02
    purity_sampling_mode: str = "targeted"
    store_purity: bool = True
    ppt_tol: float = 1e-10
    qutrit_script: Optional[str] = None
    reject_ppt_qutrit_without_script: bool = True
    random_state: Optional[int] = 42


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for model evaluation."""

    test_size: float = 0.30
    random_state: int = 42
    use_rfe: bool = False
    rfe_step: float = 0.10
    rfe_cv: int = 5
    n_jobs: int = -1
    scoring: str = "balanced_accuracy"


# ---------------------------------------------------------------------------
# Basic quantum utilities
# ---------------------------------------------------------------------------

def canonical_metric(metric: str) -> str:
    """Return a canonical metric name: "bures" or "hs"."""

    m = str(metric).strip().lower().replace("_", "-")
    if m == "bures":
        return "bures"
    if m in {"hs", "hilbert", "hilbert-schmidt", "hilbert schmidt"}:
        return "hs"
    raise ValueError("metric must be 'bures' or 'hs'/'hilbert-schmidt'.")


def dims_from_system(system: str) -> Tuple[int, int]:
    """Parse a system string and return subsystem dimensions."""

    s = str(system).strip().lower().replace(" ", "")
    aliases = {
        "2x2": (2, 2),
        "qubit-qubit": (2, 2),
        "qubit": (2, 2),
        "3x3": (3, 3),
        "qutrit-qutrit": (3, 3),
        "qutrit": (3, 3),
    }
    if s not in aliases:
        raise ValueError("system must be '2x2' or '3x3'.")
    return aliases[s]


def purity(rho: np.ndarray) -> float:
    """Return the purity Tr(rho^2) of a density matrix."""

    return float(np.real(np.trace(rho @ rho)))


def validate_eta(eta: float) -> float:
    """Validate and normalize an eta value for purity-window filtering."""

    eta_value = float(eta)
    if eta_value < 0.0:
        raise ValueError("eta must be non-negative.")
    return eta_value


def purity_window_bounds(total_dim: int, eta: float) -> Dict[str, Tuple[float, float]]:
    """Return low/high extreme-purity interval bounds for total dimension D."""

    eta_value = validate_eta(eta)
    d = int(total_dim)
    if d <= 0:
        raise ValueError("total_dim must be positive.")
    min_purity = 1.0 / float(d)
    return {
        "low": (min_purity, min(min_purity + eta_value, 1.0)),
        "high": (max(1.0 - eta_value, min_purity), 1.0),
    }


def maximal_eta_for_dimension(total_dim: int) -> float:
    """Return the eta value for which the two purity windows cover [1/D, 1]."""

    d = int(total_dim)
    if d <= 0:
        raise ValueError("total_dim must be positive.")
    return (d - 1.0) / (2.0 * d)


def purity_windows_cover_full_range(total_dim: int, eta: float, atol: float = 1e-12) -> bool:
    """True when the configured windows cover the full physical purity range."""

    return validate_eta(eta) >= maximal_eta_for_dimension(total_dim) - atol


def purity_regime(value: float, total_dim: int, eta: float, atol: float = 1e-12) -> str:
    """Classify a purity value as low, high, or middle for the chosen eta."""

    bounds = purity_window_bounds(total_dim, eta)
    p = float(value)
    lo0, lo1 = bounds["low"]
    hi0, hi1 = bounds["high"]
    if (lo0 - atol) <= p <= (lo1 + atol):
        return "low"
    if (hi0 - atol) <= p <= (hi1 + atol):
        return "high"
    return "middle"


def purity_in_extreme_regime(value: float, total_dim: int, eta: float, atol: float = 1e-12) -> bool:
    """True iff purity lies in the low- or high-purity acceptance window."""

    return purity_regime(value, total_dim=total_dim, eta=eta, atol=atol) in {"low", "high"}


def as_density_matrix(rho: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    """Hermitize and normalize a density matrix candidate.

    This is useful after floating-point operations that introduce tiny
    anti-Hermitian components.
    """

    rho = np.asarray(rho, dtype=np.complex128)
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) < atol:
        raise ValueError("density matrix candidate has near-zero trace.")
    rho = rho / tr
    return 0.5 * (rho + rho.conj().T)


def complex_normal(shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Complex standard normal entries with E|z|^2 = 1."""

    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)


def haar_random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Haar-random unitary by QR decomposition of a Ginibre matrix."""

    z = complex_normal((d, d), rng)
    q, r = np.linalg.qr(z)
    diag = np.diag(r)
    phases = np.ones_like(diag)
    nonzero = np.abs(diag) > 0
    phases[nonzero] = diag[nonzero] / np.abs(diag[nonzero])
    return q @ np.diag(np.conj(phases))


def haar_random_pure_state(d: int, rng: np.random.Generator) -> np.ndarray:
    """Return |psi><psi| for a Haar-random pure state in C^d."""

    psi = complex_normal((d,), rng)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def random_hs_density(d: int, rng: np.random.Generator, k: Optional[int] = None) -> np.ndarray:
    """Sample a d x d density matrix from the Hilbert-Schmidt ensemble."""

    if k is None:
        k = d
    g = complex_normal((d, k), rng)
    x = g @ g.conj().T
    return as_density_matrix(x)


def random_bures_density(d: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a d x d density matrix from the Bures ensemble.

    The construction is
        rho ∝ (I + U) G G† (I + U)†,
    with U Haar-random and G complex Ginibre.
    """

    g = complex_normal((d, d), rng)
    u = haar_random_unitary(d, rng)
    a = np.eye(d, dtype=np.complex128) + u
    x = a @ (g @ g.conj().T) @ a.conj().T
    return as_density_matrix(x)


def sample_density_by_metric(metric: str, d: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a density matrix using either Bures or Hilbert-Schmidt metric."""

    m = canonical_metric(metric)
    if m == "bures":
        return random_bures_density(d, rng)
    return random_hs_density(d, rng)


def sample_purity_target(total_dim: int, eta: float, rng: np.random.Generator, regime: str) -> float:
    """Sample a target purity uniformly from the requested low/high window."""

    bounds = purity_window_bounds(total_dim, eta)
    if regime not in bounds:
        raise ValueError("regime must be 'low' or 'high'.")
    lower, upper = bounds[regime]
    if upper < lower:
        raise ValueError(f"Invalid purity bounds for regime {regime}: {(lower, upper)}")
    if np.isclose(lower, upper):
        return float(lower)
    return float(rng.uniform(lower, upper))


def sample_reachable_purity_target(
    seed_rho: np.ndarray,
    total_dim: int,
    eta: float,
    rng: np.random.Generator,
    regime: str,
) -> float:
    """Sample a purity-window target reachable by mixing seed_rho with I/D.

    Since rho(lambda) = (1-lambda)I/D + lambda*sigma can only move from the
    maximally mixed purity 1/D up to purity(sigma), the target interval must be
    intersected with [1/D, purity(seed_rho)].
    """

    bounds = purity_window_bounds(total_dim, eta)
    if regime not in bounds:
        raise ValueError("regime must be 'low' or 'high'.")

    min_purity = 1.0 / float(total_dim)
    seed_purity = purity(as_density_matrix(seed_rho))
    if seed_purity < min_purity - 1e-12:
        raise ValueError(f"seed purity {seed_purity} is below the physical minimum {min_purity}.")

    if regime == "low":
        lower = min_purity
        upper = min(bounds["low"][1], seed_purity)
    else:
        lower = max(bounds["high"][0], min_purity)
        upper = seed_purity

    if upper < lower - 1e-12:
        raise ValueError(
            f"No reachable {regime} purity target for seed purity {seed_purity}; "
            f"reachable interval [1/D, seed_purity]=[{min_purity}, {seed_purity}], "
            f"window={bounds[regime]}."
        )

    upper = max(lower, upper)
    if np.isclose(lower, upper):
        return float(lower)
    return float(rng.uniform(lower, upper))


def mix_with_identity_to_purity(seed_rho: np.ndarray, target_purity: float) -> np.ndarray:
    """Mix a seed state with I/D so the output has the requested purity.

    For rho(lambda) = (1 - lambda) I/D + lambda sigma,
    Tr[rho(lambda)^2] = 1/D + lambda^2 (Tr[sigma^2] - 1/D).
    """

    sigma = as_density_matrix(seed_rho)
    d = sigma.shape[0]
    min_purity = 1.0 / float(d)
    seed_purity = purity(sigma)
    target = float(target_purity)

    if target < min_purity - 1e-12:
        raise ValueError("target_purity is below the maximally mixed purity 1/D.")
    if target > seed_purity + 1e-12:
        raise ValueError(
            f"target_purity={target} exceeds seed purity={seed_purity}; choose a purer seed state."
        )
    if seed_purity <= min_purity + 1e-15:
        return np.eye(d, dtype=np.complex128) / float(d)

    lam = np.sqrt(max(target - min_purity, 0.0) / max(seed_purity - min_purity, 1e-15))
    lam = float(np.clip(lam, 0.0, 1.0))
    rho = (1.0 - lam) * np.eye(d, dtype=np.complex128) / float(d) + lam * sigma
    return as_density_matrix(rho)


def random_product_state(d_a: int, d_b: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a pure product state rho_A ⊗ rho_B."""

    rho_a = haar_random_pure_state(d_a, rng)
    rho_b = haar_random_pure_state(d_b, rng)
    return np.kron(rho_a, rho_b)


def random_separable_state(
    d_a: int,
    d_b: int,
    rng: np.random.Generator,
    mixture_terms: int = 1,
) -> np.ndarray:
    """Generate a known separable state as a convex mixture of product states."""

    if mixture_terms < 1:
        raise ValueError("mixture_terms must be at least 1.")
    if mixture_terms == 1:
        return random_product_state(d_a, d_b, rng)

    weights = rng.dirichlet(np.ones(mixture_terms))
    rho = np.zeros((d_a * d_b, d_a * d_b), dtype=np.complex128)
    for w in weights:
        rho += w * random_product_state(d_a, d_b, rng)
    return as_density_matrix(rho)


def sample_extreme_purity_separable_state(
    d_a: int,
    d_b: int,
    rng: np.random.Generator,
    eta: float,
    regime: str,
    mixture_terms: int = 1,
) -> np.ndarray:
    """Efficiently sample a known separable state in a target purity regime."""

    total_dim = d_a * d_b
    if regime == "high":
        seed = random_product_state(d_a, d_b, rng)
    else:
        seed = random_separable_state(d_a, d_b, rng, mixture_terms=max(1, mixture_terms))
    target = sample_reachable_purity_target(seed, total_dim, eta, rng, regime=regime)
    return mix_with_identity_to_purity(seed, target)


def sample_extreme_purity_entangled_candidate(
    total_dim: int,
    rng: np.random.Generator,
    eta: float,
) -> np.ndarray:
    """Efficiently sample a high-purity full-system candidate.

    This targeted mode is a controlled extreme-purity distribution. It is not
    a strict Bures- or Hilbert-Schmidt-conditioned sample. Existing PPT/DPS
    labelling still verifies the final entangled label.
    """

    seed = haar_random_pure_state(total_dim, rng)
    target = sample_reachable_purity_target(seed, total_dim, eta, rng, regime="high")
    return mix_with_identity_to_purity(seed, target)


def partial_transpose(
    rho: np.ndarray,
    dims: Tuple[int, int],
    subsystem: int = 0,
) -> np.ndarray:
    """Partial transpose of rho on subsystem A (0) or B (1)."""

    d_a, d_b = dims
    rho4 = np.asarray(rho, dtype=np.complex128).reshape(d_a, d_b, d_a, d_b)
    if subsystem == 0:
        rho_pt = rho4.transpose(2, 1, 0, 3)
    elif subsystem == 1:
        rho_pt = rho4.transpose(0, 3, 2, 1)
    else:
        raise ValueError("subsystem must be 0 for A or 1 for B.")
    return rho_pt.reshape(d_a * d_b, d_a * d_b)


def partial_trace(
    rho: np.ndarray,
    dims: Tuple[int, int],
    trace_over: int,
) -> np.ndarray:
    """Partial trace over subsystem A (0) or B (1).

    Returns
    -------
    np.ndarray
        If trace_over == 0, returns Tr_A(rho), a d_B x d_B matrix.
        If trace_over == 1, returns Tr_B(rho), a d_A x d_A matrix.
    """

    d_a, d_b = dims
    rho4 = np.asarray(rho, dtype=np.complex128).reshape(d_a, d_b, d_a, d_b)
    if trace_over == 0:
        return np.trace(rho4, axis1=0, axis2=2)
    if trace_over == 1:
        return np.trace(rho4, axis1=1, axis2=3)
    raise ValueError("trace_over must be 0 for A or 1 for B.")


def ppt_label(
    rho: np.ndarray,
    dims: Tuple[int, int],
    tol: float = 1e-10,
    subsystem: int = 0,
) -> int:
    """Return -1 if rho is NPT entangled, otherwise +1 as PPT/non-NPT.

    For 2x2 this is a complete separability test. For 3x3 this is only a fast
    entanglement detector; PPT states require a stronger qutrit labeller.
    """

    eigvals = np.linalg.eigvalsh(partial_transpose(rho, dims=dims, subsystem=subsystem))
    return ENTANGLED_LABEL if np.min(eigvals) < -tol else SEPARABLE_LABEL


def is_npt(rho: np.ndarray, dims: Tuple[int, int], tol: float = 1e-10) -> bool:
    """True when the partial transpose has a negative eigenvalue."""

    return ppt_label(rho, dims=dims, tol=tol) == ENTANGLED_LABEL


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def pauli_basis() -> Tuple[List[str], List[np.ndarray]]:
    """Identity plus Pauli matrices in the order I, X, Y, Z."""

    names = ["I", "X", "Y", "Z"]
    mats = [
        np.array([[1, 0], [0, 1]], dtype=np.complex128),
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.array([[1, 0], [0, -1]], dtype=np.complex128),
    ]
    return names, mats


def gell_mann_basis_with_identity() -> Tuple[List[str], List[np.ndarray]]:
    """Identity plus the eight standard Gell-Mann matrices."""

    z = np.zeros((3, 3), dtype=np.complex128)
    mats: List[np.ndarray] = []
    names = ["I", "lambda1", "lambda2", "lambda3", "lambda4", "lambda5", "lambda6", "lambda7", "lambda8"]

    mats.append(np.eye(3, dtype=np.complex128))

    lam1 = z.copy(); lam1[0, 1] = lam1[1, 0] = 1
    lam2 = z.copy(); lam2[0, 1] = -1j; lam2[1, 0] = 1j
    lam3 = np.diag([1, -1, 0]).astype(np.complex128)
    lam4 = z.copy(); lam4[0, 2] = lam4[2, 0] = 1
    lam5 = z.copy(); lam5[0, 2] = -1j; lam5[2, 0] = 1j
    lam6 = z.copy(); lam6[1, 2] = lam6[2, 1] = 1
    lam7 = z.copy(); lam7[1, 2] = -1j; lam7[2, 1] = 1j
    lam8 = (1.0 / np.sqrt(3.0)) * np.diag([1, 1, -2]).astype(np.complex128)

    mats.extend([lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8])
    return names, mats


def su_basis_for_dim(d: int) -> Tuple[List[str], List[np.ndarray]]:
    """Return the identity+generator basis for local dimension 2 or 3."""

    if d == 2:
        return pauli_basis()
    if d == 3:
        return gell_mann_basis_with_identity()
    raise NotImplementedError("SU features are implemented for local dimensions 2 and 3 only.")


def su_feature_names(dims: Tuple[int, int]) -> List[str]:
    """Column names for SU/Pauli/Gell-Mann expectation features."""

    names_a, _ = su_basis_for_dim(dims[0])
    names_b, _ = su_basis_for_dim(dims[1])
    return [f"SU_{a}x{b}" for a in names_a for b in names_b]


def su_features(rho: np.ndarray, dims: Tuple[int, int]) -> np.ndarray:
    """Compute expectation values Tr[rho (A_i ⊗ B_j)]."""

    _, basis_a = su_basis_for_dim(dims[0])
    _, basis_b = su_basis_for_dim(dims[1])
    values = [np.trace(rho @ np.kron(a, b)) for a in basis_a for b in basis_b]
    return np.real_if_close(np.asarray(values, dtype=np.complex128), tol=1000).real.astype(float)


def moment_feature_names() -> List[str]:
    """Column names for partial-transpose moment features."""

    return [
        "Moment_Tr_PT_1",
        "Moment_Tr_PT_2",
        "Moment_Tr_PT_3",
        "Moment_Tr_PT_4",
        "Moment_det_PT",
    ]


def pt_moment_features(
    rho: np.ndarray,
    dims: Tuple[int, int],
    pt_subsystem: int = 0,
) -> np.ndarray:
    """Compute Tr[(rho^Gamma)^k], k=1..4, and det(rho^Gamma)."""

    rho_pt = partial_transpose(rho, dims=dims, subsystem=pt_subsystem)
    powers = []
    running = np.eye(rho_pt.shape[0], dtype=np.complex128)
    for _ in range(1, 5):
        running = running @ rho_pt
        powers.append(np.trace(running))
    powers.append(np.linalg.det(rho_pt))
    return np.real_if_close(np.asarray(powers, dtype=np.complex128), tol=1000).real.astype(float)


def rm_invariant_names() -> List[str]:
    """Column names for randomized-measurement invariant features."""

    return [f"RMI_x{i}" for i in range(10)]


def rm_invariant_features(rho: np.ndarray, dims: Tuple[int, int]) -> np.ndarray:
    """Compute the ten randomized-measurement invariant features.

    Notation follows the prompt:
    * Tr_A(rho) is rho_B, the reduced state on subsystem B.
    * Tr_B(rho) is rho_A, the reduced state on subsystem A.
    """

    rho = np.asarray(rho, dtype=np.complex128)
    rho_a = partial_trace(rho, dims=dims, trace_over=1)  # Tr_B rho
    rho_b = partial_trace(rho, dims=dims, trace_over=0)  # Tr_A rho

    rho2 = rho @ rho
    rho3 = rho2 @ rho
    rho2_a = partial_trace(rho2, dims=dims, trace_over=1)  # Tr_B rho^2
    rho2_b = partial_trace(rho2, dims=dims, trace_over=0)  # Tr_A rho^2
    rho_pt_a = partial_transpose(rho, dims=dims, subsystem=0)

    x0 = np.trace(rho) ** 3
    x1 = np.trace(rho_b @ rho_b)
    x2 = np.trace(rho_b @ rho_b @ rho_b)
    x3 = np.trace(rho_a @ rho_a)
    x4 = np.trace(np.kron(rho_a, rho_b) @ rho)
    x5 = np.trace(rho2)
    x6 = np.trace(rho2_b @ rho_b)
    x7 = np.trace(rho_a @ rho_a @ rho_a)
    x8 = np.trace(rho2_a @ rho_a)
    x9 = 0.5 * (np.trace(rho3) + np.trace(rho_pt_a @ rho_pt_a @ rho_pt_a))

    values = np.asarray([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9], dtype=np.complex128)
    return np.real_if_close(values, tol=1000).real.astype(float)


def feature_columns_for_dims(dims: Tuple[int, int]) -> FeatureColumns:
    """Return grouped feature column names for the requested system."""

    return {
        "SU": su_feature_names(dims),
        "Moment": moment_feature_names(),
        "RMInvariant": rm_invariant_names(),
    }


def extract_feature_groups(rho: np.ndarray, dims: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """Extract all feature families from a state."""

    return {
        "SU": su_features(rho, dims),
        "Moment": pt_moment_features(rho, dims),
        "RMInvariant": rm_invariant_features(rho, dims),
    }


def feature_row(
    rho: np.ndarray,
    dims: Tuple[int, int],
    y: int,
    include_purity: bool = False,
) -> Dict[str, float]:
    """Return a flat feature row suitable for a pandas DataFrame."""

    groups = extract_feature_groups(rho, dims)
    columns = feature_columns_for_dims(dims)
    row: Dict[str, float] = {}
    for group_name in ("SU", "Moment", "RMInvariant"):
        for col, val in zip(columns[group_name], groups[group_name]):
            row[col] = float(val)
    if include_purity:
        row["purity"] = purity(rho)
    row["y"] = int(y)
    return row


# ---------------------------------------------------------------------------
# Safe adapter for attached 3x3 DPS/Gilbert logic
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def load_attached_3x3_module(script_path: str) -> types.ModuleType:
    """Load definitions from 3x3_svm.py without running its bottom simulation.

    The provided script calls simulate_all_parameters() at module import time.
    This loader executes only the definitions above that final call.
    """

    path = Path(script_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cannot find qutrit script: {path}")

    source = path.read_text(encoding="utf-8")
    sentinel = "\nsimulate_all_parameters()"
    idx = source.rfind(sentinel)
    if idx != -1:
        source = source[:idx]

    module = types.ModuleType("attached_3x3_svm_safe")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


class QutritDPSGilbertLabeler:
    """Label qutrit-qutrit states with fast PPT plus attached DPS/Gilbert logic.

    Returns
    -------
    int | None
        -1 for entangled, +1 for separable, None for inconclusive/rejected.
    """

    def __init__(
        self,
        script_path: Optional[str],
        ppt_tol: float = 1e-10,
        reject_ppt_without_script: bool = True,
        verbose_external: bool = False,
    ) -> None:
        self.script_path = script_path
        self.ppt_tol = ppt_tol
        self.reject_ppt_without_script = reject_ppt_without_script
        self.verbose_external = verbose_external

    def __call__(self, rho: np.ndarray) -> Optional[int]:
        dims = (3, 3)

        # Fast, rigorous entanglement certificate for NPT qutrit-qutrit states.
        if is_npt(rho, dims=dims, tol=self.ppt_tol):
            return ENTANGLED_LABEL

        # PPT qutrit states need the stronger attached logic.
        if self.script_path is None:
            if self.reject_ppt_without_script:
                return None
            raise RuntimeError(
                "A PPT 3x3 candidate requires the attached DPS/Gilbert script. "
                "Pass --qutrit-script /path/to/3x3_svm.py or enable rejection."
            )

        try:
            import torch  # Imported lazily because only the attached script needs it.

            module = load_attached_3x3_module(self.script_path)
            rho_torch = torch.as_tensor(rho, dtype=torch.complex128)

            # The attached Gilbert code writes temporary MatrixMarket files with
            # fixed names. Run it in an isolated temporary working directory.
            with tempfile.TemporaryDirectory(prefix="qutrit_dps_gilbert_") as tmp:
                old_cwd = os.getcwd()
                os.chdir(tmp)
                try:
                    if self.verbose_external:
                        external_label = module.entanglement_label(rho_torch, 3, 3)
                    else:
                        with open(os.devnull, "w", encoding="utf-8", errors="replace") as devnull:
                            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                                external_label = module.entanglement_label(rho_torch, 3, 3)
                finally:
                    os.chdir(old_cwd)
        except Exception as exc:
            LOGGER.warning("Attached 3x3 labeller failed; rejecting state. Error: %s", exc)
            return None

        if float(external_label) == 1.0:
            return ENTANGLED_LABEL
        if float(external_label) == 0.0:
            return SEPARABLE_LABEL
        return None


# ---------------------------------------------------------------------------
# Dataset generation and storage
# ---------------------------------------------------------------------------

def canonical_generation_method(method: int | str) -> int:
    """Return a validated state-generation method id."""

    try:
        method_id = int(method)
    except (TypeError, ValueError) as exc:
        raise ValueError("generation_method must be one of 1, 2, 3, or 4.") from exc
    if method_id not in {1, 2, 3, 4}:
        raise ValueError("generation_method must be one of 1, 2, 3, or 4.")
    return method_id


def build_qutrit_labeler(config: DatasetConfig, dims: Tuple[int, int]) -> Optional[QutritDPSGilbertLabeler]:
    """Create the attached qutrit DPS/Gilbert labeler only when needed."""

    if dims != (3, 3):
        return None
    return QutritDPSGilbertLabeler(
        script_path=config.qutrit_script,
        ppt_tol=config.ppt_tol,
        reject_ppt_without_script=config.reject_ppt_qutrit_without_script,
    )


def label_with_existing_criteria(
    rho: np.ndarray,
    dims: Tuple[int, int],
    config: DatasetConfig,
    qutrit_labeler: Optional[QutritDPSGilbertLabeler],
) -> Optional[int]:
    """Label a state with the existing PPT/DPS/Gilbert decision logic."""

    if dims == (2, 2):
        return ppt_label(rho, dims=dims, tol=config.ppt_tol, subsystem=0)
    if dims == (3, 3):
        assert qutrit_labeler is not None
        return qutrit_labeler(rho)
    raise NotImplementedError("Only 2x2 and 3x3 systems are supported.")


def canonical_purity_sampling_mode(mode: str) -> str:
    """Return a validated purity sampling mode."""

    m = str(mode).strip().lower().replace("_", "-")
    if m in {"targeted", "rejection"}:
        return m
    raise ValueError("purity_sampling_mode must be 'targeted' or 'rejection'.")


def validate_purity_generation_config(config: DatasetConfig) -> Tuple[float, str]:
    """Validate purity-related config fields and return normalized values."""

    eta = validate_eta(config.eta)
    mode = canonical_purity_sampling_mode(config.purity_sampling_mode)
    if config.purity_filter and eta <= 0.0:
        raise ValueError("purity_filter requires eta > 0.")
    return eta, mode


def purity_accepts_state(rho: np.ndarray, dims: Tuple[int, int], config: DatasetConfig) -> bool:
    """Return True when rho satisfies the configured purity filter."""

    if not config.purity_filter:
        return True
    d = dims[0] * dims[1]
    return purity_in_extreme_regime(purity(rho), total_dim=d, eta=config.eta)


def feature_row_for_config(
    rho: np.ndarray,
    dims: Tuple[int, int],
    label: int,
    config: DatasetConfig,
) -> Dict[str, float]:
    """Build a row, optionally carrying the diagnostic purity column."""

    return feature_row(rho, dims, label, include_purity=config.store_purity)


def append_state_row(
    rows: List[Dict[str, float]],
    rho: np.ndarray,
    dims: Tuple[int, int],
    label: int,
    config: DatasetConfig,
) -> None:
    """Append a generated state using the configured diagnostic columns."""

    rows.append(feature_row_for_config(rho, dims, label, config))


def targeted_purity_regime(rng: np.random.Generator) -> str:
    """Pick a low/high extreme-purity regime for targeted sampling."""

    return "high" if bool(rng.integers(0, 2)) else "low"


def sample_full_state_candidate_for_config(
    metric: str,
    total_dim: int,
    rng: np.random.Generator,
    config: DatasetConfig,
) -> np.ndarray:
    """Sample a full-system candidate respecting the configured purity mode."""

    _, mode = validate_purity_generation_config(config)
    if (
        config.purity_filter
        and mode == "targeted"
        and not purity_windows_cover_full_range(total_dim, config.eta)
    ):
        regime = targeted_purity_regime(rng)
        if regime == "high":
            seed = haar_random_pure_state(total_dim, rng)
        else:
            seed = sample_density_by_metric(metric, total_dim, rng)
        target = sample_reachable_purity_target(seed, total_dim, config.eta, rng, regime=regime)
        return mix_with_identity_to_purity(seed, target)
    return sample_density_by_metric(metric, total_dim, rng)


def sample_separable_state_for_config(
    dims: Tuple[int, int],
    metric: str,
    rng: np.random.Generator,
    config: DatasetConfig,
    metric_local: bool = False,
) -> np.ndarray:
    """Sample a separable candidate respecting the configured purity mode."""

    _, mode = validate_purity_generation_config(config)
    d_a, d_b = dims
    if (
        config.purity_filter
        and mode == "targeted"
        and not purity_windows_cover_full_range(d_a * d_b, config.eta)
    ):
        return sample_extreme_purity_separable_state(
            d_a,
            d_b,
            rng=rng,
            eta=config.eta,
            regime=targeted_purity_regime(rng),
            mixture_terms=config.sep_mixture_terms,
        )
    if metric_local:
        return metric_distributed_separable_state(
            d_a,
            d_b,
            metric=metric,
            rng=rng,
            mixture_terms=config.sep_mixture_terms,
        )
    return random_separable_state(
        d_a,
        d_b,
        rng=rng,
        mixture_terms=config.sep_mixture_terms,
    )


def finalize_dataset_rows(rows: List[Dict[str, float]], config: DatasetConfig) -> pd.DataFrame:
    """Shuffle and return the feature-only dataset in the existing format."""

    if not rows:
        raise RuntimeError("No states were accepted; cannot build an empty dataset.")
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=config.random_state).reset_index(drop=True)
    df["y"] = df["y"].astype(int)
    return df


def quota_remaining(label: int, accepted_counts: Mapping[int, int], config: DatasetConfig) -> bool:
    """Return True when another row of this label is still requested."""

    if label == ENTANGLED_LABEL:
        return accepted_counts.get(ENTANGLED_LABEL, 0) < config.n_entangled
    if label == SEPARABLE_LABEL:
        return accepted_counts.get(SEPARABLE_LABEL, 0) < config.n_separable
    return False


def requested_quotas_met(accepted_counts: Mapping[int, int], config: DatasetConfig) -> bool:
    """Return True when both requested label counts have been accepted."""

    return (
        accepted_counts.get(ENTANGLED_LABEL, 0) >= config.n_entangled
        and accepted_counts.get(SEPARABLE_LABEL, 0) >= config.n_separable
    )


def accept_labelled_state_if_needed(
    rows: List[Dict[str, float]],
    rho: np.ndarray,
    dims: Tuple[int, int],
    label: Optional[int],
    accepted_counts: Dict[int, int],
    config: DatasetConfig,
) -> bool:
    """Append a labelled state when its requested class still has capacity."""

    if (
        label is None
        or not quota_remaining(label, accepted_counts, config)
        or not purity_accepts_state(rho, dims, config)
    ):
        return False
    append_state_row(rows, rho, dims, label, config)
    accepted_counts[label] = accepted_counts.get(label, 0) + 1
    return True


def generate_dataset_method_1(config: DatasetConfig) -> pd.DataFrame:
    """Generate a balanced feature-only dataset.

    The returned DataFrame contains no raw density matrices. It contains flat
    feature columns and the label column y.

    Method 1 is the original implementation: separable states are convex
    mixtures of product states, while entangled states are accepted from metric
    candidates with the existing PPT/DPS/Gilbert logic.
    """

    dims = dims_from_system(config.system)
    d = dims[0] * dims[1]
    metric = canonical_metric(config.metric)
    validate_purity_generation_config(config)
    rng = np.random.default_rng(config.random_state)

    rows: List[Dict[str, float]] = []

    LOGGER.info("Generating %d known separable %s states.", config.n_separable, config.system)
    accepted_separable = 0
    separable_draws = 0
    rejected_separable_by_purity = 0
    while accepted_separable < config.n_separable and separable_draws < config.max_draws:
        separable_draws += 1
        rho_sep = sample_separable_state_for_config(dims, metric, rng, config)
        if not purity_accepts_state(rho_sep, dims, config):
            rejected_separable_by_purity += 1
            continue
        append_state_row(rows, rho_sep, dims, SEPARABLE_LABEL, config)
        accepted_separable += 1
        if accepted_separable % 100 == 0:
            LOGGER.info("  separable: %d/%d", accepted_separable, config.n_separable)

    if accepted_separable < config.n_separable:
        raise RuntimeError(
            f"Reached max_draws={config.max_draws} before collecting "
            f"{config.n_separable} separable states. Accepted {accepted_separable}; "
            f"rejected by purity {rejected_separable_by_purity}."
        )

    qutrit_labeler = build_qutrit_labeler(config, dims)

    LOGGER.info(
        "Generating %d entangled %s states from %s metric candidates.",
        config.n_entangled,
        config.system,
        metric,
    )
    accepted_entangled = 0
    draws = 0
    rejected_or_wrong_label = 0
    rejected_by_purity = 0
    rejected_entangled_by_purity = 0

    while accepted_entangled < config.n_entangled and draws < config.max_draws:
        draws += 1
        rho = sample_full_state_candidate_for_config(metric, d, rng, config)

        if not purity_accepts_state(rho, dims, config):
            rejected_entangled_by_purity += 1
            continue

        label = label_with_existing_criteria(rho, dims, config, qutrit_labeler)

        if label == ENTANGLED_LABEL:
            append_state_row(rows, rho, dims, ENTANGLED_LABEL, config)
            accepted_entangled += 1
            if accepted_entangled % 100 == 0:
                LOGGER.info("  entangled: %d/%d", accepted_entangled, config.n_entangled)
        else:
            rejected_or_wrong_label += 1

    if accepted_entangled < config.n_entangled:
        raise RuntimeError(
            f"Reached max_draws={config.max_draws} before collecting "
            f"{config.n_entangled} entangled states. Accepted {accepted_entangled}; "
            f"rejected/non-entangled/inconclusive {rejected_or_wrong_label}; "
            f"rejected by purity {rejected_entangled_by_purity}."
        )

    return finalize_dataset_rows(rows, config)


def label_name(label: int) -> str:
    """Human-readable label name for logging."""

    return "entangled" if label == ENTANGLED_LABEL else "separable"


def requested_count_for_label(label: int, config: DatasetConfig) -> int:
    """Requested sample count for a label."""

    if label == ENTANGLED_LABEL:
        return config.n_entangled
    if label == SEPARABLE_LABEL:
        return config.n_separable
    raise ValueError(f"Unknown label: {label}")


def collect_metric_rejection_samples(
    rows: List[Dict[str, float]],
    dims: Tuple[int, int],
    metric: str,
    rng: np.random.Generator,
    config: DatasetConfig,
    qutrit_labeler: Optional[QutritDPSGilbertLabeler],
    target_label: int,
) -> int:
    """Method 2 helper: accept metric samples only when their label matches."""

    requested = requested_count_for_label(target_label, config)
    if requested <= 0:
        return 0

    accepted = 0
    draws = 0
    rejected_or_inconclusive = 0
    rejected_by_purity = 0
    name = label_name(target_label)

    LOGGER.info(
        "Method 2: generating %d %s %s states from %s metric rejection sampling.",
        requested,
        name,
        config.system,
        metric,
    )
    while accepted < requested and draws < config.max_draws:
        draws += 1
        rho = sample_full_state_candidate_for_config(metric, dims[0] * dims[1], rng, config)
        if not purity_accepts_state(rho, dims, config):
            rejected_by_purity += 1
            continue
        label = label_with_existing_criteria(rho, dims, config, qutrit_labeler)

        if label == target_label:
            append_state_row(rows, rho, dims, target_label, config)
            accepted += 1
            if accepted % 100 == 0:
                LOGGER.info("  %s: %d/%d", name, accepted, requested)
        else:
            rejected_or_inconclusive += 1

    if accepted < requested:
        LOGGER.warning(
            "Method 2 stopped after max_draws=%d while collecting %s states. "
            "Accepted %d/%d; rejected/wrong-label/inconclusive %d. "
            "Saving the accepted rows so downstream balancing can be handled separately.",
            config.max_draws,
            name,
            accepted,
            requested,
            rejected_or_inconclusive + rejected_by_purity,
        )
    return accepted


def generate_dataset_method_2(config: DatasetConfig) -> pd.DataFrame:
    """Metric-consistent rejection sampling for both labels.

    Every candidate state is drawn from the selected metric distribution.
    Entangled and separable examples differ only by the acceptance criterion.
    """

    dims = dims_from_system(config.system)
    metric = canonical_metric(config.metric)
    validate_purity_generation_config(config)
    rng = np.random.default_rng(config.random_state)
    qutrit_labeler = build_qutrit_labeler(config, dims)
    rows: List[Dict[str, float]] = []

    collect_metric_rejection_samples(
        rows,
        dims,
        metric,
        rng,
        config,
        qutrit_labeler,
        ENTANGLED_LABEL,
    )
    collect_metric_rejection_samples(
        rows,
        dims,
        metric,
        rng,
        config,
        qutrit_labeler,
        SEPARABLE_LABEL,
    )
    return finalize_dataset_rows(rows, config)


def random_metric_product_state(
    d_a: int,
    d_b: int,
    metric: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate rho_A tensor rho_B with local states drawn from the metric."""

    rho_a = sample_density_by_metric(metric, d_a, rng)
    rho_b = sample_density_by_metric(metric, d_b, rng)
    return as_density_matrix(np.kron(rho_a, rho_b))


def metric_distributed_separable_state(
    d_a: int,
    d_b: int,
    metric: str,
    rng: np.random.Generator,
    mixture_terms: int = 1,
) -> np.ndarray:
    """Generate a guaranteed separable state from metric-local product terms."""

    if mixture_terms < 1:
        raise ValueError("mixture_terms must be at least 1.")
    if mixture_terms == 1:
        return random_metric_product_state(d_a, d_b, metric, rng)

    weights = rng.dirichlet(np.ones(mixture_terms))
    rho = np.zeros((d_a * d_b, d_a * d_b), dtype=np.complex128)
    for w in weights:
        rho += w * random_metric_product_state(d_a, d_b, metric, rng)
    return as_density_matrix(rho)


def collect_method_1_entangled_samples(
    rows: List[Dict[str, float]],
    dims: Tuple[int, int],
    metric: str,
    rng: np.random.Generator,
    config: DatasetConfig,
    qutrit_labeler: Optional[QutritDPSGilbertLabeler],
) -> int:
    """Collect entangled metric candidates using Method 1's label logic."""

    accepted_entangled = 0
    draws = 0
    rejected_or_wrong_label = 0

    LOGGER.info(
        "Generating %d entangled %s states from %s metric candidates.",
        config.n_entangled,
        config.system,
        metric,
    )
    while accepted_entangled < config.n_entangled and draws < config.max_draws:
        draws += 1
        rho = sample_full_state_candidate_for_config(metric, dims[0] * dims[1], rng, config)
        if not purity_accepts_state(rho, dims, config):
            rejected_by_purity += 1
            continue
        label = label_with_existing_criteria(rho, dims, config, qutrit_labeler)

        if label == ENTANGLED_LABEL:
            append_state_row(rows, rho, dims, ENTANGLED_LABEL, config)
            accepted_entangled += 1
            if accepted_entangled % 100 == 0:
                LOGGER.info("  entangled: %d/%d", accepted_entangled, config.n_entangled)
        else:
            rejected_or_wrong_label += 1

    if accepted_entangled < config.n_entangled:
        raise RuntimeError(
            f"Reached max_draws={config.max_draws} before collecting "
            f"{config.n_entangled} entangled states. Accepted {accepted_entangled}; "
            f"rejected/non-entangled/inconclusive {rejected_or_wrong_label}; "
            f"rejected by purity {rejected_by_purity}."
        )
    return accepted_entangled


def generate_dataset_method_3(config: DatasetConfig) -> pd.DataFrame:
    """Metric-distributed convex combinations for guaranteed separability.

    Separable rows are labelled by construction and never sent to PPT/DPS/Gilbert.
    Entangled rows deliberately fall back to Method 1 metric-candidate sampling.
    """

    dims = dims_from_system(config.system)
    metric = canonical_metric(config.metric)
    validate_purity_generation_config(config)
    rng = np.random.default_rng(config.random_state)
    rows: List[Dict[str, float]] = []

    LOGGER.info(
        "Method 3: generating %d guaranteed separable %s states from metric-local product mixtures.",
        config.n_separable,
        config.system,
    )
    accepted_separable = 0
    separable_draws = 0
    rejected_by_purity = 0
    while accepted_separable < config.n_separable and separable_draws < config.max_draws:
        separable_draws += 1
        rho_sep = sample_separable_state_for_config(
            dims,
            metric,
            rng,
            config,
            metric_local=True,
        )
        if not purity_accepts_state(rho_sep, dims, config):
            rejected_by_purity += 1
            continue
        append_state_row(rows, rho_sep, dims, SEPARABLE_LABEL, config)
        accepted_separable += 1
        if accepted_separable % 100 == 0:
            LOGGER.info("  separable: %d/%d", accepted_separable, config.n_separable)

    if accepted_separable < config.n_separable:
        raise RuntimeError(
            f"Reached max_draws={config.max_draws} before collecting "
            f"{config.n_separable} separable states. Accepted {accepted_separable}; "
            f"rejected by purity {rejected_by_purity}."
        )

    qutrit_labeler = build_qutrit_labeler(config, dims)
    collect_method_1_entangled_samples(rows, dims, metric, rng, config, qutrit_labeler)
    return finalize_dataset_rows(rows, config)


def depolarize_until_ppt(
    rho: np.ndarray,
    dims: Tuple[int, int],
    ppt_tol: float,
    p_step: float,
) -> Tuple[np.ndarray, float]:
    """Decrease p monotonically in p*rho+(1-p)I/D until the state is PPT."""

    if p_step <= 0.0 or p_step > 1.0:
        raise ValueError("method4_depolarize_step must be in the interval (0, 1].")

    total_dim = dims[0] * dims[1]
    maximally_mixed = np.eye(total_dim, dtype=np.complex128) / total_dim
    p = 1.0
    candidate = as_density_matrix(rho)
    if not is_npt(candidate, dims=dims, tol=ppt_tol):
        return candidate, p

    while p > 0.0:
        p = max(0.0, p - p_step)
        candidate = as_density_matrix(p * rho + (1.0 - p) * maximally_mixed)
        if not is_npt(candidate, dims=dims, tol=ppt_tol):
            return candidate, p

    return maximally_mixed, 0.0


def generate_dataset_method_4(config: DatasetConfig) -> pd.DataFrame:
    """Controlled depolarization of metric candidates.

    Metric candidates are labelled directly when possible. After the rejection
    counter reaches the configured threshold, failed candidates are mixed
    monotonically with the maximally mixed state until they become PPT, then
    labelled with the appropriate 2x2 PPT or 3x3 DPS/Gilbert logic.
    """

    if config.method4_depolarize_after < 1:
        raise ValueError("method4_depolarize_after must be at least 1.")

    dims = dims_from_system(config.system)
    metric = canonical_metric(config.metric)
    validate_purity_generation_config(config)
    rng = np.random.default_rng(config.random_state)
    qutrit_labeler = build_qutrit_labeler(config, dims)
    rows: List[Dict[str, float]] = []
    accepted_counts: Dict[int, int] = {
        ENTANGLED_LABEL: 0,
        SEPARABLE_LABEL: 0,
    }

    draws = 0
    rejection_counter = 0
    depolarized_attempts = 0

    LOGGER.info(
        "Method 4: generating %d entangled and %d separable %s states from %s metric candidates.",
        config.n_entangled,
        config.n_separable,
        config.system,
        metric,
    )
    while not requested_quotas_met(accepted_counts, config) and draws < config.max_draws:
        draws += 1
        rho = sample_full_state_candidate_for_config(metric, dims[0] * dims[1], rng, config)
        label = label_with_existing_criteria(rho, dims, config, qutrit_labeler)

        if accept_labelled_state_if_needed(rows, rho, dims, label, accepted_counts, config):
            rejection_counter = 0
            LOGGER.debug(
                "Method 4 accepted raw %s state: entangled=%d/%d, separable=%d/%d",
                label_name(label),
                accepted_counts[ENTANGLED_LABEL],
                config.n_entangled,
                accepted_counts[SEPARABLE_LABEL],
                config.n_separable,
            )
            continue

        rejection_counter += 1
        if rejection_counter < config.method4_depolarize_after:
            continue

        rho_ppt, p_value = depolarize_until_ppt(
            rho,
            dims=dims,
            ppt_tol=config.ppt_tol,
            p_step=config.method4_depolarize_step,
        )
        depolarized_attempts += 1
        depolarized_label = label_with_existing_criteria(rho_ppt, dims, config, qutrit_labeler)

        if accept_labelled_state_if_needed(rows, rho_ppt, dims, depolarized_label, accepted_counts, config):
            rejection_counter = 0
            LOGGER.debug(
                "Method 4 accepted depolarized %s state at p=%.6f: entangled=%d/%d, separable=%d/%d",
                label_name(depolarized_label),
                p_value,
                accepted_counts[ENTANGLED_LABEL],
                config.n_entangled,
                accepted_counts[SEPARABLE_LABEL],
                config.n_separable,
            )

    if not requested_quotas_met(accepted_counts, config):
        raise RuntimeError(
            f"Reached max_draws={config.max_draws} before collecting the requested Method 4 dataset. "
            f"Accepted entangled {accepted_counts[ENTANGLED_LABEL]}/{config.n_entangled}, "
            f"separable {accepted_counts[SEPARABLE_LABEL]}/{config.n_separable}; "
            f"controlled depolarization attempts {depolarized_attempts}."
        )
    return finalize_dataset_rows(rows, config)


def generate_dataset(config: DatasetConfig) -> pd.DataFrame:
    """Generate a feature-only dataset with the configured state method."""

    method = canonical_generation_method(config.generation_method)
    if method == 1:
        return generate_dataset_method_1(config)
    if method == 2:
        return generate_dataset_method_2(config)
    if method == 3:
        return generate_dataset_method_3(config)
    return generate_dataset_method_4(config)


def save_dataset_bundle(
    df: pd.DataFrame,
    out_csv: str | Path,
    config: DatasetConfig,
    write_npz: bool = True,
) -> None:
    """Save a feature-only dataset as CSV plus metadata and optional NPZ arrays."""

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    dims = dims_from_system(config.system)
    columns = feature_columns_for_dims(dims)
    metadata = {
        "config": asdict(config),
        "labels": {"entangled": ENTANGLED_LABEL, "separable": SEPARABLE_LABEL},
        "feature_columns": columns,
        "diagnostic_columns": [c for c in ["purity"] if c in df.columns],
        "csv": str(out_csv),
        "contains_raw_density_matrices": False,
    }
    metadata_path = out_csv.with_suffix(out_csv.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if write_npz:
        npz_path = out_csv.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            SU_features=df[columns["SU"]].to_numpy(dtype=float),
            Moment_features=df[columns["Moment"]].to_numpy(dtype=float),
            RMInvariant_features=df[columns["RMInvariant"]].to_numpy(dtype=float),
            y=df["y"].to_numpy(dtype=int),
        )
        metadata["npz"] = str(npz_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Load a CSV or NPZ dataset created by this pipeline."""

    path = Path(dataset_path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        df["y"] = df["y"].astype(int)
        return df

    if path.suffix.lower() == ".npz":
        metadata_candidates = [
            path.with_suffix(".csv.metadata.json"),
            path.with_suffix(path.suffix + ".metadata.json"),
        ]
        metadata = None
        for candidate in metadata_candidates:
            if candidate.exists():
                metadata = json.loads(candidate.read_text(encoding="utf-8"))
                break
        if metadata is None:
            raise FileNotFoundError(
                "NPZ loading requires the metadata JSON generated with the dataset. "
                "Load the CSV instead or keep the .metadata.json file."
            )
        arrays = np.load(path)
        cols = metadata["feature_columns"]
        df = pd.DataFrame(
            np.hstack([
                arrays["SU_features"],
                arrays["Moment_features"],
                arrays["RMInvariant_features"],
            ]),
            columns=cols["SU"] + cols["Moment"] + cols["RMInvariant"],
        )
        df["y"] = arrays["y"].astype(int)
        return df

    raise ValueError("dataset_path must point to a .csv or .npz file.")


# ---------------------------------------------------------------------------
# ML feature scenarios, RFE, model evaluation
# ---------------------------------------------------------------------------

DEFAULT_SCENARIOS: Mapping[str, Tuple[str, ...]] = {
    "SU": ("SU",),
    "Moment": ("Moment",),
    "RMInvariant": ("RMInvariant",),
    "SU+Moment": ("SU", "Moment"),
    "SU+RMInvariant": ("SU", "RMInvariant"),
    "Moment+RMInvariant": ("Moment", "RMInvariant"),
    "ALL": ("SU", "Moment", "RMInvariant"),
}

RFE_ABLATION_SCENARIOS: Mapping[str, Tuple[str, ...]] = {
    "SU": DEFAULT_SCENARIOS["SU"],
    "Moment": DEFAULT_SCENARIOS["Moment"],
    "RMInvariant": DEFAULT_SCENARIOS["RMInvariant"],
    "ALL": DEFAULT_SCENARIOS["ALL"],
}

# User-facing aliases for the advanced visualization interface requested in
# this version. These aliases deliberately keep the physics feature groups
# unchanged: "Moments" maps to the existing "Moment" group and "RM" maps to
# the existing "RMInvariant" group.
ANALYSIS_SCENARIOS: Mapping[str, Tuple[str, ...]] = {
    "SU": ("SU",),
    "Moments": ("Moment",),
    "RM": ("RMInvariant",),
    "All": ("SU", "Moment", "RMInvariant"),
}

PLOT_CHOICES: Tuple[str, ...] = (
    "RocCurve",
    "ResidualPlot",
    "SHAPPlot",
    "MarginDistributionPlot",
)

DEFAULT_DATASET_FILE_CANDIDATES: Mapping[str, Tuple[str, ...]] = {
    "2x2": (
        "data_2x2.csv",
        "2x2.csv",
        "qubit_bures.csv",
        "qubit_hs.csv",
        "data/qubit_bures.csv",
        "data/qubit_hs.csv",
    ),
    "3x3": (
        "data_3x3.csv",
        "3x3.csv",
        "qutrit_hs.csv",
        "qutrit_bures.csv",
        "data/qutrit_hs.csv",
        "data/qutrit_bures.csv",
    ),
}


def infer_feature_columns(df: pd.DataFrame) -> FeatureColumns:
    """Infer feature groups from column prefixes."""

    return {
        "SU": [c for c in df.columns if c.startswith("SU_")],
        "Moment": [c for c in df.columns if c.startswith("Moment_")],
        "RMInvariant": [c for c in df.columns if c.startswith("RMI_")],
    }


def columns_for_scenario(
    df: pd.DataFrame,
    scenario_groups: Sequence[str],
) -> List[str]:
    """Return flat DataFrame columns for a feature scenario."""

    grouped = infer_feature_columns(df)
    cols: List[str] = []
    for group in scenario_groups:
        if group not in grouped:
            raise KeyError(f"Unknown feature group: {group}")
        cols.extend(grouped[group])
    if not cols:
        raise ValueError(f"Scenario {scenario_groups} contains no columns.")
    return cols


def build_models(random_state: int = 42, n_jobs: int = -1) -> Dict[str, object]:
    """Construct the requested classifiers."""

    linear_svm = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LinearSVC(
                C=1.0,
                penalty="l2",
                dual=False,
                class_weight="balanced",
                max_iter=50_000,
                random_state=random_state,
            ),
        ),
    ])

    rbf_svm = GridSearchCV(
        Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]),
        param_grid={
            "clf__C": [0.1, 1.0, 10.0, 100.0],
            "clf__gamma": ["scale", "auto", 0.01, 0.1, 1.0],
        },
        scoring="balanced_accuracy",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
        n_jobs=n_jobs,
        refit=True,
    )

    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=n_jobs,
    )

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size="auto",
                learning_rate="adaptive",
                max_iter=500,
                early_stopping=True,
                random_state=random_state,
            ),
        ),
    ])

    return {
        "Linear SVM": linear_svm,
        "RBF SVM optimized": rbf_svm,
        "Random Forest": random_forest,
        "MLP": mlp,
    }


MODEL_CLI_ALIASES: Mapping[str, str] = {
    "LinearSVC": "Linear SVM",
    "Linear_SVC": "Linear SVM",
    "LinearSVM": "Linear SVM",
    "Linear_SVM": "Linear SVM",
    "Linear SVM": "Linear SVM",
    "RBF_SVM": "RBF SVM optimized",
    "RBF SVM": "RBF SVM optimized",
    "RBF_SVM_optimized": "RBF SVM optimized",
    "RBF SVM optimized": "RBF SVM optimized",
    "RandomForest": "Random Forest",
    "Random_Forest": "Random Forest",
    "Random Forest": "Random Forest",
    "MLP": "MLP",
}

MODEL_CLI_CHOICES: Tuple[str, ...] = (
    "LinearSVC",
    "RBF_SVM",
    "RandomForest",
    "MLP",
)


def resolve_model_cli_name(model_name: str) -> str:
    """Map a compact CLI model name to the existing build_models key."""

    if model_name in MODEL_CLI_ALIASES:
        return MODEL_CLI_ALIASES[model_name]
    allowed = ", ".join(MODEL_CLI_CHOICES)
    raise ValueError(f"Unknown model '{model_name}'. Choose one of: {allowed}.")


def build_model_from_cli_name(
    model_name: str,
    random_state: int = 42,
    n_jobs: int = -1,
    scoring: str = "balanced_accuracy",
    cv_folds: int = 5,
) -> object:
    """Build one of the existing models selected by the compact CLI name."""

    resolved = resolve_model_cli_name(model_name)
    model = build_models(random_state=random_state, n_jobs=n_jobs)[resolved]

    if isinstance(model, GridSearchCV):
        model.scoring = scoring
        model.cv = StratifiedKFold(
            n_splits=max(2, int(cv_folds)),
            shuffle=True,
            random_state=random_state,
        )
    return model


def rfe_importance_getter_for_model(model: object) -> str:
    """Return the sklearn RFE importance getter for supported estimators."""

    if isinstance(model, Pipeline):
        clf = model.named_steps.get("clf")
        if isinstance(clf, LinearSVC):
            return "named_steps.clf.coef_"
        if isinstance(clf, RandomForestClassifier):
            return "named_steps.clf.feature_importances_"
        raise ValueError(
            "RFE ranking requires a model with coefficients or feature importances. "
            "Use --model LinearSVC or --model RandomForest for this command."
        )

    if isinstance(model, RandomForestClassifier):
        return "feature_importances_"

    raise ValueError(
        "RFE ranking requires a model with coefficients or feature importances. "
        "Use --model LinearSVC or --model RandomForest for this command."
    )


def extract_fitted_feature_importance(model: object, n_features: int) -> np.ndarray:
    """Extract non-negative importances from a fitted RFE-compatible model."""

    fitted = EntanglementVisualizer._unwrap_grid_search(model)
    if isinstance(fitted, Pipeline):
        fitted = fitted.named_steps["clf"]

    if hasattr(fitted, "coef_"):
        coef = np.asarray(fitted.coef_, dtype=float)
        if coef.ndim == 1:
            importance = np.abs(coef)
        else:
            importance = np.mean(np.abs(coef), axis=0)
    elif hasattr(fitted, "feature_importances_"):
        importance = np.asarray(fitted.feature_importances_, dtype=float)
    else:
        importance = np.ones(n_features, dtype=float)

    importance = np.asarray(importance, dtype=float).reshape(-1)
    if importance.size != n_features:
        return np.ones(n_features, dtype=float)
    return importance


def fit_rfe_ablation_ranking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    model: object,
    config: EvaluationConfig,
) -> Tuple[List[str], Dict[str, object]]:
    """Fit RFECV/RFE on training data and return features most-to-least important.

    The returned order is intentionally most important first, because the
    ablation curve removes the first k ranked features at each step.
    """

    n_features = int(X_train.shape[1])
    feature_names = list(feature_names)
    if n_features != len(feature_names):
        raise ValueError("Feature-name count does not match X_train columns.")
    if n_features == 1:
        return feature_names, {
            "method": "single_feature",
            "ranking": {feature_names[0]: 1},
            "ordered_features_most_to_least_important": feature_names,
        }

    importance_getter = rfe_importance_getter_for_model(model)
    class_counts = np.unique(y_train, return_counts=True)[1]
    min_class_count = int(np.min(class_counts)) if len(class_counts) else 0

    selector = None
    selector_method = "RFECV"
    if config.use_rfe and min_class_count >= 2:
        try:
            cv = StratifiedKFold(
                n_splits=min(config.rfe_cv, min_class_count),
                shuffle=True,
                random_state=config.random_state,
            )
            selector = RFECV(
                estimator=clone(model),
                step=config.rfe_step,
                min_features_to_select=1,
                cv=cv,
                scoring=config.scoring,
                n_jobs=config.n_jobs,
                importance_getter=importance_getter,
            )
            selector.fit(X_train, y_train)
        except Exception as exc:
            LOGGER.warning("RFECV failed for ablation ranking; falling back to RFE(step=1). Error: %s", exc)
            selector = None

    if selector is None:
        selector_method = "RFE"
        selector = RFE(
            estimator=clone(model),
            n_features_to_select=1,
            step=1,
            importance_getter=importance_getter,
        )
        selector.fit(X_train, y_train)

    fitted_full_model = clone(model)
    fitted_full_model.fit(X_train, y_train)
    importances = extract_fitted_feature_importance(fitted_full_model, n_features)
    ranks = np.asarray(selector.ranking_, dtype=int)
    order_idx = sorted(
        range(n_features),
        key=lambda idx: (int(ranks[idx]), -float(importances[idx]), feature_names[idx]),
    )
    ordered_features = [feature_names[idx] for idx in order_idx]

    info: Dict[str, object] = {
        "method": selector_method,
        "ranking": {name: int(rank) for name, rank in zip(feature_names, ranks)},
        "importance": {name: float(value) for name, value in zip(feature_names, importances)},
        "ordered_features_most_to_least_important": ordered_features,
    }
    if hasattr(selector, "n_features_"):
        info["n_features_selected_by_selector"] = int(selector.n_features_)
    if hasattr(selector, "cv_results_"):
        info["cv_mean_test_score"] = [
            float(x) for x in selector.cv_results_.get("mean_test_score", [])
        ]
        info["cv_std_test_score"] = [
            float(x) for x in selector.cv_results_.get("std_test_score", [])
        ]
    return ordered_features, info


def safe_filename_component(text: str) -> str:
    """Return a filesystem-safe component for model/feature/plot filenames."""

    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unnamed"


class EntanglementVisualizer:
    """Advanced model-interpretability plots for separability classifiers.

    The class works with fitted scikit-learn estimators, Pipelines, and
    GridSearchCV objects. It treats separability (+1) as the positive class for
    ROC, residual, probability, and SHAP plots.
    """

    def __init__(
        self,
        output_dir: str | Path,
        feature_names: Sequence[str],
        random_state: int = 42,
        max_shap_background: int = 50,
        max_shap_samples: int = 100,
        shap_nsamples: int = 100,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = list(feature_names)
        self.random_state = int(random_state)
        self.max_shap_background = int(max_shap_background)
        self.max_shap_samples = int(max_shap_samples)
        self.shap_nsamples = int(shap_nsamples)

    def _path(self, model_name: str, feature_set: str, plot_type: str) -> Path:
        filename = "_".join([
            safe_filename_component(model_name),
            safe_filename_component(feature_set),
            safe_filename_component(plot_type),
        ]) + ".png"
        return self.output_dir / filename

    def _as_frame(self, X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.feature_names]
        return pd.DataFrame(np.asarray(X, dtype=float), columns=self.feature_names)

    @staticmethod
    def _binary_labels(y: np.ndarray) -> np.ndarray:
        """Map {-1,+1} labels to {0,1}, with separable (+1) positive."""

        return (np.asarray(y, dtype=int) == SEPARABLE_LABEL).astype(int)

    @staticmethod
    def _unwrap_grid_search(model: object) -> object:
        if isinstance(model, GridSearchCV) and hasattr(model, "best_estimator_"):
            return model.best_estimator_
        return model

    @classmethod
    def _final_estimator(cls, model: object) -> object:
        base = cls._unwrap_grid_search(model)
        if isinstance(base, Pipeline):
            return base.steps[-1][1]
        return base

    @staticmethod
    def _positive_class_index(model: object) -> int:
        classes = getattr(model, "classes_", None)
        if classes is None and isinstance(model, GridSearchCV) and hasattr(model, "best_estimator_"):
            classes = getattr(model.best_estimator_, "classes_", None)
        if classes is None:
            return -1
        classes = np.asarray(classes)
        matches = np.where(classes == SEPARABLE_LABEL)[0]
        return int(matches[0]) if len(matches) else int(len(classes) - 1)

    def positive_class_score(
        self,
        model: object,
        X: np.ndarray | pd.DataFrame,
        as_probability: bool = False,
    ) -> np.ndarray:
        """Return scores/probabilities for the positive separable class.

        If the estimator lacks probabilities, the decision function is used.
        For residual plots, decision scores are converted to a bounded
        pseudo-probability with a logistic transform. These values are suitable
        for diagnostics, not for calibrated uncertainty statements.
        """

        if hasattr(model, "predict_proba"):
            try:
                proba = np.asarray(model.predict_proba(X), dtype=float)
                if proba.ndim == 2 and proba.shape[1] > 1:
                    return proba[:, self._positive_class_index(model)]
                return proba.reshape(-1)
            except Exception:
                pass

        if hasattr(model, "decision_function"):
            decision = np.asarray(model.decision_function(X), dtype=float)
            if decision.ndim == 2:
                pos_idx = self._positive_class_index(model)
                if 0 <= pos_idx < decision.shape[1]:
                    decision = decision[:, pos_idx]
                else:
                    decision = decision[:, -1]
            decision = decision.reshape(-1)

            classes = getattr(model, "classes_", None)
            if classes is not None:
                classes = np.asarray(classes)
                if len(classes) == 2 and classes[-1] != SEPARABLE_LABEL:
                    decision = -decision

            if as_probability:
                decision = np.clip(decision, -50.0, 50.0)
                return 1.0 / (1.0 + np.exp(-decision))
            return decision

        pred = np.asarray(model.predict(X), dtype=int)
        return (pred == SEPARABLE_LABEL).astype(float)

    def plot_roc_curve(
        self,
        model: object,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        feature_set: str,
    ) -> Optional[Path]:
        """Plot ROC curve and calculate AUC for separability classification."""

        y_binary = self._binary_labels(y_test)
        if len(np.unique(y_binary)) < 2:
            LOGGER.warning("Skipping ROC curve for %s/%s: test split has one class.", model_name, feature_set)
            return None

        scores = self.positive_class_score(model, X_test, as_probability=False)
        fpr, tpr, _ = sk_metrics.roc_curve(y_binary, scores)
        auc_value = sk_metrics.roc_auc_score(y_binary, scores)

        out_path = self._path(model_name, feature_set, "RocCurve")
        plt.figure(figsize=(6.5, 5.0))
        plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Random baseline")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"ROC curve: {model_name} / {feature_set}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path

    def plot_residuals(
        self,
        model: object,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        feature_set: str,
    ) -> Optional[Path]:
        """Plot predicted-probability residuals against predicted values.

        Residual = predicted P(separable) - actual binary label, where
        entangled=-1 maps to 0 and separable=+1 maps to 1.
        """

        y_binary = self._binary_labels(y_test)
        predicted_probability = self.positive_class_score(model, X_test, as_probability=True)
        residual = predicted_probability - y_binary

        plot_df = pd.DataFrame({
            "predicted_probability": predicted_probability,
            "residual": residual,
            "actual_label": np.where(y_binary == 1, "separable (+1)", "entangled (-1)"),
        })

        out_path = self._path(model_name, feature_set, "ResidualPlot")
        plt.figure(figsize=(6.5, 5.0))
        sns.scatterplot(
            data=plot_df,
            x="predicted_probability",
            y="residual",
            hue="actual_label",
            alpha=0.75,
            edgecolor=None,
        )
        plt.axhline(0.0, linestyle="--", linewidth=1.0)
        plt.xlabel("Predicted P(separable)")
        plt.ylabel("Residual: predicted probability - actual label")
        plt.title(f"Probability residuals: {model_name} / {feature_set}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path

    def _sample_frame(self, X: np.ndarray, max_rows: int) -> pd.DataFrame:
        frame = self._as_frame(X)
        if len(frame) <= max_rows:
            return frame
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(frame), size=max_rows, replace=False)
        return frame.iloc[np.sort(idx)].reset_index(drop=True)

    @staticmethod
    def _is_tree_model(model: object) -> bool:
        return isinstance(EntanglementVisualizer._final_estimator(model), RandomForestClassifier)

    @staticmethod
    def _is_svm_model(model: object) -> bool:
        return isinstance(EntanglementVisualizer._final_estimator(model), (SVC, LinearSVC))

    def _select_positive_shap_values(self, shap_values: object, model: object) -> np.ndarray:
        """Convert SHAP output to an (n_samples, n_features) matrix."""

        if isinstance(shap_values, list):
            pos_idx = self._positive_class_index(model)
            pos_idx = pos_idx if 0 <= pos_idx < len(shap_values) else -1
            return np.asarray(shap_values[pos_idx], dtype=float)

        values = getattr(shap_values, "values", shap_values)
        values = np.asarray(values, dtype=float)
        if values.ndim == 3:
            pos_idx = self._positive_class_index(model)
            pos_idx = pos_idx if 0 <= pos_idx < values.shape[-1] else -1
            values = values[:, :, pos_idx]
        return values

    def plot_shap_summary(
        self,
        model: object,
        X_train: np.ndarray,
        X_test: np.ndarray,
        model_name: str,
        feature_set: str,
    ) -> Optional[Path]:
        """Generate and save a SHAP summary plot for the fitted model.

        Random forests use SHAP's tree-aware Explainer. SVMs and other
        Pipeline/GridSearchCV estimators are treated as black boxes using their
        separable-class probability or decision score.
        """

        if X_train.shape[1] == 0 or X_test.shape[1] == 0:
            LOGGER.warning("Skipping SHAP for %s/%s: no features.", model_name, feature_set)
            return None

        background = self._sample_frame(X_train, max(1, self.max_shap_background))
        explain = self._sample_frame(X_test, max(1, self.max_shap_samples))
        out_path = self._path(model_name, feature_set, "SHAPPlot")

        try:
            if self._is_tree_model(model):
                # For RandomForestClassifier, shap.Explainer dispatches to the
                # appropriate tree-based algorithm and preserves feature names.
                tree_model = self._final_estimator(model)
                explainer = shap.Explainer(tree_model, background)
                raw_shap_values = explainer(explain)
                shap_values = self._select_positive_shap_values(raw_shap_values, model)
            elif self._is_svm_model(model):
                # SVMs after scaling are black-box functions for SHAP. The
                # KernelExplainer path is slower but correct for kernel SVMs and
                # LinearSVC pipelines that lack native probabilities.
                def predict_fn(data: np.ndarray) -> np.ndarray:
                    return self.positive_class_score(model, np.asarray(data, dtype=float), as_probability=True)

                explainer = shap.KernelExplainer(predict_fn, background.to_numpy(dtype=float))
                raw_shap_values = explainer.shap_values(
                    explain.to_numpy(dtype=float),
                    nsamples=self.shap_nsamples,
                    silent=True,
                )
                shap_values = self._select_positive_shap_values(raw_shap_values, model)
            else:
                # Model-agnostic fallback using shap.Explainer. This covers MLPs
                # and any future estimator with predict_proba/decision_function.
                def predict_fn(data: np.ndarray) -> np.ndarray:
                    return self.positive_class_score(model, np.asarray(data, dtype=float), as_probability=True)

                masker = shap.maskers.Independent(background)
                explainer = shap.Explainer(predict_fn, masker, algorithm="permutation")
                max_evals = max(self.shap_nsamples, 2 * len(self.feature_names) + 1)
                raw_shap_values = explainer(explain, max_evals=max_evals, silent=True)
                shap_values = self._select_positive_shap_values(raw_shap_values, model)

            plt.figure(figsize=(8.0, 6.0))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                shap.summary_plot(
                    shap_values,
                    explain,
                    feature_names=self.feature_names,
                    show=False,
                    max_display=min(25, len(self.feature_names)),
                )
            plt.title(f"SHAP summary: {model_name} / {feature_set}")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            return out_path
        except Exception as exc:
            plt.close()
            LOGGER.warning("Skipping SHAP plot for %s/%s: %s", model_name, feature_set, exc)
            return None

    def plot_margin_distribution(
        self,
        model: object,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        feature_set: str,
    ) -> Optional[Path]:
        """Plot signed SVM decision-function margins for near-miss analysis."""

        if not hasattr(model, "decision_function"):
            LOGGER.warning(
                "Skipping margin distribution for %s/%s: estimator has no decision_function.",
                model_name,
                feature_set,
            )
            return None

        margins = np.asarray(model.decision_function(X_test), dtype=float)
        if margins.ndim == 2:
            margins = margins[:, self._positive_class_index(model)]
        margins = margins.reshape(-1)

        classes = getattr(model, "classes_", None)
        if classes is not None:
            classes = np.asarray(classes)
            if len(classes) == 2 and classes[-1] != SEPARABLE_LABEL:
                margins = -margins

        plot_df = pd.DataFrame({
            "signed_margin": margins,
            "actual_label": np.where(np.asarray(y_test) == SEPARABLE_LABEL, "separable (+1)", "entangled (-1)"),
        })

        out_path = self._path(model_name, feature_set, "MarginDistributionPlot")
        plt.figure(figsize=(6.5, 5.0))
        sns.histplot(
            data=plot_df,
            x="signed_margin",
            hue="actual_label",
            bins=40,
            kde=True,
            element="step",
            stat="density",
            common_norm=False,
        )
        plt.yscale("log")   # ← add this line
        plt.axvline(0.0, linestyle="--", linewidth=1.0)
        plt.xlabel("Signed distance / decision-function value")
        plt.ylabel("Density")
        plt.title(f"SVM margin distribution: {model_name} / {feature_set}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path


def run_rfecv_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    config: EvaluationConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Run RFECV with a linear SVM and return the selected feature mask."""

    if X_train.shape[1] == 1:
        return np.array([True]), {
            "n_features_selected": 1,
            "selected_features": list(feature_names),
            "ranking": {feature_names[0]: 1},
            "support": {feature_names[0]: True},
        }

    class_counts = np.unique(y_train, return_counts=True)[1]
    min_class_count = int(np.min(class_counts)) if len(class_counts) else 0
    if min_class_count < 2:
        # RFECV cannot run with fewer than two samples per class in the
        # training split. Keep all features and record the reason.
        return np.ones(X_train.shape[1], dtype=bool), {
            "n_features_selected": int(X_train.shape[1]),
            "selected_features": list(feature_names),
            "ranking": {name: 1 for name in feature_names},
            "support": {name: True for name in feature_names},
            "rfe_skipped_reason": "RFECV requires at least two training samples per class.",
        }

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    estimator = LinearSVC(
        C=1.0,
        penalty="l2",
        dual=False,
        class_weight="balanced",
        max_iter=50_000,
        random_state=config.random_state,
    )
    cv = StratifiedKFold(
        n_splits=min(config.rfe_cv, min_class_count),
        shuffle=True,
        random_state=config.random_state,
    )
    selector = RFECV(
        estimator=estimator,
        step=config.rfe_step,
        min_features_to_select=1,
        cv=cv,
        scoring=config.scoring,
        n_jobs=config.n_jobs,
    )
    selector.fit(X_train_scaled, y_train)

    support = np.asarray(selector.support_, dtype=bool)
    selected = [name for name, keep in zip(feature_names, support) if keep]
    ranking = {name: int(rank) for name, rank in zip(feature_names, selector.ranking_)}
    support_map = {name: bool(keep) for name, keep in zip(feature_names, support)}

    info = {
        "n_features_selected": int(selector.n_features_),
        "selected_features": selected,
        "ranking": ranking,
        "support": support_map,
    }

    if hasattr(selector, "cv_results_"):
        # Newer scikit-learn exposes a result dictionary; keep serializable fields.
        info["cv_mean_test_score"] = [float(x) for x in selector.cv_results_.get("mean_test_score", [])]
        info["cv_std_test_score"] = [float(x) for x in selector.cv_results_.get("std_test_score", [])]

    return support, info


def evaluate_feature_scenarios(
    df: pd.DataFrame,
    out_dir: Optional[str | Path] = None,
    config: EvaluationConfig = EvaluationConfig(),
    scenarios: Mapping[str, Tuple[str, ...]] = DEFAULT_SCENARIOS,
    plot_types: Optional[Sequence[str]] = None,
    plot_output_dir: Optional[str | Path] = None,
    max_shap_background: int = 50,
    max_shap_samples: int = 100,
    shap_nsamples: int = 100,
) -> Dict[str, object]:
    """Run RFE, train models, and optionally generate performance plots."""

    if "y" not in df.columns:
        raise ValueError("Dataset must contain a y column.")

    requested_plots = list(plot_types or [])
    invalid_plots = sorted(set(requested_plots) - set(PLOT_CHOICES))
    if invalid_plots:
        raise ValueError(f"Unknown plot type(s): {invalid_plots}. Choose from {list(PLOT_CHOICES)}.")

    y = df["y"].to_numpy(dtype=int)
    row_indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        row_indices,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    results: Dict[str, object] = {
        "config": asdict(config),
        "labels": {"entangled": ENTANGLED_LABEL, "separable": SEPARABLE_LABEL},
        "scenarios": {},
    }

    for scenario_name, groups in scenarios.items():
        LOGGER.info("Evaluating feature scenario: %s", scenario_name)
        cols = columns_for_scenario(df, groups)
        X = df[cols].to_numpy(dtype=float)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if config.use_rfe:
            mask, rfe_info = run_rfecv_selection(X_train, y_train, cols, config)
            selected_cols = [c for c, keep in zip(cols, mask) if keep]
            X_train_eval = X_train[:, mask]
            X_test_eval = X_test[:, mask]
        else:
            rfe_info = {
                "n_features_selected": len(cols),
                "selected_features": list(cols),
                "ranking": {c: 1 for c in cols},
                "support": {c: True for c in cols},
            }
            selected_cols = cols
            X_train_eval = X_train
            X_test_eval = X_test

        models = build_models(random_state=config.random_state, n_jobs=config.n_jobs)
        scenario_result: Dict[str, object] = {
            "groups": list(groups),
            "all_features": list(cols),
            "rfe": rfe_info,
            "plots_requested": requested_plots,
            "models": {},
        }

        visualizer: Optional[EntanglementVisualizer] = None
        if requested_plots:
            effective_plot_dir = plot_output_dir or out_dir or "./plots"
            visualizer = EntanglementVisualizer(
                output_dir=effective_plot_dir,
                feature_names=selected_cols,
                random_state=config.random_state,
                max_shap_background=max_shap_background,
                max_shap_samples=max_shap_samples,
                shap_nsamples=shap_nsamples,
            )

        for model_name, model in models.items():
            LOGGER.info("  fitting %s on %d features", model_name, len(selected_cols))
            fitted = clone(model)

            # GridSearchCV with StratifiedKFold needs enough training examples
            # in every class. For tiny smoke tests, fall back to a single RBF
            # SVM fit rather than failing. For normal experiments this branch
            # is not used.
            if isinstance(fitted, GridSearchCV):
                class_counts = np.unique(y_train, return_counts=True)[1]
                min_class_count = int(np.min(class_counts)) if len(class_counts) else 0
                if min_class_count < 2:
                    fitted = Pipeline([
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            SVC(
                                kernel="rbf",
                                C=1.0,
                                gamma="scale",
                                class_weight="balanced",
                                probability=True,
                                random_state=config.random_state,
                            ),
                        ),
                    ])
                else:
                    fitted.cv = StratifiedKFold(
                        n_splits=min(3, min_class_count),
                        shuffle=True,
                        random_state=config.random_state,
                    )

            if model_name == "MLP":
                class_counts = np.unique(y_train, return_counts=True)[1]
                min_class_count = int(np.min(class_counts)) if len(class_counts) else 0
                if min_class_count < 3 or len(y_train) < 20:
                    # Early stopping uses a stratified validation split inside
                    # MLPClassifier; disable it for tiny smoke-test datasets.
                    try:
                        fitted.set_params(clf__early_stopping=False)
                    except ValueError:
                        pass

            fitted.fit(X_train_eval, y_train)
            y_pred = fitted.predict(X_test_eval)

            report_dict = classification_report(
                y_test,
                y_pred,
                labels=[ENTANGLED_LABEL, SEPARABLE_LABEL],
                target_names=["entangled (-1)", "separable (+1)"],
                output_dict=True,
                zero_division=0,
            )
            report_text = classification_report(
                y_test,
                y_pred,
                labels=[ENTANGLED_LABEL, SEPARABLE_LABEL],
                target_names=["entangled (-1)", "separable (+1)"],
                digits=4,
                zero_division=0,
            )
            cm = confusion_matrix(y_test, y_pred, labels=[ENTANGLED_LABEL, SEPARABLE_LABEL])

            model_result: Dict[str, object] = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "confusion_matrix_labels": [ENTANGLED_LABEL, SEPARABLE_LABEL],
                "confusion_matrix": cm.astype(int).tolist(),
                "classification_report": report_dict,
                "classification_report_text": report_text,
            }
            if isinstance(fitted, GridSearchCV):
                model_result["best_params"] = fitted.best_params_
                model_result["best_cv_score"] = float(fitted.best_score_)

            if visualizer is not None:
                plot_paths: Dict[str, Optional[str]] = {}
                for plot_type in requested_plots:
                    plot_path: Optional[Path]
                    if plot_type == "RocCurve":
                        plot_path = visualizer.plot_roc_curve(
                            fitted, X_test_eval, y_test, model_name, scenario_name
                        )
                    elif plot_type == "ResidualPlot":
                        plot_path = visualizer.plot_residuals(
                            fitted, X_test_eval, y_test, model_name, scenario_name
                        )
                    elif plot_type == "SHAPPlot":
                        plot_path = visualizer.plot_shap_summary(
                            fitted, X_train_eval, X_test_eval, model_name, scenario_name
                        )
                    elif plot_type == "MarginDistributionPlot":
                        plot_path = visualizer.plot_margin_distribution(
                            fitted, X_test_eval, y_test, model_name, scenario_name
                        )
                    else:
                        plot_path = None
                    plot_paths[plot_type] = str(plot_path) if plot_path is not None else None
                model_result["plots"] = plot_paths

            scenario_result["models"][model_name] = model_result

        results["scenarios"][scenario_name] = scenario_result

    if out_dir is not None:
        write_evaluation_results(results, out_dir)

    return results


def evaluate_feature_scenarios_fixed_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Optional[str | Path] = None,
    config: EvaluationConfig = EvaluationConfig(),
    scenarios: Mapping[str, Tuple[str, ...]] = DEFAULT_SCENARIOS,
) -> Dict[str, object]:
    """Evaluate models using explicit train and unrestricted test dataframes.

    This is used by the purity-constrained experiment so the training set can
    be filtered by purity while the held-out test set remains sampled from the
    unrestricted data-generating distribution.
    """

    if "y" not in train_df.columns or "y" not in test_df.columns:
        raise ValueError("Both train_df and test_df must contain a y column.")

    y_train = train_df["y"].to_numpy(dtype=int)
    y_test = test_df["y"].to_numpy(dtype=int)

    results: Dict[str, object] = {
        "config": asdict(config),
        "split": {
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
            "train_class_counts": {str(k): int(v) for k, v in train_df["y"].value_counts().to_dict().items()},
            "test_class_counts": {str(k): int(v) for k, v in test_df["y"].value_counts().to_dict().items()},
        },
        "labels": {"entangled": ENTANGLED_LABEL, "separable": SEPARABLE_LABEL},
        "scenarios": {},
    }

    for scenario_name, groups in scenarios.items():
        LOGGER.info("Evaluating fixed-split feature scenario: %s", scenario_name)
        cols = columns_for_scenario(train_df, groups)
        missing = sorted(set(cols) - set(test_df.columns))
        if missing:
            raise ValueError(f"Test set is missing feature columns required by {scenario_name}: {missing}")

        X_train = train_df[cols].to_numpy(dtype=float)
        X_test = test_df[cols].to_numpy(dtype=float)

        if config.use_rfe:
            mask, rfe_info = run_rfecv_selection(X_train, y_train, cols, config)
            selected_cols = [c for c, keep in zip(cols, mask) if keep]
            X_train_eval = X_train[:, mask]
            X_test_eval = X_test[:, mask]
        else:
            rfe_info = {
                "n_features_selected": len(cols),
                "selected_features": list(cols),
                "ranking": {c: 1 for c in cols},
                "support": {c: True for c in cols},
            }
            selected_cols = cols
            X_train_eval = X_train
            X_test_eval = X_test

        models = build_models(random_state=config.random_state, n_jobs=config.n_jobs)
        scenario_result: Dict[str, object] = {
            "groups": list(groups),
            "all_features": list(cols),
            "rfe": rfe_info,
            "models": {},
        }

        for model_name, model in models.items():
            LOGGER.info("  fitting %s on %d features", model_name, len(selected_cols))
            fitted = clone(model)

            if isinstance(fitted, GridSearchCV):
                class_counts = np.unique(y_train, return_counts=True)[1]
                min_class_count = int(np.min(class_counts)) if len(class_counts) else 0
                if min_class_count < 2:
                    fitted = Pipeline([
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            SVC(
                                kernel="rbf",
                                C=1.0,
                                gamma="scale",
                                class_weight="balanced",
                                probability=True,
                                random_state=config.random_state,
                            ),
                        ),
                    ])
                else:
                    fitted.scoring = config.scoring
                    fitted.cv = StratifiedKFold(
                        n_splits=min(3, min_class_count),
                        shuffle=True,
                        random_state=config.random_state,
                    )

            if model_name == "MLP":
                class_counts = np.unique(y_train, return_counts=True)[1]
                min_class_count = int(np.min(class_counts)) if len(class_counts) else 0
                if min_class_count < 3 or len(y_train) < 20:
                    try:
                        fitted.set_params(clf__early_stopping=False)
                    except ValueError:
                        pass

            fitted.fit(X_train_eval, y_train)
            y_pred = fitted.predict(X_test_eval)

            report_dict = classification_report(
                y_test,
                y_pred,
                labels=[ENTANGLED_LABEL, SEPARABLE_LABEL],
                target_names=["entangled (-1)", "separable (+1)"],
                output_dict=True,
                zero_division=0,
            )
            report_text = classification_report(
                y_test,
                y_pred,
                labels=[ENTANGLED_LABEL, SEPARABLE_LABEL],
                target_names=["entangled (-1)", "separable (+1)"],
                digits=4,
                zero_division=0,
            )
            cm = confusion_matrix(y_test, y_pred, labels=[ENTANGLED_LABEL, SEPARABLE_LABEL])

            model_result: Dict[str, object] = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "confusion_matrix_labels": [ENTANGLED_LABEL, SEPARABLE_LABEL],
                "confusion_matrix": cm.astype(int).tolist(),
                "classification_report": report_dict,
                "classification_report_text": report_text,
            }
            if isinstance(fitted, GridSearchCV):
                model_result["best_params"] = fitted.best_params_
                model_result["best_cv_score"] = float(fitted.best_score_)

            scenario_result["models"][model_name] = model_result

        results["scenarios"][scenario_name] = scenario_result

    if out_dir is not None:
        write_evaluation_results(results, out_dir)

    return results


def write_evaluation_results(results: Mapping[str, object], out_dir: str | Path) -> None:
    """Write JSON and human-readable text reports."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "classification_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines: List[str] = []
    scenarios = results.get("scenarios", {})
    if isinstance(scenarios, Mapping):
        for scenario_name, scenario_result in scenarios.items():
            lines.append("=" * 88)
            lines.append(f"Feature scenario: {scenario_name}")
            rfe = scenario_result.get("rfe", {}) if isinstance(scenario_result, Mapping) else {}
            lines.append(f"RFE selected features: {rfe.get('n_features_selected')}")
            selected = rfe.get("selected_features", [])
            if isinstance(selected, list):
                lines.append("Selected feature names: " + ", ".join(map(str, selected)))
            lines.append("")

            models = scenario_result.get("models", {}) if isinstance(scenario_result, Mapping) else {}
            if isinstance(models, Mapping):
                for model_name, model_result in models.items():
                    lines.append("-" * 88)
                    lines.append(f"Model: {model_name}")
                    lines.append(f"Accuracy: {model_result.get('accuracy'):.6f}")
                    lines.append(f"Balanced accuracy: {model_result.get('balanced_accuracy'):.6f}")
                    if "best_params" in model_result:
                        lines.append(f"Best params: {model_result['best_params']}")
                    lines.append("Confusion matrix labels: [-1 entangled, +1 separable]")
                    lines.append(str(model_result.get("confusion_matrix")))
                    lines.append(str(model_result.get("classification_report_text")))
                    lines.append("")

    text_path = out / "classification_reports.txt"
    text_path.write_text("\n".join(lines), encoding="utf-8")

    # Store RFE rankings per scenario as CSV for easy inspection.
    rankings_path = out / "rfe_rankings.csv"
    with rankings_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "feature", "selected", "rank"])
        if isinstance(scenarios, Mapping):
            for scenario_name, scenario_result in scenarios.items():
                if not isinstance(scenario_result, Mapping):
                    continue
                rfe = scenario_result.get("rfe", {})
                if not isinstance(rfe, Mapping):
                    continue
                ranking = rfe.get("ranking", {})
                support = rfe.get("support", {})
                if isinstance(ranking, Mapping):
                    for feature, rank in ranking.items():
                        writer.writerow([scenario_name, feature, bool(support.get(feature, False)), rank])


# ---------------------------------------------------------------------------
# Purity-constrained training experiment
# ---------------------------------------------------------------------------

def balanced_class_counts(total_size: int) -> Tuple[int, int]:
    """Split a total dataset size into entangled/separable class counts."""

    total = int(total_size)
    if total < 2:
        raise ValueError("Dataset sizes must be at least 2 so both classes are represented.")
    n_entangled = total // 2
    n_separable = total - n_entangled
    return n_entangled, n_separable


def offset_random_state(random_state: Optional[int], offset: int) -> Optional[int]:
    """Derive deterministic, distinct seeds for related generated datasets."""

    if random_state is None:
        return None
    return int(random_state) + int(offset)


def eta_path_component(eta: float) -> str:
    """Filesystem-safe path component for an eta value."""

    return "eta_" + safe_filename_component(f"{float(eta):.8g}".replace(".", "p").replace("-", "m"))


def save_purity_distribution(
    df: pd.DataFrame,
    out_dir: str | Path,
    name: str,
    total_dim: int,
    eta: Optional[float] = None,
) -> Dict[str, str]:
    """Save purity values and a histogram for a generated train/test set."""

    if "purity" not in df.columns:
        return {}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_name = safe_filename_component(name)

    dist = pd.DataFrame({
        "purity": df["purity"].to_numpy(dtype=float),
        "y": df["y"].to_numpy(dtype=int),
    })
    dist["label"] = np.where(dist["y"] == ENTANGLED_LABEL, "entangled (-1)", "separable (+1)")
    if eta is not None:
        dist["purity_regime"] = [
            purity_regime(p, total_dim=total_dim, eta=eta)
            for p in dist["purity"]
        ]
    else:
        dist["purity_regime"] = "not_filtered"

    csv_path = out / f"{safe_name}_purity_distribution.csv"
    dist.to_csv(csv_path, index=False)

    plot_path = out / f"{safe_name}_purity_distribution.png"
    plt.figure(figsize=(7.0, 5.0))
    sns.histplot(
        data=dist,
        x="purity",
        hue="label",
        bins=40,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.xlabel(r"Purity $\mathrm{Tr}(\rho^2)$")
    plt.ylabel("Density")
    title = f"Purity distribution: {name}"
    if eta is not None:
        bounds = purity_window_bounds(total_dim, eta)
        for bound in bounds["low"] + bounds["high"]:
            plt.axvline(bound, linestyle="--", linewidth=1.0)
        title += f" (eta={eta:g})"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {"csv": str(csv_path), "plot": str(plot_path)}


def collect_performance_rows(
    results: Mapping[str, object],
    training_condition: str,
    eta: float,
) -> List[Dict[str, object]]:
    """Flatten nested evaluation results into one row per scenario/model."""

    rows: List[Dict[str, object]] = []
    log_eta = float(np.log10(float(eta)))
    scenarios = results.get("scenarios", {})
    if not isinstance(scenarios, Mapping):
        return rows

    for scenario_name, scenario_result in scenarios.items():
        if not isinstance(scenario_result, Mapping):
            continue
        models = scenario_result.get("models", {})
        if not isinstance(models, Mapping):
            continue
        for model_name, model_result in models.items():
            if not isinstance(model_result, Mapping):
                continue
            rows.append({
                "training_condition": training_condition,
                "eta": float(eta),
                "log10_eta": log_eta,
                "scenario": str(scenario_name),
                "model": str(model_name),
                "accuracy": float(model_result.get("accuracy", np.nan)),
                "balanced_accuracy": float(model_result.get("balanced_accuracy", np.nan)),
                "confusion_matrix": json.dumps(model_result.get("confusion_matrix")),
            })
    return rows


def plot_performance_vs_eta(performance_df: pd.DataFrame, out_dir: str | Path) -> List[str]:
    """Plot accuracy against log10(eta), with one PNG per model."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if performance_df.empty:
        return []

    plot_paths: List[str] = []
    for model_name in sorted(performance_df["model"].dropna().unique()):
        model_df = performance_df[performance_df["model"] == model_name].copy()
        if model_df.empty:
            continue

        plt.figure(figsize=(8.0, 5.5))
        for (condition, scenario), group in model_df.groupby(["training_condition", "scenario"]):
            group = group.sort_values("eta")
            linestyle = "--" if condition == "baseline" else "-"
            plt.plot(
                group["log10_eta"].to_numpy(dtype=float),
                group["accuracy"].to_numpy(dtype=float),
                marker="o",
                linestyle=linestyle,
                label=f"{condition}: {scenario}",
            )

        plt.xlabel(r"$\log_{10}(\eta)$")
        plt.ylabel("Accuracy")
        plt.ylim(0.0, 1.02)
        plt.title(f"Accuracy vs log10(eta): {model_name}")
        plt.legend(fontsize=7, loc="best")
        plt.tight_layout()

        plot_path = out / f"performance_vs_eta_{safe_filename_component(model_name)}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths.append(str(plot_path))

    return plot_paths


def write_purity_comparison_report(performance_df: pd.DataFrame, out_path: str | Path) -> None:
    """Write a compact baseline-vs-purity text comparison."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("Purity-constrained training comparison")
    lines.append("Baseline trains on an unrestricted distribution; constrained trains only on extreme-purity states.")
    lines.append("The test set is the same unrestricted dataset for both conditions.")
    lines.append("Targeted sampling is a controlled extreme-purity distribution, not a strict metric-conditioned sample.")
    lines.append("")

    constrained = performance_df[performance_df["training_condition"] == "purity_constrained"]
    baseline = performance_df[performance_df["training_condition"] == "baseline"]
    for eta in sorted(constrained["eta"].unique()):
        lines.append("=" * 88)
        lines.append(f"eta = {eta:g}  (log10 eta = {np.log10(float(eta)):.6f})")
        eta_constrained = constrained[constrained["eta"] == eta]
        eta_baseline = baseline[baseline["eta"] == eta]
        for scenario in sorted(eta_constrained["scenario"].unique()):
            lines.append(f"Feature scenario: {scenario}")
            for model_name in sorted(eta_constrained[eta_constrained["scenario"] == scenario]["model"].unique()):
                c_row = eta_constrained[
                    (eta_constrained["scenario"] == scenario)
                    & (eta_constrained["model"] == model_name)
                ].iloc[0]
                b_match = eta_baseline[
                    (eta_baseline["scenario"] == scenario)
                    & (eta_baseline["model"] == model_name)
                ]
                if b_match.empty:
                    continue
                b_row = b_match.iloc[0]
                delta_acc = float(c_row["accuracy"]) - float(b_row["accuracy"])
                delta_bal = float(c_row["balanced_accuracy"]) - float(b_row["balanced_accuracy"])
                lines.append(
                    f"  {model_name}: baseline acc={float(b_row['accuracy']):.6f}, "
                    f"constrained acc={float(c_row['accuracy']):.6f}, "
                    f"delta={delta_acc:+.6f}; "
                    f"baseline bal_acc={float(b_row['balanced_accuracy']):.6f}, "
                    f"constrained bal_acc={float(c_row['balanced_accuracy']):.6f}, "
                    f"delta={delta_bal:+.6f}"
                )
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_purity_experiment(
    system: str,
    metric: str,
    eta_values: Sequence[float],
    n_train: int,
    n_test: int,
    out_dir: str | Path,
    sep_mixture_terms: int = 1,
    max_draws: int = 10_000_000,
    ppt_tol: float = 1e-10,
    qutrit_script: Optional[str] = None,
    reject_ppt_qutrit_without_script: bool = True,
    random_state: int = 42,
    eval_config: EvaluationConfig = EvaluationConfig(),
    scenarios: Mapping[str, Tuple[str, ...]] = DEFAULT_SCENARIOS,
    write_npz: bool = True,
    purity_sampling_mode: str = "targeted",
    generation_method: int = 1,
) -> Dict[str, object]:
    """Run baseline and purity-constrained training with an unrestricted test set."""

    if not eta_values:
        raise ValueError("At least one eta value is required.")
    eta_values = [validate_eta(e) for e in eta_values]
    if any(e <= 0.0 for e in eta_values):
        raise ValueError("purity_experiment requires eta > 0 for the log10(eta) performance plot.")
    canonical_purity_sampling_mode(purity_sampling_mode)

    dims = dims_from_system(system)
    total_dim = dims[0] * dims[1]
    n_train_entangled, n_train_separable = balanced_class_counts(n_train)
    n_test_entangled, n_test_separable = balanced_class_counts(n_test)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    datasets_dir = out / "datasets"
    purity_dir = out / "purity_distributions"
    plots_dir = out / "performance_vs_eta"

    def make_dataset_config(
        *,
        n_entangled: int,
        n_separable: int,
        purity_filter_enabled: bool,
        eta: float,
        seed_offset: int,
    ) -> DatasetConfig:
        return DatasetConfig(
            system=system,
            metric=metric,
            generation_method=generation_method,
            n_entangled=n_entangled,
            n_separable=n_separable,
            sep_mixture_terms=sep_mixture_terms,
            max_draws=max_draws,
            ppt_tol=ppt_tol,
            qutrit_script=qutrit_script,
            reject_ppt_qutrit_without_script=reject_ppt_qutrit_without_script,
            purity_filter=purity_filter_enabled,
            eta=eta,
            purity_sampling_mode=purity_sampling_mode,
            store_purity=True,
            random_state=offset_random_state(random_state, seed_offset),
        )

    LOGGER.info("Generating unrestricted baseline training set.")
    baseline_train_cfg = make_dataset_config(
        n_entangled=n_train_entangled,
        n_separable=n_train_separable,
        purity_filter_enabled=False,
        eta=eta_values[0],
        seed_offset=101,
    )
    baseline_train_df = generate_dataset(baseline_train_cfg)
    save_dataset_bundle(baseline_train_df, datasets_dir / "baseline_train.csv", baseline_train_cfg, write_npz=write_npz)

    LOGGER.info("Generating unrestricted test set shared by all experiment conditions.")
    test_cfg = make_dataset_config(
        n_entangled=n_test_entangled,
        n_separable=n_test_separable,
        purity_filter_enabled=False,
        eta=eta_values[0],
        seed_offset=202,
    )
    test_df = generate_dataset(test_cfg)
    save_dataset_bundle(test_df, datasets_dir / "test_unrestricted.csv", test_cfg, write_npz=write_npz)

    purity_artifacts: Dict[str, object] = {
        "baseline_train": save_purity_distribution(
            baseline_train_df, purity_dir, "baseline_train_unrestricted", total_dim=total_dim
        ),
        "test_unrestricted": save_purity_distribution(
            test_df, purity_dir, "test_unrestricted", total_dim=total_dim
        ),
    }

    LOGGER.info("Evaluating unrestricted baseline training condition.")
    baseline_out = out / "baseline"
    baseline_results = evaluate_feature_scenarios_fixed_split(
        baseline_train_df,
        test_df,
        out_dir=baseline_out,
        config=eval_config,
        scenarios=scenarios,
    )

    performance_rows: List[Dict[str, object]] = []
    constrained_result_paths: Dict[str, str] = {}
    constrained_dataset_paths: Dict[str, str] = {}

    for eta_index, eta in enumerate(eta_values):
        eta_component = eta_path_component(eta)
        eta_dataset_dir = datasets_dir / eta_component
        eta_out = out / eta_component
        eta_purity_dir = purity_dir / eta_component

        performance_rows.extend(collect_performance_rows(baseline_results, "baseline", eta))

        LOGGER.info("Generating purity-constrained training set for eta=%g.", eta)
        constrained_train_cfg = make_dataset_config(
            n_entangled=n_train_entangled,
            n_separable=n_train_separable,
            purity_filter_enabled=True,
            eta=eta,
            seed_offset=1000 + eta_index,
        )
        constrained_train_df = generate_dataset(constrained_train_cfg)
        constrained_train_path = eta_dataset_dir / "purity_constrained_train.csv"
        save_dataset_bundle(constrained_train_df, constrained_train_path, constrained_train_cfg, write_npz=write_npz)
        constrained_dataset_paths[str(eta)] = str(constrained_train_path)

        purity_artifacts[str(eta)] = {
            "purity_constrained_train": save_purity_distribution(
                constrained_train_df,
                eta_purity_dir,
                "purity_constrained_train",
                total_dim=total_dim,
                eta=eta,
            ),
            "test_unrestricted_with_eta_windows": save_purity_distribution(
                test_df,
                eta_purity_dir,
                "test_unrestricted",
                total_dim=total_dim,
                eta=eta,
            ),
        }

        LOGGER.info("Evaluating purity-constrained training condition for eta=%g.", eta)
        constrained_results = evaluate_feature_scenarios_fixed_split(
            constrained_train_df,
            test_df,
            out_dir=eta_out / "purity_constrained",
            config=eval_config,
            scenarios=scenarios,
        )
        constrained_result_paths[str(eta)] = str(eta_out / "purity_constrained" / "classification_results.json")
        performance_rows.extend(collect_performance_rows(constrained_results, "purity_constrained", eta))

    performance_df = pd.DataFrame(performance_rows)
    performance_csv = out / "performance_summary.csv"
    performance_df.to_csv(performance_csv, index=False)

    comparison_report = out / "comparison_summary.txt"
    write_purity_comparison_report(performance_df, comparison_report)
    performance_plot_paths = plot_performance_vs_eta(performance_df, plots_dir)

    summary: Dict[str, object] = {
        "experiment": "purity_constrained_training",
        "system": system,
        "metric": canonical_metric(metric),
        "total_dimension": total_dim,
        "eta_values": [float(e) for e in eta_values],
        "n_train_total": int(n_train),
        "n_test_total": int(n_test),
        "class_counts": {
            "train": {"entangled": n_train_entangled, "separable": n_train_separable},
            "test": {"entangled": n_test_entangled, "separable": n_test_separable},
        },
        "baseline_results": str(baseline_out / "classification_results.json"),
        "constrained_results": constrained_result_paths,
        "datasets": {
            "baseline_train": str(datasets_dir / "baseline_train.csv"),
            "test_unrestricted": str(datasets_dir / "test_unrestricted.csv"),
            "purity_constrained_train": constrained_dataset_paths,
        },
        "purity_artifacts": purity_artifacts,
        "performance_summary_csv": str(performance_csv),
        "comparison_report": str(comparison_report),
        "performance_plots": performance_plot_paths,
        "evaluation_config": asdict(eval_config),
        "purity_sampling_mode": purity_sampling_mode,
        "sampling_caveat": (
            "targeted mode is a controlled extreme-purity distribution, "
            "not a strict Bures/Hilbert-Schmidt-conditioned sample"
        ),
    }

    summary_path = out / "purity_experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


# ---------------------------------------------------------------------------
# RFE ablation visualization
# ---------------------------------------------------------------------------

def output_path_for_feature_set(path: str | Path, feature_set: str, multiple_feature_sets: bool) -> Path:
    """Append the feature-set name to an output path when several plots are requested."""

    out_path = Path(path)
    if not multiple_feature_sets:
        return out_path
    return out_path.with_name(
        f"{out_path.stem}_{safe_filename_component(feature_set)}{out_path.suffix}"
    )


def scoring_axis_label(scoring: str) -> str:
    """Human-readable y-axis label for a sklearn scoring name."""

    if scoring == "balanced_accuracy":
        return "Balanced accuracy"
    if scoring == "accuracy":
        return "Accuracy"
    return scoring.replace("_", " ").capitalize()


def configure_ablation_fit_model(
    model: object,
    y_train: np.ndarray,
    random_state: int,
    scoring: str,
    cv_folds: int,
) -> object:
    """Adjust CV-backed models for the current train split."""

    fitted_model = clone(model)
    class_counts = np.unique(y_train, return_counts=True)[1]
    min_class_count = int(np.min(class_counts)) if len(class_counts) else 0

    if isinstance(fitted_model, GridSearchCV):
        if min_class_count < 2:
            fitted_model = Pipeline([
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ])
        else:
            fitted_model.scoring = scoring
            fitted_model.cv = StratifiedKFold(
                n_splits=min(max(2, int(cv_folds)), min_class_count),
                shuffle=True,
                random_state=random_state,
            )

    if isinstance(fitted_model, Pipeline):
        clf = fitted_model.named_steps.get("clf")
        if isinstance(clf, MLPClassifier) and (min_class_count < 3 or len(y_train) < 20):
            try:
                fitted_model.set_params(clf__early_stopping=False)
            except ValueError:
                pass

    return fitted_model


def rfe_ablation_curve(
    df: pd.DataFrame,
    feature_set: str,
    model_name: str = "LinearSVC",
    config: EvaluationConfig = EvaluationConfig(use_rfe=True),
    repeats: int = 5,
    out_path: str | Path = "rfe_ablation.png",
    out_csv: Optional[str | Path] = None,
    threshold_fraction: float = 0.90,
) -> Dict[str, object]:
    """Compute and plot accuracy degradation after removing RFE-ranked features.

    A single train/test split is held fixed for all ablation steps. The RFE
    ranking is fitted only on the training split, then the top-k ranked features
    are removed for k=0..n_features-1. At each k, the selected model is refit
    ``repeats`` times with reproducibly varied random seeds and scored on the
    unchanged held-out test split.
    """

    if "y" not in df.columns:
        raise ValueError("Dataset must contain a y column.")
    if feature_set not in RFE_ABLATION_SCENARIOS:
        allowed = ", ".join(RFE_ABLATION_SCENARIOS)
        raise ValueError(f"Unknown feature_set '{feature_set}'. Choose one of: {allowed}.")
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if config.rfe_cv < 2:
        raise ValueError("cv_folds must be at least 2.")
    if threshold_fraction <= 0.0:
        raise ValueError("threshold_fraction must be positive.")

    try:
        scorer = get_scorer(config.scoring)
    except Exception as exc:
        raise ValueError(f"Unknown or unsupported scoring value '{config.scoring}'.") from exc

    cols = columns_for_scenario(df, RFE_ABLATION_SCENARIOS[feature_set])
    n_features = len(cols)
    if n_features <= 3:
        LOGGER.warning(
            "Feature set %s has only %d feature(s); the ablation curve will be coarse.",
            feature_set,
            n_features,
        )

    y = df["y"].to_numpy(dtype=int)
    X = df[cols].to_numpy(dtype=float)
    row_indices = np.arange(len(df))
    try:
        train_idx, test_idx = train_test_split(
            row_indices,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y,
        )
    except ValueError as exc:
        LOGGER.warning("Stratified train/test split failed; using an unstratified split. Error: %s", exc)
        train_idx, test_idx = train_test_split(
            row_indices,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=None,
        )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    ranking_model = build_model_from_cli_name(
        model_name,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        scoring=config.scoring,
        cv_folds=config.rfe_cv,
    )
    ordered_features, rfe_info = fit_rfe_ablation_ranking(
        X_train,
        y_train,
        cols,
        ranking_model,
        config,
    )
    ordered_indices = [cols.index(name) for name in ordered_features]

    mean_scores: List[float] = []
    std_scores: List[float] = []
    all_scores: List[List[float]] = []

    for n_removed in range(n_features):
        remaining_idx = ordered_indices[n_removed:]
        X_train_reduced = X_train[:, remaining_idx]
        X_test_reduced = X_test[:, remaining_idx]
        repeat_scores: List[float] = []

        for repeat_idx in range(repeats):
            rng = np.random.default_rng(int(config.random_state) + repeat_idx)
            repeat_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            base_model = build_model_from_cli_name(
                model_name,
                random_state=repeat_seed,
                n_jobs=config.n_jobs,
                scoring=config.scoring,
                cv_folds=config.rfe_cv,
            )
            fit_model = configure_ablation_fit_model(
                base_model,
                y_train=y_train,
                random_state=repeat_seed,
                scoring=config.scoring,
                cv_folds=config.rfe_cv,
            )
            fit_model.fit(X_train_reduced, y_train)
            try:
                repeat_scores.append(float(scorer(fit_model, X_test_reduced, y_test)))
            except Exception as exc:
                raise ValueError(
                    f"Scoring '{config.scoring}' is not supported by model '{model_name}'."
                ) from exc

        all_scores.append(repeat_scores)
        mean_scores.append(float(np.mean(repeat_scores)))
        std_scores.append(float(np.std(repeat_scores, ddof=1)) if repeats > 1 else 0.0)

    n_removed_values = np.arange(n_features, dtype=int)
    n_remaining_values = n_features - n_removed_values
    result_df = pd.DataFrame({
        "n_removed": n_removed_values,
        "n_remaining": n_remaining_values,
        "mean_score": mean_scores,
        "std_score": std_scores,
    })

    baseline_score = float(mean_scores[0])
    threshold_score = threshold_fraction * baseline_score
    below_threshold = np.where(np.asarray(mean_scores, dtype=float) < threshold_score)[0]
    threshold_k = int(below_threshold[0]) if len(below_threshold) else None
    best_k = int(np.argmax(mean_scores))
    best_score = float(mean_scores[best_k])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_label = scoring_axis_label(config.scoring)

    plt.figure(figsize=(7.5, 5.0))
    plt.errorbar(
        n_removed_values,
        mean_scores,
        yerr=std_scores,
        marker="o",
        linewidth=1.7,
        capsize=3,
        label="Mean score +/- 1 std",
    )
    plt.axhline(
        baseline_score,
        linestyle="--",
        linewidth=1.0,
        color="0.35",
        label=f"Full-feature baseline = {baseline_score:.3f}",
    )
    if threshold_k is not None:
        threshold_value = float(mean_scores[threshold_k])
        plt.axvline(threshold_k, linestyle="--", linewidth=1.0, color="tab:red")
        plt.annotate(
            f"threshold at k={threshold_k} (score={threshold_value:.3f})",
            xy=(threshold_k, threshold_value),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
            color="tab:red",
        )
    plt.xlabel("Number of features removed")
    plt.ylabel(y_label)
    plt.title(f"RFE ablation: accuracy vs features removed — {feature_set}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    out_csv_path: Optional[Path] = Path(out_csv) if out_csv is not None else None
    if out_csv_path is not None:
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_csv_path, index=False)

    ranking_base = out_csv_path if out_csv_path is not None else out_path
    ranking_path = ranking_base.with_name(f"{ranking_base.stem}_ranking.json")
    ranking_payload: Dict[str, object] = {
        "feature_set": feature_set,
        "model": model_name,
        "resolved_model": resolve_model_cli_name(model_name),
        "scoring": config.scoring,
        "random_state": int(config.random_state),
        "test_size": float(config.test_size),
        "cv_folds": int(config.rfe_cv),
        "repeats": int(repeats),
        "threshold_fraction": float(threshold_fraction),
        "baseline_score": baseline_score,
        "threshold_k": threshold_k,
        "best_k": best_k,
        "best_score": best_score,
        "ranking": ordered_features,
        "rfe_info": rfe_info,
        "repeat_scores": {
            str(int(k)): [float(score) for score in scores]
            for k, scores in zip(n_removed_values, all_scores)
        },
        "plot": str(out_path),
        "csv": str(out_csv_path) if out_csv_path is not None else None,
    }
    ranking_path.write_text(json.dumps(ranking_payload, indent=2), encoding="utf-8")

    threshold_text = "not reached" if threshold_k is None else str(threshold_k)
    print(
        f"RFE ablation {feature_set}: baseline={baseline_score:.6f}, "
        f"threshold_k={threshold_text}, best_k={best_k} (score={best_score:.6f})"
    )
    print(f"Saved plot: {out_path}")
    if out_csv_path is not None:
        print(f"Saved CSV: {out_csv_path}")
    print(f"Saved ranking: {ranking_path}")

    return {
        "feature_set": feature_set,
        "plot": str(out_path),
        "csv": str(out_csv_path) if out_csv_path is not None else None,
        "ranking_json": str(ranking_path),
        "baseline_score": baseline_score,
        "threshold_k": threshold_k,
        "best_k": best_k,
        "best_score": best_score,
        "results": result_df,
        "ranking": ordered_features,
    }


# ---------------------------------------------------------------------------
# t-SNE visualization
# ---------------------------------------------------------------------------

def plot_tsne(
    df: pd.DataFrame,
    feature_set: str,
    out_path: str | Path,
    max_samples: int = 3000,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Compute and save a t-SNE plot for a feature scenario.

    Returns the 2D embedding as a NumPy array.
    """

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    if feature_set not in DEFAULT_SCENARIOS:
        raise ValueError(f"feature_set must be one of: {list(DEFAULT_SCENARIOS)}")

    cols = columns_for_scenario(df, DEFAULT_SCENARIOS[feature_set])
    y = df["y"].to_numpy(dtype=int)
    X = df[cols].to_numpy(dtype=float)

    rng = np.random.default_rng(random_state)
    if len(X) > max_samples:
        # Stratified subsampling for visual balance.
        selected_parts = []
        per_class = max_samples // 2
        for label in [ENTANGLED_LABEL, SEPARABLE_LABEL]:
            idx = np.where(y == label)[0]
            take = min(per_class, len(idx))
            selected_parts.append(rng.choice(idx, size=take, replace=False))
        selected = np.concatenate(selected_parts)
        rng.shuffle(selected)
        X = X[selected]
        y = y[selected]

    X_scaled = StandardScaler().fit_transform(X)
    n = len(X_scaled)
    if n < 3:
        raise ValueError("t-SNE requires at least three samples.")
    effective_perplexity = min(float(perplexity), max(2.0, (n - 1) / 3.0))

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    embedding = tsne.fit_transform(X_scaled)

    ent_mask = y == ENTANGLED_LABEL
    sep_mask = y == SEPARABLE_LABEL

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(embedding[ent_mask, 0], embedding[ent_mask, 1], alpha=0.70, s=18, label="Entangled (-1)")
    plt.scatter(embedding[sep_mask, 0], embedding[sep_mask, 1], alpha=0.70, s=18, label="Separable (+1)")
    plt.title(f"t-SNE embedding: {feature_set}")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    emb_path = out_path.with_suffix(".embedding.csv")
    pd.DataFrame({"tsne_1": embedding[:, 0], "tsne_2": embedding[:, 1], "y": y}).to_csv(emb_path, index=False)
    return embedding


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_analysis_dataset_path(dataset_key: str, dataset_path: Optional[str | Path] = None) -> Path:
    """Resolve the dataset selected by --dataset {2x2,3x3} to a CSV/NPZ path."""

    if dataset_path is not None:
        path = Path(dataset_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {path}")
        return path

    candidates = DEFAULT_DATASET_FILE_CANDIDATES.get(dataset_key, ())
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No default dataset file found for --dataset {dataset_key}. "
        f"Looked for: {', '.join(candidates)}. "
        "Generate one first or pass --dataset_path /path/to/file.csv."
    )


def run_advanced_analysis(
    dataset_key: str,
    features: str,
    plot_types: Sequence[str],
    output_dir: str | Path,
    dataset_path: Optional[str | Path] = None,
    test_size: float = 0.30,
    random_state: int = 42,
    use_rfe: bool = True,
    rfe_step: float = 0.10,
    rfe_cv: int = 5,
    n_jobs: int = -1,
    max_shap_background: int = 50,
    max_shap_samples: int = 100,
    shap_nsamples: int = 100,
) -> Dict[str, object]:
    """Train selected models/features and save requested performance plots."""

    if features not in ANALYSIS_SCENARIOS:
        raise ValueError(f"features must be one of {list(ANALYSIS_SCENARIOS)}")
    invalid_plots = sorted(set(plot_types) - set(PLOT_CHOICES))
    if invalid_plots:
        raise ValueError(f"Unknown plot type(s): {invalid_plots}. Choose from {list(PLOT_CHOICES)}.")

    resolved_path = resolve_analysis_dataset_path(dataset_key, dataset_path)
    LOGGER.info("Using dataset file: %s", resolved_path)
    df = load_dataset(resolved_path)

    cfg = EvaluationConfig(
        test_size=test_size,
        random_state=random_state,
        use_rfe=use_rfe,
        rfe_step=rfe_step,
        rfe_cv=rfe_cv,
        n_jobs=n_jobs,
    )
    scenarios = {features: ANALYSIS_SCENARIOS[features]}
    return evaluate_feature_scenarios(
        df,
        out_dir=output_dir,
        config=cfg,
        scenarios=scenarios,
        plot_types=plot_types,
        plot_output_dir=output_dir,
        max_shap_background=max_shap_background,
        max_shap_samples=max_shap_samples,
        shap_nsamples=shap_nsamples,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate datasets, evaluate ML models, and create advanced "
            "visualizations for bipartite entanglement separability studies."
        )
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Top-level advanced-analysis mode requested by the user. This allows:
    # python entanglement_ml_pipeline_v2.py --dataset 2x2 --features All --plot RocCurve ...
    parser.add_argument("--dataset", choices=["2x2", "3x3"], default=None,
                        help="Dataset preset for advanced analysis mode.")
    parser.add_argument("--dataset_path", "--dataset-path", dest="dataset_path", default=None,
                        help="Optional explicit CSV/NPZ path for advanced analysis mode.")
    parser.add_argument("--features", choices=list(ANALYSIS_SCENARIOS), default=None,
                        help="Feature set for advanced analysis mode: SU, Moments, RM, or All.")
    parser.add_argument("--plot", nargs="+", choices=list(PLOT_CHOICES), default=None,
                        help="One or more plots for advanced analysis mode.")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="./plots",
                        help="Directory for PNG plots and analysis reports in advanced analysis mode.")
    parser.add_argument("--max-shap-background", type=int, default=50,
                        help="Maximum background samples for SHAP in advanced analysis mode.")
    parser.add_argument("--max-shap-samples", type=int, default=100,
                        help="Maximum test samples explained by SHAP in advanced analysis mode.")
    parser.add_argument("--shap-nsamples", type=int, default=100,
                        help="Approximate number of samples for SHAP KernelExplainer in advanced analysis mode.")
    parser.add_argument("--test-size", type=float, default=0.30,
                        help="Test split fraction for advanced analysis mode.")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for advanced analysis mode.")
    parser.add_argument("--no-rfe", action="store_true",
                        help="Disable RFE in advanced analysis mode.")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel jobs for advanced analysis mode.")

    subparsers = parser.add_subparsers(dest="command", required=False)

    gen = subparsers.add_parser("generate", help="Generate a feature-only dataset.")
    gen.add_argument("--system", required=True, choices=["2x2", "3x3"])
    gen.add_argument("--metric", default="bures", choices=["bures", "hs", "hilbert-schmidt"])
    # CLI documentation: choose the state generator with --generation-method {1,2,3,4}.
    gen.add_argument(
        "--generation-method",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help=(
            "State generation method: 1=existing logic, "
            "2=metric rejection sampling, "
            "3=metric-local separable mixtures plus Method 1 entangled, "
            "4=controlled depolarization."
        ),
    )
    gen.add_argument("--n-entangled", type=int, required=True)
    gen.add_argument("--n-separable", type=int, required=True)
    gen.add_argument("--sep-mixture-terms", type=int, default=1)
    gen.add_argument("--max-draws", type=int, default=10_000_000)
    gen.add_argument(
        "--method4-depolarize-after",
        type=int,
        default=1000,
        help="Method 4 starts depolarizing only after this many rejected/unneeded candidates.",
    )
    gen.add_argument(
        "--method4-depolarize-step",
        type=float,
        default=0.01,
        help="Method 4 monotonic p decrement for rho(p)=p*rho+(1-p)I/D.",
    )
    gen.add_argument("--ppt-tol", type=float, default=1e-10)
    gen.add_argument("--qutrit-script", default=None, help="Path to attached 3x3_svm.py for PPT qutrit states.")
    gen.add_argument("--no-reject-ppt-qutrit-without-script", action="store_true")
    gen.add_argument(
        "--purity-filter",
        action="store_true",
        help="Accept only states with purity in [1/D, 1/D+eta] or [1-eta, 1].",
    )
    gen.add_argument("--eta", type=float, default=0.02,
                     help="Purity-window width used with --purity-filter.")
    gen.add_argument(
        "--purity-sampling-mode",
        choices=["targeted", "rejection"],
        default="targeted",
        help=(
            "How to sample when --purity-filter is enabled. 'targeted' is a controlled "
            "extreme-purity distribution; 'rejection' keeps the original candidate samplers."
        ),
    )
    gen.add_argument("--no-purity-column", action="store_true",
                     help="Do not store the diagnostic purity column in the output CSV.")
    gen.add_argument("--random-state", type=int, default=42)
    gen.add_argument("--out", required=True, help="Output CSV path. A matching .npz and metadata JSON are also written.")
    gen.add_argument("--no-npz", action="store_true", help="Do not write the grouped NPZ bundle.")

    ev = subparsers.add_parser("evaluate", help="Run RFE and model evaluation.")
    ev.add_argument("--dataset", required=True, help="CSV or NPZ dataset path.")
    ev.add_argument("--out-dir", required=True)
    ev.add_argument("--test-size", type=float, default=0.30)
    ev.add_argument("--random-state", type=int, default=42)
    ev.add_argument("--no-rfe", action="store_true")
    ev.add_argument("--rfe-step", type=float, default=0.10)
    ev.add_argument("--rfe-cv", type=int, default=5)
    ev.add_argument("--n-jobs", type=int, default=-1)
    ev.add_argument("--features", choices=list(ANALYSIS_SCENARIOS), default=None,
                    help="Optional single feature set to evaluate: SU, Moments, RM, or All.")
    ev.add_argument("--plot", nargs="+", choices=list(PLOT_CHOICES), default=None,
                    help="Optional advanced plots to save after each model is fitted.")
    ev.add_argument("--plot-output-dir", "--output_dir", "--output-dir", dest="plot_output_dir", default=None,
                    help="Where to save PNG plots; defaults to --out-dir.")
    ev.add_argument("--max-shap-background", type=int, default=50)
    ev.add_argument("--max-shap-samples", type=int, default=100)
    ev.add_argument("--shap-nsamples", type=int, default=100)

    tsne = subparsers.add_parser("tsne", help="Compute and save a t-SNE plot.")
    tsne.add_argument("--dataset", required=True, help="CSV or NPZ dataset path.")
    tsne.add_argument("--feature-set", default="ALL", choices=list(DEFAULT_SCENARIOS))
    tsne.add_argument("--out", required=True)
    tsne.add_argument("--max-samples", type=int, default=3000)
    tsne.add_argument("--perplexity", type=float, default=30.0)
    tsne.add_argument("--random-state", type=int, default=42)

    rfe_ab = subparsers.add_parser(
        "ts_rfe_ablation",
        help="Plot held-out score vs number of top-ranked RFE features removed.",
    )
    rfe_ab.add_argument("--dataset", required=True, help="CSV or NPZ dataset path.")
    rfe_ab.add_argument(
        "--feature-set",
        nargs="+",
        required=True,
        choices=list(RFE_ABLATION_SCENARIOS),
        help="One or more feature sets: SU, Moment, RMInvariant, or ALL.",
    )
    rfe_ab.add_argument(
        "--model",
        default="LinearSVC",
        choices=list(MODEL_CLI_CHOICES),
        help="Model used for RFE ranking and repeated ablation fits.",
    )
    rfe_ab.add_argument("--cv-folds", type=int, default=5,
                        help="RFECV folds; capped by the smallest training class count.")
    rfe_ab.add_argument("--repeats", type=int, default=5,
                        help="Repeated refits per ablation point.")
    rfe_ab.add_argument("--scoring", default="balanced_accuracy",
                        help="Any sklearn scorer name supported by the selected model.")
    rfe_ab.add_argument("--test-size", type=float, default=0.30,
                        help="Fixed held-out test fraction.")
    rfe_ab.add_argument("--random-state", type=int, default=42)
    rfe_ab.add_argument("--n-jobs", type=int, default=-1)
    rfe_ab.add_argument("--rfe-step", type=float, default=0.10,
                        help="RFECV feature-removal step.")
    rfe_ab.add_argument("--no-rfecv", action="store_true",
                        help="Use plain RFE(step=1) instead of RFECV for the ranking.")
    rfe_ab.add_argument("--baseline-fraction", type=float, default=0.90,
                        help="Annotate first k whose mean score drops below this fraction of baseline.")
    rfe_ab.add_argument("--out", required=True, help="Output PNG path.")
    rfe_ab.add_argument("--out-csv", default=None,
                        help="Optional CSV path for n_removed, n_remaining, mean_score, std_score.")

    purity_exp = subparsers.add_parser(
        "purity_experiment",
        help="Compare baseline training with purity-constrained training on an unrestricted test set.",
    )
    purity_exp.add_argument("--system", required=True, choices=["2x2", "3x3"])
    purity_exp.add_argument("--metric", default="bures", choices=["bures", "hs", "hilbert-schmidt"])
    purity_exp.add_argument("--generation-method", type=int, default=1, choices=[1, 2, 3, 4],
                            help="Dataset-generation method used for baseline, test, and constrained training sets.")
    purity_exp.add_argument("--eta", type=float, default=0.02,
                            help="Single eta value for the purity-constrained training experiment.")
    purity_exp.add_argument("--eta-grid", type=float, nargs="+", default=None,
                            help="Optional list of eta values. Overrides --eta and enables performance-vs-eta curves.")
    purity_exp.add_argument(
        "--purity-sampling-mode",
        choices=["targeted", "rejection"],
        default="targeted",
        help=(
            "How to sample constrained training data. 'targeted' is efficient but not "
            "a strict metric-conditioned sample; 'rejection' keeps original candidate samplers."
        ),
    )
    purity_exp.add_argument("--n-train", type=int, required=True,
                            help="Total training rows per condition; split as evenly as possible by class.")
    purity_exp.add_argument("--n-test", type=int, required=True,
                            help="Total unrestricted test rows; split as evenly as possible by class.")
    purity_exp.add_argument("--sep-mixture-terms", type=int, default=1)
    purity_exp.add_argument("--max-draws", type=int, default=10_000_000)
    purity_exp.add_argument("--ppt-tol", type=float, default=1e-10)
    purity_exp.add_argument("--qutrit-script", default=None, help="Path to attached 3x3_svm.py for PPT qutrit states.")
    purity_exp.add_argument("--no-reject-ppt-qutrit-without-script", action="store_true")
    purity_exp.add_argument("--random-state", type=int, default=42)
    purity_exp.add_argument("--out-dir", required=True)
    purity_exp.add_argument("--no-npz", action="store_true", help="Do not write grouped NPZ bundles for generated datasets.")
    purity_exp.add_argument("--no-rfe", action="store_true")
    purity_exp.add_argument("--rfe-step", type=float, default=0.10)
    purity_exp.add_argument("--rfe-cv", type=int, default=5)
    purity_exp.add_argument("--n-jobs", type=int, default=-1)
    purity_exp.add_argument("--features", choices=list(ANALYSIS_SCENARIOS), default=None,
                            help="Optional single feature set to evaluate: SU, Moments, RM, or All. Defaults to all scenarios.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    logging.getLogger("shap").setLevel(logging.WARNING)

    if args.command is None:
        if args.dataset is None or args.features is None or args.plot is None:
            parser.print_help()
            return 2
        run_advanced_analysis(
            dataset_key=args.dataset,
            dataset_path=args.dataset_path,
            features=args.features,
            plot_types=args.plot,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            use_rfe=not args.no_rfe,
            n_jobs=args.n_jobs,
            max_shap_background=args.max_shap_background,
            max_shap_samples=args.max_shap_samples,
            shap_nsamples=args.shap_nsamples,
        )
        LOGGER.info("Saved advanced analysis outputs to %s", args.output_dir)
        return 0

    if args.command == "generate":
        cfg = DatasetConfig(
            system=args.system,
            metric=args.metric,
            generation_method=args.generation_method,
            n_entangled=args.n_entangled,
            n_separable=args.n_separable,
            sep_mixture_terms=args.sep_mixture_terms,
            max_draws=args.max_draws,
            method4_depolarize_after=args.method4_depolarize_after,
            method4_depolarize_step=args.method4_depolarize_step,
            ppt_tol=args.ppt_tol,
            qutrit_script=args.qutrit_script,
            reject_ppt_qutrit_without_script=not args.no_reject_ppt_qutrit_without_script,
            purity_filter=args.purity_filter,
            eta=args.eta,
            purity_sampling_mode=args.purity_sampling_mode,
            store_purity=not args.no_purity_column,
            random_state=args.random_state,
        )
        df = generate_dataset(cfg)
        save_dataset_bundle(df, args.out, cfg, write_npz=not args.no_npz)
        LOGGER.info("Saved dataset with shape %s to %s", df.shape, args.out)
        LOGGER.info("Class counts: %s", df["y"].value_counts().to_dict())
        return 0

    if args.command == "evaluate":
        df = load_dataset(args.dataset)
        cfg = EvaluationConfig(
            test_size=args.test_size,
            random_state=args.random_state,
            use_rfe=not args.no_rfe,
            rfe_step=args.rfe_step,
            rfe_cv=args.rfe_cv,
            n_jobs=args.n_jobs,
        )
        scenarios = DEFAULT_SCENARIOS if args.features is None else {args.features: ANALYSIS_SCENARIOS[args.features]}
        evaluate_feature_scenarios(
            df,
            out_dir=args.out_dir,
            config=cfg,
            scenarios=scenarios,
            plot_types=args.plot,
            plot_output_dir=args.plot_output_dir or args.out_dir,
            max_shap_background=args.max_shap_background,
            max_shap_samples=args.max_shap_samples,
            shap_nsamples=args.shap_nsamples,
        )
        LOGGER.info("Saved evaluation outputs to %s", args.out_dir)
        return 0

    if args.command == "purity_experiment":
        eta_values = args.eta_grid if args.eta_grid is not None else [args.eta]
        eval_cfg = EvaluationConfig(
            test_size=0.0,
            random_state=args.random_state,
            use_rfe=not args.no_rfe,
            rfe_step=args.rfe_step,
            rfe_cv=args.rfe_cv,
            n_jobs=args.n_jobs,
        )
        scenarios = DEFAULT_SCENARIOS if args.features is None else {args.features: ANALYSIS_SCENARIOS[args.features]}
        summary = run_purity_experiment(
            system=args.system,
            metric=args.metric,
            eta_values=eta_values,
            n_train=args.n_train,
            n_test=args.n_test,
            out_dir=args.out_dir,
            sep_mixture_terms=args.sep_mixture_terms,
            max_draws=args.max_draws,
            ppt_tol=args.ppt_tol,
            qutrit_script=args.qutrit_script,
            reject_ppt_qutrit_without_script=not args.no_reject_ppt_qutrit_without_script,
            random_state=args.random_state,
            eval_config=eval_cfg,
            scenarios=scenarios,
            write_npz=not args.no_npz,
            purity_sampling_mode=args.purity_sampling_mode,
            generation_method=args.generation_method,
        )
        LOGGER.info("Saved purity experiment summary to %s", summary.get("summary_json"))
        return 0

    if args.command == "tsne":
        df = load_dataset(args.dataset)
        plot_tsne(
            df,
            feature_set=args.feature_set,
            out_path=args.out,
            max_samples=args.max_samples,
            perplexity=args.perplexity,
            random_state=args.random_state,
        )
        LOGGER.info("Saved t-SNE plot to %s", args.out)
        return 0

    if args.command == "ts_rfe_ablation":
        df = load_dataset(args.dataset)
        cfg = EvaluationConfig(
            test_size=args.test_size,
            random_state=args.random_state,
            use_rfe=not args.no_rfecv,
            rfe_step=args.rfe_step,
            rfe_cv=args.cv_folds,
            n_jobs=args.n_jobs,
            scoring=args.scoring,
        )
        multiple_feature_sets = len(args.feature_set) > 1
        for feature_set in args.feature_set:
            out_path = output_path_for_feature_set(args.out, feature_set, multiple_feature_sets)
            out_csv = (
                output_path_for_feature_set(args.out_csv, feature_set, multiple_feature_sets)
                if args.out_csv is not None
                else None
            )
            rfe_ablation_curve(
                df,
                feature_set=feature_set,
                model_name=args.model,
                config=cfg,
                repeats=args.repeats,
                out_path=out_path,
                out_csv=out_csv,
                threshold_fraction=args.baseline_fraction,
            )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

# End of entanglement_ml_pipeline_v3.py
