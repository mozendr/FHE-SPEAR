#!/usr/bin/env python3
import sys
import numpy as np
from scipy.linalg import svd
from scipy import stats as sp_stats
from pathlib import Path

try:
    from .simulator import FHEAccuracySimulator
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from fhesim.simulator import FHEAccuracySimulator


def measure_correlation(embeddings, dim, n_pairs=1000, noise_std=None,
                               c=0.002798):
    n = len(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    X = (embeddings / (norms + 1e-8)).astype(np.float32)

    _, _, Vt = svd(X, full_matrices=False)
    actual_dim = min(dim, Vt.shape[0])
    Z = X @ Vt[:actual_dim].T
    Z_norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = (Z / (Z_norms + 1e-8)).astype(np.float32)

    pairs = []
    while len(pairs) < n_pairs:
        i, j = np.random.randint(n), np.random.randint(n)
        if i != j:
            pairs.append((i, j))

    true_sims = np.array([Z[i] @ Z[j] for i, j in pairs])
    sigma_eps = c * np.sqrt(actual_dim) if noise_std is None else noise_std
    noisy_sims = true_sims + np.random.normal(0, sigma_eps, len(true_sims))

    corr = sp_stats.pearsonr(true_sims, noisy_sims)[0]
    return corr


def validated_fhesim(n_total=500, original_dim=256, seed=42):
    np.random.seed(seed)

    n_clusters = 10
    centers = np.random.randn(n_clusters, original_dim).astype(np.float32)
    labels = np.random.randint(0, n_clusters, n_total)
    X = centers[labels] + 0.3 * np.random.randn(n_total, original_dim).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    # 60/40 train/test split
    n_train = int(0.6 * n_total)
    idx = np.random.permutation(n_total)
    X_train = X[idx[:n_train]]
    X_test = X[idx[n_train:]]

    print(f"Data: {n_total} total, {n_train} train, {n_total - n_train} test")
    print(f"Original dim: {original_dim}")
    print()

    # Calibrate on train set
    print("Calibrating on train set")
    sim_train = FHEAccuracySimulator(poly_modulus_degree=4096)
    cal_dims = [16, 32, 64]
    actual_corrs_train = {}
    for d in cal_dims:
        corr = measure_correlation(X_train, d, n_pairs=500)
        actual_corrs_train[d] = corr
        print(f"  dim={d}: actual correlation = {corr:.4f}")

    c_calibrated = sim_train.calibrate(X_train, actual_corrs_train)
    print(f"  Calibrated c = {c_calibrated:.6f}")
    print()

    print("Predicting on test set")
    sim_test = FHEAccuracySimulator(poly_modulus_degree=4096,
                                     noise_constant=c_calibrated)

    test_dims = [8, 16, 32, 48, 64, 96, 128]
    print(f"  {'Dim':>5} {'Predicted':>10} {'Actual':>10} {'Error':>10}")
    print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*10}")

    errors = []
    for d in test_dims:
        if d >= X_test.shape[1]:
            continue
        result = sim_test.predict(X_test, target_dim=d, n_samples=500)
        predicted = result.predicted_correlation
        actual = measure_correlation(X_test, d, n_pairs=500,
                                            c=c_calibrated)
        error = abs(predicted - actual)
        errors.append(error)
        print(f"  {d:>5} {predicted:>10.4f} {actual:>10.4f} {error:>10.4f}")

    mean_err = np.mean(errors)
    max_err = np.max(errors)
    print()
    print(f"  Mean absolute error (test set): {mean_err:.4f} ({mean_err*100:.1f}%)")
    print(f"  Max absolute error (test set):  {max_err:.4f} ({max_err*100:.1f}%)")
    print()

    print("Circular validation (reference)")
    sim_circular = FHEAccuracySimulator(poly_modulus_degree=4096)

    actual_corrs_all = {}
    for d in cal_dims:
        actual_corrs_all[d] = measure_correlation(X, d, n_pairs=500)
    c_circular = sim_circular.calibrate(X, actual_corrs_all)

    sim_circular2 = FHEAccuracySimulator(poly_modulus_degree=4096,
                                          noise_constant=c_circular)
    errors_circular = []
    for d in test_dims:
        if d >= X.shape[1]:
            continue
        result = sim_circular2.predict(X, target_dim=d, n_samples=500)
        predicted = result.predicted_correlation
        actual = measure_correlation(X, d, n_pairs=500, c=c_circular)
        error = abs(predicted - actual)
        errors_circular.append(error)

    mean_err_circ = np.mean(errors_circular)
    print(f"  Circular mean error:   {mean_err_circ:.4f} ({mean_err_circ*100:.1f}%)")
    print(f"  Train/test mean error: {mean_err:.4f} ({mean_err*100:.1f}%)")
    print(f"  Difference:            {abs(mean_err - mean_err_circ):.4f}")
    print()

    print("Summary:")
    print(f"  Train/test: {mean_err*100:.1f}% mean error")
    print(f"  Circular:   {mean_err_circ*100:.1f}% mean error")

    return mean_err, mean_err_circ


if __name__ == "__main__":
    validated_fhesim()
