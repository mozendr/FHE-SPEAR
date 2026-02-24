#!/usr/bin/env python3
import os
import numpy as np
import time
from scipy.stats import pearsonr
from scipy.linalg import svd
import json

try:
    import tenseal as ts
except ImportError:
    print("TenSEAL required")
    exit(1)


def create_context(poly_degree=4096):
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_degree,
        coeff_mod_bit_sizes=[40, 20, 40] if poly_degree == 4096 else [60, 40, 40, 60]
    )
    ctx.global_scale = 2**20 if poly_degree == 4096 else 2**40
    ctx.generate_galois_keys()
    return ctx


def measure_noise(ctx, dim, n_trials=100):
    np.random.seed(42)
    noises, true_sims, fhe_sims = [], [], []

    for _ in range(n_trials):
        x = np.random.randn(dim).astype(np.float64)
        x = x / np.linalg.norm(x)
        y = np.random.randn(dim).astype(np.float64)
        y = y / np.linalg.norm(y)

        true_sim = float(np.dot(x, y))
        enc_x = ts.ckks_vector(ctx, x.tolist())
        fhe_sim = enc_x.dot(y.tolist()).decrypt()[0]

        noises.append(fhe_sim - true_sim)
        true_sims.append(true_sim)
        fhe_sims.append(fhe_sim)

    return {
        'noise_mean': np.mean(noises),
        'noise_std': np.std(noises),
        'correlation': pearsonr(true_sims, fhe_sims)[0]
    }


def calibrate_noise_constant():
    dims = [8, 16, 32, 64, 128, 256]
    n_contexts = 5
    all_results = []

    for dim in dims:
        dim_results = []
        for _ in range(n_contexts):
            ctx = create_context(4096)
            result = measure_noise(ctx, dim, n_trials=100)
            dim_results.append(result)

        avg_noise_std = np.mean([r['noise_std'] for r in dim_results])
        std_noise_std = np.std([r['noise_std'] for r in dim_results])
        avg_corr = np.mean([r['correlation'] for r in dim_results])
        c_estimate = avg_noise_std / np.sqrt(dim)

        all_results.append({
            'dim': dim,
            'noise_std': avg_noise_std,
            'noise_std_err': std_noise_std,
            'correlation': avg_corr,
            'c_estimate': c_estimate
        })
        print(f"d={dim}: noise_std={avg_noise_std:.6f}, c={c_estimate:.6f}")

    dims_arr = np.array([r['dim'] for r in all_results])
    noise_stds = np.array([r['noise_std'] for r in all_results])
    c_fitted = np.sum(noise_stds * np.sqrt(dims_arr)) / np.sum(dims_arr)

    print(f"\nFitted c = {c_fitted:.6f}")
    return c_fitted, all_results


def test_on_embeddings(c):
    np.random.seed(42)
    n_docs, original_dim, rank = 300, 384, 50

    U = np.random.randn(n_docs, rank)
    V = np.random.randn(rank, original_dim)
    embeddings = U @ V
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    _, _, Vt = svd(embeddings, full_matrices=False)
    results = []

    for target_dim in [16, 32, 64]:
        Z = embeddings @ Vt[:target_dim].T
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)

        idx = np.random.randint(0, n_docs, (500, 2))
        true_sims = np.array([Z[i] @ Z[j] for i, j in idx if i != j])
        sigma_z = np.std(true_sims)

        sigma_eps = c * np.sqrt(target_dim)
        predicted_corr = sigma_z / np.sqrt(sigma_z**2 + sigma_eps**2)

        ctx = create_context(4096)
        fhe_sims, true_sims_sample = [], []

        for i, j in idx[:200]:
            if i == j:
                continue
            true_sim = float(Z[i] @ Z[j])
            enc_i = ts.ckks_vector(ctx, Z[i].tolist())
            fhe_sim = enc_i.dot(Z[j].tolist()).decrypt()[0]
            true_sims_sample.append(true_sim)
            fhe_sims.append(fhe_sim)

        actual_corr = pearsonr(true_sims_sample, fhe_sims)[0]
        error = abs(predicted_corr - actual_corr)
        results.append({'dim': target_dim, 'predicted': predicted_corr, 'actual': actual_corr, 'error': error})
        print(f"d={target_dim}: predicted={predicted_corr:.4f}, actual={actual_corr:.4f}, error={error:.4f}")

    return results


def main():
    start = time.time()
    c, noise_data = calibrate_noise_constant()
    emb_results = test_on_embeddings(c)
    elapsed = time.time() - start

    avg_error = np.mean([r['error'] for r in emb_results])
    print(f"\nAvg prediction error: {avg_error*100:.1f}%")
    print(f"Time: {elapsed:.1f}s")

    results = {
        'calibrated_c': c,
        'noise_data': noise_data,
        'validation_error': avg_error
    }

    out_path = os.path.join(os.path.dirname(__file__), 'fhesim_calibration.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
