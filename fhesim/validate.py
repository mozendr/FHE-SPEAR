#!/usr/bin/env python3
import numpy as np
from collections import Counter
from scipy.stats import pearsonr, ks_2samp
from scipy.linalg import svd

try:
    from .simulator import FHEAccuracySimulator
except ImportError:
    from simulator import FHEAccuracySimulator

try:
    import tenseal as ts
    HAS_TENSEAL = True
except ImportError:
    HAS_TENSEAL = False

try:
    from Pyfhel import Pyfhel
    HAS_PYFHEL = True
except ImportError:
    HAS_PYFHEL = False


def create_tenseal_context(poly_degree=4096):
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_degree,
        coeff_mod_bit_sizes=[40, 20, 40] if poly_degree == 4096 else [60, 40, 40, 60]
    )
    ctx.global_scale = 2**20 if poly_degree == 4096 else 2**40
    ctx.generate_galois_keys()
    return ctx


def create_pyfhel_context(poly_degree=4096):
    HE = Pyfhel()
    if poly_degree == 4096:
        HE.contextGen(scheme='ckks', n=4096, scale=2**20, qi_sizes=[40, 20, 40])
    else:
        HE.contextGen(scheme='ckks', n=8192, scale=2**40, qi_sizes=[60, 40, 40, 60])
    HE.keyGen()
    HE.relinKeyGen()
    return HE


def measure_noise_tenseal(dim=64, n_trials=100):
    ctx = create_tenseal_context(4096)
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
        'noise_std': np.std(noises),
        'correlation': pearsonr(true_sims, fhe_sims)[0],
    }


def test_noise_constant():
    if not HAS_TENSEAL:
        return None

    sim = FHEAccuracySimulator(poly_modulus_degree=4096)
    ratios = []

    for dim in [16, 32, 64]:
        predicted_std = sim.c * np.sqrt(dim)
        ts_result = measure_noise_tenseal(dim, n_trials=100)
        ratio = ts_result['noise_std'] / predicted_std
        ratios.append(ratio)
        print(f"d={dim}: predicted={predicted_std:.6f}, actual={ts_result['noise_std']:.6f}, ratio={ratio:.2f}")

    avg_ratio = np.mean(ratios)
    return 0.8 <= avg_ratio <= 1.2


def test_correlation_formula():
    if not HAS_TENSEAL:
        return None

    np.random.seed(42)
    n_docs, original_dim = 200, 128

    embeddings = np.random.randn(n_docs, original_dim).astype(np.float64)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    _, _, Vt = svd(embeddings, full_matrices=False)

    errors = []

    for target_dim in [16, 32, 64]:
        Z = embeddings @ Vt[:target_dim].T
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)

        sim = FHEAccuracySimulator(poly_modulus_degree=4096)
        result = sim.predict(Z, target_dim=target_dim)
        predicted_corr = result.predicted_correlation

        ctx = create_tenseal_context(4096)
        true_sims, fhe_sims = [], []

        for _ in range(100):
            i, j = np.random.randint(n_docs), np.random.randint(n_docs)
            if i == j:
                continue
            true_sim = float(Z[i] @ Z[j])
            enc_i = ts.ckks_vector(ctx, Z[i].tolist())
            fhe_sim = enc_i.dot(Z[j].tolist()).decrypt()[0]
            true_sims.append(true_sim)
            fhe_sims.append(fhe_sim)

        actual_corr = pearsonr(true_sims, fhe_sims)[0]
        error = abs(predicted_corr - actual_corr)
        errors.append(error)
        print(f"d={target_dim}: predicted={predicted_corr:.4f}, actual={actual_corr:.4f}, error={error:.4f}")

    return np.mean(errors) < 0.10


def test_bias_simulation():
    if not HAS_TENSEAL:
        return None

    np.random.seed(42)
    dim, n_contexts, n_trials = 64, 10, 50

    real_biases = []
    for _ in range(n_contexts):
        ctx = create_tenseal_context(4096)
        noises = []
        for _ in range(n_trials):
            x = np.random.randn(dim).astype(np.float64)
            x = x / np.linalg.norm(x)
            y = np.random.randn(dim).astype(np.float64)
            y = y / np.linalg.norm(y)

            true_sim = float(np.dot(x, y))
            enc_x = ts.ckks_vector(ctx, x.tolist())
            fhe_sim = enc_x.dot(y.tolist()).decrypt()[0]
            noises.append(fhe_sim - true_sim)
        real_biases.append(np.mean(noises))

    real_bias_std = np.std(real_biases)

    sim = FHEAccuracySimulator(poly_modulus_degree=4096, simulate_bias=True)
    sim_biases = [sim.new_context() for _ in range(1000)]
    sim_bias_std = np.std(sim_biases)

    ratio = sim_bias_std / real_bias_std
    print(f"Bias std: real={real_bias_std:.4f}, sim={sim_bias_std:.4f}, ratio={ratio:.2f}")
    return 0.7 <= ratio <= 1.3


def test_retrieval_accuracy():
    if not HAS_TENSEAL:
        return None

    np.random.seed(42)
    n_docs, dim, k, n_queries = 100, 32, 10, 10

    embeddings = np.random.randn(n_docs, dim).astype(np.float64)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    sim = FHEAccuracySimulator(poly_modulus_degree=4096)
    overlaps = []

    for q_idx in range(n_queries):
        true_sims = embeddings @ embeddings[q_idx]
        true_sims[q_idx] = -np.inf
        true_top_k = set(np.argsort(true_sims)[-k:])

        ctx = create_tenseal_context(4096)
        enc_query = ts.ckks_vector(ctx, embeddings[q_idx].tolist())

        real_fhe_sims = []
        for i in range(n_docs):
            if i == q_idx:
                real_fhe_sims.append(-np.inf)
            else:
                fhe_sim = enc_query.dot(embeddings[i].tolist()).decrypt()[0]
                real_fhe_sims.append(fhe_sim)

        real_top_k = set(np.argsort(real_fhe_sims)[-k:])

        sim_top_k_runs = []
        for _ in range(3):
            sim_top_k, _ = sim.simulate_retrieval(embeddings, q_idx, k)
            sim_top_k_runs.extend(sim_top_k)
        counts = Counter(sim_top_k_runs)
        sim_top_k = set([idx for idx, _ in counts.most_common(k)])

        overlaps.append(len(real_top_k & sim_top_k))

    avg_overlap = np.mean(overlaps)
    print(f"Avg real-sim overlap: {avg_overlap:.1f}/{k}")
    return avg_overlap >= 6


def main():
    print(f"TenSEAL: {HAS_TENSEAL}, Pyfhel: {HAS_PYFHEL}")

    if not HAS_TENSEAL and not HAS_PYFHEL:
        print("Need at least one FHE library")
        return

    results = {
        'noise_constant': test_noise_constant(),
        'correlation': test_correlation_formula(),
        'bias': test_bias_simulation(),
        'retrieval': test_retrieval_accuracy(),
    }

    passed = sum(1 for r in results.values() if r is True)
    total = sum(1 for r in results.values() if r is not None)
    print(f"\n{passed}/{total} tests passed")


if __name__ == "__main__":
    main()
