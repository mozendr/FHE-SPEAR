#!/usr/bin/env python3
import numpy as np
import time
import tenseal as ts

try:
    from .simulator import FHEAccuracySimulator
except ImportError:
    from simulator import FHEAccuracySimulator


def benchmark_config(N, coeff_bits, n_ops=50):
    np.random.seed(42)
    d = 64
    vectors = np.random.randn(n_ops * 2, d).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=N, coeff_mod_bit_sizes=coeff_bits)
    ctx.global_scale = 2 ** 40 if N > 4096 else 2 ** 20
    ctx.generate_galois_keys()

    start = time.perf_counter()
    for i in range(n_ops):
        v1, v2 = vectors[i*2], vectors[i*2+1]
        enc = ts.ckks_vector(ctx, v1.tolist())
        _ = enc.dot(v2.tolist()).decrypt()[0]
    tenseal_time = time.perf_counter() - start
    tenseal_ms = tenseal_time / n_ops * 1000

    sim = FHEAccuracySimulator(poly_modulus_degree=N)
    start = time.perf_counter()
    for i in range(n_ops):
        v1, v2 = vectors[i*2], vectors[i*2+1]
        _ = sim.simulate_dot_product(v1, v2)
    sim_time = time.perf_counter() - start
    sim_ms = sim_time / n_ops * 1000

    return tenseal_ms, sim_ms, tenseal_time / sim_time


def main():
    configs = [
        (4096, [40, 20, 40], "N=4096"),
        (8192, [60, 40, 40, 60], "N=8192"),
        (16384, [60, 40, 40, 40, 40, 60], "N=16384"),
    ]

    for N, coeff_bits, label in configs:
        n_ops = 20 if N == 16384 else 50
        _, _, speedup = benchmark_config(N, coeff_bits, n_ops)
        print(f"{label}: {speedup:,.0f}x")


if __name__ == "__main__":
    main()
