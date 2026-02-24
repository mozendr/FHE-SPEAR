#!/usr/bin/env python3
import numpy as np
from scipy.linalg import svd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Compatibility(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MARGINAL = "marginal"
    POOR = "poor"
    INCOMPATIBLE = "incompatible"


@dataclass
class SimulatorResult:
    predicted_correlation: float
    optimal_dimension: int
    compatibility: Compatibility
    uniformity: float
    similarity_std: float
    recommendation: str
    details: Dict

    def __repr__(self):
        return f"""FHE-Sim: {self.predicted_correlation:.1%} correlation, {self.optimal_dimension}d optimal, {self.compatibility.value}"""


class FHEAccuracySimulator:
    NOISE_CONSTANTS = {4096: 0.002798, 8192: 0.00140, 16384: 0.00070}
    BIAS_STD = {4096: 0.36, 8192: 0.18, 16384: 0.09}

    def __init__(self, poly_modulus_degree: int = 4096,
                 noise_constant: Optional[float] = None,
                 simulate_bias: bool = False):
        self.poly_modulus_degree = poly_modulus_degree
        self.simulate_bias = simulate_bias

        if noise_constant is not None:
            self.c = noise_constant
        elif poly_modulus_degree in self.NOISE_CONSTANTS:
            self.c = self.NOISE_CONSTANTS[poly_modulus_degree]
        else:
            self.c = 0.002798 * (4096 / poly_modulus_degree) ** 0.5

        self._bias_std = self.BIAS_STD.get(poly_modulus_degree, 0.36)
        self.context_bias = 0.0
        if simulate_bias:
            self.new_context()

    def new_context(self) -> float:
        self.context_bias = np.random.normal(0, self._bias_std)
        return self.context_bias

    def predict(self, embeddings: np.ndarray,
                target_dim: Optional[int] = None,
                n_samples: int = 1000) -> SimulatorResult:
        embeddings = self._normalize(embeddings)
        n, original_dim = embeddings.shape

        uniformity, mean_sim, sim_std = self._compute_stats(embeddings, n_samples)
        optimal_dim = self._find_optimal_dim(embeddings)
        dim = target_dim if target_dim is not None else optimal_dim

        _, _, Vt = svd(embeddings, full_matrices=False)
        actual_dim = min(dim, Vt.shape[0])
        Z = self._normalize(embeddings @ Vt[:actual_dim].T)

        pairs = [(np.random.randint(n), np.random.randint(n)) for _ in range(min(n_samples, n*n//2))]
        pairs = [(i, j) for i, j in pairs if i != j][:n_samples]
        orig_sims = np.array([embeddings[i] @ embeddings[j] for i, j in pairs])
        comp_sims = np.array([Z[i] @ Z[j] for i, j in pairs])

        from scipy import stats as sp_stats
        rho_compression = sp_stats.pearsonr(orig_sims, comp_sims)[0] if len(orig_sims) > 2 else 1.0

        sigma_z = np.std(comp_sims)
        rho_noise = self._predict_snr(sigma_z, actual_dim)
        predicted_rho = np.clip(rho_compression * rho_noise, 0, 1)

        compatibility = self._assess(uniformity, sim_std, predicted_rho)
        recommendation = self._recommend(uniformity, sim_std, predicted_rho, optimal_dim, target_dim)

        return SimulatorResult(
            predicted_correlation=predicted_rho,
            optimal_dimension=optimal_dim,
            compatibility=compatibility,
            uniformity=uniformity,
            similarity_std=sim_std,
            recommendation=recommendation,
            details={
                'original_dim': original_dim,
                'target_dim': actual_dim,
                'noise_constant': self.c,
                'rho_compression': rho_compression,
                'rho_noise': rho_noise,
            }
        )

    def simulate_dot_product(self, x: np.ndarray, y: np.ndarray) -> float:
        d = len(x)
        noise = np.random.normal(0, self.c * np.sqrt(d))
        return float(np.dot(x, y) + noise + self.context_bias)

    def simulate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = self._normalize(embeddings)
        n = len(embeddings)
        sim = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                sim[i, j] = self.simulate_dot_product(embeddings[i], embeddings[j])
        return sim

    def simulate_retrieval(self, embeddings: np.ndarray,
                           query_idx: int, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        embeddings = self._normalize(embeddings)
        query = embeddings[query_idx]
        sims = np.array([self.simulate_dot_product(query, e) for e in embeddings])
        sims[query_idx] = -np.inf
        top_k = np.argsort(sims)[-k:][::-1]
        return top_k, sims[top_k]

    def estimate_retrieval_accuracy(self, embeddings: np.ndarray,
                                    n_queries: int = 100, k: int = 10,
                                    n_runs: int = 5) -> Dict:
        embeddings = self._normalize(embeddings)
        n = len(embeddings)
        true_sim = embeddings @ embeddings.T

        precisions = []
        for q_idx in np.random.choice(n, min(n_queries, n), replace=False):
            true_sims = true_sim[q_idx].copy()
            true_sims[q_idx] = -np.inf
            true_top_k = set(np.argsort(true_sims)[-k:])

            hits = {}
            for _ in range(n_runs):
                fhe_top_k, _ = self.simulate_retrieval(embeddings, q_idx, k)
                for idx in fhe_top_k:
                    hits[idx] = hits.get(idx, 0) + 1
            fhe_top_k_final = set(sorted(hits.keys(), key=lambda x: -hits[x])[:k])
            precisions.append(len(true_top_k & fhe_top_k_final) / k)

        return {'precision_at_k': np.mean(precisions), 'precision_std': np.std(precisions), 'k': k}

    def calibrate(self, embeddings: np.ndarray, actual_correlations: Dict[int, float]) -> float:
        _, _, Vt = svd(embeddings, full_matrices=False)
        c_estimates = []

        for d, actual_rho in actual_correlations.items():
            Z = self._normalize(embeddings @ Vt[:d].T)
            n = len(Z)
            pairs = [(np.random.randint(n), np.random.randint(n)) for _ in range(500)]
            sims = [Z[i] @ Z[j] for i, j in pairs if i != j]
            sigma_z = np.std(sims)

            if 0 < actual_rho < 1:
                c_sq = (sigma_z**2 / actual_rho**2 - sigma_z**2) / d
                if c_sq > 0:
                    c_estimates.append(np.sqrt(c_sq))

        if c_estimates:
            self.c = np.mean(c_estimates)
        return self.c

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return (X / (norms + 1e-8)).astype(np.float32)

    def _compute_stats(self, X: np.ndarray, n: int) -> Tuple[float, float, float]:
        idx = np.random.randint(0, len(X), (n, 2))
        sims = np.array([X[i] @ X[j] for i, j in idx if i != j])
        return 1 - abs(np.mean(sims)), np.mean(sims), np.std(sims)

    def _predict_snr(self, sigma_z: float, d: int) -> float:
        sigma_eps = self.c * np.sqrt(d)
        return sigma_z / np.sqrt(sigma_z**2 + sigma_eps**2) if sigma_z > 1e-6 else 0.0

    def _find_optimal_dim(self, X: np.ndarray) -> int:
        from scipy import stats as sp_stats
        _, S, Vt = svd(X, full_matrices=False)
        dims = [d for d in [8, 16, 32, 48, 64, 96, 128] if d < len(S)]
        if not dims:
            return min(64, len(S))

        n = len(X)
        pairs = [(np.random.randint(n), np.random.randint(n)) for _ in range(300)]
        pairs = [(i, j) for i, j in pairs if i != j]
        orig_sims = np.array([X[i] @ X[j] for i, j in pairs])

        best_dim, best_rho = dims[0], -1
        for d in dims:
            Z = self._normalize(X @ Vt[:d].T)
            comp_sims = np.array([Z[i] @ Z[j] for i, j in pairs])
            sigma_z = np.std(comp_sims)
            rho_compression = sp_stats.pearsonr(orig_sims, comp_sims)[0] if len(orig_sims) > 2 else 1.0
            rho_noise = self._predict_snr(sigma_z, d)
            rho = rho_compression * rho_noise
            if rho > best_rho:
                best_dim, best_rho = d, rho
        return best_dim

    def _assess(self, U: float, sigma: float, rho: float) -> Compatibility:
        if sigma < 0.01:
            return Compatibility.INCOMPATIBLE
        if rho >= 0.95:
            return Compatibility.EXCELLENT
        if rho >= 0.85:
            return Compatibility.GOOD
        if rho >= 0.70:
            return Compatibility.MARGINAL
        if rho >= 0.50:
            return Compatibility.POOR
        return Compatibility.INCOMPATIBLE

    def _recommend(self, U: float, sigma: float, rho: float,
                   opt_dim: int, target: Optional[int]) -> str:
        if sigma < 0.01:
            return "Embeddings have no variance."
        dim = target or opt_dim
        if rho >= 0.90:
            return f"Use SVD to {dim}d. Expected {rho:.0%} correlation."
        if rho >= 0.70:
            return f"Use {opt_dim}d. Expected {rho:.0%} correlation."
        if rho >= 0.50:
            return f"Expected {rho:.0%}. Consider larger N (8192)."
        return f"Expected {rho:.0%}. May need parameter changes."


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(500, 256).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    sim = FHEAccuracySimulator(poly_modulus_degree=4096)
    result = sim.predict(X, target_dim=64)
    print(result)

    _, _, Vt = svd(X, full_matrices=False)
    Z = X @ Vt[:64].T
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)

    acc = sim.estimate_retrieval_accuracy(Z, n_queries=50, k=10, n_runs=3)
    print(f"Precision@10: {acc['precision_at_k']:.1%}")
