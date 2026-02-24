#!/usr/bin/env python3
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fhe_common import PHANTOM_PATH  # noqa: F401

try:
    import pyPhantom as phantom
    HAS_PHANTOM = True
except ImportError:
    HAS_PHANTOM = False


class EncryptedSimilarityJoins:
    def __init__(self, embed_dim=16, poly_degree=8192, scale_bits=40):
        if not HAS_PHANTOM:
            raise RuntimeError("PhantomFHE required")

        self.embed_dim = embed_dim
        self.poly_degree = poly_degree
        self.scale = 2.0 ** scale_bits
        self.slot_count = poly_degree // 2

        params = phantom.params(phantom.scheme_type.ckks)
        params.set_poly_modulus_degree(poly_degree)
        params.set_coeff_modulus(phantom.create_coeff_modulus(poly_degree, [60, 40, 40, 60]))
        params.set_special_modulus_size(1)

        self.ctx = phantom.context(params)
        self.sk = phantom.secret_key(self.ctx)
        self.pk = self.sk.gen_publickey(self.ctx)
        self.relin_keys = self.sk.gen_relinkey(self.ctx)
        self.encoder = phantom.ckks_encoder(self.ctx)
        self.n_cts_per_vec = embed_dim // 2

    def _complex_pack_vectors(self, vectors, conjugate=False):
        n_vecs = len(vectors)
        packed = []
        for j in range(self.n_cts_per_vec):
            slots = np.zeros(self.slot_count, dtype=np.complex128)
            for i in range(min(n_vecs, self.slot_count)):
                real_part = vectors[i, 2*j]
                imag_part = vectors[i, 2*j + 1]
                if conjugate:
                    slots[i] = real_part - 1j * imag_part
                else:
                    slots[i] = real_part + 1j * imag_part
            packed.append(slots)
        return packed

    def encrypt_database(self, embeddings):
        n_docs = len(embeddings)
        docs_per_chunk = self.slot_count
        n_chunks = (n_docs + docs_per_chunk - 1) // docs_per_chunk

        encrypted_chunks = []
        for c in range(n_chunks):
            start = c * docs_per_chunk
            end = min(start + docs_per_chunk, n_docs)
            chunk_vecs = embeddings[start:end]

            if len(chunk_vecs) < docs_per_chunk:
                pad = np.zeros((docs_per_chunk - len(chunk_vecs), self.embed_dim))
                chunk_vecs = np.vstack([chunk_vecs, pad])

            packed = self._complex_pack_vectors(chunk_vecs, conjugate=False)
            chunk_cts = []
            for slots in packed:
                pt = self.encoder.encode_complex_vector(self.ctx, list(slots), self.scale)
                ct = self.pk.encrypt_asymmetric(self.ctx, pt)
                chunk_cts.append(ct)
            encrypted_chunks.append(chunk_cts)

        return {'chunks': encrypted_chunks, 'n_docs': n_docs, 'n_chunks': n_chunks, 'docs_per_chunk': docs_per_chunk}

    def encrypt_query(self, query):
        query = query.reshape(1, -1)
        query_cts = []
        for j in range(self.n_cts_per_vec):
            slots = np.zeros(self.slot_count, dtype=np.complex128)
            real_part = query[0, 2*j]
            imag_part = query[0, 2*j + 1]
            slots[:] = real_part - 1j * imag_part
            pt = self.encoder.encode_complex_vector(self.ctx, list(slots), self.scale)
            ct = self.pk.encrypt_asymmetric(self.ctx, pt)
            query_cts.append(ct)
        return query_cts

    def search(self, encrypted_query, encrypted_db):
        encrypted_scores = []
        for chunk_cts in encrypted_db['chunks']:
            result = phantom.multiply(self.ctx, encrypted_query[0], chunk_cts[0])
            result = phantom.relinearize(self.ctx, result, self.relin_keys)
            result = phantom.rescale_to_next(self.ctx, result)

            for j in range(1, self.n_cts_per_vec):
                product = phantom.multiply(self.ctx, encrypted_query[j], chunk_cts[j])
                product = phantom.relinearize(self.ctx, product, self.relin_keys)
                product = phantom.rescale_to_next(self.ctx, product)
                result = phantom.add(self.ctx, result, product)

            encrypted_scores.append(result)
        return encrypted_scores

    def decrypt_scores(self, encrypted_scores, n_docs):
        all_scores = []
        for ct in encrypted_scores:
            dec = self.sk.decrypt(self.ctx, ct)
            scores = self.encoder.decode_complex_vector(self.ctx, dec)
            all_scores.extend([s.real for s in scores[:self.slot_count]])
        return np.array(all_scores[:n_docs])


def compute_recall_at_k(scores, relevance, ks=[1, 5, 10, 50, 100]):
    results = {}
    for k in ks:
        hits, total = 0, 0
        for q_idx, doc_idx in relevance:
            if q_idx < len(scores):
                top_k = np.argsort(-scores[q_idx])[:k]
                if doc_idx in top_k:
                    hits += 1
                total += 1
        if total > 0:
            results[f"R@{k}"] = hits / total
    return results


def demo_with_real_data(n_docs=10000, n_queries=100, embed_dim=64):
    qwen3_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "qwen3_emb_10k.npz")
    msmarco_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "msmarco_10k_emb.npz")

    if os.path.exists(qwen3_path):
        data = np.load(qwen3_path, allow_pickle=True)
        queries_full = data['query_emb']
        docs_full = data['doc_emb']
        relevance = list(data['relevance'])
        embedding_type = "qwen3"
    elif os.path.exists(msmarco_path):
        data = np.load(msmarco_path, allow_pickle=True)
        queries_full = data['queries']
        docs_full = data['docs']
        relevance = list(data['relevance'])
        embedding_type = "msmarco"
    else:
        print("No embeddings found")
        return

    TRAIN_CUTOFF = 800
    test_rel = [(q, d) for q, d in relevance if q >= TRAIN_CUTOFF]
    n_docs = min(n_docs, docs_full.shape[0])
    valid_test_rel = [(q, d) for q, d in test_rel if d < n_docs]

    if embedding_type == "qwen3":
        proj_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f"qwen3_distilled_{embed_dim}d.npy")
        combined_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f"qwen3_combined_{embed_dim}d.npy")

        if os.path.exists(combined_path):
            proj_matrix = np.load(combined_path).T
            queries_proj = queries_full @ proj_matrix
            docs_proj = docs_full[:n_docs] @ proj_matrix
        elif os.path.exists(proj_path):
            proj_matrix = np.load(proj_path).T
            queries_proj = queries_full @ proj_matrix
            docs_proj = docs_full[:n_docs] @ proj_matrix
        else:
            queries_proj = queries_full[:, :embed_dim]
            docs_proj = docs_full[:n_docs, :embed_dim]
    else:
        proj_path = os.path.join(os.path.dirname(__file__), f"learned_projection_{embed_dim}d.npy")
        if os.path.exists(proj_path):
            proj_matrix = np.load(proj_path).T
        else:
            np.random.seed(42)
            proj_matrix = np.random.randn(384, embed_dim).astype(np.float32)
            proj_matrix = proj_matrix / np.linalg.norm(proj_matrix, axis=0)
        queries_proj = queries_full @ proj_matrix
        docs_proj = docs_full[:n_docs] @ proj_matrix

    queries_proj = queries_proj / np.linalg.norm(queries_proj, axis=1, keepdims=True)
    docs_proj = docs_proj / np.linalg.norm(docs_proj, axis=1, keepdims=True)

    test_queries = queries_proj[TRAIN_CUTOFF:]
    n_test_queries = min(n_queries, len(test_queries))
    test_queries = test_queries[:n_test_queries]

    true_scores = test_queries @ docs_proj.T

    esj = EncryptedSimilarityJoins(embed_dim=embed_dim)

    t0 = time.perf_counter()
    encrypted_db = esj.encrypt_database(docs_proj)
    encrypt_db_time = time.perf_counter() - t0
    print(f"DB encryption: {encrypt_db_time:.2f}s")

    total_search_time = 0
    all_encrypted_scores = []

    for i in range(n_test_queries):
        t0 = time.perf_counter()
        encrypted_query = esj.encrypt_query(test_queries[i])
        encrypted_scores = esj.search(encrypted_query, encrypted_db)
        scores = esj.decrypt_scores(encrypted_scores, n_docs)
        search_time = time.perf_counter() - t0
        total_search_time += search_time
        all_encrypted_scores.append(scores)

    all_encrypted_scores = np.array(all_encrypted_scores)
    correlation = np.corrcoef(all_encrypted_scores.flatten(), true_scores.flatten())[0, 1]

    local_test_rel = [(q - TRAIN_CUTOFF, d) for q, d in valid_test_rel if q - TRAIN_CUTOFF < n_test_queries]
    recall_results = compute_recall_at_k(all_encrypted_scores, local_test_rel, ks=[1, 5, 10, 50, 100])

    avg_time = total_search_time / n_test_queries
    print(f"Avg query: {avg_time*1000:.1f}ms, correlation: {correlation:.6f}")
    for k, v in sorted(recall_results.items(), key=lambda x: int(x[0].split('@')[1])):
        print(f"  {k}: {v*100:.1f}%")

    return {'avg_time_ms': avg_time * 1000, 'correlation': correlation, 'recall': recall_results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs', type=int, default=10000)
    parser.add_argument('--queries', type=int, default=100)
    parser.add_argument('--dim', type=int, default=64, choices=[16, 32, 64])
    args = parser.parse_args()

    demo_with_real_data(n_docs=args.docs, n_queries=args.queries, embed_dim=args.dim)
