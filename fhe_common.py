#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
import time
from pathlib import Path

PHANTOM_PATH = os.path.expanduser("~/gpu_fhe_libs/phantom-fhe/build/lib")
if PHANTOM_PATH not in sys.path:
    sys.path.insert(0, PHANTOM_PATH)

try:
    import pyPhantom as phantom
    USE_PHANTOM_GPU = True
except ImportError:
    USE_PHANTOM_GPU = False

if not USE_PHANTOM_GPU:
    try:
        import tenseal as ts
    except ImportError:
        ts = None

PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data"

FALLBACK_MODEL_URL = "https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth"


def download_fallback_model():
    path = MODEL_DIR / "RWKV-7-1.5B-World.pth"
    if path.exists():
        return path
    print(f"Downloading fallback model to {path}...")
    import urllib.request
    urllib.request.urlretrieve(FALLBACK_MODEL_URL, path)
    return path


def euclidean_to_lorentz(v):
    v_norm_sq = (v ** 2).sum(axis=-1, keepdims=True)
    x0 = np.sqrt(1 + v_norm_sq)
    return np.concatenate([x0, v], axis=-1)


def lorentz_inner_product_batch(queries, docs):
    q0 = queries[:, 0:1]
    q_space = queries[:, 1:]
    d0 = docs[:, 0:1]
    d_space = docs[:, 1:]
    return -q0 @ d0.T + q_space @ d_space.T


def pack_complex(real_vec):
    n = len(real_vec)
    if n % 2 != 0:
        real_vec = np.concatenate([real_vec, [0.0]])
    return real_vec[0::2] + 1j * real_vec[1::2]


def pack_complex_conjugate(real_vec):
    n = len(real_vec)
    if n % 2 != 0:
        real_vec = np.concatenate([real_vec, [0.0]])
    return real_vec[0::2] - 1j * real_vec[1::2]


def get_embeddings(model, tokenizer, texts, batch_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens_list = [tokenizer.encode(t, add_eos=True) for t in batch]
        max_len = max(len(t) for t in tokens_list)
        padded = [[0] * (max_len - len(t)) + t for t in tokens_list]
        with torch.no_grad():
            embs, _ = model.forward_text_only(padded, None)
            embeddings.append(embs.cpu().numpy())
    return np.vstack(embeddings)


class PhantomFHE:
    def __init__(self, poly_degree=8192):
        params = phantom.params(phantom.scheme_type.ckks)
        params.set_poly_modulus_degree(poly_degree)
        params.set_coeff_modulus(phantom.create_coeff_modulus(poly_degree, [60, 40, 40, 60]))
        params.set_special_modulus_size(1)

        self.ctx = phantom.context(params)
        self.sk = phantom.secret_key(self.ctx)
        self.pk = self.sk.gen_publickey(self.ctx)
        self.rlk = self.sk.gen_relinkey(self.ctx)
        self.encoder = phantom.ckks_encoder(self.ctx)
        self.scale = 2.0 ** 40
        self.slot_count = self.encoder.slot_count()

    def encrypt_complex(self, vec):
        slots = list(vec) + [complex(0, 0)] * (self.slot_count - len(vec))
        pt = self.encoder.encode_complex_vector(self.ctx, slots, self.scale)
        return self.pk.encrypt_asymmetric(self.ctx, pt)

    def dot_product(self, enc_query, doc_vec):
        doc_slots = list(doc_vec) + [complex(0, 0)] * (self.slot_count - len(doc_vec))
        doc_pt = self.encoder.encode_complex_vector(self.ctx, doc_slots, self.scale)
        result = phantom.multiply_plain(self.ctx, enc_query, doc_pt)
        result = phantom.rescale_to_next(self.ctx, result)
        dec_pt = self.sk.decrypt(self.ctx, result)
        dec_vec = self.encoder.decode_complex_vector(self.ctx, dec_pt)
        return sum(c.real for c in dec_vec[:len(doc_vec)])

    def batched_dot_products_ctpt(self, query_packed, docs_packed, slots_per_doc):
        n_docs = len(docs_packed)
        batch_size = self.slot_count // slots_per_doc
        scores = []

        for batch_start in range(0, n_docs, batch_size):
            batch_end = min(batch_start + batch_size, n_docs)
            batch_docs = docs_packed[batch_start:batch_end]
            actual_batch = len(batch_docs)

            query_slots = []
            for _ in range(actual_batch):
                query_slots.extend(list(query_packed))
            query_slots = query_slots + [complex(0, 0)] * (self.slot_count - len(query_slots))

            doc_slots = []
            for doc in batch_docs:
                doc_slots.extend(list(doc))
            doc_slots = doc_slots + [complex(0, 0)] * (self.slot_count - len(doc_slots))

            q_pt = self.encoder.encode_complex_vector(self.ctx, query_slots, self.scale)
            enc_q = self.pk.encrypt_asymmetric(self.ctx, q_pt)
            doc_pt = self.encoder.encode_complex_vector(self.ctx, doc_slots, self.scale)
            result = phantom.multiply_plain(self.ctx, enc_q, doc_pt)
            result = phantom.rescale_to_next(self.ctx, result)

            dec_pt = self.sk.decrypt(self.ctx, result)
            dec_vec = self.encoder.decode_complex_vector(self.ctx, dec_pt)

            for i in range(actual_batch):
                start = i * slots_per_doc
                end = start + slots_per_doc
                score = sum(c.real for c in dec_vec[start:end])
                scores.append(score)

        return np.array(scores)

    def encrypt_docs_batch(self, docs_packed, slots_per_doc):
        n_docs = len(docs_packed)
        batch_size = self.slot_count // slots_per_doc
        encrypted_batches = []

        for batch_start in range(0, n_docs, batch_size):
            batch_end = min(batch_start + batch_size, n_docs)
            batch_docs = docs_packed[batch_start:batch_end]

            doc_slots = []
            for doc in batch_docs:
                doc_slots.extend(list(doc))
            doc_slots = doc_slots + [complex(0, 0)] * (self.slot_count - len(doc_slots))

            doc_pt = self.encoder.encode_complex_vector(self.ctx, doc_slots, self.scale)
            enc_docs = self.pk.encrypt_asymmetric(self.ctx, doc_pt)
            encrypted_batches.append((enc_docs, len(batch_docs)))

        return encrypted_batches

    def batched_dot_products_ctct(self, query_packed, encrypted_doc_batches, slots_per_doc):
        scores = []

        for enc_docs, actual_batch in encrypted_doc_batches:
            query_slots = []
            for _ in range(actual_batch):
                query_slots.extend(list(query_packed))
            query_slots = query_slots + [complex(0, 0)] * (self.slot_count - len(query_slots))

            q_pt = self.encoder.encode_complex_vector(self.ctx, query_slots, self.scale)
            enc_q = self.pk.encrypt_asymmetric(self.ctx, q_pt)

            result = phantom.multiply(self.ctx, enc_q, enc_docs)
            result = phantom.relinearize(self.ctx, result, self.rlk)
            result = phantom.rescale_to_next(self.ctx, result)

            dec_pt = self.sk.decrypt(self.ctx, result)
            dec_vec = self.encoder.decode_complex_vector(self.ctx, dec_pt)

            for i in range(actual_batch):
                start = i * slots_per_doc
                end = start + slots_per_doc
                score = sum(c.real for c in dec_vec[start:end])
                scores.append(score)

        return np.array(scores)


class TenSEALFHE:
    def __init__(self, poly_degree=8192):
        self.ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly_degree,
                              coeff_mod_bit_sizes=[60, 40, 40, 60])
        self.ctx.global_scale = 2**40
        self.ctx.generate_galois_keys()

    def encrypt(self, vec):
        return ts.ckks_vector(self.ctx, vec.astype(np.float64).tolist())

    def encrypt_complex(self, vec):
        return ts.ckks_vector(self.ctx, vec.tolist())

    def dot_product(self, enc_query, doc_vec):
        return enc_query.dot(doc_vec.tolist()).decrypt()[0]

    def dot_product_complex(self, enc_query, doc_vec):
        result = enc_query.dot(doc_vec.tolist())
        dec = result.decrypt()
        return sum(c.real for c in dec[:len(doc_vec)])
