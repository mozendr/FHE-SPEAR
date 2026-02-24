#!/usr/bin/env python3
import numpy as np
import time
import torch
from pathlib import Path

from fhe_common import PHANTOM_PATH  # noqa: F401

import pyPhantom as phantom

PROJECT_DIR = Path(__file__).parent
_MODELS_DIR = PROJECT_DIR / "models"
_G1D = _MODELS_DIR / "rwkv7-g1d-1.5b-20260212-ctx8192"
_G1C = _MODELS_DIR / "rwkv7-g1c-1.5b-20260110-ctx8192"
MODEL_PATH = _G1D if (_G1D.parent / (_G1D.name + ".pth")).exists() else _G1C


def load_weights(model_path=MODEL_PATH):
    raw = torch.load(str(model_path) + '.pth', map_location='cpu', mmap=True)
    w = {}
    for k, v in raw.items():
        tensor = v.float().squeeze()
        if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
            tensor = tensor.t().contiguous()
        w[k] = tensor
    return w


class CKKSContext:
    def __init__(self, poly_modulus_degree=32768, depth=9, prime_bits=40):
        bit_sizes = [60] + [prime_bits] * depth + [60]
        params = phantom.params(phantom.scheme_type.ckks)
        params.set_poly_modulus_degree(poly_modulus_degree)
        params.set_coeff_modulus(phantom.create_coeff_modulus(poly_modulus_degree, bit_sizes))
        params.set_special_modulus_size(1)

        self.ctx = phantom.context(params)
        self.sk = phantom.secret_key(self.ctx)
        self.pk = self.sk.gen_publickey(self.ctx)
        self.rlk = self.sk.gen_relinkey(self.ctx)
        self.gk = self.sk.create_galois_keys(self.ctx)
        self.encoder = phantom.ckks_encoder(self.ctx)
        self.scale = 2.0 ** prime_bits
        self.slots = self.encoder.slot_count()
        self.depth = depth

    def encrypt(self, vec):
        padded = list(vec) + [0.0] * (self.slots - len(vec))
        pt = self.encoder.encode_double_vector(self.ctx, padded, self.scale)
        return self.pk.encrypt_asymmetric(self.ctx, pt)

    def decrypt_slot0(self, ct):
        pt = self.sk.decrypt(self.ctx, ct)
        return self.encoder.decode_double_vector(self.ctx, pt)[0]


def normalize_columns(W):
    W = W.copy()
    for j in range(W.shape[1]):
        s = np.std(W[:, j])
        if s > 1e-6:
            W[:, j] /= s
    return W


def ct_pt_dot(ckks, ct, weights, dim):
    w_padded = list(weights) + [0.0] * (ckks.slots - dim)
    w_pt = ckks.encoder.encode_double_vector(ckks.ctx, w_padded, ckks.scale)
    prod = phantom.multiply_plain(ckks.ctx, ct, w_pt)
    prod = phantom.rescale_to_next(ckks.ctx, prod)
    step = 1
    while step < dim:
        rotated = phantom.rotate(ckks.ctx, prod, step, ckks.gk)
        prod = phantom.add(ckks.ctx, prod, rotated)
        step *= 2
    return prod


def ct_pt_weighted_sum(ckks, ct_list, weights, level):
    w_const = [weights[0]] * ckks.slots
    w_pt = ckks.encoder.encode_double_vector(ckks.ctx, w_const, ckks.scale)
    w_pt = phantom.mod_switch_to(ckks.ctx, w_pt, level)
    result = phantom.multiply_plain(ckks.ctx, ct_list[0], w_pt)
    result = phantom.rescale_to_next(ckks.ctx, result)

    for j in range(1, len(ct_list)):
        w_const = [weights[j]] * ckks.slots
        w_pt = ckks.encoder.encode_double_vector(ckks.ctx, w_const, ckks.scale)
        w_pt = phantom.mod_switch_to(ckks.ctx, w_pt, level)
        term = phantom.multiply_plain(ckks.ctx, ct_list[j], w_pt)
        term = phantom.rescale_to_next(ckks.ctx, term)
        result = phantom.add(ckks.ctx, result, term)

    return result


def ct_ct_square(ckks, ct):
    sq = phantom.multiply(ckks.ctx, ct, ct)
    sq = phantom.relinearize(ckks.ctx, sq, ckks.rlk)
    sq = phantom.rescale_to_next(ckks.ctx, sq)
    return sq


def ct_ct_multiply(ckks, ct1, ct2):
    prod = phantom.multiply(ckks.ctx, ct1, ct2)
    prod = phantom.relinearize(ckks.ctx, prod, ckks.rlk)
    prod = phantom.rescale_to_next(ckks.ctx, prod)
    return prod


def run_inference(embed_dim, ffn_dim, vocab_dim, ckks, w=None):
    print(f"\n{embed_dim}x{ffn_dim}x{vocab_dim}")

    if w is None:
        w = load_weights()

    emb = w['emb.weight'][:vocab_dim, :embed_dim].numpy().astype(np.float64) * 10.0
    W_key = normalize_columns(w['blocks.0.ffn.key.weight'][:embed_dim, :ffn_dim].numpy().astype(np.float64))
    W_val = normalize_columns(w['blocks.0.ffn.value.weight'][:ffn_dim, :embed_dim].numpy().astype(np.float64))
    W_head = normalize_columns(w['head.weight'][:embed_dim, :vocab_dim].numpy().astype(np.float64))

    x = emb[3]
    k = x @ W_key
    k_sq = k ** 2
    v = k_sq @ W_val
    logits_ref = v @ W_head
    token_ref = int(np.argmax(logits_ref))

    t0 = time.perf_counter()
    ct_x = ckks.encrypt(x)

    t1 = time.perf_counter()
    ct_k = [ct_pt_dot(ckks, ct_x, W_key[:, j], embed_dim) for j in range(ffn_dim)]
    t_key = time.perf_counter() - t1

    t1 = time.perf_counter()
    ct_k_sq = [ct_ct_square(ckks, ct_k[j]) for j in range(ffn_dim)]
    del ct_k
    t_sq = time.perf_counter() - t1

    t1 = time.perf_counter()
    ct_v = []
    for i in range(embed_dim):
        ct_v.append(ct_pt_weighted_sum(ckks, ct_k_sq, W_val[:, i], level=3))
        if (i + 1) % 100 == 0:
            print(f"  value: {i+1}/{embed_dim}")
    del ct_k_sq
    t_val = time.perf_counter() - t1

    t1 = time.perf_counter()
    ct_logits = [ct_pt_weighted_sum(ckks, ct_v, W_head[:, i], level=4) for i in range(vocab_dim)]
    t_head = time.perf_counter() - t1

    t_total = time.perf_counter() - t0

    logits_fhe = np.array([ckks.decrypt_slot0(ct) for ct in ct_logits])
    token_fhe = int(np.argmax(logits_fhe))

    corr = np.corrcoef(logits_fhe, logits_ref)[0, 1]
    match = token_ref == token_fhe

    print(f"Plaintext: {token_ref}, FHE: {token_fhe}")
    print(f"Correlation: {corr:.6f}")
    print(f"Time: {t_total:.1f}s (key={t_key:.1f}, sq={t_sq:.1f}, val={t_val:.1f}, head={t_head:.1f})")

    return match, corr, t_total


def run_multilayer_inference(embed_dim, ffn_dim, vocab_dim, num_blocks, ckks, w=None):
    rescales_needed = 3 * num_blocks + 1
    print(f"\n{num_blocks}x {embed_dim}x{ffn_dim}x{vocab_dim}, depth {rescales_needed}/{ckks.depth}")

    if rescales_needed > ckks.depth:
        print(f"ERROR: need {rescales_needed} rescales but depth={ckks.depth}")
        return False, 0.0, 0.0

    if w is None:
        w = load_weights()

    emb = w['emb.weight'][:vocab_dim, :embed_dim].numpy().astype(np.float64) * 10.0

    block_weights = []
    for b in range(num_blocks):
        W_key = normalize_columns(w[f'blocks.{b}.ffn.key.weight'][:embed_dim, :ffn_dim].numpy().astype(np.float64))
        W_val = normalize_columns(w[f'blocks.{b}.ffn.value.weight'][:ffn_dim, :embed_dim].numpy().astype(np.float64))
        block_weights.append((W_key, W_val))

    W_head = normalize_columns(w['head.weight'][:embed_dim, :vocab_dim].numpy().astype(np.float64))

    x = emb[3]
    h = x.copy()
    for W_key, W_val in block_weights:
        k = h @ W_key
        k_sq = k ** 2
        h = k_sq @ W_val
    logits_ref = h @ W_head
    token_ref = int(np.argmax(logits_ref))

    t0 = time.perf_counter()
    rescales = 0

    ct_x = ckks.encrypt(x)
    W_key, W_val = block_weights[0]

    t1 = time.perf_counter()
    ct_k = [ct_pt_dot(ckks, ct_x, W_key[:, j], embed_dim) for j in range(ffn_dim)]
    rescales += 1
    del ct_x
    t_key = time.perf_counter() - t1

    t1 = time.perf_counter()
    ct_k_sq = [ct_ct_square(ckks, ct_k[j]) for j in range(ffn_dim)]
    del ct_k
    rescales += 1
    t_sq = time.perf_counter() - t1

    t1 = time.perf_counter()
    level = rescales + 1
    ct_h = []
    for i in range(embed_dim):
        ct_h.append(ct_pt_weighted_sum(ckks, ct_k_sq, W_val[:, i], level=level))
        if (i + 1) % 100 == 0:
            print(f"    block 0 value: {i+1}/{embed_dim}")
    del ct_k_sq
    rescales += 1
    t_val = time.perf_counter() - t1

    print(f"  Block 0: key={t_key:.1f}s sq={t_sq:.1f}s val={t_val:.1f}s [rescales={rescales}]")

    for b in range(1, num_blocks):
        W_key, W_val = block_weights[b]

        t1 = time.perf_counter()
        level = rescales + 1
        ct_k = []
        for j in range(ffn_dim):
            ct_k.append(ct_pt_weighted_sum(ckks, ct_h, W_key[:, j], level=level))
            if (j + 1) % 100 == 0:
                print(f"    block {b} key: {j+1}/{ffn_dim}")
        rescales += 1
        t_key = time.perf_counter() - t1

        t1 = time.perf_counter()
        ct_k_sq = [ct_ct_square(ckks, ct_k[j]) for j in range(ffn_dim)]
        del ct_k
        rescales += 1
        t_sq = time.perf_counter() - t1

        t1 = time.perf_counter()
        level = rescales + 1
        ct_h_new = []
        for i in range(embed_dim):
            ct_h_new.append(ct_pt_weighted_sum(ckks, ct_k_sq, W_val[:, i], level=level))
            if (i + 1) % 100 == 0:
                print(f"    block {b} value: {i+1}/{embed_dim}")
        del ct_k_sq
        rescales += 1
        t_val = time.perf_counter() - t1

        del ct_h
        ct_h = ct_h_new
        print(f"  Block {b}: key={t_key:.1f}s sq={t_sq:.1f}s val={t_val:.1f}s [rescales={rescales}]")

    t1 = time.perf_counter()
    level = rescales + 1
    ct_logits = [ct_pt_weighted_sum(ckks, ct_h, W_head[:, i], level=level) for i in range(vocab_dim)]
    del ct_h
    rescales += 1
    t_head = time.perf_counter() - t1

    t_total = time.perf_counter() - t0

    logits_fhe = np.array([ckks.decrypt_slot0(ct) for ct in ct_logits])
    token_fhe = int(np.argmax(logits_fhe))

    corr = np.corrcoef(logits_fhe, logits_ref)[0, 1]
    match = token_ref == token_fhe

    print(f"  Head: {t_head:.1f}s")
    print(f"Plaintext: {token_ref}, FHE: {token_fhe}, Match: {'Y' if match else 'N'}")
    print(f"Correlation: {corr:.6f}")
    print(f"Time: {t_total:.1f}s, Rescales used: {rescales}/{ckks.depth}")

    return match, corr, t_total


def ct_mod_switch_down(ckks, ct, num_levels):
    result = ct
    for _ in range(num_levels):
        result = phantom.mod_switch_to_next(ckks.ctx, result)
    return result


def run_multilayer_residual_inference(embed_dim, ffn_dim, vocab_dim, num_blocks, ckks, w=None):
    rescales_needed = 3 * num_blocks + 1
    print(f"\n{num_blocks}x+res {embed_dim}x{ffn_dim}x{vocab_dim}, depth {rescales_needed}/{ckks.depth}")

    if rescales_needed > ckks.depth:
        print(f"ERROR: need {rescales_needed} rescales but depth={ckks.depth}")
        return False, 0.0, 0.0

    if w is None:
        w = load_weights()

    emb = w['emb.weight'][:vocab_dim, :embed_dim].numpy().astype(np.float64) * 10.0

    block_weights = []
    for b in range(num_blocks):
        W_key = normalize_columns(w[f'blocks.{b}.ffn.key.weight'][:embed_dim, :ffn_dim].numpy().astype(np.float64))
        W_val = normalize_columns(w[f'blocks.{b}.ffn.value.weight'][:ffn_dim, :embed_dim].numpy().astype(np.float64))
        block_weights.append((W_key, W_val))

    W_head = normalize_columns(w['head.weight'][:embed_dim, :vocab_dim].numpy().astype(np.float64))

    x = emb[3]
    h = x.copy()
    W_key, W_val = block_weights[0]
    k = h @ W_key; k_sq = k ** 2; h = k_sq @ W_val
    for W_key, W_val in block_weights[1:]:
        h_in = h.copy()
        k = h @ W_key; k_sq = k ** 2; h = k_sq @ W_val
        h = h_in + h
    logits_ref = h @ W_head
    token_ref = int(np.argmax(logits_ref))

    t0 = time.perf_counter()
    rescales = 0

    ct_x = ckks.encrypt(x)
    W_key, W_val = block_weights[0]

    ct_k = [ct_pt_dot(ckks, ct_x, W_key[:, j], embed_dim) for j in range(ffn_dim)]
    rescales += 1
    del ct_x

    ct_k_sq = [ct_ct_square(ckks, ct_k[j]) for j in range(ffn_dim)]
    del ct_k
    rescales += 1

    level = rescales + 1
    ct_h = []
    for i in range(embed_dim):
        ct_h.append(ct_pt_weighted_sum(ckks, ct_k_sq, W_val[:, i], level=level))
    del ct_k_sq
    rescales += 1
    print(f"  Block 0: done [rescales={rescales}]")

    for b in range(1, num_blocks):
        W_key, W_val = block_weights[b]
        ct_h_in = ct_h
        rescales_at_input = rescales

        level = rescales + 1
        ct_k = [ct_pt_weighted_sum(ckks, ct_h, W_key[:, j], level=level) for j in range(ffn_dim)]
        rescales += 1

        ct_k_sq = [ct_ct_square(ckks, ct_k[j]) for j in range(ffn_dim)]
        del ct_k
        rescales += 1

        level = rescales + 1
        ct_h_new = []
        for i in range(embed_dim):
            ct_h_new.append(ct_pt_weighted_sum(ckks, ct_k_sq, W_val[:, i], level=level))
        del ct_k_sq
        rescales += 1

        levels_to_drop = rescales - rescales_at_input
        ct_h = []
        for i in range(embed_dim):
            ct_res = ct_mod_switch_down(ckks, ct_h_in[i], levels_to_drop)
            ct_res.set_scale(ckks.scale)
            ct_h_new[i].set_scale(ckks.scale)
            ct_h.append(phantom.add(ckks.ctx, ct_res, ct_h_new[i]))
        del ct_h_in, ct_h_new
        print(f"  Block {b}: done +residual [rescales={rescales}]")

    level = rescales + 1
    ct_logits = [ct_pt_weighted_sum(ckks, ct_h, W_head[:, i], level=level) for i in range(vocab_dim)]
    del ct_h
    rescales += 1

    t_total = time.perf_counter() - t0

    logits_fhe = np.array([ckks.decrypt_slot0(ct) for ct in ct_logits])
    token_fhe = int(np.argmax(logits_fhe))

    corr = np.corrcoef(logits_fhe, logits_ref)[0, 1]
    match = token_ref == token_fhe

    print(f"Plaintext: {token_ref}, FHE: {token_fhe}, Match: {'Y' if match else 'N'}")
    print(f"Correlation: {corr:.6f}")
    print(f"Time: {t_total:.1f}s, Rescales used: {rescales}/{ckks.depth}")

    return match, corr, t_total


def run_autoregressive(embed_dim, ffn_dim, vocab_dim, num_blocks, num_tokens, ckks, w=None):
    if w is None:
        w = load_weights()

    emb = w['emb.weight'][:vocab_dim, :embed_dim].numpy().astype(np.float64) * 10.0
    block_weights = []
    for b in range(num_blocks):
        W_key = normalize_columns(w[f'blocks.{b}.ffn.key.weight'][:embed_dim, :ffn_dim].numpy().astype(np.float64))
        W_val = normalize_columns(w[f'blocks.{b}.ffn.value.weight'][:ffn_dim, :embed_dim].numpy().astype(np.float64))
        block_weights.append((W_key, W_val))
    W_head = normalize_columns(w['head.weight'][:embed_dim, :vocab_dim].numpy().astype(np.float64))

    print(f"\nautoregressive {num_blocks}x {embed_dim}x{ffn_dim}x{vocab_dim}, {num_tokens} tokens")

    token_plain = 3
    token_fhe = 3
    tokens_plain = [token_plain]
    tokens_fhe = [token_fhe]
    total_time = 0.0

    for step in range(num_tokens):
        x_ref = emb[token_plain]
        h = x_ref.copy()
        for W_key, W_val in block_weights:
            k = h @ W_key; k_sq = k ** 2; h = k_sq @ W_val
        logits_ref = h @ W_head
        token_plain = int(np.argmax(logits_ref))
        tokens_plain.append(token_plain)

        t0 = time.perf_counter()
        x_fhe = emb[token_fhe]
        rescales = 0

        ct_x = ckks.encrypt(x_fhe)
        W_key, W_val = block_weights[0]
        ct_k = [ct_pt_dot(ckks, ct_x, W_key[:, j], embed_dim) for j in range(ffn_dim)]
        rescales += 1
        del ct_x
        ct_k_sq = [ct_ct_square(ckks, ct_k[j]) for j in range(ffn_dim)]
        del ct_k
        rescales += 1
        level = rescales + 1
        ct_h = [ct_pt_weighted_sum(ckks, ct_k_sq, W_val[:, i], level=level) for i in range(embed_dim)]
        del ct_k_sq
        rescales += 1

        for b in range(1, num_blocks):
            W_key, W_val = block_weights[b]
            level = rescales + 1
            ct_k = [ct_pt_weighted_sum(ckks, ct_h, W_key[:, j], level=level) for j in range(ffn_dim)]
            rescales += 1
            ct_k_sq = [ct_ct_square(ckks, ct_k[j]) for j in range(ffn_dim)]
            del ct_k
            rescales += 1
            level = rescales + 1
            ct_h_new = [ct_pt_weighted_sum(ckks, ct_k_sq, W_val[:, i], level=level) for i in range(embed_dim)]
            del ct_k_sq
            rescales += 1
            del ct_h
            ct_h = ct_h_new

        level = rescales + 1
        ct_logits = [ct_pt_weighted_sum(ckks, ct_h, W_head[:, i], level=level) for i in range(vocab_dim)]
        del ct_h

        t_step = time.perf_counter() - t0
        total_time += t_step

        logits_fhe = np.array([ckks.decrypt_slot0(ct) for ct in ct_logits])
        token_fhe = int(np.argmax(logits_fhe))
        tokens_fhe.append(token_fhe)

        match = tokens_plain[-1] == tokens_fhe[-1]
        corr = np.corrcoef(logits_fhe, logits_ref)[0, 1]
        print(f"  Token {step}: plain={tokens_plain[-1]}, fhe={tokens_fhe[-1]}, "
              f"match={'Y' if match else 'N'}, corr={corr:.6f}, time={t_step:.1f}s")

    all_match = tokens_plain == tokens_fhe
    print(f"Tokens plain: {tokens_plain}")
    print(f"Tokens FHE:   {tokens_fhe}")
    print(f"All match: {all_match}, Total time: {total_time:.1f}s")

    return all_match, total_time


def magnitude_controlled_weights(w, embed_dim, ffn_dim, vocab_dim, num_blocks,
                                  target_mag=10.0, seed_token=3):
    emb = w['emb.weight'][:vocab_dim, :embed_dim].numpy().astype(np.float64) * 10.0
    x = emb[seed_token]

    block_weights = []
    h = x.copy()
    for b in range(num_blocks):
        W_key = normalize_columns(
            w[f'blocks.{b}.ffn.key.weight'][:embed_dim, :ffn_dim]
            .numpy().astype(np.float64))
        W_val = normalize_columns(
            w[f'blocks.{b}.ffn.value.weight'][:ffn_dim, :embed_dim]
            .numpy().astype(np.float64))

        k = h @ W_key
        k_sq = k ** 2
        h_out = k_sq @ W_val

        scale = target_mag / (np.max(np.abs(h_out)) + 1e-30)
        W_val_scaled = W_val * scale
        h = k_sq @ W_val_scaled
        block_weights.append((W_key, W_val_scaled))

    W_head = normalize_columns(
        w['head.weight'][:embed_dim, :vocab_dim]
        .numpy().astype(np.float64))
    return emb, block_weights, W_head


def run_deep_inference(embed_dim, ffn_dim, vocab_dim, num_blocks, ckks, w=None):
    rescales_needed = 3 * num_blocks + 1
    print(f"deep {embed_dim}x{ffn_dim}x{vocab_dim}, {num_blocks} blocks, rescales={rescales_needed}/{ckks.depth}")

    if rescales_needed > ckks.depth:
        print(f"ERROR: need {rescales_needed} rescales but depth={ckks.depth}")
        return False, 0.0, 0.0

    if w is None:
        w = load_weights()

    emb, block_weights, W_head = magnitude_controlled_weights(
        w, embed_dim, ffn_dim, vocab_dim, num_blocks)

    x = emb[3]
    h = x.copy()
    for W_key, W_val in block_weights:
        k = h @ W_key
        k_sq = k ** 2
        h = k_sq @ W_val
    logits_ref = h @ W_head
    token_ref = int(np.argmax(logits_ref))

    t0 = time.perf_counter()
    rescales = 0

    ct_x = ckks.encrypt(x)
    W_key, W_val = block_weights[0]
    ct_k = [ct_pt_dot(ckks, ct_x, W_key[:, j], embed_dim) for j in range(ffn_dim)]
    rescales += 1
    del ct_x
    ct_k_sq = [ct_ct_square(ckks, ct_k[j]) for j in range(ffn_dim)]
    del ct_k
    rescales += 1
    level = rescales + 1
    ct_h = [ct_pt_weighted_sum(ckks, ct_k_sq, W_val[:, i], level=level)
            for i in range(embed_dim)]
    del ct_k_sq
    rescales += 1
    print(f"  Block 0: done [rescales={rescales}]")

    for b in range(1, num_blocks):
        W_key, W_val = block_weights[b]
        level = rescales + 1
        ct_k = [ct_pt_weighted_sum(ckks, ct_h, W_key[:, j], level=level)
                for j in range(ffn_dim)]
        rescales += 1
        ct_k_sq = [ct_ct_square(ckks, ct_k[j]) for j in range(ffn_dim)]
        del ct_k
        rescales += 1
        level = rescales + 1
        ct_h_new = [ct_pt_weighted_sum(ckks, ct_k_sq, W_val[:, i], level=level)
                    for i in range(embed_dim)]
        del ct_k_sq
        rescales += 1
        del ct_h
        ct_h = ct_h_new
        print(f"  Block {b}: done [rescales={rescales}]")

    level = rescales + 1
    ct_logits = [ct_pt_weighted_sum(ckks, ct_h, W_head[:, i], level=level)
                 for i in range(vocab_dim)]
    del ct_h
    rescales += 1

    t_total = time.perf_counter() - t0

    logits_fhe = np.array([ckks.decrypt_slot0(ct) for ct in ct_logits])
    token_fhe = int(np.argmax(logits_fhe))
    corr = np.corrcoef(logits_fhe, logits_ref)[0, 1]
    match = token_ref == token_fhe

    print(f"Plaintext: {token_ref}, FHE: {token_fhe}, Match: {'Y' if match else 'N'}")
    print(f"Correlation: {corr:.6f}")
    print(f"Time: {t_total:.1f}s, Rescales: {rescales}/{ckks.depth}")

    return match, corr, t_total


def fullscale_weights(w, embed_dim, ffn_dim, num_blocks,
                      target_mag=10.0, seed_token=3):
    vocab_dim = w['emb.weight'].shape[0]
    emb = w['emb.weight'][:vocab_dim, :embed_dim].numpy().astype(np.float64) * 10.0
    x = emb[seed_token]

    block_weights = []
    h = x.copy()
    for b in range(num_blocks):
        W_key = normalize_columns(
            w[f'blocks.{b}.ffn.key.weight'][:embed_dim, :ffn_dim]
            .numpy().astype(np.float64))
        W_val = normalize_columns(
            w[f'blocks.{b}.ffn.value.weight'][:ffn_dim, :embed_dim]
            .numpy().astype(np.float64))

        k = h @ W_key
        k_sq = k ** 2
        h_out = k_sq @ W_val
        scale = target_mag / (np.max(np.abs(h_out)) + 1e-30)
        W_val_scaled = W_val * scale
        h = k_sq @ W_val_scaled
        block_weights.append((W_key, W_val_scaled))

    W_head = normalize_columns(
        w['head.weight'][:embed_dim, :vocab_dim].numpy().astype(np.float64))
    print(f"  Weights: {embed_dim}x{ffn_dim}x{vocab_dim}, {num_blocks} blocks, mag_ctrl")

    return emb, block_weights, W_head


def run_fullscale_inference(embed_dim, ffn_dim, num_blocks, ckks, w=None):
    if w is None:
        w = load_weights()

    vocab_dim = w['emb.weight'].shape[0]
    rescales_needed = 3 * num_blocks
    print(f"fullscale {embed_dim}x{ffn_dim}x{vocab_dim}, {num_blocks} blocks, rescales={rescales_needed}/{ckks.depth}")

    if rescales_needed > ckks.depth:
        print(f"ERROR: need {rescales_needed} rescales but depth={ckks.depth}")
        return False, 0.0, 0.0

    emb, block_weights, W_head = fullscale_weights(
        w, embed_dim, ffn_dim, num_blocks)

    x = emb[3]
    h = x.copy()
    for W_key, W_val in block_weights:
        k = h @ W_key; k_sq = k ** 2; h = k_sq @ W_val
    logits_ref = h @ W_head
    token_ref = int(np.argmax(logits_ref))
    print(f"  plaintext token: {token_ref}")

    batch_size = min(ffn_dim, 1024)
    t0 = time.perf_counter()
    rescales = 0

    ct_x = ckks.encrypt(x)
    W_key, W_val = block_weights[0]

    ct_h = None
    for bs in range(0, ffn_dim, batch_size):
        be = min(bs + batch_size, ffn_dim)
        ct_k_sq_batch = []
        for j in range(bs, be):
            ct_kj = ct_pt_dot(ckks, ct_x, W_key[:, j], embed_dim)
            ct_k_sq_batch.append(ct_ct_square(ckks, ct_kj))
            del ct_kj
        level_val = 3
        for i in range(embed_dim):
            partial = ct_pt_weighted_sum(ckks, ct_k_sq_batch,
                                         W_val[bs:be, i], level=level_val)
            if ct_h is None:
                ct_h = [partial]
            elif bs == 0:
                ct_h.append(partial)
            else:
                ct_h[i] = phantom.add(ckks.ctx, ct_h[i], partial)
        del ct_k_sq_batch
        print(f"    Block 0 batch {bs}-{be}/{ffn_dim} {time.perf_counter()-t0:.0f}s")
    del ct_x
    rescales = 3
    print(f"  Block 0: done [{rescales}] {time.perf_counter()-t0:.0f}s")

    for b in range(1, num_blocks):
        W_key, W_val = block_weights[b]
        ct_h_new = None
        for bs in range(0, ffn_dim, batch_size):
            be = min(bs + batch_size, ffn_dim)
            level_key = rescales + 1
            ct_k_sq_batch = []
            for j in range(bs, be):
                ct_kj = ct_pt_weighted_sum(ckks, ct_h, W_key[:, j], level=level_key)
                ct_k_sq_batch.append(ct_ct_square(ckks, ct_kj))
                del ct_kj
            level_val = rescales + 3
            for i in range(embed_dim):
                partial = ct_pt_weighted_sum(ckks, ct_k_sq_batch,
                                             W_val[bs:be, i], level=level_val)
                if ct_h_new is None:
                    ct_h_new = [partial]
                elif bs == 0:
                    ct_h_new.append(partial)
                else:
                    ct_h_new[i] = phantom.add(ckks.ctx, ct_h_new[i], partial)
            del ct_k_sq_batch
            print(f"    Block {b} batch {bs}-{be}/{ffn_dim} {time.perf_counter()-t0:.0f}s")
        del ct_h
        ct_h = ct_h_new
        rescales += 3
        print(f"  Block {b}: done [{rescales}] {time.perf_counter()-t0:.0f}s")

    t_fhe = time.perf_counter() - t0

    t_client = time.perf_counter()
    h_fhe = np.array([ckks.decrypt_slot0(ct) for ct in ct_h])
    del ct_h
    logits_fhe = h_fhe @ W_head
    token_fhe = int(np.argmax(logits_fhe))
    t_client_done = time.perf_counter() - t_client
    t_total = time.perf_counter() - t0

    corr = np.corrcoef(logits_fhe, logits_ref)[0, 1]
    match = token_ref == token_fhe

    print(f"Plaintext: {token_ref}, FHE: {token_fhe}, Match: {'Y' if match else 'N'}")
    print(f"Correlation: {corr:.6f}")
    print(f"Time: {t_total:.1f}s (fhe={t_fhe:.1f}s, client={t_client_done:.1f}s)")
    print(f"Rescales: {rescales}/{ckks.depth}")
    print(f"fullscale {num_blocks}x{embed_dim}x{ffn_dim}x{vocab_dim}: "
          f"match={'Y' if match else 'N'}, corr={corr:.6f}, time={t_total:.1f}s")

    return match, corr, t_total


def run_timemix_ffn_inference(embed_dim, ffn_dim, vocab_dim, ckks, w=None):
    # 7 rescales: 4 for time-mix + 3 for FFN
    print(f"\ntimemix+ffn {embed_dim}x{ffn_dim}x{vocab_dim}, depth={ckks.depth}")

    if w is None:
        w = load_weights()

    emb = w['emb.weight'][:vocab_dim, :embed_dim].numpy().astype(np.float64) * 10.0

    W_r = normalize_columns(
        w['blocks.0.att.receptance.weight'][:embed_dim, :embed_dim]
        .numpy().astype(np.float64))
    W_k_att = normalize_columns(
        w['blocks.0.att.key.weight'][:embed_dim, :embed_dim]
        .numpy().astype(np.float64))
    W_v_att = normalize_columns(
        w['blocks.0.att.value.weight'][:embed_dim, :embed_dim]
        .numpy().astype(np.float64))
    W_o = normalize_columns(
        w['blocks.0.att.output.weight'][:embed_dim, :embed_dim]
        .numpy().astype(np.float64))

    # sigmoid(x) ~ 0.25x + 0.5, absorb 0.25 into W_r
    W_r_scaled = W_r * 0.25

    W_key = normalize_columns(
        w['blocks.0.ffn.key.weight'][:embed_dim, :ffn_dim]
        .numpy().astype(np.float64))
    W_val = normalize_columns(
        w['blocks.0.ffn.value.weight'][:ffn_dim, :embed_dim]
        .numpy().astype(np.float64))

    W_head = normalize_columns(
        w['head.weight'][:embed_dim, :vocab_dim]
        .numpy().astype(np.float64))

    test_tokens = [0, 3, 7]
    results = []

    for seed_token in test_tokens:
        x = emb[seed_token]

        # Plaintext reference
        r = x @ W_r_scaled + 0.5
        k_att = x @ W_k_att
        v_att = x @ W_v_att
        gated = r * k_att * v_att
        tm_out = gated @ W_o
        fk = tm_out @ W_key
        fk_sq = fk ** 2
        ffn_out = fk_sq @ W_val
        logits_ref = ffn_out @ W_head
        token_ref = int(np.argmax(logits_ref))

        t0 = time.perf_counter()

        ct_x = ckks.encrypt(x)

        t1 = time.perf_counter()
        ct_r = [ct_pt_dot(ckks, ct_x, W_r_scaled[:, j], embed_dim)
                for j in range(embed_dim)]
        ct_k_att = [ct_pt_dot(ckks, ct_x, W_k_att[:, j], embed_dim)
                    for j in range(embed_dim)]
        ct_v_att = [ct_pt_dot(ckks, ct_x, W_v_att[:, j], embed_dim)
                    for j in range(embed_dim)]
        del ct_x
        rescales = 1
        t_proj = time.perf_counter() - t1

        # sigmoid bias (+0.5)
        has_bias = False
        try:
            bias_vec = [0.5] * ckks.slots
            bias_pt = ckks.encoder.encode_double_vector(
                ckks.ctx, bias_vec, ckks.scale)
            bias_pt = phantom.mod_switch_to(ckks.ctx, bias_pt, 2)
            for j in range(embed_dim):
                ct_r[j].set_scale(ckks.scale)
                ct_r[j] = phantom.add_plain(ckks.ctx, ct_r[j], bias_pt)
            has_bias = True
        except Exception as e:
            print(f"  (sigmoid bias skipped: {e})")

        t1 = time.perf_counter()
        for j in range(embed_dim):
            ct_r[j].set_scale(ckks.scale)
            ct_k_att[j].set_scale(ckks.scale)
        ct_rk = [ct_ct_multiply(ckks, ct_r[j], ct_k_att[j])
                 for j in range(embed_dim)]
        del ct_r, ct_k_att
        rescales = 2
        t_rk = time.perf_counter() - t1

        # mod_switch v to match multiply depth
        t1 = time.perf_counter()
        for j in range(embed_dim):
            ct_v_att[j] = phantom.mod_switch_to_next(ckks.ctx, ct_v_att[j])
            ct_v_att[j].set_scale(ckks.scale)
            ct_rk[j].set_scale(ckks.scale)

        ct_rkv = [ct_ct_multiply(ckks, ct_rk[j], ct_v_att[j])
                  for j in range(embed_dim)]
        del ct_rk, ct_v_att
        rescales = 3
        t_rkv = time.perf_counter() - t1

        t1 = time.perf_counter()
        level = rescales + 1
        ct_tm = [ct_pt_weighted_sum(ckks, ct_rkv, W_o[:, i], level=level)
                 for i in range(embed_dim)]
        del ct_rkv
        rescales = 4
        t_wo = time.perf_counter() - t1

        t1 = time.perf_counter()
        level = rescales + 1
        ct_fk = [ct_pt_weighted_sum(ckks, ct_tm, W_key[:, j], level=level)
                 for j in range(ffn_dim)]
        del ct_tm
        rescales = 5
        t_fk = time.perf_counter() - t1

        t1 = time.perf_counter()
        ct_fk_sq = [ct_ct_square(ckks, ct_fk[j]) for j in range(ffn_dim)]
        del ct_fk
        rescales = 6
        t_sq = time.perf_counter() - t1

        t1 = time.perf_counter()
        level = rescales + 1
        ct_ffn = [ct_pt_weighted_sum(ckks, ct_fk_sq, W_val[:, i], level=level)
                  for i in range(embed_dim)]
        del ct_fk_sq
        rescales = 7
        t_fv = time.perf_counter() - t1

        t_fhe = time.perf_counter() - t0

        h_fhe = np.array([ckks.decrypt_slot0(ct) for ct in ct_ffn])
        del ct_ffn
        logits_fhe = h_fhe @ W_head
        token_fhe = int(np.argmax(logits_fhe))

        t_total = time.perf_counter() - t0

        corr = np.corrcoef(logits_fhe, logits_ref)[0, 1]
        match = token_ref == token_fhe

        print(f"  Token {seed_token}: plain={token_ref}, fhe={token_fhe}, "
              f"match={'Y' if match else 'N'}, corr={corr:.6f}, "
              f"rescales={rescales}, time={t_total:.1f}s"
              f"{' +bias' if has_bias else ''}")
        print(f"    proj={t_proj:.1f}s r*k={t_rk:.1f}s (r*k)*v={t_rkv:.1f}s "
              f"Wo={t_wo:.1f}s fk={t_fk:.1f}s sq={t_sq:.1f}s fv={t_fv:.1f}s")

        results.append((seed_token, match, corr, t_total))

    all_match = all(m for _, m, _, _ in results)
    print(f"  All tokens match: {all_match}")
    return results


def main():
    print("fhe_rwkv benchmark")

    print("Loading weights...")
    w = load_weights()

    ckks = CKKSContext(depth=9)
    print(f"\nN=32768, slots={ckks.slots}, depth=9")

    single_configs = [
        (64, 128, 32),
        (128, 256, 64),
        (256, 512, 64),
        (512, 1024, 64),
        (1024, 2048, 64),
    ]

    results = []
    for e, f, v in single_configs:
        try:
            match, corr, t = run_inference(e, f, v, ckks, w=w)
            results.append((f"1x{e}x{f}x{v}", corr, match, t))
        except Exception as ex:
            print(f"Error ({e}x{f}x{v}): {ex}")
            break

    multi_configs = [
        (64, 128, 32, 2),
        (128, 256, 64, 2),
        (64, 128, 32, 3),
        (128, 256, 64, 3),
    ]

    ckks_cache = {9: ckks}
    for e, f, v, nb in multi_configs:
        depth_needed = 3 * nb + 2
        if depth_needed not in ckks_cache:
            print(f"\nInitializing CKKS with depth={depth_needed}")
            ckks_cache[depth_needed] = CKKSContext(depth=depth_needed)
        ctx = ckks_cache[depth_needed] if depth_needed > 9 else ckks

        try:
            match, corr, t = run_multilayer_inference(e, f, v, nb, ctx, w=w)
            results.append((f"{nb}x{e}x{f}x{v}", corr, match, t))
        except Exception as ex:
            print(f"Error ({nb}x{e}x{f}x{v}): {ex}")

    print("\n--- Time-mixing + FFN ---")
    try:
        run_timemix_ffn_inference(64, 128, 32, ckks, w=w)
    except Exception as ex:
        print(f"Error (timemix): {ex}")
        import traceback; traceback.print_exc()

    print("\nResults:")
    for cfg, corr, match, t in results:
        print(f"{cfg:<20} corr={corr:.4f} match={'Y' if match else 'N'} t={t:.1f}s")


if __name__ == "__main__":
    main()
