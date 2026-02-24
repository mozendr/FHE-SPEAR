#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fhe_common import PHANTOM_PATH  # noqa: F401
import pyPhantom as ph

from fhe_rwkv_inference import load_weights

def compute_rotation_galois_elements(poly_degree, max_dim):
    M = 2 * poly_degree
    elts = set()
    elts.add(M - 1)
    step = 1
    while step <= max_dim:
        elts.add(pow(5, step, M))
        step *= 2
    return list(elts)


def compute_bsgs_params(D):
    G = int(np.ceil(np.sqrt(D)))
    B = int(np.ceil(D / G))
    return G, B


def compute_bsgs_galois_elements(poly_degree, D):
    G, B = compute_bsgs_params(D)
    steps = []
    for b in range(1, G):
        steps.append(b)
    for g in range(1, B):
        steps.append(g * G)
    return ph.get_elts_from_steps(steps, poly_degree)


def compute_diagonals(W, D):
    diags = []
    for k in range(D):
        d = np.array([W[j, (j + k) % D] for j in range(D)])
        diags.append(d)
    return diags


def replicate_vector(vec, slots):
    D = len(vec)
    reps = slots // D
    remainder = slots % D
    result = list(vec) * reps + list(vec[:remainder])
    return result


class CKKSBootstrapContext:
    def __init__(self, poly_degree=32768, L0=24, prime_bits=59,
                 special_mod_size=3, level_budget=None, max_rot_dim=256,
                 bsgs_dim=0, skip_bootstrap=False):
        if level_budget is None:
            level_budget = [2, 2]

        print(f"[CKKS] Setting up: N={poly_degree}, L0={L0}, bits={prime_bits}, "
              f"P={special_mod_size}"
              + ("" if skip_bootstrap else f", budget={level_budget}"))

        if not skip_bootstrap:
            boot_elts = ph.ckks_bootstrapper.get_galois_elements(
                poly_degree, 0, level_budget)
        else:
            boot_elts = []
        rot_elts = compute_rotation_galois_elements(poly_degree, max_dim=max_rot_dim)
        dims = bsgs_dim if isinstance(bsgs_dim, (list, tuple)) else [bsgs_dim]
        dims = [d for d in dims if d > 0]
        bsgs_elts = []
        for d in sorted(set(dims)):
            elts = compute_bsgs_galois_elements(poly_degree, d)
            bsgs_elts.extend(elts)
            G, B = compute_bsgs_params(d)
            print(f"[CKKS] BSGS: D={d}, G={G} baby, B={B} giant, "
                  f"{len(elts)} galois elements")
        bsgs_elts = list(set(bsgs_elts))
        all_elts = sorted(set(boot_elts) | set(rot_elts) | set(bsgs_elts))
        print(f"[CKKS] Galois elements: {len(boot_elts)} boot + "
              f"{len(rot_elts)} rot + {len(bsgs_elts)} bsgs = {len(all_elts)} total")

        parms = ph.params(ph.scheme_type.ckks)
        parms.set_poly_modulus_degree(poly_degree)
        parms.set_special_modulus_size(special_mod_size)
        parms.set_galois_elts(all_elts)
        bits = [prime_bits] * (L0 + special_mod_size)
        parms.set_coeff_modulus(ph.create_coeff_modulus(poly_degree, bits))

        self.ctx = ph.context(parms)
        self.sk = ph.secret_key(self.ctx)
        self.encoder = ph.ckks_encoder(self.ctx)
        self.scale = 2.0 ** prime_bits
        # L0<=2: half-scale diags to avoid overflow in multiply_plain
        self.diag_scale = 2.0 ** (prime_bits // 2) if L0 <= 2 else self.scale
        self.slots = self.encoder.slot_count()
        self.L0 = L0
        self.rlk = self.sk.gen_relinkey(self.ctx)
        self.gk = self.sk.create_galois_keys(self.ctx)

        self.bt = None
        if not skip_bootstrap:
            self.bt = ph.ckks_bootstrapper(self.encoder)
            self.bt.setup(self.ctx, level_budget)
            self.bt.keygen(self.ctx, self.sk)
            bt_depth = ph.ckks_bootstrapper.get_bootstrap_depth(level_budget)
            print(f"[CKKS] Bootstrap depth={bt_depth}, post-bootstrap levels={L0 - bt_depth - 1}")
        print(f"[CKKS] Slots={self.slots}")

    def encrypt(self, vec):
        padded = list(vec) + [0.0] * (self.slots - len(vec))
        pt = self.encoder.encode_double_vector(self.ctx, padded, self.scale)
        return self.sk.encrypt_symmetric(self.ctx, pt)

    def encrypt_replicated(self, vec):
        rep = replicate_vector(vec, self.slots)
        pt = self.encoder.encode_double_vector(self.ctx, rep, self.scale)
        return self.sk.encrypt_symmetric(self.ctx, pt)

    def encrypt_replicated_complex(self, vec_real, vec_imag):
        combined = np.array(vec_real) + 1j * np.array(vec_imag)
        rep = replicate_vector(combined, self.slots)
        pt = self.encoder.encode_complex_vector(self.ctx, rep, self.scale)
        return self.sk.encrypt_symmetric(self.ctx, pt)

    def decrypt_vec(self, ct, dim):
        pt = self.sk.decrypt(self.ctx, ct)
        vals = self.encoder.decode_double_vector(self.ctx, pt)
        return np.array(vals[:dim])

    def decrypt_vec_complex(self, ct, dim):
        pt = self.sk.decrypt(self.ctx, ct)
        vals = self.encoder.decode_complex_vector(self.ctx, pt)
        return np.array(vals[:dim])

    def decrypt_slot0(self, ct):
        pt = self.sk.decrypt(self.ctx, ct)
        return self.encoder.decode_double_vector(self.ctx, pt)[0]

    def bootstrap(self, ct):
        if self.bt is None:
            raise RuntimeError("Bootstrap not available (skip_bootstrap=True)")
        while ct.coeff_modulus_size() > 2:
            ct = ph.mod_switch_to_next(self.ctx, ct)
        return self.bt.bootstrap(self.ctx, ct)


def ct_pt_dot(ckks, ct, weights, dim):
    w_padded = list(weights[:dim]) + [0.0] * (ckks.slots - dim)
    w_pt = ckks.encoder.encode_double_vector(ckks.ctx, w_padded, ckks.scale)
    prod = ph.multiply_plain(ckks.ctx, ct, w_pt)
    prod = ph.rescale_to_next(ckks.ctx, prod)
    step = 1
    while step < dim:
        rotated = ph.rotate(ckks.ctx, prod, step, ckks.gk)
        prod = ph.add(ckks.ctx, prod, rotated)
        step *= 2
    return prod


def ct_ct_square(ckks, ct):
    sq = ph.multiply(ckks.ctx, ct, ct)
    sq = ph.relinearize(ckks.ctx, sq, ckks.rlk)
    sq = ph.rescale_to_next(ckks.ctx, sq)
    return sq


def ct_pt_weighted_sum(ckks, ct_list, weights):
    level = ct_list[0].chain_index()
    w_const = [float(weights[0])] * ckks.slots
    w_pt = ckks.encoder.encode_double_vector(ckks.ctx, w_const, ckks.scale)
    w_pt = ph.mod_switch_to(ckks.ctx, w_pt, level)
    result = ph.multiply_plain(ckks.ctx, ct_list[0], w_pt)
    result = ph.rescale_to_next(ckks.ctx, result)

    for j in range(1, len(ct_list)):
        w_const = [float(weights[j])] * ckks.slots
        w_pt = ckks.encoder.encode_double_vector(ckks.ctx, w_const, ckks.scale)
        w_pt = ph.mod_switch_to(ckks.ctx, w_pt, level)
        term = ph.multiply_plain(ckks.ctx, ct_list[j], w_pt)
        term = ph.rescale_to_next(ckks.ctx, term)
        result = ph.add(ckks.ctx, result, term)

    return result


# BSGS matrix-vector multiply

def _extract_diagonals(W, D):
    # d_k[j] = W[j, (j+k) mod D], returns (D, D) array
    j = np.arange(D)
    k = np.arange(D)
    cols = (j[np.newaxis, :] + k[:, np.newaxis]) % D  # (D, D)
    return W[j[np.newaxis, :], cols]


def _replicate_to_slots(vec, slots):
    D = len(vec)
    reps = slots // D
    remainder = slots % D
    if remainder == 0:
        return np.tile(vec, reps)
    return np.concatenate([np.tile(vec, reps), vec[:remainder]])


def _compute_baby_rotations(ckks, ct_x_rep, G):
    ct_baby = [None] * G
    ct_baby[0] = ct_x_rep
    for b in range(1, G):
        ct_baby[b] = ph.rotate(ckks.ctx, ct_x_rep, b, ckks.gk)
    return ct_baby


_bsgs_pool = None


def _get_bsgs_pool(n=4):
    global _bsgs_pool
    if _bsgs_pool is None:
        _bsgs_pool = ThreadPoolExecutor(max_workers=n)
    return _bsgs_pool


def _parallel_bsgs_projections(ckks, inputs, cpu_offloaded_list, D):
    G, B = compute_bsgs_params(D)
    n = len(inputs)
    ct_list = [ckks.encrypt_replicated(x) for x in inputs]

    pool = _get_bsgs_pool(n)
    futures = []
    for ct_x, cpu_data in zip(ct_list, cpu_offloaded_list):
        data, ci, sc, cms, pmd = cpu_data
        f = pool.submit(ph.bsgs_complete_from_cpu,
                        ckks.ctx, ct_x, data, ci, sc, cms, pmd,
                        G, B, D, ckks.gk)
        futures.append(f)

    results = [ckks.decrypt_vec(f.result(), D) for f in futures]
    del ct_list
    return results


def pre_encode_real_diags(ckks, W, D, G, B, level):
    diags = _extract_diagonals(W, D)
    slots = ckks.slots
    return _batch_encode_diags_real(ckks, diags, D, G, slots, level)


def pre_encode_complex_diags(ckks, W1, W2, D, G, B, level):
    diags1 = _extract_diagonals(W1, D)
    diags2 = _extract_diagonals(W2, D)
    slots = ckks.slots
    return _batch_encode_diags_complex(ckks, diags1, diags2, D, G, slots, level)


def pre_encode_block(ckks, block, D, F, G=None, B=None):
    if G is None or B is None:
        G, B = compute_bsgs_params(D)

    dummy_vec = [0.0] * ckks.slots
    dummy_pt = ckks.encoder.encode_double_vector(ckks.ctx, dummy_vec, ckks.scale)
    dummy_ct = ckks.sk.encrypt_symmetric(ckks.ctx, dummy_pt)
    level = dummy_ct.chain_index()
    del dummy_pt, dummy_ct

    pe = {}

    # ATT projections (W.T because BSGS computes M@x, we want x@W)
    for name, W in [('r', block.W_r), ('k', block.W_k),
                    ('v', block.W_v), ('o', block.W_o)]:
        pe[name] = pre_encode_real_diags(ckks, W.T, D, G, B, level)

    # FFN key: complex-packed pairs of output chunks
    n_chunks = int(np.ceil(F / D))
    pe['ffn_key'] = []
    c = 0
    while c < n_chunks:
        out_s1 = c * D
        out_e1 = min(out_s1 + D, F)
        cols1 = out_e1 - out_s1
        M1 = np.zeros((D, D))
        M1[:cols1, :] = block.W_key_ffn[:, out_s1:out_e1].T

        if c + 1 < n_chunks:
            out_s2 = (c + 1) * D
            out_e2 = min(out_s2 + D, F)
            cols2 = out_e2 - out_s2
            M2 = np.zeros((D, D))
            M2[:cols2, :] = block.W_key_ffn[:, out_s2:out_e2].T
            pe['ffn_key'].append(
                pre_encode_complex_diags(ckks, M1, M2, D, G, B, level))
            c += 2
        else:
            pe['ffn_key'].append(
                pre_encode_real_diags(ckks, M1, D, G, B, level))
            c += 1

    # FFN val: conjugate trick, Enc(x0+ix1) * (d0-id1) real part = M0@x0 + M1@x1
    G_out, B_out = compute_bsgs_params(D)
    n_chunks_val = int(np.ceil(F / D))
    pe['ffn_val'] = []
    c = 0
    while c < n_chunks_val:
        in_s = c * D
        in_e = min(in_s + D, F)
        rows = in_e - in_s
        M0 = np.zeros((D, D))
        M0[:, :rows] = block.W_val_ffn[in_s:in_e, :].T

        if c + 1 < n_chunks_val:
            in_s1 = (c + 1) * D
            in_e1 = min(in_s1 + D, F)
            rows1 = in_e1 - in_s1
            M1_neg = np.zeros((D, D))
            M1_neg[:, :rows1] = -block.W_val_ffn[in_s1:in_e1, :].T
            pe['ffn_val'].append(
                pre_encode_complex_diags(ckks, M0, M1_neg, D, G_out, B_out, level))
            c += 2
        else:
            pe['ffn_val'].append(
                pre_encode_real_diags(ckks, M0, D, G_out, B_out, level))
            c += 1

    return pe


def offload_block_plaintexts(pe_block):
    cpu = {}
    for key in ['r', 'k', 'v', 'o']:
        cpu[key] = ph.offload_plaintexts(pe_block[key])
    cpu['ffn_key'] = [ph.offload_plaintexts(pts) for pts in pe_block['ffn_key']]
    cpu['ffn_val'] = [ph.offload_plaintexts(pts) for pts in pe_block['ffn_val']]
    return cpu


def upload_block_plaintexts(cpu_block):
    pe = {}
    for key in ['r', 'k', 'v', 'o']:
        data, ci, sc, cms, pmd = cpu_block[key]
        pe[key] = ph.upload_plaintexts(data, ci, sc, cms, pmd)
    pe['ffn_key'] = []
    for item in cpu_block['ffn_key']:
        data, ci, sc, cms, pmd = item
        pe['ffn_key'].append(ph.upload_plaintexts(data, ci, sc, cms, pmd))
    pe['ffn_val'] = []
    for item in cpu_block['ffn_val']:
        data, ci, sc, cms, pmd = item
        pe['ffn_val'].append(ph.upload_plaintexts(data, ci, sc, cms, pmd))
    return pe


def _batch_encode_diags_real(ckks, diags, D, G, slots, level):
    B_groups = (D + G - 1) // G
    diags_rolled = np.empty_like(diags)
    diags_rolled[:G] = diags[:G]
    for g in range(1, B_groups):
        s, e = g * G, min((g + 1) * G, D)
        shift = g * G
        diags_rolled[s:e, shift:] = diags[s:e, :D - shift]
        diags_rolled[s:e, :shift] = diags[s:e, D - shift:]

    reps = slots // D
    remainder = slots % D
    if remainder == 0:
        diag_vecs = np.tile(diags_rolled, (1, reps))
    else:
        diag_vecs = np.concatenate(
            [np.tile(diags_rolled, (1, reps)), diags_rolled[:, :remainder]],
            axis=1)

    enc_scale = ckks.diag_scale
    try:
        return ckks.encoder.encode_double_vector_batch(
            ckks.ctx, diag_vecs, enc_scale, chain_index=level)
    except AttributeError:
        pts = []
        for k in range(D):
            d_pt = ckks.encoder.encode_double_vector(
                ckks.ctx, list(diag_vecs[k]), enc_scale)
            d_pt = ph.mod_switch_to(ckks.ctx, d_pt, level)
            pts.append(d_pt)
        return pts


def _batch_encode_diags_complex(ckks, diags1, diags2, D, G, slots, level):
    B_groups = (D + G - 1) // G
    d1_rolled = np.empty_like(diags1)
    d2_rolled = np.empty_like(diags2)
    d1_rolled[:G] = diags1[:G]
    d2_rolled[:G] = diags2[:G]
    for g in range(1, B_groups):
        s, e = g * G, min((g + 1) * G, D)
        shift = g * G
        d1_rolled[s:e, shift:] = diags1[s:e, :D - shift]
        d1_rolled[s:e, :shift] = diags1[s:e, D - shift:]
        d2_rolled[s:e, shift:] = diags2[s:e, :D - shift]
        d2_rolled[s:e, :shift] = diags2[s:e, D - shift:]

    reps = slots // D
    remainder = slots % D
    if remainder == 0:
        d1_rep = np.tile(d1_rolled, (1, reps))
        d2_rep = np.tile(d2_rolled, (1, reps))
    else:
        d1_rep = np.concatenate(
            [np.tile(d1_rolled, (1, reps)), d1_rolled[:, :remainder]], axis=1)
        d2_rep = np.concatenate(
            [np.tile(d2_rolled, (1, reps)), d2_rolled[:, :remainder]], axis=1)

    diag_vecs = d1_rep + 1j * d2_rep

    enc_scale = ckks.diag_scale
    try:
        return ckks.encoder.encode_complex_vector_batch(
            ckks.ctx, diag_vecs, enc_scale, chain_index=level)
    except AttributeError:
        pts = []
        for k in range(D):
            d_pt = ckks.encoder.encode_complex_vector(
                ckks.ctx, list(diag_vecs[k]), enc_scale)
            d_pt = ph.mod_switch_to(ckks.ctx, d_pt, level)
            pts.append(d_pt)
        return pts


def fhe_matmul_bsgs(ckks, ct_x_rep, W, D, G=None, B=None, ct_baby=None,
                    preencoded=None, cpu_offloaded=None):
    if G is None or B is None:
        G, B = compute_bsgs_params(D)
    slots = ckks.slots

    if ct_baby is None:
        ct_baby = _compute_baby_rotations(ckks, ct_x_rep, G)

    level = ct_x_rep.chain_index()

    if cpu_offloaded is not None:
        data, ci, sc, cms, pmd = cpu_offloaded
        try:
            return ph.bsgs_from_cpu(
                ckks.ctx, ct_baby, data, ci, sc, cms, pmd, G, B, D, ckks.gk)
        except AttributeError:
            preencoded = ph.upload_plaintexts(data, ci, sc, cms, pmd)

    if preencoded is None:
        diags = _extract_diagonals(W, D)
        preencoded = _batch_encode_diags_real(ckks, diags, D, G, slots, level)

    try:
        return ph.bsgs_multiply_accumulate(
            ckks.ctx, ct_baby, preencoded, G, B, D, ckks.gk)
    except AttributeError:
        pass

    ct_result = None
    for g in range(B):
        ct_inner = None
        for b in range(G):
            k = g * G + b
            if k >= D:
                continue
            term = ph.multiply_plain(ckks.ctx, ct_baby[b], preencoded[k])
            if ct_inner is None:
                ct_inner = term
            else:
                ct_inner = ph.add(ckks.ctx, ct_inner, term)
        if ct_inner is None:
            continue
        if g > 0:
            ct_inner = ph.rotate(ckks.ctx, ct_inner, g * G, ckks.gk)
        if ct_result is None:
            ct_result = ct_inner
        else:
            ct_result = ph.add(ckks.ctx, ct_result, ct_inner)
    ct_result = ph.rescale_to_next(ckks.ctx, ct_result)
    return ct_result


def fhe_matmul_bsgs_complex(ckks, ct_x_rep, W1, W2, D, G=None, B=None,
                             ct_baby=None, preencoded=None,
                             cpu_offloaded=None):
    if G is None or B is None:
        G, B = compute_bsgs_params(D)
    slots = ckks.slots

    if ct_baby is None:
        ct_baby = _compute_baby_rotations(ckks, ct_x_rep, G)

    level = ct_x_rep.chain_index()

    if cpu_offloaded is not None:
        data, ci, sc, cms, pmd = cpu_offloaded
        try:
            return ph.bsgs_from_cpu(
                ckks.ctx, ct_baby, data, ci, sc, cms, pmd, G, B, D, ckks.gk)
        except AttributeError:
            preencoded = ph.upload_plaintexts(data, ci, sc, cms, pmd)

    if preencoded is None:
        diags1 = _extract_diagonals(W1, D)
        diags2 = _extract_diagonals(W2, D)
        preencoded = _batch_encode_diags_complex(
            ckks, diags1, diags2, D, G, slots, level)

    try:
        return ph.bsgs_multiply_accumulate(
            ckks.ctx, ct_baby, preencoded, G, B, D, ckks.gk)
    except AttributeError:
        pass

    # Fallback: Python loop
    ct_result = None
    for g in range(B):
        ct_inner = None
        for b in range(G):
            k = g * G + b
            if k >= D:
                continue
            term = ph.multiply_plain(ckks.ctx, ct_baby[b], preencoded[k])
            if ct_inner is None:
                ct_inner = term
            else:
                ct_inner = ph.add(ckks.ctx, ct_inner, term)
        if ct_inner is None:
            continue
        if g > 0:
            ct_inner = ph.rotate(ckks.ctx, ct_inner, g * G, ckks.gk)
        if ct_result is None:
            ct_result = ct_inner
        else:
            ct_result = ph.add(ckks.ctx, ct_result, ct_inner)
    ct_result = ph.rescale_to_next(ckks.ctx, ct_result)
    return ct_result


def fhe_projection_bsgs(ckks, x, W, D_in, D_out, label="",
                        preencoded_diags=None, cpu_offloaded_diags=None):
    if D_in == D_out:
        G, B = compute_bsgs_params(D_in)
        ct_x = ckks.encrypt_replicated(x)
        pe = preencoded_diags[0] if preencoded_diags else None
        cpu = cpu_offloaded_diags[0] if cpu_offloaded_diags else None
        ct_y = fhe_matmul_bsgs(ckks, ct_x, W.T, D_in, G, B,
                                preencoded=pe, cpu_offloaded=cpu)
        result = ckks.decrypt_vec(ct_y, D_in)
        del ct_x, ct_y
        return result

    elif D_out > D_in:
        G, B = compute_bsgs_params(D_in)
        n_chunks = int(np.ceil(D_out / D_in))
        ct_x = ckks.encrypt_replicated(x)
        result = np.zeros(D_out)
        ct_baby = _compute_baby_rotations(ckks, ct_x, G)

        pe_idx = 0
        c = 0
        while c < n_chunks:
            c_end = min(c + 2, n_chunks)
            out_start1 = c * D_in
            out_end1 = min(out_start1 + D_in, D_out)
            cols1 = out_end1 - out_start1
            M1 = np.zeros((D_in, D_in))
            M1[:cols1, :] = W[:, out_start1:out_end1].T

            if c + 1 < n_chunks:
                out_start2 = (c + 1) * D_in
                out_end2 = min(out_start2 + D_in, D_out)
                cols2 = out_end2 - out_start2
                M2 = np.zeros((D_in, D_in))
                M2[:cols2, :] = W[:, out_start2:out_end2].T

                pe = preencoded_diags[pe_idx] if preencoded_diags else None
                cpu = cpu_offloaded_diags[pe_idx] if cpu_offloaded_diags else None
                ct_y = fhe_matmul_bsgs_complex(ckks, ct_x, M1, M2,
                                                D_in, G, B,
                                                ct_baby=ct_baby,
                                                preencoded=pe,
                                                cpu_offloaded=cpu)
                vals = ckks.decrypt_vec_complex(ct_y, D_in)
                result[out_start1:out_end1] = np.real(vals[:cols1])
                result[out_start2:out_end2] = np.imag(vals[:cols2])
                del ct_y
            else:
                pe = preencoded_diags[pe_idx] if preencoded_diags else None
                cpu = cpu_offloaded_diags[pe_idx] if cpu_offloaded_diags else None
                ct_y = fhe_matmul_bsgs(ckks, ct_x, M1, D_in, G, B,
                                        ct_baby=ct_baby, preencoded=pe,
                                        cpu_offloaded=cpu)
                vals = ckks.decrypt_vec(ct_y, D_in)
                result[out_start1:out_end1] = vals[:cols1]
                del ct_y
            pe_idx += 1
            c = c_end

        del ct_x, ct_baby
        return result

    else:
        # conjugate trick, 2 input chunks per BSGS
        G_out, B_out = compute_bsgs_params(D_out)
        n_chunks = int(np.ceil(D_in / D_out))
        result = np.zeros(D_out)

        pe_idx = 0
        c = 0
        while c < n_chunks:
            in_start0 = c * D_out
            in_end0 = min(in_start0 + D_out, D_in)
            rows0 = in_end0 - in_start0
            x0 = np.zeros(D_out)
            x0[:rows0] = x[in_start0:in_end0]

            if c + 1 < n_chunks:
                in_start1 = (c + 1) * D_out
                in_end1 = min(in_start1 + D_out, D_in)
                rows1 = in_end1 - in_start1
                x1 = np.zeros(D_out)
                x1[:rows1] = x[in_start1:in_end1]

                M0 = np.zeros((D_out, D_out))
                M0[:, :rows0] = W[in_start0:in_end0, :].T
                M1_neg = np.zeros((D_out, D_out))
                M1_neg[:, :rows1] = -W[in_start1:in_end1, :].T

                pe = preencoded_diags[pe_idx] if preencoded_diags else None
                cpu = cpu_offloaded_diags[pe_idx] if cpu_offloaded_diags else None
                ct_pair = ckks.encrypt_replicated_complex(x0, x1)
                ct_y = fhe_matmul_bsgs_complex(ckks, ct_pair, M0, M1_neg,
                                                D_out, G_out, B_out,
                                                preencoded=pe, cpu_offloaded=cpu)
                vals = ckks.decrypt_vec_complex(ct_y, D_out)
                result += np.real(vals)
                del ct_pair, ct_y
                c += 2
            else:
                M = np.zeros((D_out, D_out))
                M[:, :rows0] = W[in_start0:in_end0, :].T
                pe = preencoded_diags[pe_idx] if preencoded_diags else None
                cpu = cpu_offloaded_diags[pe_idx] if cpu_offloaded_diags else None
                ct_chunk = ckks.encrypt_replicated(x0)
                ct_y = fhe_matmul_bsgs(ckks, ct_chunk, M, D_out, G_out, B_out,
                                        preencoded=pe, cpu_offloaded=cpu)
                partial = ckks.decrypt_vec(ct_y, D_out)
                result += partial
                del ct_chunk, ct_y
                c += 1
            pe_idx += 1

        return result


class RWKVBlockWeights:
    def __init__(self, w, block_idx, D, F, n_head, head_size):
        b = f'blocks.{block_idx}.'
        self.D = D
        self.F = F
        self.n_head = n_head
        self.head_size = head_size
        self.block_idx = block_idx

        self.ln1_w = w[b + 'ln1.weight'][:D].numpy().astype(np.float64)
        self.ln1_b = w[b + 'ln1.bias'][:D].numpy().astype(np.float64)
        self.ln2_w = w[b + 'ln2.weight'][:D].numpy().astype(np.float64)
        self.ln2_b = w[b + 'ln2.bias'][:D].numpy().astype(np.float64)
        self.ln_x_w = w[b + 'att.ln_x.weight'][:D].numpy().astype(np.float64)
        self.ln_x_b = w[b + 'att.ln_x.bias'][:D].numpy().astype(np.float64)

        self.x_r = w[b + 'att.x_r'].float().squeeze()[:D].numpy().astype(np.float64)
        self.x_k = w[b + 'att.x_k'].float().squeeze()[:D].numpy().astype(np.float64)
        self.x_v = w[b + 'att.x_v'].float().squeeze()[:D].numpy().astype(np.float64)
        self.x_g = w[b + 'att.x_g'].float().squeeze()[:D].numpy().astype(np.float64)
        self.x_w = w[b + 'att.x_w'].float().squeeze()[:D].numpy().astype(np.float64)
        self.x_a = w[b + 'att.x_a'].float().squeeze()[:D].numpy().astype(np.float64)
        self.x_k_ffn = w[b + 'ffn.x_k'].float().squeeze()[:D].numpy().astype(np.float64)

        self.k_k = w[b + 'att.k_k'].float().squeeze()[:D].numpy().astype(np.float64)
        self.k_a = w[b + 'att.k_a'].float().squeeze()[:D].numpy().astype(np.float64)

        self.w0 = w[b + 'att.w0'][:D].numpy().astype(np.float64)
        self.w1 = w[b + 'att.w1'][:D, :].numpy().astype(np.float64)  # (D, 96)
        self.w2 = w[b + 'att.w2'][:, :D].numpy().astype(np.float64)  # (96, D)

        self.a0 = w[b + 'att.a0'][:D].numpy().astype(np.float64)
        self.a1 = w[b + 'att.a1'][:D, :].numpy().astype(np.float64)  # (D, 96)
        self.a2 = w[b + 'att.a2'][:, :D].numpy().astype(np.float64)  # (96, D)

        if b + 'att.v0' in w:
            self.v0 = w[b + 'att.v0'][:D].numpy().astype(np.float64)
            self.v1 = w[b + 'att.v1'][:D, :].numpy().astype(np.float64)  # (D, 64)
            self.v2 = w[b + 'att.v2'][:, :D].numpy().astype(np.float64)  # (64, D)
        else:
            self.v0 = np.zeros(D)
            self.v1 = np.zeros((D, 64))
            self.v2 = np.zeros((64, D))

        self.r_k = w[b + 'att.r_k'][:n_head, :head_size].numpy().astype(np.float64)
        self.g1 = w[b + 'att.g1'][:D, :].numpy().astype(np.float64)
        self.g2 = w[b + 'att.g2'][:, :D].numpy().astype(np.float64)

        # load_weights transposes to [in, out]
        self.W_r = w[b + 'att.receptance.weight'][:D, :D].numpy().astype(np.float64)
        self.W_k = w[b + 'att.key.weight'][:D, :D].numpy().astype(np.float64)
        self.W_v = w[b + 'att.value.weight'][:D, :D].numpy().astype(np.float64)
        self.W_o = w[b + 'att.output.weight'][:D, :D].numpy().astype(np.float64)
        self.W_key_ffn = w[b + 'ffn.key.weight'][:D, :F].numpy().astype(np.float64)
        self.W_val_ffn = w[b + 'ffn.value.weight'][:F, :D].numpy().astype(np.float64)


def layer_norm(x, weight, bias, eps=1e-5):
    mean = np.mean(x)
    var = np.var(x)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def group_norm(x, n_groups, weight, bias, eps=64e-5):
    D = len(x)
    group_size = D // n_groups
    out = np.zeros_like(x)
    for g in range(n_groups):
        s = g * group_size
        e = s + group_size
        chunk = x[s:e]
        mean = np.mean(chunk)
        var = np.var(chunk)
        out[s:e] = (chunk - mean) / np.sqrt(var + eps)
    return out * weight + bias


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def fhe_projection(ckks, x_or_ct, W, dim_in, dim_out, label=""):
    if isinstance(x_or_ct, np.ndarray):
        ct_in = ckks.encrypt(x_or_ct)
    else:
        ct_in = x_or_ct
    result = np.zeros(dim_out)
    for j in range(dim_out):
        ct_j = ct_pt_dot(ckks, ct_in, W[:, j], dim_in)
        result[j] = ckks.decrypt_slot0(ct_j)
        del ct_j
    return result


def client_aided_block(ckks, block, x, x_prev_att, x_prev_ffn, state, v_first,
                       use_bsgs=False, preencoded_block=None,
                       cpu_offloaded_block=None):
    D = block.D
    F = block.F
    n_head = block.n_head
    head_size = block.head_size
    timings = {}

    # Client: LayerNorm + mixing
    t0 = time.perf_counter()
    x_ln = layer_norm(x, block.ln1_w, block.ln1_b)
    new_x_prev_att = x_ln.copy()

    xx = x_prev_att - x_ln
    xr = x_ln + xx * block.x_r
    xk = x_ln + xx * block.x_k
    xv = x_ln + xx * block.x_v
    xg = x_ln + xx * block.x_g
    xw = x_ln + xx * block.x_w
    xa = x_ln + xx * block.x_a
    timings['client_mix'] = time.perf_counter() - t0

    # Server: r, k, v projections
    t0 = time.perf_counter()
    pe = preencoded_block
    cpu = cpu_offloaded_block
    if use_bsgs:
        r = fhe_projection_bsgs(ckks, xr, block.W_r, D, D, "r",
                                preencoded_diags=[pe['r']] if pe else None,
                                cpu_offloaded_diags=[cpu['r']] if cpu else None)
        k = fhe_projection_bsgs(ckks, xk, block.W_k, D, D, "k",
                                preencoded_diags=[pe['k']] if pe else None,
                                cpu_offloaded_diags=[cpu['k']] if cpu else None)
        v = fhe_projection_bsgs(ckks, xv, block.W_v, D, D, "v",
                                preencoded_diags=[pe['v']] if pe else None,
                                cpu_offloaded_diags=[cpu['v']] if cpu else None)
    else:
        r = fhe_projection(ckks, xr, block.W_r, D, D, "r")
        k = fhe_projection(ckks, xk, block.W_k, D, D, "k")
        v = fhe_projection(ckks, xv, block.W_v, D, D, "v")
    timings['server_rkv'] = time.perf_counter() - t0

    # Client: WKV state update + GroupNorm + gating
    t0 = time.perf_counter()
    r_h = r.reshape(n_head, head_size)
    k_h = k.reshape(n_head, head_size)
    v_h = v.reshape(n_head, head_size)

    w_vec = sigmoid(block.w0 + np.tanh(xw @ block.w1) @ block.w2)
    w_h = w_vec.reshape(n_head, head_size)
    # decay = exp(-exp(-0.5) * sigmoid(w)) -- matches CUDA kernel
    decay = np.exp(-np.exp(-0.5) * w_h)

    a_vec = sigmoid(block.a0 + (xa @ block.a1) @ block.a2)
    a_h = a_vec.reshape(n_head, head_size)

    k_k_h = block.k_k.reshape(n_head, head_size)
    kk_h = k_h * k_k_h
    kk_norms = np.linalg.norm(kk_h, axis=1, keepdims=True)
    kk_h = kk_h / (kk_norms + 1e-12)

    k_a_h = block.k_a.reshape(n_head, head_size)
    k_h = k_h * (1.0 + (a_h - 1.0) * k_a_h)

    if block.block_idx == 0:
        v_first_out = v.copy()
    else:
        v_gate = sigmoid(block.v0 + (xv @ block.v1) @ block.v2)
        v = v + (v_first - v) * v_gate
        v_h = v.reshape(n_head, head_size)
        v_first_out = v_first

    # WKV state update
    new_state = state.copy()
    wkv_heads = np.zeros((n_head, head_size))
    for h in range(n_head):
        sa = new_state[h] @ (-kk_h[h])
        sab = np.outer(sa, kk_h[h] * a_h[h])
        # decay per-COLUMN (j index), not per-row
        new_state[h] = new_state[h] * decay[h] + sab + np.outer(v_h[h], k_h[h])
        wkv_heads[h] = new_state[h] @ r_h[h]

    wkv = wkv_heads.reshape(D)
    wkv = group_norm(wkv, n_head, block.ln_x_w, block.ln_x_b)

    rkrk = (r_h * k_h * block.r_k).sum(axis=1, keepdims=True)
    wkv = wkv + (rkrk * v_h).reshape(D)

    g = sigmoid(xg @ block.g1) @ block.g2
    gated_out = wkv * g
    timings['client_wkv_gate'] = time.perf_counter() - t0

    # Server: W_o projection
    t0 = time.perf_counter()
    if use_bsgs:
        att_out = fhe_projection_bsgs(ckks, gated_out, block.W_o, D, D, "o",
                                      preencoded_diags=[pe['o']] if pe else None,
                                      cpu_offloaded_diags=[cpu['o']] if cpu else None)
    else:
        att_out = fhe_projection(ckks, gated_out, block.W_o, D, D, "o")
    timings['server_wo'] = time.perf_counter() - t0

    # Client: residual + FFN mixing
    t0 = time.perf_counter()
    x = x + att_out

    x_ffn_ln = layer_norm(x, block.ln2_w, block.ln2_b)
    new_x_prev_ffn = x_ffn_ln.copy()

    xx_ffn = x_prev_ffn - x_ffn_ln
    x_k_ffn = x_ffn_ln + xx_ffn * block.x_k_ffn
    timings['client_ffn_prep'] = time.perf_counter() - t0

    # Server: FFN key projection
    t0 = time.perf_counter()
    if use_bsgs:
        fk = fhe_projection_bsgs(ckks, x_k_ffn, block.W_key_ffn, D, F, "ffn_key",
                                 preencoded_diags=pe.get('ffn_key') if pe else None,
                                 cpu_offloaded_diags=cpu.get('ffn_key') if cpu else None)
    else:
        fk = fhe_projection(ckks, x_k_ffn, block.W_key_ffn, D, F, "ffn_key")
    timings['server_ffn_key'] = time.perf_counter() - t0

    # Client: ReLU + square
    t0 = time.perf_counter()
    fk_sq = np.maximum(fk, 0.0) ** 2
    timings['client_relu_sq'] = time.perf_counter() - t0

    # Server: FFN value projection
    t0 = time.perf_counter()
    if use_bsgs:
        v_ffn = fhe_projection_bsgs(ckks, fk_sq, block.W_val_ffn, F, D, "ffn_val",
                                    preencoded_diags=pe.get('ffn_val') if pe else None,
                                    cpu_offloaded_diags=cpu.get('ffn_val') if cpu else None)
    else:
        v_ffn = fhe_projection(ckks, fk_sq, block.W_val_ffn, F, D, "ffn_val")
    timings['server_ffn_val'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    x = x + v_ffn
    timings['client_residual'] = time.perf_counter() - t0

    return x, new_x_prev_att, new_x_prev_ffn, new_state, v_first_out, timings


def plaintext_block(block, x, x_prev_att, x_prev_ffn, state, v_first):
    D = block.D
    n_head = block.n_head
    head_size = block.head_size

    x_ln = layer_norm(x, block.ln1_w, block.ln1_b)
    new_x_prev_att = x_ln.copy()

    xx = x_prev_att - x_ln
    xr = x_ln + xx * block.x_r
    xk = x_ln + xx * block.x_k
    xv = x_ln + xx * block.x_v
    xg = x_ln + xx * block.x_g
    xw = x_ln + xx * block.x_w
    xa = x_ln + xx * block.x_a

    r = xr @ block.W_r
    k = xk @ block.W_k
    v = xv @ block.W_v

    r_h = r.reshape(n_head, head_size)
    k_h = k.reshape(n_head, head_size)
    v_h = v.reshape(n_head, head_size)

    w_vec = sigmoid(block.w0 + np.tanh(xw @ block.w1) @ block.w2)
    w_h = w_vec.reshape(n_head, head_size)
    decay = np.exp(-np.exp(-0.5) * w_h)

    a_vec = sigmoid(block.a0 + (xa @ block.a1) @ block.a2)
    a_h = a_vec.reshape(n_head, head_size)

    k_k_h = block.k_k.reshape(n_head, head_size)
    kk_h = k_h * k_k_h
    kk_norms = np.linalg.norm(kk_h, axis=1, keepdims=True)
    kk_h = kk_h / (kk_norms + 1e-12)

    k_a_h = block.k_a.reshape(n_head, head_size)
    k_h = k_h * (1.0 + (a_h - 1.0) * k_a_h)

    if block.block_idx == 0:
        v_first_out = v.copy()
    else:
        v_gate = sigmoid(block.v0 + (xv @ block.v1) @ block.v2)
        v = v + (v_first - v) * v_gate
        v_h = v.reshape(n_head, head_size)
        v_first_out = v_first

    # WKV state update
    new_state = state.copy()
    wkv_heads = np.zeros((n_head, head_size))
    for h in range(n_head):
        sa = new_state[h] @ (-kk_h[h])
        sab = np.outer(sa, kk_h[h] * a_h[h])
        # decay per-COLUMN (j index), not per-row
        new_state[h] = new_state[h] * decay[h] + sab + np.outer(v_h[h], k_h[h])
        wkv_heads[h] = new_state[h] @ r_h[h]

    wkv = wkv_heads.reshape(D)
    wkv = group_norm(wkv, n_head, block.ln_x_w, block.ln_x_b)

    rkrk = (r_h * k_h * block.r_k).sum(axis=1, keepdims=True)
    wkv = wkv + (rkrk * v_h).reshape(D)

    g = sigmoid(xg @ block.g1) @ block.g2
    att_out = (wkv * g) @ block.W_o
    x = x + att_out

    x_ffn_ln = layer_norm(x, block.ln2_w, block.ln2_b)
    new_x_prev_ffn = x_ffn_ln.copy()

    xx_ffn = x_prev_ffn - x_ffn_ln
    x_k_ffn = x_ffn_ln + xx_ffn * block.x_k_ffn

    k_ffn = x_k_ffn @ block.W_key_ffn
    k_ffn = np.maximum(k_ffn, 0.0) ** 2
    v_ffn = k_ffn @ block.W_val_ffn
    x = x + v_ffn

    return x, new_x_prev_att, new_x_prev_ffn, new_state, v_first_out


def generate_token_fhe(ckks, blocks, emb, head_w, ln_out_w, ln_out_b,
                       ln0_w, ln0_b, token_id, x_prevs_att, x_prevs_ffn,
                       states, D, use_bsgs=False, preencoded_blocks=None,
                       cpu_offloaded_blocks=None):
    x = layer_norm(emb[token_id].copy(), ln0_w, ln0_b)
    new_x_prevs_att = []
    new_x_prevs_ffn = []
    new_states = []
    block_timings = []
    v_first = None

    for i, block in enumerate(blocks):
        pe_block = preencoded_blocks[i] if preencoded_blocks else None
        cpu_block = cpu_offloaded_blocks[i] if cpu_offloaded_blocks else None

        x, xpa, xpf, new_state, v_first, timings = client_aided_block(
            ckks, block, x, x_prevs_att[i], x_prevs_ffn[i], states[i], v_first,
            use_bsgs=use_bsgs, preencoded_block=pe_block,
            cpu_offloaded_block=cpu_block)

        new_x_prevs_att.append(xpa)
        new_x_prevs_ffn.append(xpf)
        new_states.append(new_state)
        block_timings.append(timings)

    x = layer_norm(x, ln_out_w, ln_out_b)
    logits = x @ head_w

    return logits, new_x_prevs_att, new_x_prevs_ffn, new_states, block_timings


def generate_token_plaintext(blocks, emb, head_w, ln_out_w, ln_out_b,
                             ln0_w, ln0_b, token_id, x_prevs_att,
                             x_prevs_ffn, states, D):
    x = layer_norm(emb[token_id].copy(), ln0_w, ln0_b)
    new_x_prevs_att = []
    new_x_prevs_ffn = []
    new_states = []
    v_first = None

    for i, block in enumerate(blocks):
        x, xpa, xpf, new_state, v_first = plaintext_block(
            block, x, x_prevs_att[i], x_prevs_ffn[i], states[i], v_first)
        new_x_prevs_att.append(xpa)
        new_x_prevs_ffn.append(xpf)
        new_states.append(new_state)

    x = layer_norm(x, ln_out_w, ln_out_b)
    logits = x @ head_w
    return logits, new_x_prevs_att, new_x_prevs_ffn, new_states


def bootstrap_spot_check(ckks, block, x, x_prev, D):
    print("\nBootstrap spot-check:")
    N_CHECK = 4

    x_ffn_ln = layer_norm(x, block.ln2_w, block.ln2_b)
    xx_ffn = x_prev - x_ffn_ln
    x_k_ffn = x_ffn_ln + xx_ffn * block.x_k_ffn

    k_ref = x_k_ffn @ block.W_key_ffn[:, :N_CHECK]
    ct_in = ckks.encrypt(x_k_ffn)
    fk_fhe = np.zeros(N_CHECK)
    for j in range(N_CHECK):
        ct_j = ct_pt_dot(ckks, ct_in, block.W_key_ffn[:, j], D)
        fk_fhe[j] = ckks.decrypt_slot0(ct_j)
        del ct_j
    del ct_in

    print(f"  FHE key projection (first {N_CHECK} dims):")
    for j in range(N_CHECK):
        err = abs(fk_fhe[j] - k_ref[j])
        print(f"    [{j}] ref={k_ref[j]:.6f} fhe={fk_fhe[j]:.6f} err={err:.6e}")

    fk_relu_sq = np.maximum(fk_fhe, 0.0) ** 2

    t0 = time.perf_counter()
    ct_packed = ckks.encrypt(fk_relu_sq)
    ct_bt = ckks.bootstrap(ct_packed)
    t_bootstrap = time.perf_counter() - t0
    del ct_packed

    vals_after = ckks.decrypt_vec(ct_bt, N_CHECK)
    del ct_bt

    print(f"\n  Bootstrap ({t_bootstrap:.2f}s):")
    max_err = 0.0
    for j in range(N_CHECK):
        err = abs(vals_after[j] - fk_relu_sq[j])
        max_err = max(max_err, err)
        print(f"    [{j}] before={fk_relu_sq[j]:.6f} after={vals_after[j]:.6f} err={err:.6e}")

    print(f"  Bootstrap max error: {max_err:.6e}")
    print(f"  Bootstrap {'ok' if max_err < 0.1 else 'error too large'}")
    return max_err


def load_model_weights(D, F, n_blocks):
    w = load_weights()

    full_D = w['emb.weight'].shape[1]
    full_n_head = w['blocks.0.att.r_k'].shape[0]
    full_head_size = full_D // full_n_head

    n_head = min(full_n_head, max(1, D // full_head_size))
    head_size = D // n_head
    D = n_head * head_size

    print(f"  Full model: D={full_D}, n_head={full_n_head}, head_size={full_head_size}")
    print(f"  Using: D={D}, n_head={n_head}, head_size={head_size}, F={F}")

    blocks = [RWKVBlockWeights(w, i, D, F, n_head, head_size)
              for i in range(n_blocks)]

    emb = w['emb.weight'][:, :D].numpy().astype(np.float64)
    vocab_size = emb.shape[0]
    head_w = w['head.weight'][:D, :].numpy().astype(np.float64)
    ln_out_w = w['ln_out.weight'][:D].numpy().astype(np.float64)
    ln_out_b = w['ln_out.bias'][:D].numpy().astype(np.float64)
    ln0_w = w['blocks.0.ln0.weight'][:D].numpy().astype(np.float64)
    ln0_b = w['blocks.0.ln0.bias'][:D].numpy().astype(np.float64)

    return blocks, emb, head_w, ln_out_w, ln_out_b, ln0_w, ln0_b, D, n_head, head_size, vocab_size


def run_generation(ckks, blocks, emb, head_w, ln_out_w, ln_out_b,
                   ln0_w, ln0_b, D, n_head, head_size,
                   seed_tokens, num_gen_tokens, tokenizer=None,
                   use_bsgs=False, use_preencoded=False):
    n_blocks = len(blocks)
    F = blocks[0].F
    xpa_fhe = [np.zeros(D) for _ in range(n_blocks)]
    xpf_fhe = [np.zeros(D) for _ in range(n_blocks)]
    xpa_ref = [np.zeros(D) for _ in range(n_blocks)]
    xpf_ref = [np.zeros(D) for _ in range(n_blocks)]
    states_fhe = [np.zeros((n_head, head_size, head_size)) for _ in range(n_blocks)]
    states_ref = [np.zeros((n_head, head_size, head_size)) for _ in range(n_blocks)]

    preencoded_blocks = None
    cpu_offloaded_blocks = None
    preencode_time = 0.0
    if use_preencoded and use_bsgs:
        G, B = compute_bsgs_params(D)
        gb_per_block = D * 10 * 256 / 1024 / 1024
        print(f"  Pre-encoding diagonal plaintexts for {n_blocks} blocks...")
        print(f"  (~{gb_per_block:.1f}GB/block at N={ckks.slots * 2})")

        has_offload = hasattr(ph, 'offload_plaintexts') and hasattr(ph, 'upload_plaintexts')
        if has_offload:
            print(f"  Mode: encode on GPU -> offload to CPU -> upload per-block")
            cpu_offloaded_blocks = [None] * n_blocks

        t_pe = time.perf_counter()
        if not has_offload:
            preencoded_blocks = [None] * n_blocks
        n_cached = 0
        for i, block in enumerate(blocks):
            t_block = time.perf_counter()
            try:
                pe = pre_encode_block(ckks, block, D, F, G, B)
                dt_enc = time.perf_counter() - t_block

                n_pts = 0
                for v in pe.values():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                        n_pts += sum(len(vv) for vv in v)
                    else:
                        n_pts += len(v)

                if has_offload:
                    t_off = time.perf_counter()
                    cpu_offloaded_blocks[i] = offload_block_plaintexts(pe)
                    dt_off = time.perf_counter() - t_off
                    del pe
                    print(f"    Block {i}: encode={dt_enc:.1f}s offload={dt_off:.1f}s "
                          f"({n_pts} pts)")
                else:
                    preencoded_blocks[i] = pe
                    print(f"    Block {i}: {dt_enc:.1f}s, {n_pts} plaintexts")

                n_cached += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    dt = time.perf_counter() - t_block
                    print(f"    Block {i}: OOM after {dt:.1f}s -- stopping pre-encoding")
                    print(f"    (remaining {n_blocks - i} blocks will use on-the-fly encoding)")
                    break
                raise
        preencode_time = time.perf_counter() - t_pe
        mode = "offloaded to CPU" if has_offload else "cached on GPU"
        print(f"  Pre-encoding done: {n_cached}/{n_blocks} blocks {mode} in "
              f"{preencode_time:.1f}s")

    prefill_time = 0.0
    if len(seed_tokens) > 1:
        n_prefill = len(seed_tokens) - 1
        print(f"  Prefilling {n_prefill} tokens (plaintext, building state)...")
        t_prefill = time.perf_counter()
        for i, tok in enumerate(seed_tokens[:-1]):
            _, xpa_fhe, xpf_fhe, states_fhe = generate_token_plaintext(
                blocks, emb, head_w, ln_out_w, ln_out_b, ln0_w, ln0_b,
                tok, xpa_fhe, xpf_fhe, states_fhe, D)
            _, xpa_ref, xpf_ref, states_ref = generate_token_plaintext(
                blocks, emb, head_w, ln_out_w, ln_out_b, ln0_w, ln0_b,
                tok, xpa_ref, xpf_ref, states_ref, D)
        prefill_time = time.perf_counter() - t_prefill
        print(f"  Prefill done: {prefill_time:.1f}s ({prefill_time/n_prefill:.3f}s/token)")

    token_fhe = seed_tokens[-1]
    token_ref = seed_tokens[-1]
    tokens_fhe = [token_fhe]
    tokens_ref = [token_ref]
    total_fhe_time = 0.0

    for step in range(num_gen_tokens):
        print(f"\n  --- Token {step+1}/{num_gen_tokens} ---")

        logits_ref, xpa_ref, xpf_ref, states_ref = generate_token_plaintext(
            blocks, emb, head_w, ln_out_w, ln_out_b, ln0_w, ln0_b,
            token_ref, xpa_ref, xpf_ref, states_ref, D)
        token_ref = int(np.argmax(logits_ref))
        tokens_ref.append(token_ref)

        t_step = time.perf_counter()
        logits_fhe, xpa_fhe, xpf_fhe, states_fhe, block_timings = generate_token_fhe(
            ckks, blocks, emb, head_w, ln_out_w, ln_out_b, ln0_w, ln0_b,
            token_fhe, xpa_fhe, xpf_fhe, states_fhe, D, use_bsgs=use_bsgs,
            preencoded_blocks=preencoded_blocks,
            cpu_offloaded_blocks=cpu_offloaded_blocks)
        t_step = time.perf_counter() - t_step
        total_fhe_time += t_step

        token_fhe = int(np.argmax(logits_fhe))
        tokens_fhe.append(token_fhe)

        match = (token_fhe == token_ref)
        if len(logits_fhe) > 1 and len(logits_ref) > 1:
            corr = np.corrcoef(logits_fhe[:1000], logits_ref[:1000])[0, 1]
        else:
            corr = 1.0 if match else 0.0

        for bi, bt in enumerate(block_timings):
            server_t = bt.get('server_rkv', 0) + bt.get('server_wo', 0) + \
                       bt.get('server_ffn_key', 0) + bt.get('server_ffn_val', 0)
            client_t = bt.get('client_mix', 0) + bt.get('client_wkv_gate', 0) + \
                       bt.get('client_ffn_prep', 0) + bt.get('client_relu_sq', 0) + \
                       bt.get('client_residual', 0)
            upload_t = bt.get('upload', 0) + bt.get('gpu_upload', 0)
            upload_str = f" upload={upload_t:.2f}s" if upload_t > 0 else ""
            print(f"    Block {bi}: server={server_t:.1f}s client={client_t:.2f}s{upload_str}")

        tok_text = ""
        if tokenizer:
            try:
                tok_text = f" '{tokenizer.decode([token_fhe])}'"
            except Exception:
                pass

        print(f"  Token: ref={token_ref} fhe={token_fhe}{tok_text} "
              f"{'match' if match else 'mismatch'} corr={corr:.6f} time={t_step:.1f}s")

    if preencode_time > 0:
        print(f"\n  Pre-encode setup: {preencode_time:.1f}s (one-time, amortized over tokens)")

    return tokens_fhe, tokens_ref, total_fhe_time, prefill_time


def run_rag_demo(args):
    from scipy.linalg import svd
    from fhe_common import (
        PhantomFHE, pack_complex, pack_complex_conjugate,
        euclidean_to_lorentz, get_embeddings, MODEL_DIR,
    )
    from fhe_spear_retrieval import load_msmarco_samples, load_squad_samples
    from rwkv_emb.model import EmbeddingRWKV
    from rwkv_emb.tokenizer import RWKVTokenizer

    D = args.embed_dim
    F = args.ffn_dim
    dataset_name = getattr(args, 'dataset', 'msmarco')

    ret_mode = getattr(args, 'retrieval_mode', 'ctpt')
    print(f"E2E RAG (D={D}, F={F}, blocks={args.num_blocks}, "
          f"dataset={dataset_name}, ret={ret_mode})")

    print("\nLoading models and data...")
    t0 = time.perf_counter()

    emb_model = EmbeddingRWKV(str(MODEL_DIR / "rwkv0b4-emb-curriculum.pth"))
    tokenizer = RWKVTokenizer()
    if dataset_name == 'squad':
        dataset_path = getattr(args, 'dataset_path', None)
        passages, test_qa = load_squad_samples(args.n_docs, args.n_queries,
                                                data_path=dataset_path)
    else:
        passages, test_qa = load_msmarco_samples(args.n_docs, args.n_queries)

    blocks, emb, head_w, ln_out_w, ln_out_b, ln0_w, ln0_b, D, n_head, head_size, vocab_size = \
        load_model_weights(D, F, args.num_blocks)
    args.embed_dim = D

    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    print(f"  Vocab={vocab_size}, passages={len(passages)}, queries={len(test_qa)}")

    ret_mode = getattr(args, 'retrieval_mode', 'ctpt')
    mode_labels = {'ctpt': 'CT-PT (query encrypted)', 'ctct': 'CT-CT (both encrypted)',
                   'plaintext': 'Plaintext (baseline)'}
    print(f"\nRetrieval [{mode_labels[ret_mode]}] ({args.n_queries} queries)...")
    t0 = time.perf_counter()
    doc_embs = get_embeddings(emb_model, tokenizer, passages)
    doc_embs = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
    _, S, Vt = svd(doc_embs, full_matrices=False)
    proj = Vt[:64].T
    doc_proj = doc_embs @ proj
    doc_proj = doc_proj / (np.linalg.norm(doc_proj, axis=1, keepdims=True) + 1e-8)
    doc_lorentz = euclidean_to_lorentz(doc_proj)
    docs_packed = [pack_complex(doc.astype(np.float64)) for doc in doc_lorentz]
    slots_per_doc = len(docs_packed[0])

    if ret_mode == 'plaintext':
        fhe_ret = None
    else:
        fhe_ret = PhantomFHE()

    encrypted_doc_batches = None
    if ret_mode == 'ctct' and fhe_ret is not None:
        encrypted_doc_batches = fhe_ret.encrypt_docs_batch(docs_packed, slots_per_doc)
        print(f"  Encrypted {len(encrypted_doc_batches)} doc batches for CT-CT")

    retrieval_results = []
    for qi, (question, gold_answer, gold_context) in enumerate(test_qa):
        q_emb = get_embeddings(emb_model, tokenizer, [question])[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        q_proj = q_emb @ proj
        q_proj = q_proj / (np.linalg.norm(q_proj) + 1e-8)
        q_lorentz = euclidean_to_lorentz(q_proj.reshape(1, -1)).flatten()
        q_fhe = q_lorentz.copy()
        q_fhe[0] = -q_fhe[0]
        q_packed = pack_complex_conjugate(q_fhe.astype(np.float64))

        if ret_mode == 'plaintext':
            scores = np.array([np.sum(q_fhe * doc) for doc in doc_lorentz])
        elif ret_mode == 'ctct':
            scores = fhe_ret.batched_dot_products_ctct(
                q_packed, encrypted_doc_batches, slots_per_doc)
        else:  # ctpt
            scores = fhe_ret.batched_dot_products_ctpt(
                q_packed, docs_packed, slots_per_doc)

        top_idx = int(np.argmax(scores))
        retrieved = passages[top_idx]
        gold_match = (retrieved == gold_context)
        retrieval_results.append((question, gold_answer, gold_context,
                                  retrieved, gold_match))
        print(f"  Query {qi+1}: gold={'Y' if gold_match else 'N'} - {retrieved[:60]}...")

    if fhe_ret is not None:
        del fhe_ret
    del docs_packed, doc_embs, doc_proj, doc_lorentz
    del emb_model
    if encrypted_doc_batches is not None:
        del encrypted_doc_batches
    import gc; gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"  Retrieval done in {time.perf_counter() - t0:.1f}s, GPU freed for generation")

    use_bsgs = getattr(args, 'bsgs', False)
    a100_mode = getattr(args, 'a100', False)
    if use_bsgs:
        if a100_mode:
            N, P, L0 = 8192, 1, 3
        else:
            N, P, L0 = 32768, 3, 24
        print(f"\nSetting up CKKS (BSGS, no bootstrap)...")
        if a100_mode:
            print(f"  A100 mode: N={N}, P={P}, L0={L0} (sm_80 rotate workaround)")
        t0 = time.perf_counter()
        prime_bits = 54 if a100_mode else 59
        ckks = CKKSBootstrapContext(poly_degree=N, L0=L0, prime_bits=prime_bits,
                                    special_mod_size=P,
                                    max_rot_dim=max(D, F),
                                    bsgs_dim=[D, F], skip_bootstrap=True)
        print(f"  CKKS setup: {time.perf_counter() - t0:.1f}s")
        print(f"\nBootstrap spot-check skipped (BSGS mode)")
    else:
        print(f"\nSetting up CKKS (bootstrap)...")
        t0 = time.perf_counter()
        ckks = CKKSBootstrapContext(L0=24, prime_bits=59, max_rot_dim=max(D, F))
        print(f"  CKKS setup: {time.perf_counter() - t0:.1f}s")

        if not getattr(args, 'no_bootstrap', False):
            print(f"\nBootstrap spot-check...")
            x_test = emb[3].copy()
            bootstrap_spot_check(ckks, blocks[0], x_test, np.zeros(D), D)
        else:
            print(f"\nBootstrap spot-check skipped")

    print(f"\nGenerating answers ({args.n_queries} queries)...")
    results = []

    for qi, (question, gold_answer, gold_context, retrieved, gold_match) in \
            enumerate(retrieval_results):
        print(f"\nQuery {qi+1}/{args.n_queries}: {question[:80]}...")
        print(f"  [Retrieval] gold={'Y' if gold_match else 'N'}")
        print(f"  Retrieved: {retrieved[:100]}...")

        use_prefill = getattr(args, 'prefill', False)
        if use_prefill:
            prompt = f"Context: {retrieved}\nQuestion: {question}\nAnswer:"
            seed_tokens = tokenizer.encode(prompt)
            seed_tokens = [t for t in seed_tokens if t != 65535]
            print(f"  Prefill prompt: {len(seed_tokens)} tokens")
            print(f"    '{prompt[:120]}...'")
        else:
            passage_tokens = tokenizer.encode(retrieved)
            seed_tokens = passage_tokens[:min(5, len(passage_tokens))]
            if not seed_tokens:
                seed_tokens = [3]
            seed_text = tokenizer.decode(seed_tokens)
            print(f"  Seed tokens: {seed_tokens} = '{seed_text}'")

        use_preencoded = getattr(args, 'preencoded', False)
        tokens_fhe, tokens_ref, gen_time, prefill_t = run_generation(
            ckks, blocks, emb, head_w, ln_out_w, ln_out_b,
            ln0_w, ln0_b, D, n_head, head_size,
            seed_tokens, args.num_tokens, tokenizer,
            use_bsgs=use_bsgs, use_preencoded=use_preencoded)

        all_match = (tokens_fhe == tokens_ref)
        gen_text = tokenizer.decode(tokens_fhe)

        print(f"\n  [Generation] {gen_time:.1f}s FHE + {prefill_t:.1f}s prefill, "
              f"match={'Y' if all_match else 'N'}")
        print(f"  Generated text: '{gen_text}'")
        print(f"  Gold answer: {gold_answer[:100]}...")

        results.append({
            'query': question[:60],
            'retrieval_gold': gold_match,
            'generation_time': gen_time,
            'prefill_time': prefill_t,
            'n_prefill_tokens': max(0, len(seed_tokens) - 1),
            'tokens_match': all_match,
            'generated_text': gen_text,
        })

    print(f"\nE2E RAG summary:")
    n_ret = sum(1 for r in results if r['retrieval_gold'])
    n_match = sum(1 for r in results if r['tokens_match'])
    avg_gen = np.mean([r['generation_time'] for r in results])

    avg_prefill = np.mean([r['prefill_time'] for r in results])
    avg_prefill_tok = np.mean([r['n_prefill_tokens'] for r in results])
    print(f"  Retrieval:  {n_ret}/{len(results)} gold top-1")
    print(f"  Generation: {n_match}/{len(results)} FHE=plaintext match")
    print(f"  Avg prefill: {avg_prefill:.1f}s ({avg_prefill_tok:.0f} tokens, plaintext)")
    print(f"  Avg FHE gen: {avg_gen:.1f}s ({avg_gen/args.num_tokens:.1f}s/token)")
    print(f"  Config: D={D}, F={F}, {args.num_blocks} blocks, vocab={vocab_size}")
    print(f"  Protocol: encrypted retrieval -> {args.num_blocks*4} FHE round-trips/token")
    print(f"\n  Generated outputs:")
    for i, r in enumerate(results):
        print(f"    [{i+1}] '{r['generated_text']}'")



def run_retrieval_only(args):
    from scipy.linalg import svd
    from fhe_common import (
        PhantomFHE, pack_complex, pack_complex_conjugate,
        euclidean_to_lorentz, get_embeddings, MODEL_DIR,
    )
    from fhe_spear_retrieval import load_msmarco_samples, load_squad_samples
    from rwkv_emb.model import EmbeddingRWKV
    from rwkv_emb.tokenizer import RWKVTokenizer

    dataset_name = getattr(args, 'dataset', 'msmarco')
    print(f"Retrieval benchmark (dataset={dataset_name}, "
          f"n_docs={args.n_docs}, n_queries={args.n_queries})")

    emb_model = EmbeddingRWKV(str(MODEL_DIR / "rwkv0b4-emb-curriculum.pth"))
    tokenizer = RWKVTokenizer()

    if dataset_name == 'squad':
        dataset_path = getattr(args, 'dataset_path', None)
        passages, test_qa = load_squad_samples(args.n_docs, args.n_queries,
                                                data_path=dataset_path)
    else:
        passages, test_qa = load_msmarco_samples(args.n_docs, args.n_queries)

    doc_embs = get_embeddings(emb_model, tokenizer, passages)
    doc_embs = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
    _, S, Vt = svd(doc_embs, full_matrices=False)
    proj = Vt[:64].T
    doc_proj = doc_embs @ proj
    doc_proj = doc_proj / (np.linalg.norm(doc_proj, axis=1, keepdims=True) + 1e-8)
    doc_lorentz = euclidean_to_lorentz(doc_proj)
    docs_packed = [pack_complex(doc.astype(np.float64)) for doc in doc_lorentz]
    slots_per_doc = len(docs_packed[0])

    q_data = []
    for question, gold_answer, gold_context in test_qa:
        q_emb = get_embeddings(emb_model, tokenizer, [question])[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        q_proj = q_emb @ proj
        q_proj = q_proj / (np.linalg.norm(q_proj) + 1e-8)
        q_lorentz = euclidean_to_lorentz(q_proj.reshape(1, -1)).flatten()
        q_fhe = q_lorentz.copy()
        q_fhe[0] = -q_fhe[0]
        q_packed = pack_complex_conjugate(q_fhe.astype(np.float64))
        q_data.append((question, gold_answer, gold_context, q_fhe, q_packed))

    for mode in ['plaintext', 'ctpt', 'ctct']:
        print(f"\n--- {mode.upper()} ---")

        fhe_ret = None
        encrypted_doc_batches = None
        if mode != 'plaintext':
            fhe_ret = PhantomFHE()
        if mode == 'ctct' and fhe_ret is not None:
            t_enc = time.perf_counter()
            encrypted_doc_batches = fhe_ret.encrypt_docs_batch(docs_packed, slots_per_doc)
            print(f"  Doc encryption: {time.perf_counter() - t_enc:.3f}s ({len(encrypted_doc_batches)} batches)")

        correct = 0
        t0 = time.perf_counter()
        for qi, (question, gold_answer, gold_context, q_fhe, q_packed) in enumerate(q_data):
            if mode == 'plaintext':
                scores = np.array([np.sum(q_fhe * doc) for doc in doc_lorentz])
            elif mode == 'ctct':
                scores = fhe_ret.batched_dot_products_ctct(
                    q_packed, encrypted_doc_batches, slots_per_doc)
            else:
                scores = fhe_ret.batched_dot_products_ctpt(
                    q_packed, docs_packed, slots_per_doc)

            top_idx = int(np.argmax(scores))
            retrieved = passages[top_idx]
            gold_match = (retrieved == gold_context)
            if gold_match:
                correct += 1
            if qi < 5:
                print(f"  Q{qi+1}: {'Y' if gold_match else 'N'} - {retrieved[:60]}...")

        total_time = time.perf_counter() - t0
        print(f"  R@1: {correct}/{len(test_qa)} ({100*correct/len(test_qa):.0f}%)")
        print(f"  Total: {total_time:.3f}s, {1000*total_time/len(test_qa):.1f}ms/query")

        if fhe_ret is not None:
            del fhe_ret
        if encrypted_doc_batches is not None:
            del encrypted_doc_batches

    print(f"\nDone.")


def run_standalone(args):
    D = args.embed_dim
    F = args.ffn_dim

    tokenizer = None
    try:
        from rwkv_emb.tokenizer import RWKVTokenizer
        tokenizer = RWKVTokenizer()
    except Exception:
        try:
            from rwkv.utils import PIPELINE
            from rwkv.model import RWKV as _RWKV_unused  # noqa: ensure rwkv installed
            tokenizer = PIPELINE(None, "rwkv_vocab_v20230424").tokenizer
        except Exception:
            pass

    print(f"Client-aided FHE-RWKV (D={D}, F={F}, blocks={args.num_blocks}, "
          f"tokens={args.num_tokens})")

    print("\nLoading RWKV-7 weights...")
    t0 = time.perf_counter()
    blocks, emb, head_w, ln_out_w, ln_out_b, ln0_w, ln0_b, D, n_head, head_size, vocab_size = \
        load_model_weights(D, F, args.num_blocks)
    args.embed_dim = D
    print(f"  Vocab={vocab_size}, loaded in {time.perf_counter() - t0:.1f}s")

    use_bsgs = args.bsgs
    a100_mode = getattr(args, 'a100', False)
    if use_bsgs:
        if a100_mode:
            N, P, L0 = 8192, 1, 3
        else:
            N, P, L0 = 32768, 3, 24
        print(f"\nSetting up CKKS (BSGS, no bootstrap)...")
        if a100_mode:
            print(f"  A100 mode: N={N}, P={P}, L0={L0} (sm_80 rotate workaround)")
        t0 = time.perf_counter()
        prime_bits = 54 if a100_mode else 59
        ckks = CKKSBootstrapContext(poly_degree=N, L0=L0, prime_bits=prime_bits,
                                    special_mod_size=P,
                                    max_rot_dim=max(D, F),
                                    bsgs_dim=[D, F], skip_bootstrap=True)
        print(f"  CKKS setup: {time.perf_counter() - t0:.1f}s")
        print("\nBootstrap spot-check skipped (BSGS mode)")
    else:
        print(f"\nSetting up CKKS (bootstrap)...")
        t0 = time.perf_counter()
        ckks = CKKSBootstrapContext(L0=24, prime_bits=59, max_rot_dim=max(D, F))
        print(f"  CKKS setup: {time.perf_counter() - t0:.1f}s")

        if not args.no_bootstrap:
            print("\nBootstrap spot-check...")
            x_test = layer_norm(emb[args.seed_token].copy(), ln0_w, ln0_b)
            bootstrap_spot_check(ckks, blocks[0], x_test, np.zeros(D), D)
        else:
            print("\nBootstrap spot-check skipped")

    if args.prompt and tokenizer:
        seed_tokens = tokenizer.encode(args.prompt)
        seed_tokens = [t for t in seed_tokens if t != 65535]
        print(f"\n  Prompt: '{args.prompt}' -> {len(seed_tokens)} tokens")
    else:
        seed_tokens = [args.seed_token]

    use_preencoded = getattr(args, 'preencoded', False)
    mode = "BSGS diagonal" if use_bsgs else "naive per-column"
    if use_preencoded:
        mode += " + pre-encoded diags"
    print(f"\nGenerating {args.num_tokens} tokens ({mode})...")
    tokens_fhe, tokens_ref, total_fhe_time, prefill_time = run_generation(
        ckks, blocks, emb, head_w, ln_out_w, ln_out_b,
        ln0_w, ln0_b, D, n_head, head_size,
        seed_tokens, args.num_tokens, tokenizer,
        use_bsgs=use_bsgs, use_preencoded=use_preencoded)

    print("\nSummary:")
    all_match = (tokens_fhe == tokens_ref)
    print(f"  Tokens (ref):  {tokens_ref}")
    print(f"  Tokens (FHE):  {tokens_fhe}")
    if tokenizer:
        print(f"  Text (FHE):    '{tokenizer.decode(tokens_fhe)}'")
        print(f"  Text (ref):    '{tokenizer.decode(tokens_ref)}'")
    print(f"  All match: {'yes' if all_match else 'no'}")
    n_prefill = max(0, len(seed_tokens) - 1)
    if n_prefill > 0:
        print(f"  Prefill: {n_prefill} tokens in {prefill_time:.1f}s (plaintext)")
    print(f"  FHE generation: {total_fhe_time:.1f}s "
          f"({total_fhe_time/args.num_tokens:.1f}s/token)")
    print(f"  Config: D={D}, F={F}, blocks={args.num_blocks}, vocab={vocab_size}")
    print(f"  Protocol: {args.num_blocks*4} client round-trips/token "
          f"({args.num_blocks} blocks x 4 rounds/block)")

    print(f"  {'Match' if all_match else 'Mismatch (check correlation)'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Client-aided FHE-RWKV with bootstrap")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--ffn_dim", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--num_tokens", type=int, default=5)
    parser.add_argument("--seed_token", type=int, default=3)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (tokenized as seed, last token starts FHE)")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip bootstrap spot-check (faster)")
    parser.add_argument("--rag", action="store_true",
                        help="Run full E2E RAG demo (retrieval + generation)")
    parser.add_argument("--n_docs", type=int, default=100)
    parser.add_argument("--n_queries", type=int, default=3)
    parser.add_argument("--bsgs", action="store_true",
                        help="Use BSGS diagonal matmul (fast, needs more Galois keys)")
    parser.add_argument("--a100", action="store_true",
                        help="A100 (sm_80) compatible config: N=16384, P=1, L0=5 "
                             "(workaround for ph.rotate() bug with P>=2 on sm_80)")
    parser.add_argument("--dataset", type=str, default="msmarco",
                        choices=["msmarco", "squad"],
                        help="Dataset for RAG demo (default: msmarco)")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Custom path to dataset JSONL file")
    parser.add_argument("--retrieval_mode", type=str, default="ctpt",
                        choices=["ctpt", "ctct", "plaintext"],
                        help="Retrieval mode: ctpt (query enc), ctct (both enc), plaintext")
    parser.add_argument("--retrieval_only", action="store_true",
                        help="Run retrieval benchmark only (no generation)")
    parser.add_argument("--prefill", action="store_true",
                        help="RAG: prefill full prompt in plaintext, FHE for answer only")
    parser.add_argument("--preencoded", action="store_true",
                        help="Pre-encode diagonal plaintexts (one-time setup, ~10x BSGS speedup)")
    args = parser.parse_args()

    if args.retrieval_only:
        run_retrieval_only(args)
    elif args.rag:
        run_rag_demo(args)
    else:
        run_standalone(args)


if __name__ == "__main__":
    main()
