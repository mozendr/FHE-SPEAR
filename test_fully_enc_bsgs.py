#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from fhe_common import PHANTOM_PATH  # noqa: F401
import pyPhantom as ph

from bootstrap_generation import (
    CKKSBootstrapContext,
    compute_bsgs_params,
    fhe_matmul_bsgs,
    _compute_baby_rotations,
    _extract_diagonals,
    _replicate_to_slots,
)


def fully_encrypted_ffn_block(ckks, ct_x_rep, W_key, W_val, D, F,
                               block_idx=0):
    t0 = time.time()
    G, B = compute_bsgs_params(D)
    n_chunks = int(np.ceil(F / D))

    start_level = ct_x_rep.chain_index()
    remaining = (ckks.L0 - 1) - start_level
    print(f"  Block {block_idx}: chain_index={start_level}, "
          f"remaining={remaining}, D={D}, F={F}, {n_chunks} chunks")

    ct_baby = _compute_baby_rotations(ckks, ct_x_rep, G)

    # W^T because BSGS computes M@x, we want x@W
    ct_fk_chunks = []
    for c in range(n_chunks):
        out_start = c * D
        out_end = min(out_start + D, F)
        cols = out_end - out_start

        M = np.zeros((D, D))
        M[:cols, :] = W_key[:, out_start:out_end].T
        ct_chunk = fhe_matmul_bsgs(ckks, ct_x_rep, M, D, G, B, ct_baby)
        ct_fk_chunks.append(ct_chunk)

    t_key = time.time() - t0
    print(f"    FFN key: {t_key:.1f}s, {n_chunks} BSGS calls, "
          f"chain_index now={ct_fk_chunks[0].chain_index()}")

    t1 = time.time()
    ct_sq_chunks = []
    for ct_fk in ct_fk_chunks:
        ct_sq = ph.multiply(ckks.ctx, ct_fk, ct_fk)
        ct_sq = ph.relinearize(ckks.ctx, ct_sq, ckks.rlk)
        ct_sq = ph.rescale_to_next(ckks.ctx, ct_sq)
        ct_sq_chunks.append(ct_sq)
    t_sq = time.time() - t1
    print(f"    Square: {t_sq:.1f}s, chain_index now={ct_sq_chunks[0].chain_index()}")

    t2 = time.time()
    ct_val_acc = None
    for c, ct_sq in enumerate(ct_sq_chunks):
        in_start = c * D
        in_end = min(in_start + D, F)
        rows = in_end - in_start

        M = np.zeros((D, D))
        M[:, :rows] = W_val[in_start:in_end, :].T

        ct_baby_sq = _compute_baby_rotations(ckks, ct_sq, compute_bsgs_params(D)[0])
        ct_partial = fhe_matmul_bsgs(ckks, ct_sq, M, D,
                                      compute_bsgs_params(D)[0],
                                      compute_bsgs_params(D)[1],
                                      ct_baby_sq)
        if ct_val_acc is None:
            ct_val_acc = ct_partial
        else:
            level_acc = ct_val_acc.chain_index()
            level_par = ct_partial.chain_index()
            if level_acc < level_par:
                while ct_val_acc.chain_index() < ct_partial.chain_index():
                    ct_partial = ph.mod_switch_to_next(ckks.ctx, ct_partial)
            elif level_par < level_acc:
                while ct_partial.chain_index() < ct_val_acc.chain_index():
                    ct_val_acc = ph.mod_switch_to_next(ckks.ctx, ct_val_acc)
            ct_val_acc = ph.add(ckks.ctx, ct_val_acc, ct_partial)

    t_val = time.time() - t2
    print(f"    FFN val: {t_val:.1f}s, {n_chunks} BSGS calls, "
          f"chain_index now={ct_val_acc.chain_index()}")

    # level+scale alignment for residual add
    t3 = time.time()
    ct_x_aligned = ct_x_rep
    level_x = ct_x_aligned.chain_index()
    level_v = ct_val_acc.chain_index()
    if level_x < level_v:
        while ct_x_aligned.chain_index() < level_v:
            ct_x_aligned = ph.mod_switch_to_next(ckks.ctx, ct_x_aligned)
    elif level_v < level_x:
        while ct_val_acc.chain_index() < level_x:
            ct_val_acc = ph.mod_switch_to_next(ckks.ctx, ct_val_acc)
    ct_val_acc.set_scale(ct_x_aligned.scale())
    ct_out = ph.add(ckks.ctx, ct_x_aligned, ct_val_acc)
    t_res = time.time() - t3

    total = time.time() - t0
    end_level = ct_out.chain_index()
    levels_used = end_level - start_level
    print(f"    Residual: {t_res:.1f}s | TOTAL: {total:.1f}s | "
          f"levels: {start_level} -> {end_level} (used {levels_used})")

    return ct_out, levels_used


def plaintext_ffn_block(x, W_key, W_val):
    fk = x @ W_key
    fk_sq = fk ** 2
    fv = fk_sq @ W_val
    return x + fv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--D', type=int, default=64)
    parser.add_argument('--F', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=7)
    parser.add_argument('--L0', type=int, default=23)
    parser.add_argument('--P', type=int, default=3)
    parser.add_argument('--N', type=int, default=32768)
    parser.add_argument('--no-bootstrap', action='store_true')
    parser.add_argument('--a100', action='store_true')
    parser.add_argument('--use-model-weights', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    D, F = args.D, args.F
    num_blocks = args.num_blocks
    use_bootstrap = not args.no_bootstrap
    L0 = args.L0
    P = args.P
    N = args.N

    if args.a100:
        N = 16384
        P = 1

    np.random.seed(args.seed)

    print(f"Fully encrypted FFN (D={D}, F={F}, blocks={num_blocks}, "
          f"L0={L0}, P={P}, N={N}"
          + (", bootstrap" if use_bootstrap else "") + ")")

    if args.use_model_weights:
        print("\nLoading RWKV model weights...")
        t0 = time.time()
        from fhe_rwkv_inference import load_weights
        w = load_weights()
        print(f"  Loaded in {time.time()-t0:.1f}s")
        W_keys, W_vals_raw = [], []
        for b in range(num_blocks):
            Wk = w[f'blocks.{b}.ffn.key.weight'][:D, :F].numpy().astype(np.float64)
            Wv = w[f'blocks.{b}.ffn.value.weight'][:F, :D].numpy().astype(np.float64)
            W_keys.append(Wk)
            W_vals_raw.append(Wv)
    else:
        print("\nGenerating random weights...")
        W_keys, W_vals_raw = [], []
        for b in range(num_blocks):
            Wk = np.random.randn(D, F).astype(np.float64) * 0.02
            Wv = np.random.randn(F, D).astype(np.float64) * 0.02
            W_keys.append(Wk)
            W_vals_raw.append(Wv)
        print(f"  Generated {num_blocks} blocks of random weights")

    # Calibrate magnitude control (folded into W_val)
    print("\nCalibrating magnitude control...")
    x_cal = np.random.randn(D) * 0.1
    W_vals = []
    x_ref = x_cal.copy()
    target_mag = 1.0
    for b in range(num_blocks):
        fk = x_ref @ W_keys[b]
        fk_sq = fk ** 2
        fv = fk_sq @ W_vals_raw[b]
        raw_mag = np.max(np.abs(fv))
        ms = target_mag / (raw_mag + 1e-12)
        W_vals.append(W_vals_raw[b] * ms)
        x_ref = x_ref + fv * ms
        print(f"  Block {b}: raw_mag={raw_mag:.4f}, scale={ms:.6f}, "
              f"|x|_max={np.max(np.abs(x_ref)):.4f}")

    print("\nPlaintext reference...")
    x_pt = x_cal.copy()
    x_pt_per_block = [x_cal.copy()]
    for b in range(num_blocks):
        x_pt = plaintext_ffn_block(x_pt, W_keys[b], W_vals[b])
        x_pt_per_block.append(x_pt.copy())
        print(f"  Block {b}: |x|_max={np.max(np.abs(x_pt)):.6f}")

    print(f"\nSetting up CKKS (bootstrap={'on' if use_bootstrap else 'off'})...")
    t0 = time.time()

    ckks = CKKSBootstrapContext(
        poly_degree=N,
        L0=L0,
        prime_bits=59,
        special_mod_size=P,
        level_budget=[2, 2] if use_bootstrap else None,
        max_rot_dim=max(D, F),
        bsgs_dim=[D, F],
        skip_bootstrap=not use_bootstrap,
    )

    print(f"  Setup done in {time.time()-t0:.1f}s (slots={ckks.slots})")

    usable_levels = L0 - 1
    blocks_possible = usable_levels // 3
    print(f"  Usable levels: {usable_levels}, blocks at 3 levels each: {blocks_possible}")

    print("\nEncrypting input...")
    ct = ckks.encrypt_replicated(x_cal)
    print(f"  Initial chain_index={ct.chain_index()}")

    dec_check = ckks.decrypt_vec(ct, D)
    enc_err = np.max(np.abs(dec_check - x_cal))
    print(f"  Encryption error: {enc_err:.2e}")

    print(f"\nRunning {num_blocks} fully encrypted FFN blocks...")
    total_bootstraps = 0
    blocks_completed = 0

    for b in range(num_blocks):
        current_ci = ct.chain_index()
        remaining = (L0 - 1) - current_ci
        levels_needed = 4  # 3 per block + 1 margin

        if remaining < levels_needed:
            if use_bootstrap:
                print(f"\n  >>> BOOTSTRAP before block {b} "
                      f"(chain_index={current_ci}, remaining={remaining})")
                t_bt = time.time()
                ct = ckks.bootstrap(ct)
                total_bootstraps += 1

                # Bootstrap outputs at scale ~ (2^bits)^2, must rescale
                bt_scale = ct.scale()
                ct = ph.rescale_to_next(ckks.ctx, ct)
                new_ci = ct.chain_index()
                new_remaining = (L0 - 1) - new_ci
                print(f"  >>> Bootstrap done: {time.time()-t_bt:.1f}s, "
                      f"chain_index={new_ci}, remaining={new_remaining}, "
                      f"scale: {bt_scale:.2e} -> {ct.scale():.2e}")
                dec_bt = ckks.decrypt_vec(ct, D)
                bt_err = np.max(np.abs(dec_bt - x_pt_per_block[b]))
                print(f"  >>> Post-bootstrap |vals|_max={np.max(np.abs(dec_bt)):.6f}, "
                      f"err vs ref={bt_err:.2e}")
            else:
                print(f"\n  *** OUT OF LEVELS at block {b} "
                      f"(chain_index={current_ci}, remaining={remaining})")
                break

        ct, levels_used = fully_encrypted_ffn_block(
            ckks, ct, W_keys[b], W_vals[b], D, F, block_idx=b
        )

        dec_vals = ckks.decrypt_vec(ct, D)
        fhe_max = np.max(np.abs(dec_vals))
        ref_vals = x_pt_per_block[b + 1]
        corr = np.corrcoef(dec_vals, ref_vals)[0, 1]
        max_err = np.max(np.abs(dec_vals - ref_vals))
        print(f"  Block {b} verified: corr={corr:.10f}, max_err={max_err:.2e}, "
              f"|fhe|={fhe_max:.4f}\n")
        blocks_completed = b + 1

    print("\nResults:")

    dec_final = ckks.decrypt_vec(ct, D)
    ref_final = x_pt_per_block[blocks_completed]

    corr = np.corrcoef(dec_final, ref_final)[0, 1]
    max_err = np.max(np.abs(dec_final - ref_final))
    rel_err = max_err / (np.max(np.abs(ref_final)) + 1e-12)

    print(f"  Blocks completed: {blocks_completed}/{num_blocks}")
    print(f"  Bootstraps used: {total_bootstraps}")
    print(f"  Final correlation: {corr:.10f}")
    print(f"  Max absolute error: {max_err:.6e}")
    print(f"  Relative error: {rel_err:.6e}")
    print(f"  |fhe|_max: {np.max(np.abs(dec_final)):.6f}")
    print(f"  |ref|_max: {np.max(np.abs(ref_final)):.6f}")

    print(f"  corr={corr:.8f} ({'match' if corr > 0.999 else 'degraded'})")


if __name__ == '__main__':
    main()
