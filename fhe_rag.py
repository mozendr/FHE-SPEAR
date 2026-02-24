#!/usr/bin/env python3
import os
os.environ['RWKV_V7_ON'] = '1'
os.environ['RWKV_JIT_ON'] = '1'
os.environ['RWKV_CUDA_ON'] = '1'

import numpy as np
import time
from scipy.linalg import svd

from fhe_common import (
    MODEL_DIR,
    PhantomFHE, pack_complex, pack_complex_conjugate,
    euclidean_to_lorentz, get_embeddings,
)

from fhe_rwkv_inference import (
    CKKSContext, ct_pt_dot, ct_pt_weighted_sum, ct_ct_square,
    normalize_columns, load_weights, magnitude_controlled_weights,
)

from fhe_spear_retrieval import load_msmarco_samples

from rwkv_emb.model import EmbeddingRWKV
from rwkv_emb.tokenizer import RWKVTokenizer


def run_fhe_generation_step(ckks, emb, block_weights, W_head,
                            embed_dim, ffn_dim, vocab_dim, token_id):
    x = emb[token_id]

    h = x.copy()
    for W_key, W_val in block_weights:
        k = h @ W_key
        k_sq = k ** 2
        h = k_sq @ W_val
    logits_ref = h @ W_head
    plain_token = int(np.argmax(logits_ref))

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

    for b in range(1, len(block_weights)):
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

    level = rescales + 1
    ct_logits = [ct_pt_weighted_sum(ckks, ct_h, W_head[:, i], level=level)
                 for i in range(vocab_dim)]
    del ct_h

    logits_fhe = np.array([ckks.decrypt_slot0(ct) for ct in ct_logits])
    fhe_token = int(np.argmax(logits_fhe))

    t_step = time.perf_counter() - t0
    corr = np.corrcoef(logits_fhe, logits_ref)[0, 1]

    return fhe_token, plain_token, corr, t_step


def run_e2e_pipeline(n_docs=100, n_queries=5, embed_dim=64, ffn_dim=128,
                     vocab_dim=32, num_blocks=2, num_gen_tokens=3,
                     retrieval_mode="ctpt", use_mag_ctrl=False):

    print(f"e2e benchmark (ret={retrieval_mode}, mag_ctrl={use_mag_ctrl})")

    print("\nLoading models and data...")
    t_setup = time.perf_counter()

    emb_model = EmbeddingRWKV(str(MODEL_DIR / "rwkv0b4-emb-curriculum.pth"))
    tokenizer = RWKVTokenizer()
    gen_weights = load_weights()
    passages, test_qa = load_msmarco_samples(n_docs, n_queries)

    print(f"  Setup: {time.perf_counter() - t_setup:.1f}s")

    print(f"\nRetrieval setup (N=8192, mode={retrieval_mode})")

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
    t_embed = time.perf_counter() - t0

    fhe_ret = PhantomFHE()
    batch_size = fhe_ret.slot_count // slots_per_doc

    encrypted_doc_batches = None
    if retrieval_mode == "ctct":
        t_enc = time.perf_counter()
        encrypted_doc_batches = fhe_ret.encrypt_docs_batch(docs_packed, slots_per_doc)
        print(f"  Doc encryption: {time.perf_counter() - t_enc:.2f}s "
              f"({len(encrypted_doc_batches)} batches)")

    print(f"  {len(passages)} docs, {slots_per_doc} complex slots each")
    print(f"  SIMD batch: {batch_size} docs/ciphertext")
    print(f"  Embedding + projection: {t_embed:.1f}s")

    print(f"\nGeneration setup (N=32768, {num_blocks} blocks)")

    depth = 3 * num_blocks + 2
    ckks_gen = CKKSContext(depth=depth)

    if use_mag_ctrl or num_blocks >= 4:
        emb, block_weights, W_head = magnitude_controlled_weights(
            gen_weights, embed_dim, ffn_dim, vocab_dim, num_blocks)
        print(f"  Magnitude-controlled normalization: ON")
    else:
        emb = gen_weights['emb.weight'][:vocab_dim, :embed_dim].numpy().astype(np.float64) * 10.0
        block_weights = []
        for b in range(num_blocks):
            W_key = normalize_columns(
                gen_weights[f'blocks.{b}.ffn.key.weight'][:embed_dim, :ffn_dim]
                .numpy().astype(np.float64))
            W_val = normalize_columns(
                gen_weights[f'blocks.{b}.ffn.value.weight'][:ffn_dim, :embed_dim]
                .numpy().astype(np.float64))
            block_weights.append((W_key, W_val))
        W_head = normalize_columns(
            gen_weights['head.weight'][:embed_dim, :vocab_dim]
            .numpy().astype(np.float64))

    print(f"  Context: N=32768, depth={depth}, scale=2^40")
    print(f"  Config: {embed_dim}x{ffn_dim}x{vocab_dim}, {num_blocks} blocks")

    print(f"\nRunning {n_queries} queries x {num_gen_tokens} tokens")

    results = []

    for qi, (question, gold_answer, gold_context) in enumerate(test_qa):
        print(f"\n--- Query {qi+1}/{n_queries} ---")
        print(f"  Q: {question[:80]}...")

        t_ret_start = time.perf_counter()

        q_emb = get_embeddings(emb_model, tokenizer, [question])[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        q_proj = q_emb @ proj
        q_proj = q_proj / (np.linalg.norm(q_proj) + 1e-8)
        q_lorentz = euclidean_to_lorentz(q_proj.reshape(1, -1)).flatten()
        q_fhe = q_lorentz.copy()
        q_fhe[0] = -q_fhe[0]
        q_packed = pack_complex_conjugate(q_fhe.astype(np.float64))

        if retrieval_mode == "ctct":
            scores = fhe_ret.batched_dot_products_ctct(
                q_packed, encrypted_doc_batches, slots_per_doc)
        else:
            scores = fhe_ret.batched_dot_products_ctpt(
                q_packed, docs_packed, slots_per_doc)

        top_idx = int(np.argmax(scores))
        retrieved_passage = passages[top_idx]
        gold_match = (retrieved_passage == gold_context)

        t_ret = time.perf_counter() - t_ret_start
        print(f"  [Retrieval] {t_ret*1000:.1f}ms, gold={'Y' if gold_match else 'N'}")
        print(f"  Retrieved: {retrieved_passage[:80]}...")

        seed_token = 3
        gen_tokens_fhe = [seed_token]
        gen_tokens_plain = [seed_token]
        gen_time = 0.0
        all_match = True

        for step in range(num_gen_tokens):
            fhe_tok, plain_tok, corr, t_step = run_fhe_generation_step(
                ckks_gen, emb, block_weights, W_head,
                embed_dim, ffn_dim, vocab_dim,
                gen_tokens_fhe[-1])

            gen_tokens_fhe.append(fhe_tok)
            gen_tokens_plain.append(plain_tok)
            gen_time += t_step

            match = (fhe_tok == plain_tok)
            if not match:
                all_match = False

            print(f"    Token {step}: plain={plain_tok} fhe={fhe_tok} "
                  f"{'Y' if match else 'N'} corr={corr:.6f} {t_step:.1f}s")

        print(f"  [Generation] {gen_time:.1f}s, all_match={'Y' if all_match else 'N'}")
        print(f"  Tokens: {gen_tokens_fhe}")

        e2e_time = t_ret + gen_time
        print(f"  [E2E] {e2e_time:.1f}s (ret={t_ret*1000:.0f}ms + gen={gen_time:.1f}s)")

        results.append({
            'query': question[:60],
            'retrieval_time': t_ret,
            'retrieval_gold': gold_match,
            'generation_time': gen_time,
            'tokens_match': all_match,
            'tokens_plain': gen_tokens_plain,
            'tokens_fhe': gen_tokens_fhe,
        })

    print(f"\nSummary:")

    n_ret_correct = sum(1 for r in results if r['retrieval_gold'])
    n_gen_match = sum(1 for r in results if r['tokens_match'])
    avg_ret = np.mean([r['retrieval_time'] for r in results])
    avg_gen = np.mean([r['generation_time'] for r in results])
    total = sum(r['retrieval_time'] + r['generation_time'] for r in results)

    print(f"  Queries: {n_queries}")
    print(f"  Retrieval: {n_ret_correct}/{n_queries} gold top-1, "
          f"avg {avg_ret*1000:.0f}ms/query")
    print(f"  Generation: {n_gen_match}/{n_queries} all-token-match, "
          f"{num_gen_tokens} tokens/query, avg {avg_gen:.1f}s/query")
    print(f"  Pipeline: {total:.1f}s total")
    print(f"  Config: {embed_dim}x{ffn_dim}x{vocab_dim}, "
          f"{num_blocks} blocks, {num_gen_tokens} tokens")

    print(f"\nret={n_ret_correct}/{n_queries} gen={n_gen_match}/{n_queries} "
          f"time={total:.1f}s blocks={num_blocks} tokens={num_gen_tokens}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_docs", type=int, default=100)
    parser.add_argument("--n_queries", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--ffn_dim", type=int, default=128)
    parser.add_argument("--vocab_dim", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_tokens", type=int, default=3)
    parser.add_argument("--retrieval_mode", choices=["ctpt", "ctct"], default="ctpt")
    parser.add_argument("--mag_ctrl", action="store_true")
    args = parser.parse_args()

    run_e2e_pipeline(
        n_docs=args.n_docs,
        n_queries=args.n_queries,
        embed_dim=args.embed_dim,
        ffn_dim=args.ffn_dim,
        vocab_dim=args.vocab_dim,
        num_blocks=args.num_blocks,
        num_gen_tokens=args.num_tokens,
        retrieval_mode=args.retrieval_mode,
        use_mag_ctrl=args.mag_ctrl,
    )
