#!/usr/bin/env python3
import os
import sys
os.environ['RWKV_V7_ON'] = '1'
os.environ['RWKV_JIT_ON'] = '1'
os.environ['RWKV_CUDA_ON'] = '1'

import numpy as np
import torch
import json
import time
import random
from pathlib import Path
from scipy.linalg import svd

from fhe_common import (
    USE_PHANTOM_GPU, MODEL_DIR, DATA_DIR,
    euclidean_to_lorentz, pack_complex, pack_complex_conjugate,
    get_embeddings, download_fallback_model,
    PhantomFHE, TenSEALFHE,
)

from rwkv_emb.model import EmbeddingRWKV
from rwkv_emb.tokenizer import RWKVTokenizer
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS


def load_msmarco_samples(n_docs=100, n_queries=10):
    all_qa = []

    with open(DATA_DIR / "msmarco_sft.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 500:
                break
            sample = json.loads(line)
            q = sample['query']

            if 'Context:' in q and 'Question:' in q:
                ctx_start = q.find('Context:') + 8
                q_start = q.find('Question:')
                context = q[ctx_start:q_start].strip()
                question = q[q_start+9:].strip()
                answer = sample['response']

                if len(context) > 50 and len(question) > 10:
                    all_qa.append((question, answer, context))

    random.seed(42)
    random.shuffle(all_qa)

    test_qa = all_qa[:n_queries]

    gold_passages = [ctx for _, _, ctx in test_qa]
    distractor_passages = [ctx for _, _, ctx in all_qa[n_queries:n_queries + n_docs - n_queries]]

    passages = gold_passages + distractor_passages
    random.shuffle(passages)

    print(f"  {len(passages)} passages, {len(test_qa)} queries")
    return passages, test_qa


def load_squad_samples(n_docs=100, n_queries=10, data_path=None):
    if data_path is None:
        data_path = Path(__file__).parent / "squad_sft.jsonl"

    all_qa = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break
            sample = json.loads(line)
            q = sample['query']

            if 'Context:' in q and 'Question:' in q:
                ctx_start = q.find('Context:') + 8
                q_start = q.find('Question:')
                context = q[ctx_start:q_start].strip()
                question = q[q_start+9:].strip()
                answer = sample['response']

                if len(context) > 50 and len(question) > 10:
                    all_qa.append((question, answer, context))

    random.seed(42)
    random.shuffle(all_qa)

    test_qa = all_qa[:n_queries]

    gold_passages = [ctx for _, _, ctx in test_qa]
    distractor_passages = [ctx for _, _, ctx in all_qa[n_queries:n_queries + n_docs - n_queries]]

    passages = gold_passages + distractor_passages
    random.shuffle(passages)

    print(f"  [SQuAD] {len(passages)} passages, {len(test_qa)} queries")
    return passages, test_qa


def run_benchmark(fhe, passages, test_qa, docs_packed, doc_lorentz, proj,
                   emb_model, tokenizer, gen_pipeline, mode="ctpt", encrypted_doc_batches=None):
    slots_per_doc = len(docs_packed[0])
    n_queries = len(test_qa)

    total_time = 0
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0

    for i, (question, gold_answer, gold_context) in enumerate(test_qa):
        show_detail = (i < 3)
        if show_detail:
            print(f"\n[Query {i+1}] {question[:70]}...")

        q_emb = get_embeddings(emb_model, tokenizer, [question])[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        q_proj = q_emb @ proj
        q_proj = q_proj / (np.linalg.norm(q_proj) + 1e-8)
        q_lorentz = euclidean_to_lorentz(q_proj.reshape(1, -1)).flatten()
        q_fhe = q_lorentz.copy()
        q_fhe[0] = -q_fhe[0]

        t0 = time.perf_counter()
        q_packed = pack_complex_conjugate(q_fhe.astype(np.float64))

        if isinstance(fhe, PhantomFHE):
            if mode == "ctct":
                scores = fhe.batched_dot_products_ctct(q_packed, encrypted_doc_batches, slots_per_doc)
            else:
                scores = fhe.batched_dot_products_ctpt(q_packed, docs_packed, slots_per_doc)
        else:
            enc_q = fhe.encrypt_complex(q_packed)
            scores = []
            for j in range(len(passages)):
                score = fhe.dot_product(enc_q, docs_packed[j])
                scores.append(score)
            scores = np.array(scores)

        fhe_time = time.perf_counter() - t0
        total_time += fhe_time

        top_idx = np.argmax(scores)
        retrieved = passages[top_idx]

        top10_idx = np.argsort(scores)[-10:][::-1]
        top5_idx = top10_idx[:5]
        top1_idx = top10_idx[0]
        gold_in_top1 = (passages[top1_idx] == gold_context)
        gold_in_top5 = any(passages[idx] == gold_context for idx in top5_idx)
        gold_in_top10 = any(passages[idx] == gold_context for idx in top10_idx)
        if gold_in_top1:
            correct_top1 += 1
        if gold_in_top5:
            correct_top5 += 1
        if gold_in_top10:
            correct_top10 += 1

        if show_detail:
            print(f"  FHE time: {fhe_time*1000:.1f}ms | Top-1: {gold_in_top1} | Top-5: {gold_in_top5}")
            print(f"  Retrieved: {retrieved[:100]}...")

            if gen_pipeline:
                prompt = f"""User: /no_think Answer in 1 sentence based on the context.

Context: {retrieved[:500]}

Question: {question}
Assistant:"""

                args = PIPELINE_ARGS(temperature=0.5, top_p=0.8, token_stop=['\n\nUser', '\nUser:', '<think>'])
                answer = gen_pipeline.generate(prompt, token_count=60, args=args, callback=None)
                answer = answer.strip()
                if '<think>' in answer:
                    answer = answer.split('<think>')[0].strip()
                if answer.startswith('Assistant:'):
                    answer = answer[10:].strip()

                print(f"  Generated: {answer[:100]}")
                print(f"  Gold: {gold_answer[:60]}...")
        else:
            if i == 3:
                print(f"\n... running {n_queries - 3} more queries ...")

    return {
        "correct_top1": correct_top1,
        "correct_top5": correct_top5,
        "correct_top10": correct_top10,
        "total_time": total_time,
        "n_queries": n_queries
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ctpt", "ctct", "compare"], default="compare")
    parser.add_argument("--n_docs", type=int, default=500)
    parser.add_argument("--n_queries", type=int, default=50)
    parser.add_argument("--no_gen", action="store_true")
    args = parser.parse_args()

    print("FHE-SPEAR MS-MARCO")

    passages, test_qa = load_msmarco_samples(args.n_docs, args.n_queries)

    emb_model = EmbeddingRWKV(str(MODEL_DIR / "rwkv0b4-emb-curriculum.pth"))
    tokenizer = RWKVTokenizer()

    gen_pipeline = None
    if not args.no_gen:
        g1_path = MODEL_DIR / "rwkv7-g1d-1.5b-20260212-ctx8192.pth"
        if not g1_path.exists():
            g1_path = MODEL_DIR / "rwkv7-g1c-1.5b-20260110-ctx8192.pth"
        if g1_path.exists():
            gen_model = RWKV(model=str(g1_path).replace(".pth", ""), strategy="cuda fp16")
        else:
            fallback_path = MODEL_DIR / "RWKV-7-1.5B-World.pth"
            if not fallback_path.exists():
                resp = input("G1c model not found. Download fallback RWKV-7-World 1.5B? [y/N] ")
                if resp.lower() == 'y':
                    download_fallback_model()
                else:
                    raise FileNotFoundError("No generation model available")
            gen_model = RWKV(model=str(fallback_path).replace(".pth", ""), strategy="cuda fp16")
        gen_pipeline = PIPELINE(gen_model, "rwkv_vocab_v20230424")

    doc_embs = get_embeddings(emb_model, tokenizer, passages)
    doc_embs = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)

    _, S, Vt = svd(doc_embs, full_matrices=False)
    proj = Vt[:64].T
    doc_proj = doc_embs @ proj
    doc_proj = doc_proj / (np.linalg.norm(doc_proj, axis=1, keepdims=True) + 1e-8)

    doc_lorentz = euclidean_to_lorentz(doc_proj)

    docs_packed = [pack_complex(doc.astype(np.float64)) for doc in doc_lorentz]
    slots_per_doc = len(docs_packed[0])
    print(f"  {len(docs_packed)} docs, {slots_per_doc} complex slots each")

    fhe = PhantomFHE() if USE_PHANTOM_GPU else TenSEALFHE()

    encrypted_doc_batches = None
    if USE_PHANTOM_GPU:
        batch_size = fhe.slot_count // slots_per_doc
        print(f"  SIMD batch size: {batch_size} docs per ciphertext")

        if args.mode in ["ctct", "compare"]:
            t0 = time.perf_counter()
            encrypted_doc_batches = fhe.encrypt_docs_batch(docs_packed, slots_per_doc)
            enc_time = time.perf_counter() - t0
            print(f"  Encrypted {len(encrypted_doc_batches)} batches in {enc_time:.2f}s")

    def print_results(results, mode_name):
        n_q = results["n_queries"]
        print(f"{mode_name}: R@1={100*results['correct_top1']/n_q:.0f}% R@5={100*results['correct_top5']/n_q:.0f}% R@10={100*results['correct_top10']/n_q:.0f}% {1000*results['total_time']/n_q:.1f}ms/q")

    if args.mode == "compare" and USE_PHANTOM_GPU:
        print("\nCT-PT mode")
        ctpt_results = run_benchmark(
            fhe, passages, test_qa, docs_packed, doc_lorentz, proj,
            emb_model, tokenizer, gen_pipeline, mode="ctpt"
        )

        print("\nCT-CT mode")
        ctct_results = run_benchmark(
            fhe, passages, test_qa, docs_packed, doc_lorentz, proj,
            emb_model, tokenizer, None, mode="ctct",
            encrypted_doc_batches=encrypted_doc_batches
        )

        print(f"\n{args.n_docs} docs, {args.n_queries} queries")
        print_results(ctpt_results, "CT-PT")
        print_results(ctct_results, "CT-CT")

        speedup = ctct_results['total_time'] / ctpt_results['total_time']
        print(f"\nCT-CT overhead: {speedup:.1f}x")

    else:
        mode = args.mode if args.mode != "compare" else "ctpt"
        mode_name = "CT-PT" if mode == "ctpt" else "CT-CT"

        print(f"\n{mode_name} mode")

        results = run_benchmark(
            fhe, passages, test_qa, docs_packed, doc_lorentz, proj,
            emb_model, tokenizer, gen_pipeline, mode=mode,
            encrypted_doc_batches=encrypted_doc_batches
        )

        print(f"\n{args.n_docs} docs, {args.n_queries} queries")
        print_results(results, mode_name)

    print(f"\nBackend: {'PhantomFHE GPU' if USE_PHANTOM_GPU else 'TenSEAL CPU'}")

if __name__ == "__main__":
    main()
