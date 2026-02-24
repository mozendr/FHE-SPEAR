#!/usr/bin/env python3
import os
import sys
import re
import json
import time
import random
import argparse
import numpy as np
from pathlib import Path

PHANTOM_PATH = os.path.expanduser("~/gpu_fhe_libs/phantom-fhe/build/lib")
if PHANTOM_PATH not in sys.path:
    sys.path.insert(0, PHANTOM_PATH)

import pyPhantom as phantom

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from fhe_common import (PhantomFHE, euclidean_to_lorentz, pack_complex,
                         pack_complex_conjugate, get_embeddings)
from fhe_spear_retrieval import load_msmarco_samples, load_squad_samples
from numpy.linalg import svd

PII_PATTERNS = {
    'SSN': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    'PHONE': re.compile(r'\b(?:\+1[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    'EMAIL': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'DATE': re.compile(r'\b(?:January|February|March|April|May|June|July|August|'
                       r'September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'),
    'MONEY': re.compile(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?'),
    'PERCENT': re.compile(r'\b\d+(?:\.\d+)?%'),
    'YEAR_EVENT': re.compile(r'\b(?:in|since|from|until|after|before|during)\s+\d{4}\b'),
    'ORG_PAREN': re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\([A-Z]{2,}\)'),
    'MEDICAL_STAT': re.compile(r'\b\d+(?:\.\d+)?%\s*(?:to\s+\d+(?:\.\d+)?%\s*)?of\s+'
                               r'(?:men|women|patients|people|adults|children)', re.I),
    'DOSAGE': re.compile(r'\b\d+(?:\.\d+)?\s*(?:mg|IU|mL|mcg|g/dL|mmHg)\b'),
}

CLASS_MAP = {
    'MONEY': 'financial', 'PERCENT': 'financial',
    'SSN': 'personal', 'PHONE': 'personal', 'EMAIL': 'personal',
    'DATE': 'temporal', 'YEAR_EVENT': 'temporal',
    'ORG_PAREN': 'organizational',
    'MEDICAL_STAT': 'medical', 'DOSAGE': 'medical',
}


def detect_pii(text):
    spans = []
    for pii_type, pattern in PII_PATTERNS.items():
        for m in pattern.finditer(text):
            spans.append({'type': pii_type, 'start': m.start(),
                          'end': m.end(), 'text': m.group()})
    spans.sort(key=lambda s: s['start'])
    merged = []
    for span in spans:
        if merged and span['start'] < merged[-1]['end']:
            if span['end'] > merged[-1]['end']:
                merged[-1]['end'] = span['end']
                merged[-1]['text'] = text[merged[-1]['start']:merged[-1]['end']]
                merged[-1]['type'] += '+' + span['type']
        else:
            merged.append(dict(span))
    return merged


def classify_passage(text):
    pii = detect_pii(text)
    classes = set()
    for span in pii:
        for pii_type in span['type'].split('+'):
            if pii_type in CLASS_MAP:
                classes.add(CLASS_MAP[pii_type])
    return classes


def generate_class_noise(classes, slots_per_doc, noise_scale, rng):
    noise = {}
    for c in classes:
        raw = rng.standard_normal(slots_per_doc) + 1j * rng.standard_normal(slots_per_doc)
        raw = raw / np.abs(raw).mean()
        noise[c] = raw * noise_scale
    return noise


def noise_passages(docs_packed, passage_classes, class_noise):
    noised = []
    for i, doc in enumerate(docs_packed):
        total_noise = np.zeros(len(doc), dtype=complex)
        for c in passage_classes[i]:
            total_noise += class_noise[c][:len(doc)]
        noised.append(doc + total_noise)
    return noised


def encrypt_docs_with_mapping(fhe, docs_packed, slots_per_doc):
    n_docs = len(docs_packed)
    batch_size = fhe.slot_count // slots_per_doc
    encrypted_batches = []
    mapping = []  # (batch_idx, slot_offset) per doc

    for batch_start in range(0, n_docs, batch_size):
        batch_end = min(batch_start + batch_size, n_docs)
        batch_docs = docs_packed[batch_start:batch_end]
        batch_idx = len(encrypted_batches)

        doc_slots = []
        for i, doc in enumerate(batch_docs):
            mapping.append((batch_idx, i * slots_per_doc))
            doc_slots.extend(list(doc))
        doc_slots += [complex(0, 0)] * (fhe.slot_count - len(doc_slots))

        doc_pt = fhe.encoder.encode_complex_vector(fhe.ctx, doc_slots, fhe.scale)
        enc_docs = fhe.pk.encrypt_asymmetric(fhe.ctx, doc_pt)
        encrypted_batches.append((enc_docs, len(batch_docs)))

    return encrypted_batches, mapping


def generate_corrections(fhe, class_noise, passage_classes, mapping,
                         slots_per_doc, n_batches, authorized_classes, rng):
    all_classes = sorted(set(c for cs in passage_classes for c in cs))
    corrections = {}  # keyed by (class_name, batch_idx)

    for c in all_classes:
        for b in range(n_batches):
            correction_slots = [complex(0, 0)] * fhe.slot_count

            for doc_idx, (bi, offset) in enumerate(mapping):
                if bi != b:
                    continue
                if c not in passage_classes[doc_idx]:
                    continue
                noise_vec = class_noise[c]
                for j in range(min(slots_per_doc, len(noise_vec))):
                    if c in authorized_classes:
                        correction_slots[offset + j] = -noise_vec[j]
                    else:
                        # dummy: random values in same slots (indistinguishable)
                        correction_slots[offset + j] = (
                            rng.standard_normal() + 1j * rng.standard_normal()
                        ) * np.abs(noise_vec[j])

            ct_pt = fhe.encoder.encode_complex_vector(fhe.ctx, correction_slots, fhe.scale)
            ct = fhe.pk.encrypt_asymmetric(fhe.ctx, ct_pt)
            corrections[(c, b)] = ct

    return corrections, all_classes


def apply_corrections(fhe, encrypted_batches, corrections, all_classes):
    corrected = []
    for b, (enc_docs, n_docs) in enumerate(encrypted_batches):
        result = enc_docs
        for c in all_classes:
            key = (c, b)
            if key in corrections:
                result = phantom.add(fhe.ctx, result, corrections[key])
        corrected.append((result, n_docs))
    return corrected


def ctct_retrieval(fhe, query_packed, encrypted_batches, slots_per_doc):
    scores = []
    for enc_docs, actual_batch in encrypted_batches:
        query_slots = []
        for _ in range(actual_batch):
            query_slots.extend(list(query_packed))
        query_slots += [complex(0, 0)] * (fhe.slot_count - len(query_slots))

        q_pt = fhe.encoder.encode_complex_vector(fhe.ctx, query_slots, fhe.scale)
        enc_q = fhe.pk.encrypt_asymmetric(fhe.ctx, q_pt)

        result = phantom.multiply(fhe.ctx, enc_q, enc_docs)
        result = phantom.relinearize(fhe.ctx, result, fhe.rlk)
        result = phantom.rescale_to_next(fhe.ctx, result)

        dec_pt = fhe.sk.decrypt(fhe.ctx, result)
        dec_vec = fhe.encoder.decode_complex_vector(fhe.ctx, dec_pt)

        for i in range(actual_batch):
            start = i * slots_per_doc
            end = start + slots_per_doc
            score = sum(c.real for c in dec_vec[start:end])
            scores.append(score)

    return np.array(scores)


def noise_security_analysis(docs_packed, passage_classes, class_noise, noise_scales):
    print("\n  Noise Security Sweep")
    print(f"  {'Scale':>8} {'Auth mean':>10} {'Auth std':>10} {'Unauth mean':>12} {'Unauth std':>12} {'Separation':>12}")

    rng = np.random.default_rng(123)
    q = rng.standard_normal(len(docs_packed[0])) + 1j * rng.standard_normal(len(docs_packed[0]))
    q = q / np.linalg.norm(q)

    for scale in noise_scales:
        scaled_noise = {c: n * (scale / 100.0) for c, n in class_noise.items()}

        auth_scores = []
        unauth_scores = []

        for i, doc in enumerate(docs_packed):
            true_score = np.real(np.sum(q * doc))
            if passage_classes[i]:
                total_noise = sum(scaled_noise[c][:len(doc)] for c in passage_classes[i])
                noised_score = np.real(np.sum(q * (doc + total_noise)))
                unauth_scores.append(noised_score)
                auth_scores.append(true_score)
            else:
                auth_scores.append(true_score)

        auth_arr = np.array(auth_scores)
        if unauth_scores:
            unauth_arr = np.array(unauth_scores)
            sep = np.abs(unauth_arr).mean() / (np.abs(auth_arr).mean() + 1e-10)
            print(f"  {scale:>8.0f} {auth_arr.mean():>10.4f} {auth_arr.std():>10.4f} "
                  f"{unauth_arr.mean():>12.4f} {unauth_arr.std():>12.4f} {sep:>12.1f}x")
        else:
            print(f"  {scale:>8.0f} {auth_arr.mean():>10.4f} {auth_arr.std():>10.4f} "
                  f"{'(no restricted)':>12} {'':>12} {'':>12}")


def run_pipeline(args):
    if args.dataset == "squad":
        passages, test_qa = load_squad_samples(args.n_docs, args.n_queries)
    else:
        passages, test_qa = load_msmarco_samples(args.n_docs, args.n_queries)

    print("\nPassage classification")
    passage_classes = [classify_passage(p) for p in passages]
    class_counts = {}
    for cs in passage_classes:
        for c in cs:
            class_counts[c] = class_counts.get(c, 0) + 1
    n_restricted = sum(1 for cs in passage_classes if cs)
    print(f"  {len(passages)} passages, {n_restricted} restricted, "
          f"{len(passages) - n_restricted} unrestricted")
    for c, cnt in sorted(class_counts.items()):
        print(f"    {c}: {cnt} passages")

    all_pii = []
    for p in passages:
        all_pii.extend(detect_pii(p))
    pii_type_counts = {}
    for s in all_pii:
        for t in s['type'].split('+'):
            pii_type_counts[t] = pii_type_counts.get(t, 0) + 1
    print(f"  Total PII spans: {len(all_pii)}")
    for t, cnt in sorted(pii_type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t}: {cnt}")

    print("\nEmbedding + projection")
    from fhe_common import MODEL_DIR
    from rwkv_emb.model import EmbeddingRWKV
    from rwkv_emb.tokenizer import RWKVTokenizer

    emb_model = EmbeddingRWKV(str(MODEL_DIR / "rwkv0b4-emb-curriculum.pth"))
    tokenizer = RWKVTokenizer()

    doc_embs = get_embeddings(emb_model, tokenizer, passages).astype(np.float32)
    doc_embs = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
    _, S, Vt = svd(doc_embs, full_matrices=False)
    proj = Vt[:64].T
    doc_proj = doc_embs @ proj
    doc_proj = doc_proj / (np.linalg.norm(doc_proj, axis=1, keepdims=True) + 1e-8)
    doc_lorentz = euclidean_to_lorentz(doc_proj)
    docs_packed = [pack_complex(doc.astype(np.float64)) for doc in doc_lorentz]
    slots_per_doc = len(docs_packed[0])
    avg_emb_norm = np.mean([np.linalg.norm(d) for d in docs_packed])
    print(f"  {len(docs_packed)} docs, {slots_per_doc} complex slots, avg ||e||={avg_emb_norm:.4f}")

    print("\nNoise masking")
    rng = np.random.default_rng(42)
    all_classes_set = sorted(set(c for cs in passage_classes for c in cs))
    class_noise = generate_class_noise(all_classes_set, slots_per_doc, args.noise_scale, rng)
    for c in all_classes_set:
        n_mag = np.mean(np.abs(class_noise[c]))
        print(f"  {c}: ||n||_avg={n_mag:.2f} ({n_mag/avg_emb_norm:.0f}x avg embedding)")

    noised_docs = noise_passages(docs_packed, passage_classes, class_noise)

    if args.noise_only:
        noise_security_analysis(docs_packed, passage_classes, class_noise,
                                [1, 10, 50, 100, 500])
        return

    print("\nEncrypted corpus + corrections")
    fhe = PhantomFHE()
    batch_size = fhe.slot_count // slots_per_doc
    print(f"  SIMD: {batch_size} docs/ct, {fhe.slot_count} slots")

    t0 = time.perf_counter()
    encrypted_batches, mapping = encrypt_docs_with_mapping(fhe, noised_docs, slots_per_doc)
    enc_time = time.perf_counter() - t0
    n_batches = len(encrypted_batches)
    print(f"  Encrypted {n_batches} batch(es) in {enc_time:.3f}s")

    t0 = time.perf_counter()
    baseline_batches, _ = encrypt_docs_with_mapping(fhe, docs_packed, slots_per_doc)
    base_enc_time = time.perf_counter() - t0
    print(f"  Baseline (no noise) encrypted in {base_enc_time:.3f}s")

    alice_auth = set(all_classes_set)
    bob_auth = {'temporal'} if 'temporal' in all_classes_set else set()

    t0 = time.perf_counter()
    alice_corr, all_cls = generate_corrections(
        fhe, class_noise, passage_classes, mapping, slots_per_doc,
        n_batches, alice_auth, rng)
    alice_corr_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    bob_corr, _ = generate_corrections(
        fhe, class_noise, passage_classes, mapping, slots_per_doc,
        n_batches, bob_auth, rng)
    bob_corr_time = time.perf_counter() - t0

    n_real_alice = sum(1 for (c, b) in alice_corr if c in alice_auth)
    n_dummy_alice = len(alice_corr) - n_real_alice
    n_real_bob = sum(1 for (c, b) in bob_corr if c in bob_auth)
    n_dummy_bob = len(bob_corr) - n_real_bob
    print(f"  Alice (full): {n_real_alice} real + {n_dummy_alice} dummy corrections ({alice_corr_time:.3f}s)")
    print(f"  Bob (limited={bob_auth}): {n_real_bob} real + {n_dummy_bob} dummy corrections ({bob_corr_time:.3f}s)")

    print("\nServer applies corrections")
    pre_chain = encrypted_batches[0][0].chain_index()

    t0 = time.perf_counter()
    alice_batches = apply_corrections(fhe, encrypted_batches, alice_corr, all_cls)
    alice_apply_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    bob_batches = apply_corrections(fhe, encrypted_batches, bob_corr, all_cls)
    bob_apply_time = time.perf_counter() - t0

    # CT-CT add consumes 0 levels
    post_chain_alice = alice_batches[0][0].chain_index()
    post_chain_bob = bob_batches[0][0].chain_index()
    print(f"  Alice: {alice_apply_time*1000:.2f}ms, chain_index {pre_chain}->{post_chain_alice} (0 levels)")
    print(f"  Bob:   {bob_apply_time*1000:.2f}ms, chain_index {pre_chain}->{post_chain_bob} (0 levels)")

    print("\nCT-CT retrieval")
    results = {'queries': [], 'dataset': args.dataset, 'noise_scale': args.noise_scale}

    for qi, (question, answer, gold_ctx) in enumerate(test_qa):
        gold_idx = None
        for pi, p in enumerate(passages):
            if p == gold_ctx:
                gold_idx = pi
                break

        q_emb = get_embeddings(emb_model, tokenizer, [question]).astype(np.float32)[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        q_proj = q_emb @ proj
        q_proj = q_proj / (np.linalg.norm(q_proj) + 1e-8)
        q_lorentz = euclidean_to_lorentz(q_proj.reshape(1, -1)).flatten()
        q_fhe = q_lorentz.copy()
        q_fhe[0] = -q_fhe[0]
        q_packed = pack_complex_conjugate(q_fhe.astype(np.float64))

        t0 = time.perf_counter()
        baseline_scores = ctct_retrieval(fhe, q_packed, baseline_batches, slots_per_doc)
        t_base = time.perf_counter() - t0
        baseline_top = int(np.argmax(baseline_scores))

        t0 = time.perf_counter()
        alice_scores = ctct_retrieval(fhe, q_packed, alice_batches, slots_per_doc)
        t_alice = time.perf_counter() - t0
        alice_top = int(np.argmax(alice_scores))

        t0 = time.perf_counter()
        bob_scores = ctct_retrieval(fhe, q_packed, bob_batches, slots_per_doc)
        t_bob = time.perf_counter() - t0
        bob_top = int(np.argmax(bob_scores))

        gold_class = passage_classes[gold_idx] if gold_idx is not None else set()
        gold_is_restricted = bool(gold_class)

        print(f"\n  Query {qi+1}: \"{question[:60]}...\"")
        print(f"    Gold passage idx={gold_idx}, classes={gold_class or 'unrestricted'}")
        print(f"    Baseline: top={baseline_top} {'gold' if baseline_top==gold_idx else 'miss'} ({t_base*1000:.1f}ms)")
        print(f"    Alice:    top={alice_top} {'gold' if alice_top==gold_idx else 'miss'} ({t_alice*1000:.1f}ms)")
        print(f"    Bob:      top={bob_top} {'gold' if bob_top==gold_idx else 'miss'} ({t_bob*1000:.1f}ms)")

        if gold_idx is not None and gold_is_restricted:
            gold_restricted_classes = gold_class
            bob_has_access = gold_restricted_classes.issubset(bob_auth)
            print(f"    Gold needs {gold_restricted_classes}, Bob has {bob_auth} -> "
                  f"{'can' if bob_has_access else 'cannot'} retrieve")

        results['queries'].append({
            'question': question[:80],
            'gold_idx': gold_idx,
            'gold_classes': sorted(gold_class),
            'baseline_top': baseline_top,
            'alice_top': alice_top,
            'bob_top': bob_top,
            'baseline_gold': baseline_top == gold_idx,
            'alice_gold': alice_top == gold_idx,
            'bob_gold': bob_top == gold_idx,
        })

    n_q = len(test_qa)
    baseline_correct = sum(1 for r in results['queries'] if r['baseline_gold'])
    alice_correct = sum(1 for r in results['queries'] if r['alice_gold'])
    bob_correct = sum(1 for r in results['queries'] if r['bob_gold'])

    print(f"\n  Retrieval summary ({n_q} queries)")
    print(f"    Baseline (no noise): {baseline_correct}/{n_q} gold top-1")
    print(f"    Alice (full access): {alice_correct}/{n_q} gold top-1")
    print(f"    Bob (limited):       {bob_correct}/{n_q} gold top-1")

    results['baseline_r1'] = baseline_correct
    results['alice_r1'] = alice_correct
    results['bob_r1'] = bob_correct
    results['correction_time_ms'] = {
        'alice': alice_apply_time * 1000,
        'bob': bob_apply_time * 1000,
    }
    results['levels_consumed'] = 0
    results['chain_index_before'] = pre_chain
    results['chain_index_after'] = post_chain_alice

    def run_targeted_demo(target_class_name, label):
        tidx = None
        for pi, p in enumerate(passages):
            if target_class_name in passage_classes[pi] and \
               not passage_classes[pi].issubset(bob_auth):
                tidx = pi
                break
        if tidx is None:
            print(f"\n  {label} DEMO: no {target_class_name}-class passage found")
            return None, None, None

        tc = passage_classes[tidx]
        tquery = " ".join(passages[tidx].split()[:8])
        print(f"\n  {label} DEMO ({target_class_name})")
        print(f"  Target passage idx={tidx}, classes={tc}")
        print(f"  Query: \"{tquery}\"")
        print(f"  Passage: \"{passages[tidx][:100]}...\"")

        tq_emb = get_embeddings(emb_model, tokenizer, [tquery]).astype(np.float32)[0]
        tq_emb = tq_emb / (np.linalg.norm(tq_emb) + 1e-8)
        tq_proj = tq_emb @ proj
        tq_proj = tq_proj / (np.linalg.norm(tq_proj) + 1e-8)
        tq_lorentz = euclidean_to_lorentz(tq_proj.reshape(1, -1)).flatten()
        tq_fhe = tq_lorentz.copy()
        tq_fhe[0] = -tq_fhe[0]
        tq_packed = pack_complex_conjugate(tq_fhe.astype(np.float64))

        d_base = ctct_retrieval(fhe, tq_packed, baseline_batches, slots_per_doc)
        d_alice = ctct_retrieval(fhe, tq_packed, alice_batches, slots_per_doc)
        d_bob = ctct_retrieval(fhe, tq_packed, bob_batches, slots_per_doc)

        base_top = int(np.argmax(d_base))
        alice_top = int(np.argmax(d_alice))
        bob_top = int(np.argmax(d_bob))

        print(f"  Baseline: top={base_top} {'hit' if base_top==tidx else 'miss'}")
        print(f"  Alice:    top={alice_top} {'hit' if alice_top==tidx else 'miss'}")
        print(f"  Bob:      top={bob_top} {'hit' if bob_top==tidx else 'miss'}")

        demo = {
            'target_idx': tidx,
            'target_classes': sorted(tc),
            'query': tquery,
            'baseline_top': base_top, 'alice_top': alice_top, 'bob_top': bob_top,
            'baseline_hit': base_top == tidx,
            'alice_hit': alice_top == tidx,
            'bob_hit': bob_top == tidx,
            'alice_passage': passages[alice_top][:100],
            'bob_passage': passages[bob_top][:100],
        }
        return passages[alice_top], passages[bob_top], demo

    fin_alice, fin_bob, fin_demo = run_targeted_demo('financial', 'FINANCIAL')
    if fin_demo:
        results['access_demo'] = fin_demo

    med_alice, med_bob, med_demo = run_targeted_demo('medical', 'MEDICAL')
    if med_demo:
        results['medical_demo'] = med_demo

    gen_demos = []
    if fin_alice and fin_bob and fin_alice != fin_bob:
        gen_demos.append(('financial',
            "Based on the text above, Stephanie Seymour's net worth is",
            fin_alice, fin_bob))
    if med_alice and med_bob and med_alice != med_bob:
        gen_demos.append(('medical',
            "Based on the text above, men with diabetes develop erectile dysfunction",
            med_alice, med_bob))

    if not args.skip_generation and gen_demos:
        os.environ['RWKV_V7_ON'] = '1'
        sys.path.insert(0, str(PROJECT_DIR / "scripts"))
        from bootstrap_generation import (CKKSBootstrapContext, load_model_weights,
                                          run_generation)

        D = args.embed_dim
        n_blocks = args.num_blocks
        ffn_dim = D * 4

        if args.a100:
            poly_degree, special_mod_size, L0, prime_bits = 8192, 1, 3, 54
        else:
            poly_degree, special_mod_size, L0, prime_bits = 32768, 3, 5, 59

        blocks, emb, head_w, ln_out_w, ln_out_b, ln0_w, ln0_b, \
            D, n_head, head_size, vocab_size = \
            load_model_weights(D, ffn_dim, n_blocks)

        ckks_gen = CKKSBootstrapContext(
            poly_degree=poly_degree, L0=L0, prime_bits=prime_bits,
            special_mod_size=special_mod_size, bsgs_dim=D,
            skip_bootstrap=True)

        from rwkv_emb.tokenizer import RWKVTokenizer
        gen_tokenizer = RWKVTokenizer()

        def gen_for_passage(passage_text, gen_question, label):
            first_sent = passage_text.split('.')[0] + '.'
            prompt = f"{first_sent}\n{gen_question}"
            seed = gen_tokenizer.encode(prompt)
            seed = [t for t in seed if t < 65535]
            seed_text = gen_tokenizer.decode(seed)
            print(f"    {label} seed ({len(seed)} tok): \"{seed_text}\"")
            t0 = time.perf_counter()
            tok_fhe, tok_ref, total_t, _ = run_generation(
                ckks_gen, blocks, emb, head_w, ln_out_w, ln_out_b,
                ln0_w, ln0_b, D, n_head, head_size,
                seed, args.num_tokens, gen_tokenizer, use_bsgs=True)
            gen_time = time.perf_counter() - t0
            all_match = all(f == r for f, r in zip(tok_fhe, tok_ref))
            text_fhe = gen_tokenizer.decode(tok_fhe)
            print(f"    {label}: {tok_fhe} = \"{text_fhe}\" "
                  f"match={all_match} {gen_time:.1f}s")
            return tok_fhe, text_fhe, all_match, gen_time, seed_text

        for demo_type, gen_question, a_passage, b_passage in gen_demos:
            print(f"\n{demo_type} generation")
            print(f"  Alice context: \"{a_passage[:80]}...\"")
            print(f"  Bob context:   \"{b_passage[:80]}...\"")

            a_tok, a_text, a_match, a_time, a_seed = \
                gen_for_passage(a_passage, gen_question, "Alice")
            b_tok, b_text, b_match, b_time, b_seed = \
                gen_for_passage(b_passage, gen_question, "Bob")

            differ = a_tok != b_tok
            print(f"    Outputs differ: {differ}")

            gen_key = 'generation' if demo_type == 'financial' else f'generation_{demo_type}'
            results[gen_key] = {
                'demo_type': demo_type,
                'gen_question': gen_question,
                'alice_seed': a_seed,
                'bob_seed': b_seed,
                'alice_tokens': a_tok,
                'bob_tokens': b_tok,
                'alice_text': a_text,
                'bob_text': b_text,
                'alice_match': a_match,
                'bob_match': b_match,
                'outputs_differ': differ,
                'alice_time': a_time,
                'bob_time': b_time,
            }

    print("\nSecurity analysis")
    noise_security_analysis(docs_packed, passage_classes, class_noise,
                            [1, 10, 50, 100, 500])

    out_path = PROJECT_DIR / "results" / "fhe_access_control.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_docs", type=int, default=100)
    parser.add_argument("--n_queries", type=int, default=3)
    parser.add_argument("--dataset", choices=["squad", "msmarco"], default="msmarco")
    parser.add_argument("--noise_scale", type=float, default=100.0)
    parser.add_argument("--embed_dim", type=int, default=2048)
    parser.add_argument("--num_blocks", type=int, default=24)
    parser.add_argument("--num_tokens", type=int, default=10)
    parser.add_argument("--a100", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--noise_only", action="store_true")
    args = parser.parse_args()
    run_pipeline(args)
