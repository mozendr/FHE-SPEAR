#!/usr/bin/env python3
import os, sys, time
import numpy as np

PHANTOM_PATH = os.path.expanduser("~/gpu_fhe_libs/phantom-fhe/build/lib")
if PHANTOM_PATH not in sys.path:
    sys.path.insert(0, PHANTOM_PATH)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from fhe_common import (PhantomFHE, euclidean_to_lorentz, pack_complex,
                         pack_complex_conjugate, get_embeddings)
from fhe_spear_msmarco import load_msmarco_samples
from numpy.linalg import svd
import re

PII_PATTERNS = {
    'MONEY': re.compile(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?'),
    'PERCENT': re.compile(r'\b\d+(?:\.\d+)?%'),
    'YEAR_EVENT': re.compile(r'\b(?:in|since|from|until|after|before|during)\s+\d{4}\b'),
    'PHONE': re.compile(r'\b(?:\+1[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    'EMAIL': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'DATE': re.compile(r'\b(?:January|February|March|April|May|June|July|August|'
                       r'September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'),
    'ORG_PAREN': re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\([A-Z]{2,}\)'),
}
CLASS_MAP = {
    'MONEY': 'financial', 'PERCENT': 'financial',
    'PHONE': 'personal', 'EMAIL': 'personal',
    'DATE': 'temporal', 'YEAR_EVENT': 'temporal',
    'ORG_PAREN': 'organizational',
}

def classify_passage(text):
    classes = set()
    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            if pii_type in CLASS_MAP:
                classes.add(CLASS_MAP[pii_type])
    return classes


def encrypt_docs(fhe, docs, slots_per_doc):
    import pyPhantom as phantom
    batch_sz = fhe.slot_count // slots_per_doc
    batches = []
    mapping = []
    for start in range(0, len(docs), batch_sz):
        end = min(start + batch_sz, len(docs))
        batch_docs = docs[start:end]
        batch_idx = len(batches)
        doc_slots = []
        for i, doc in enumerate(batch_docs):
            mapping.append((batch_idx, i * slots_per_doc))
            doc_slots.extend(list(doc))
        doc_slots += [complex(0, 0)] * (fhe.slot_count - len(doc_slots))
        pt = fhe.encoder.encode_complex_vector(fhe.ctx, doc_slots, fhe.scale)
        ct = fhe.pk.encrypt_asymmetric(fhe.ctx, pt)
        batches.append((ct, len(batch_docs)))
    return batches, mapping


def ctct_retrieval(fhe, q_packed, enc_batches, slots_per_doc):
    import pyPhantom as phantom
    scores = []
    for enc_docs, n_batch in enc_batches:
        qslots = []
        for _ in range(n_batch):
            qslots.extend(list(q_packed))
        qslots += [complex(0, 0)] * (fhe.slot_count - len(qslots))
        q_pt = fhe.encoder.encode_complex_vector(fhe.ctx, qslots, fhe.scale)
        enc_q = fhe.pk.encrypt_asymmetric(fhe.ctx, q_pt)
        result = phantom.multiply(fhe.ctx, enc_q, enc_docs)
        result = phantom.relinearize(fhe.ctx, result, fhe.rlk)
        result = phantom.rescale_to_next(fhe.ctx, result)
        dec_pt = fhe.sk.decrypt(fhe.ctx, result)
        dec_vec = fhe.encoder.decode_complex_vector(fhe.ctx, dec_pt)
        for i in range(n_batch):
            s = i * slots_per_doc
            score = sum(c.real for c in dec_vec[s:s + slots_per_doc])
            scores.append(score)
    return np.array(scores)


def main():
    print("Per-passage vs per-class noise attack demonstration")

    print("\n[1] Loading data and embeddings...")
    from fhe_common import MODEL_DIR
    from rwkv_emb.model import EmbeddingRWKV
    from rwkv_emb.tokenizer import RWKVTokenizer

    passages, test_qa = load_msmarco_samples(100, 10)
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

    passage_classes = [classify_passage(p) for p in passages]
    all_classes = sorted(set(c for cs in passage_classes for c in cs))

    class_members = {}
    for i, cs in enumerate(passage_classes):
        for c in cs:
            class_members.setdefault(c, []).append(i)

    target_class = max(class_members, key=lambda c: len(class_members[c]))
    members = class_members[target_class]
    print(f"  {len(passages)} passages, {len(all_classes)} classes")
    print(f"  Target class '{target_class}': {len(members)} members")

    # Group by exact class set -- attack only works for identical class sets
    classset_groups = {}
    for i in members:
        key = frozenset(passage_classes[i])
        classset_groups.setdefault(key, []).append(i)
    print(f"  Class-set groups within '{target_class}':")
    for cs, idxs in sorted(classset_groups.items(), key=lambda x: -len(x[1])):
        print(f"    {set(cs)}: {len(idxs)} passages")

    largest_key = max(classset_groups, key=lambda k: len(classset_groups[k]))
    same_set = classset_groups[largest_key]
    print(f"  Using group {set(largest_key)}: {len(same_set)} passages")

    rng = np.random.default_rng(42)
    alpha = 100.0
    avg_norm = np.mean([np.linalg.norm(d) for d in docs_packed])

    n_attack_queries = 20
    queries = []
    for _ in range(n_attack_queries):
        q = rng.standard_normal(slots_per_doc) + 1j * rng.standard_normal(slots_per_doc)
        q = q / np.linalg.norm(q)
        queries.append(q)

    print(f"\n--- Per-class noise (shared n_c per class) ---")

    class_noise = {}
    for c in all_classes:
        raw = rng.standard_normal(slots_per_doc) + 1j * rng.standard_normal(slots_per_doc)
        raw = raw / np.abs(raw).mean()
        class_noise[c] = raw * alpha * avg_norm

    noised_class = []
    for i, doc in enumerate(docs_packed):
        total_noise = np.zeros(slots_per_doc, dtype=complex)
        for c in passage_classes[i]:
            total_noise += class_noise[c][:slots_per_doc]
        noised_class.append(doc + total_noise)

    true_diffs = []
    noised_diffs = []
    for q in queries:
        true_scores = [np.real(np.sum(q * docs_packed[i])) for i in same_set]
        noised_scores = [np.real(np.sum(q * noised_class[i])) for i in same_set]
        for a in range(len(same_set)):
            for b in range(a + 1, len(same_set)):
                true_diffs.append(true_scores[a] - true_scores[b])
                noised_diffs.append(noised_scores[a] - noised_scores[b])

    true_diffs = np.array(true_diffs)
    noised_diffs = np.array(noised_diffs)
    corr_class = np.corrcoef(true_diffs, noised_diffs)[0, 1]

    print(f"  Same class-set pairs ({len(same_set)} passages, {set(largest_key)}):")
    print(f"  Correlation(true_diffs, noised_diffs) = {corr_class:.6f}")
    print(f"  Max |true - noised| = {np.max(np.abs(true_diffs - noised_diffs)):.2e}")
    print(f"  Number of pairs: {len(true_diffs)}")

    print(f"\n--- Per-passage noise (independent n_j per passage) ---")

    passage_noise = []
    for i in range(len(docs_packed)):
        if passage_classes[i]:
            raw = rng.standard_normal(slots_per_doc) + 1j * rng.standard_normal(slots_per_doc)
            raw = raw / np.abs(raw).mean()
            passage_noise.append(raw * alpha * avg_norm)
        else:
            passage_noise.append(np.zeros(slots_per_doc, dtype=complex))

    noised_passage = [doc + passage_noise[i] for i, doc in enumerate(docs_packed)]

    true_diffs2 = []
    noised_diffs2 = []
    for q in queries:
        true_scores = [np.real(np.sum(q * docs_packed[i])) for i in same_set]
        noised_scores = [np.real(np.sum(q * noised_passage[i])) for i in same_set]
        for a in range(len(same_set)):
            for b in range(a + 1, len(same_set)):
                true_diffs2.append(true_scores[a] - true_scores[b])
                noised_diffs2.append(noised_scores[a] - noised_scores[b])

    true_diffs2 = np.array(true_diffs2)
    noised_diffs2 = np.array(noised_diffs2)
    corr_passage = np.corrcoef(true_diffs2, noised_diffs2)[0, 1]

    print(f"  Correlation(true_diffs, noised_diffs) = {corr_passage:.6f}")
    print(f"  Std(true_diffs) = {np.std(true_diffs2):.4f}")
    print(f"  Std(noised_diffs) = {np.std(noised_diffs2):.4f}")
    print(f"  Noise-to-signal ratio = {np.std(noised_diffs2)/np.std(true_diffs2):.1f}x")

    print(f"\n--- Authorized retrieval under FHE with per-passage corrections ---")

    import pyPhantom as phantom

    fhe = PhantomFHE()
    batch_size = fhe.slot_count // slots_per_doc
    print(f"  SIMD: {batch_size} docs/ct, {fhe.slot_count} slots")

    t0 = time.perf_counter()
    noised_batches, mapping = encrypt_docs(fhe, noised_passage, slots_per_doc)
    print(f"  Encrypted {len(noised_batches)} batches in {time.perf_counter()-t0:.2f}s")

    # Per-passage corrections: add -n_j to cancel noise for authorized user
    t0 = time.perf_counter()
    corrected_batches = []
    for b, (enc_docs, n_in_batch) in enumerate(noised_batches):
        corr_slots = [complex(0, 0)] * fhe.slot_count
        for doc_idx, (bi, offset) in enumerate(mapping):
            if bi != b:
                continue
            if passage_classes[doc_idx]:
                for j in range(slots_per_doc):
                    corr_slots[offset + j] = -passage_noise[doc_idx][j]
        corr_pt = fhe.encoder.encode_complex_vector(fhe.ctx, corr_slots, fhe.scale)
        corr_ct = fhe.pk.encrypt_asymmetric(fhe.ctx, corr_pt)
        result = phantom.add(fhe.ctx, enc_docs, corr_ct)
        corrected_batches.append((result, n_in_batch))
    corr_time = time.perf_counter() - t0
    print(f"  Corrections applied in {corr_time*1000:.1f}ms")
    print(f"  Corrections: {sum(1 for pn in passage_noise if np.any(pn != 0))} passages")
    print(f"  Chain index: {noised_batches[0][0].chain_index()} -> {corrected_batches[0][0].chain_index()} (0 levels)")

    baseline_batches, _ = encrypt_docs(fhe, docs_packed, slots_per_doc)

    print(f"\n  Retrieval ({len(test_qa)} queries):")
    baseline_correct = 0
    corrected_correct = 0
    noised_correct = 0

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

        s_base = ctct_retrieval(fhe, q_packed, baseline_batches, slots_per_doc)
        s_corr = ctct_retrieval(fhe, q_packed, corrected_batches, slots_per_doc)
        s_noised = ctct_retrieval(fhe, q_packed, noised_batches, slots_per_doc)

        base_top = int(np.argmax(s_base))
        corr_top = int(np.argmax(s_corr))
        noised_top = int(np.argmax(s_noised))

        if base_top == gold_idx: baseline_correct += 1
        if corr_top == gold_idx: corrected_correct += 1
        if noised_top == gold_idx: noised_correct += 1

        if qi < 5:
            print(f"    Q{qi+1}: baseline={'Y' if base_top==gold_idx else 'N'} "
                  f"corrected={'Y' if corr_top==gold_idx else 'N'} "
                  f"unauthorized={'Y' if noised_top==gold_idx else 'N'}")

    nq = len(test_qa)
    print(f"\n  R@1 Summary:")
    print(f"    Baseline (no noise):     {baseline_correct}/{nq}")
    print(f"    Authorized (corrected):  {corrected_correct}/{nq}")
    print(f"    Unauthorized (noised):   {noised_correct}/{nq}")
    baseline_match = (baseline_correct == corrected_correct)
    print(f"    Authorized = Baseline:   {'yes' if baseline_match else 'no'}")

    print(f"\n--- Summary ---")
    print(f"  Per-class noise:    corr = {corr_class:.6f}")
    print(f"  Per-passage noise:  corr = {corr_passage:.6f}, noise/signal = {np.std(noised_diffs2)/np.std(true_diffs2):.1f}x")
    print(f"  Authorized R@1 = Baseline R@1: {'yes' if baseline_match else 'no'}")
    print(f"  Levels consumed: 0 (homomorphic add only)")


if __name__ == '__main__':
    main()
