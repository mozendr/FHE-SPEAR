#!/usr/bin/env python3
import os
os.environ['RWKV_V7_ON'] = '1'
os.environ['RWKV_JIT_ON'] = '1'
os.environ['RWKV_CUDA_ON'] = '1'

import numpy as np
import re
import time
from scipy.linalg import svd

from fhe_common import (
    USE_PHANTOM_GPU, MODEL_DIR, PROJECT_DIR,
    euclidean_to_lorentz, lorentz_inner_product_batch,
    pack_complex, pack_complex_conjugate,
    get_embeddings, download_fallback_model,
    PhantomFHE, TenSEALFHE,
)

from rwkv_emb.model import EmbeddingRWKV
from rwkv_emb.tokenizer import RWKVTokenizer
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS


def load_research_documents():
    documents = []

    readme_path = PROJECT_DIR / "README.md"
    if readme_path.exists():
        content = readme_path.read_text()
        sections = re.split(r'\n## ', content)
        for section in sections:
            if len(section.strip()) > 50:
                title = section.split('\n')[0].strip('#').strip()
                documents.append({'source': 'README.md', 'title': title, 'content': section[:2000]})

    paper_path = PROJECT_DIR / "paper" / "main.tex"
    if paper_path.exists():
        content = paper_path.read_text()
        abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)
        if abstract_match:
            documents.append({'source': 'main.tex', 'title': 'Paper Abstract',
                           'content': abstract_match.group(1).strip()[:2000]})

    return documents


class FHE_SPEAR:
    def __init__(self, documents, emb_model, tokenizer, gen_model, gen_pipeline,
                 proj_dim=64, use_hyperbolic=True, use_complex_packing=True):
        self.documents = documents
        self.emb_model = emb_model
        self.tokenizer = tokenizer
        self.gen_model = gen_model
        self.gen_pipeline = gen_pipeline
        self.proj_dim = proj_dim
        self.use_hyperbolic = use_hyperbolic
        self.use_complex_packing = use_complex_packing

        doc_texts = [f"{d['title']}: {d['content']}" for d in documents]
        self.doc_embs = get_embeddings(emb_model, tokenizer, doc_texts)
        self.doc_embs = self.doc_embs / (np.linalg.norm(self.doc_embs, axis=1, keepdims=True) + 1e-8)

        _, S, Vt = svd(self.doc_embs, full_matrices=False)
        self.proj = Vt[:proj_dim].T

        self.doc_embs_proj = self.doc_embs @ self.proj
        self.doc_embs_proj = self.doc_embs_proj / (np.linalg.norm(self.doc_embs_proj, axis=1, keepdims=True) + 1e-8)

        if use_hyperbolic:
            self.doc_embs_lorentz = euclidean_to_lorentz(self.doc_embs_proj)
            self.lorentz_dim = proj_dim + 1
        else:
            self.doc_embs_lorentz = self.doc_embs_proj
            self.lorentz_dim = proj_dim

        if USE_PHANTOM_GPU:
            self.fhe = PhantomFHE()
        else:
            self.fhe = TenSEALFHE()

    def retrieve_plaintext(self, query, top_k=3):
        q_emb = get_embeddings(self.emb_model, self.tokenizer, [query])[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        q_proj = q_emb @ self.proj
        q_proj = q_proj / (np.linalg.norm(q_proj) + 1e-8)

        if self.use_hyperbolic:
            q_lorentz = euclidean_to_lorentz(q_proj.reshape(1, -1))
            scores = lorentz_inner_product_batch(q_lorentz, self.doc_embs_lorentz).flatten()
        else:
            scores = self.doc_embs_proj @ q_proj

        top_indices = np.argsort(-scores)[:top_k]
        return [(self.documents[i], float(scores[i])) for i in top_indices]

    def retrieve_fhe(self, query, top_k=3):
        q_emb = get_embeddings(self.emb_model, self.tokenizer, [query])[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        q_proj = q_emb @ self.proj
        q_proj = q_proj / (np.linalg.norm(q_proj) + 1e-8)

        if self.use_hyperbolic:
            q_lorentz = euclidean_to_lorentz(q_proj.reshape(1, -1)).flatten()
            q_fhe = q_lorentz.copy()
            q_fhe[0] = -q_fhe[0]
        else:
            q_fhe = q_proj

        t0 = time.perf_counter()
        enc_scores = []

        if self.use_complex_packing:
            q_packed = pack_complex_conjugate(q_fhe.astype(np.float64))
            enc_q = self.fhe.encrypt_complex(q_packed)

            for i in range(len(self.documents)):
                doc_vec = self.doc_embs_lorentz[i] if self.use_hyperbolic else self.doc_embs_proj[i]
                doc_packed = pack_complex(doc_vec.astype(np.float64))

                if USE_PHANTOM_GPU:
                    score = self.fhe.dot_product(enc_q, doc_packed)
                else:
                    score = self.fhe.dot_product_complex(enc_q, doc_packed)
                enc_scores.append(score)
        else:
            enc_q = self.fhe.encrypt(q_fhe)
            for i in range(len(self.documents)):
                doc_vec = self.doc_embs_lorentz[i] if self.use_hyperbolic else self.doc_embs_proj[i]
                enc_scores.append(self.fhe.dot_product(enc_q, doc_vec.astype(np.float64)))

        fhe_time = time.perf_counter() - t0

        scores = np.array(enc_scores)
        top_indices = np.argsort(-scores)[:top_k]

        return [(self.documents[i], float(scores[i])) for i in top_indices], fhe_time

    def generate(self, query, context_docs, max_tokens=150):
        context_parts = [f"{doc['title']}: {doc['content'][:400]}" for doc, _ in context_docs]
        context = "\n\n".join(context_parts)

        prompt = f"""User: /no_think Based on the following context, answer the question concisely in 1-2 sentences.

Context:
{context}

Question: {query}Assistant:"""

        args = PIPELINE_ARGS(
            temperature=0.7,
            top_p=0.8,
            alpha_frequency=0.2,
            alpha_presence=0.2,
            token_stop=['\n\nUser', '<think>', '\nUser:']
        )
        response = self.gen_pipeline.generate(prompt, token_count=max_tokens, args=args, callback=None)
        return response

def main():
    print("FHE-SPEAR Demo")

    documents = load_research_documents()

    print("Loading models...")
    emb_model = EmbeddingRWKV(str(MODEL_DIR / "rwkv0b4-emb-curriculum.pth"))
    tokenizer = RWKVTokenizer()

    g1_model_path = MODEL_DIR / "rwkv7-g1d-1.5b-20260212-ctx8192.pth"
    if not g1_model_path.exists():
        g1_model_path = MODEL_DIR / "rwkv7-g1c-1.5b-20260110-ctx8192.pth"
    if g1_model_path.exists():
        gen_model = RWKV(model=str(g1_model_path).replace(".pth", ""), strategy="cuda fp16")
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

    rag = FHE_SPEAR(documents, emb_model, tokenizer, gen_model, gen_pipeline,
                    proj_dim=64, use_hyperbolic=True, use_complex_packing=True)

    queries = [
        "What latency does FHE achieve over 50k documents?",
        "How does complex packing reduce ciphertext size?",
        "What is the R@10 accuracy at 64 dimensions?",
    ]

    for query in queries:
        print(f"\nQ: {query}")

        fhe_results, fhe_time = rag.retrieve_fhe(query, top_k=2)
        print(f"\n[Retrieval] FHE time: {fhe_time*1000:.1f}ms")
        for doc, score in fhe_results:
            print(f"  [{score:.3f}] {doc['source']}: {doc['title']}")

        plain_results = rag.retrieve_plaintext(query, top_k=2)
        fhe_match = all(fhe_results[i][0]['title'] == plain_results[i][0]['title']
                        for i in range(len(fhe_results)))
        print(f"  FHE matches plaintext: {fhe_match}")

        print("\n[Generation]")
        try:
            answer = rag.generate(query, fhe_results, max_tokens=80)
            answer = answer.strip()
            if answer.startswith('Assistant:'):
                answer = answer[10:].strip()
            print(f"  {answer[:250]}")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nBackend: {'PhantomFHE GPU' if USE_PHANTOM_GPU else 'TenSEAL CPU'}")

if __name__ == "__main__":
    main()
