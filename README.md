# FHE-SPEAR

Softmax-free Private Encrypted Answering and Retrieval.

![overview](assets/overview.png)

<details>
<summary><b>Figure details</b></summary>
<br>

**(a)** Two-phase client-server protocol. Phase 1 retrieves relevant passages via encrypted cosine similarity over the document corpus. Phase 2 generates tokens by running RWKV-7 blocks on the server (BSGS-accelerated diagonal matrix-vector products) with client-side nonlinear operations, requiring 96 round-trips per generated token. The server operates exclusively on CKKS ciphertexts.

**(b)** The server performs 8 BSGS matrix-vector products per block (4 attention, 2 complex-packed FFN key, 2 complex-packed FFN value). The client handles all nonlinear operations (layer normalization, WKV recurrence, gating, squared ReLU). Data is encrypted and decrypted at each boundary crossing, yielding 4 round-trips per block.

**(c)** Encoding two dimensions per CKKS slot using real and imaginary components. Queries packed as conjugates yield inner products in the real part - halves ciphertext count compared to vertical packing.

**(d)** The diagonal method decomposes W into d diagonals and computes y = sum of rotate(x, k) \* delta_k, encoding all d output elements in a single ciphertext via SIMD packing. Baby-step giant-step factoring writes k = gG + b with G = ceil(sqrt(d)), B = ceil(d/G), reducing the rotation count to G + B - 2. At d=2048: G=46, B=45, giving 89 rotations per projection versus 22,528 naive (253-fold reduction).

**(e)** Per-class random noise is added to each restricted embedding before encryption. Each user receives correction ciphertexts: Enc(-n_c) for authorized classes, Enc(r) for unauthorized classes. Real and dummy corrections are indistinguishable under CKKS IND-CPA security. The server applies all corrections via homomorphic addition, essentially zero multiplicative levels.

**(f)** The Lorentz inner product is -x0\*y0 + sum(xi\*yi). For FHE, this reduces to a standard dot product with a sign flip on the first component; no transcendental functions are needed. Since arcosh is monotonic, ranking by the Lorentz inner product preserves the distance ordering.

**(g)** Predicts FHE accuracy without running encryption. The prediction formula requires only rho_compression (measurable without FHE) and CKKS parameters, so embedding-dimension configurations can be evaluated without running encryption.

</details>

## Setup

```bash
pip install numpy torch rwkv rwkv-emb tenseal scipy
```

GPU FHE requires [PhantomFHE](https://github.com/mozendr/phantom-fhe):

```bash
git clone https://github.com/mozendr/phantom-fhe
cd phantom-fhe && mkdir build && cd build && cmake .. && make -j
```

Set `PHANTOM_PATH` to the build output, or edit `fhe_common.py`.

```bash
python download_models.py rwkv7-g1d-1.5b-20260212-ctx8192.pth
python download_models.py rwkv0b4-emb-curriculum.pth
```

## Usage

```bash
python ret_light_demo.py
python scripts/bootstrap_generation.py --bsgs --embed_dim 2048 --num_blocks 24 --num_tokens 3
python scripts/bootstrap_generation.py --bsgs --rag --dataset squad
python test_fully_enc_bsgs.py --D 2048 --F 4096 --num_blocks 24 --L0 36 --N 16384
```

## Notes

- On A100 (sm_80), pass --a100. This sets P=1 (N=8192 for generation, N=16384 for fully encrypted). PhantomFHE rotations can fail with P>1 on sm_80.
- L0 must be divisible by P or you get CUDA memory errors. With P=3, use L0=24,27,30,36,...
- --bsgs is needed for D=1024 and above to be practical. Without it, matrix-vector products are about 30x slower.
- Fully encrypted mode (test_fully_enc_bsgs.py) needs bootstrap for more than about 19 blocks at N=32768. Client-aided mode (bootstrap_generation.py) does not.
- The 0.4B model uses --embed_dim 1024 --num_blocks 24.
- Experiments were carried on RTX 3090 (sm_86) and A100 (sm_80). Other GPUs may exhibit quirks with the PhantomFHE fork - feel free to send a message regarding that if so.

## Citation

```bibtex
@article{osman2026fhespear,
  title={FHE-SPEAR: End-to-End Homomorphic RAG with Softmax-Free Generation},
  author={Osman, Alper-Ender},
  year={2026}
}
```
