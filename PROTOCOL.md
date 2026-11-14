# Measurement protocol

Everything quoted in this repository is produced by the rules below. They exist because two of
them were learned the hard way, from numbers that had already been written down.

## 1. Perplexity: **at least 80 chunks**, never 20

`llama-perplexity -c 512 --chunks N` scores only the **last 255 tokens of each 512-token window**
(`perplexity.cpp`: `count += n_ctx - first - 1`, `first = n_ctx/2`). So `--chunks 20` evaluates
**5,100 tokens = 1.7 % of `wiki.test.raw`** (303,436 Qwen tokens, 592 chunks).

That window is not representative. Measured on Qwen2.5-7B, same binary, same file, only the window
size changed:

| chunks | % of file | 7B f16 | 7B K2 3.354 bpw | Δ | Δ (nats, paired) |
|---:|---:|---:|---:|---:|---:|
| 20 | 3.4 % | 6.2321 | 6.7842 | **+8.86 %** | 0.085 ± 0.016 |
| 40 | 6.8 % | 5.7765 | 6.1045 | +5.68 % | 0.055 ± 0.010 |
| 80 | 13.5 % | 6.7890 | 7.0513 | **+3.86 %** | **0.038 ± 0.005** |

The 7B's damage is concentrated in **chunks 3–11** (the middle of one Wikipedia article, +11…+23 %),
while chunks 12–40 run +1…+7 %. A 20-chunk result therefore over-reads the damage by ~2.3× on this
model. The 1.5B control moves much less (+2.71 % → +2.26 %), which is exactly why the artefact was
invisible while only small models were being measured.

**Rule: `--chunks 80` minimum, `--chunks 592` for a published absolute number.**

## 2. Report quantisation damage in nats, with a paired error

`ln(PPL_quant / PPL_f16)` *is* the added cross-entropy per token. It does not depend on how low the
base PPL is, and it is additive across arms. Compute it **per chunk, on the same chunks**, and quote
the mean ± standard error over the chunks — not the ratio of two aggregate PPLs.

`pipeline/ppl_ladder.sh` and the evidence archive follow this; the 20-chunk numbers in
`measurement-log.md` are historical and are marked as such.

## 3. Speed

- Decode: greedy, temperature 0, a fixed prompt, discard the first generation, report t/s over a
  fixed token count; state the context size and the expert-cache hit rate (they move decode more
  than any engine change).
- Prefill: measure with the engine's own timing line, not the client's time-to-first-token.
- Never compare runs whose cache state differs (page cache, expert profile). Freeze or delete the
  learned profile before an A/B.

## 3b. Speed: name the tool, never mix them

Two numbers measured on the same build can differ by 10-15 % purely because of the measurement path:

| tool | what it reports | overhead in the number |
|---|---|---|
| `llama-bench` | tg128 / pp512 on a fixed synthetic prompt | minimal; greedy, no sampling, no tokeniser in the loop |
| `llama-server` `/slots` (`reliable_ab.py`) | `predicted_per_second` / `prompt_per_second` of a real request | sampling, streaming, tokenisation, per-request setup |

Both are legitimate; subtracting one from the other is not. Every speed table in this repository must
state the tool **per row**, and a before/after must use one tool for both arms.

This is not hypothetical: an earlier revision of the runtime note compared `/slots` 174-176 t/s with
`llama-bench` 202.7 t/s and reported a "+16 % decode, +76 % prefill" gain that was mostly the tool
swap. The kernel commits are real; that particular delta was not measured.

## 4. Hygiene

- **One job at a time.** A five-job compile running beside a reference measurement moved it 1.9 %;
  a quantiser at 500 % CPU beside a perplexity run moved that 45 %. Serialise, or grade the result.
- **`pgrep -f <pattern>` matches its own caller.** A wait loop built on `pgrep -f "llama-quantize"`
  never exits, because the shell running the loop carries the pattern in its command line. Use
  `pgrep -f "llama-quantize[ ]"` or the PID directly.
- **`-ngl` is exact.** Qwen2.5-7B f16 measured `-ngl 0` = 6.2321 and `-ngl 20` = 6.2321 (same error
  bar) — partial offload only moves tensors, it does not change the arithmetic. `-ngl 99` is
  impossible for a 15.24 GB f16 on a 12 GB card; say which arm used which.
- **Prompt cache lies.** The engines report large "reused" counts even for unseen text across fresh
  sessions; only the first read of never-seen text in a fresh engine is a true cold prefill.

## 5. Reproduce

```bash
# the ladder (one arm at a time; label=path[:ngl])
pipeline/ppl_ladder.sh -f wiki.test.raw -c 80 \
    base-f16=base-f16.gguf:20 k2-3.354=k2_ptq_7b_gen.gguf:99 ik3_ks=ik_IQ3_KS.gguf:99

# the sensitivity measurement (bit-exact streamed implementation)
python pipeline/sc_measure_stream.py -m <hf model> -t wiki.test.raw -o sc.json \
    --rows 4 --length 256 --levels 0.30,0.15 --device cuda

# a 7B SC without a >=24 GB card
modal run pipeline/modal_sc_7b.py --rows 4 --length 256
```
