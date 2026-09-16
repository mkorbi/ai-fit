# AI Fit

Live: https://mkorbi.github.io/ai-fit/ (GitHub Pages) · source: https://github.com/mkorbi/ai-fit

A capacity planner for LLMs you host yourself. Pick accelerators, how many, the fabric inside a node and the
network between nodes, then a model and a workload, and it tells you:

- how many users (and concurrent requests) the hardware serves at a given context length,
- how much context each session can have at a given load,
- tokens per second per user, aggregate throughput, time to first token, cost per million tokens and power,
- which optimizations (KV cache and weight quantization, prefix caching, speculative decoding, session retention,
  prefill/decode disaggregation, tensor vs pipeline layout) buy how much more room.

The reverse direction works too: give it users, context, model and targets and it sizes the smallest layout per
candidate accelerator, sorted by count and price.

## Run it

Open `index.html` in a browser. Everything is client-side; the only network requests are the Google Fonts.
Inputs and custom catalog entries are kept in the browser's local storage.

## Host it on GitHub Pages

The folder is a static site with relative paths, so it works from a repository root or any subfolder:

1. Push the folder to a GitHub repository (`.nojekyll` is included so Pages serves the files untouched).
2. In the repository: Settings → Pages → Build and deployment → Source "Deploy from a branch",
   branch `main`, folder `/ (root)` (or `/docs` if you put the files there).
3. The site appears at `https://<user>.github.io/<repo>/` after the first deploy, usually within a minute. This copy is served at https://mkorbi.github.io/ai-fit/.

## Publish as a Claude artifact

The artifact publisher wraps a page in its own document skeleton, so it takes the body-only variant
`artifact.html`. Regenerate it from `index.html` after every change:

```bash
sed '1,/^<body>$/d; /^<\/body>$/,$d' index.html > artifact.html
```

## Files

| File | What it holds |
|---|---|
| `catalog.js` | Accelerators (memory, bandwidth, dense TFLOPS per precision, fabric, node size, TDP, rough price, chip generation), the `ARCHS` generation → format table, and models (shape, attention layer types, MoE, context limits, shipped precision, Hugging Face repo) |
| `tools/import-model.js` | Derives or verifies model entries from Hugging Face `config.json` and safetensors metadata; finds and verifies quantized checkpoints (node 18+, no dependencies) |
| `tools/check-sources.js` | Verifies every accelerator's source page resolves and still shows its numbers |
| `tests/anchors.js` | Calibration anchors: published fit statements and measured numbers the model must reproduce |
| `engine.js` | The model: memory budget, KV geometry per attention type, roofline decode step, prefill cost, prefill/decode interleaving, speculative decoding, layout search, reverse search. Runs in node too (`require('./engine.js')`) |
| `app.js` | The UI: inputs, KPI tiles, memory bar, timing table, capacity-frontier and speed charts, optimization ledger, reverse table, compatibility matrix, catalog editor |
| `index.html` | The page: markup and styles, a complete HTML document for browsers and GitHub Pages |
| `artifact.html` | `index.html` without the document wrapper, generated for publishing as a Claude artifact |
| `.nojekyll` | Tells GitHub Pages to serve the files without a Jekyll build |

## Verifying the catalog

Model shapes are checked against the Hugging Face `config.json` of the repo named in each entry's `hf` field
(gated repos fall back to the ungated `hfMirror`, or set `HF_TOKEN`):

```bash
node tools/import-model.js --check
```

It compares layers, hidden size, heads, KV heads, head dim, MoE experts, attention layer types and windows,
MLA ranks, sparse top-k, context limits, the parameter count from the safetensors metadata (minus multi-token-
prediction layers, which plain serving does not load) and the precision the checkpoint ships in. Exit code 1
on any mismatch. To draft a new entry from a repo:

```bash
node tools/import-model.js Qwen/Qwen3-32B
```

Which number formats an accelerator computes natively, which it can only load as weight-only quantization
(dequantized to BF16, so memory shrinks but compute does not speed up) and which it cannot load at all is not
stored per device. Each device names its chip generation in `arch`, and the `ARCHS` table in `catalog.js`
maps generations to formats, with the sources listed above it. The Catalog view runs consistency checks
between that table and each device's TFLOPS columns. Per-model, `nativePrec` records the precision the
official checkpoint ships in; choosing a higher precision is flagged as an upcast, a lower one as needing a
quantized checkpoint.

### Known checkpoints per format

Each model lists known quantized checkpoints in `variants` (FP8, INT8, INT4, FP4), found with
`node tools/import-model.js --find-variants` (searches Hugging Face, reads each candidate's `quantization_config`
and checks the shape matches the base model) and re-verified with `--check-variants`. The precision dropdown shows
the checkpoint for the chosen format with its origin (official, vendor such as RedHatAI/NVIDIA/AMD/Intel, or
community) or says that none is known.

### Engine support

The serving-engine dropdown (vLLM, SGLang, TensorRT-LLM, or hardware capability only) applies the engine's own
documented support matrix on top of the silicon table: which weight formats run natively or weight-only on each
hardware family, and which KV-cache dtypes exist (no engine in the table offers INT4 KV cache; LMDeploy does).
The matrix lives in `ENGINES` in `catalog.js` with the doc URLs and the date it was transcribed; the Catalog view
renders it. Re-transcribe when the docs change.

### Provenance of hardware numbers

Every accelerator carries `source` (vendor page URL, kind: datasheet / derived / announcement, date checked) and
the cloud price page the rate came from. `node tools/check-sources.js` fetches each page, fails on broken links and
reports whether the entry's memory, bandwidth, TFLOPS (dense or the vendor's 2× sparsity figure, per-GPU or
8-/72-GPU aggregate) and TDP still appear in the page text. "Not found" is a prompt to look, not proof of error:
several vendor pages render their spec tables with JavaScript.

### Calibration anchors

`node tests/anchors.js` checks the model against published facts: fit anchors (vendor statements such as
"gpt-oss-120b runs on one 80 GB H100", "Llama 3.1 405B FP8 fits one 8× H100 node") and measured anchors with a
tolerance (vLLM's reported ~1.3M-token KV cache for Llama 70B BF16 on 8× H100). Add your own rows from
`vllm bench serve` or `sglang.bench_serving` runs; the test fails when a prediction drifts out of tolerance.

## Extending the catalog

Use the Catalog view: paste a JSON object (or array) and add it as hardware or model. Same `id` replaces the
built-in entry. To make an entry permanent for everyone, add it to `catalog.js` instead. Field reference is at the
top of that file. Entries marked `approx: true` show a `~` in the UI.

## How the numbers are made

The Method view in the app documents every formula and the assumptions panel exposes the knobs (memory
utilization, overhead, achieved bandwidth and FLOPS, collective efficiency, fragmentation, pipeline bubble,
host restore bandwidth). Treat the output as an order-of-magnitude planning estimate and calibrate the knobs
against a benchmark of your real stack before buying or renting.

Quick sanity anchors the model reproduces: Llama 3.3 70B in BF16 on 8× H100 SXM keeps roughly 1.3M KV tokens
(about 9 sessions of 128k) and decodes at 25-30 tok/s per user; DeepSeek-V3 in FP8 on 8× H200 with data-parallel
attention holds about 60 concurrent 64k sessions at 20 tok/s each.
