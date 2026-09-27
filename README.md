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

For talks there is **AI Fit Arcade** (`arcade/index.html`): the same model on a pixel-art stage. GPUs are memory tanks
that fill with weights and one block per conversation, users sit at desks, idle in the lounge or queue outside, and a
chat bubble streams an answer at the real tokens per second. Change model, GPUs, engine, context or users on stage and
the room watches the queue form or clear. See [The arcade, for talks](#the-arcade-for-talks).

## Run it

Open `index.html` in a browser. Everything is client-side; the only network requests are the Google Fonts.
Inputs and custom catalog entries are kept in the browser's local storage. The arcade is `arcade/index.html`; it makes
no network requests at all (its fonts are embedded), so it runs from a local checkout at a venue without Wi-Fi.

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

Publish it with `catalog.js`, `glossary.js`, `engine.js`, `common.js` and `app.js` as supporting files. The arcade is not
part of the artifact; the planner hides its Arcade link when it runs inside one.

## The arcade, for talks

`arcade/index.html` is a stage for a projector: one presenter, keyboard or clicker, no backend. It uses the planner's
engine and state, so every number on stage is the planner's number, and `O` opens the current setup in the planner
(the planner's **Arcade** link does the reverse).

What the room sees:

- **The server room**: one replica's accelerators as memory tanks. From the bottom: weights (bricks), runtime overhead,
  KV cache (one block per session, green when in use), free space, and the hatched headroom the engine leaves unused.
  Tensor-parallel GPUs share a bus, a layer split is drawn as a chain, other replicas are the small racks below.
  Weights that do not fit pile up over the top.
- **The office and the street**: users at desks have a request in flight, users in the lounge are between turns (with a
  suitcase when their session is parked in host RAM), users outside wait in the queue. Tokens fly from the GPUs to the
  desks. With many users one pixel person stands for several; the sign says how many.
- **The scoreboard**: capacity at the speed target (the planner's number), what it would take to serve everyone, the
  limit that binds, speed per user, first token, GPUs, cost, and a chat bubble that streams at the real speed. Pin a
  baseline (`P`) and every number shows its change.
- **The ticker and banners**: every change is spelled out as cause and effect, e.g. "KV cache BF16 → FP8: capacity
  12 → 72 users (+500%)".

The queue is a closed interactive system: users think, send a request, wait for the answer. With `C` requests running at
once, `S` seconds per request and an active share `a`, everyone is served while users × `a` ≤ `C`; beyond that `C`
requests run, and the rest wait `W = (S/C)(users − C/a)` each (the response time law). Ollama rejects what does not fit in
its queue of 512.

### Keys

| Keys | Action | Keys | Action |
|---|---|---|---|
| Space / Enter | start; replay the answer | 1-9 / Shift+1-9 | recall / save a scene |
| PageDown, Right / PageUp, Left | next / previous scene (clicker) | B or `.` | blank screen (clicker), any key returns |
| F5 | replay the scene | P / Shift+P | pin / clear the baseline |
| X | swap current and baseline | Z / Shift+Z | undo / redo |
| M, G, E (Shift: back) | model, GPU, engine | + / − | twice / half the GPUs |
| A | auto-build for the users | N (Shift: half) | Ollama parallel slots |
| W / K | weights / KV cache format | C / U (Shift: less) | context / users |
| Y / T | usage pattern / speed target | I | idle sessions (RAM, GPU, evict) |
| Q, J, V | prefix cache, speculative decoding, PD split | D | show or hide the control deck |
| F | fullscreen (the idle cursor hides) | L | day or night palette |
| S | sound (off by default) | R | export or import scenes |
| O | open in the planner | ? or H, Esc | help, close |

Letters follow the keyboard layout, so the keys work on German QWERTZ too. `?` in the arcade lists them all.

### Scenes

`arcade/scenes.js` holds nine setups that tell one story: a laptop with Ollama, the same box shared with 50 colleagues,
a serving engine, a bigger model that no longer fits, FP8, long context, FP8 KV cache and prefix caching, coding
agents with auto-build, and a frontier MoE on Blackwell. Each scene is applied on top of the planner defaults and can
pin or clear the baseline. Edit the file for your talk, or save the current stage over a slot with Shift+1-9; `R` copies
all scenes as JSON to move them to the laptop you present from.

### At the venue

- Open `arcade/index.html` from a local checkout: it needs no network, and the fonts are embedded.
- `F` for fullscreen; the screen stays awake while the page is visible. A clicker's page keys step through the scenes,
  its blank key blanks the screen, and its start key (F5) replays the scene instead of reloading.
- `L` switches to the day palette for bright rooms or weak projectors. With the system setting for reduced motion (or
  `?motion=reduce` in the URL) people and numbers jump to their places instead of walking and counting.
- The stage scales like a slide at 16:9 and 16:10, from 1280×720 to 4K.

## Files

| File | What it holds |
|---|---|
| `catalog.js` | Accelerators (memory, bandwidth, dense TFLOPS per precision, fabric, node size, TDP, rough price, chip generation), the `ARCHS` generation → format table, and models (shape, attention layer types, MoE, context limits, shipped precision, Hugging Face repo) |
| `tools/import-model.js` | Derives or verifies model entries from Hugging Face `config.json` and safetensors metadata; finds and verifies quantized checkpoints (node 18+, no dependencies) |
| `tools/check-sources.js` | Verifies every accelerator's source page resolves and still shows its numbers |
| `tests/anchors.js` | Calibration anchors: published fit statements and measured numbers the model must reproduce |
| `tests/golden.js`, `tests/golden.json` | Regression snapshot of the engine over an engine × hardware × model × format × context × retention matrix, reverse sizing and the compatibility matrix |
| `tests/engines.js`, `tests/common.js`, `tests/arcade-model.js` | Engine serving behavior and hardware coverage; the shared state helpers; the arcade's view model |
| `tools/embed-fonts.js` | Writes `arcade/fonts.css` from the font files |
| `engine.js` | The model: memory budget, KV geometry per attention type, roofline decode step, prefill cost, prefill/decode interleaving, speculative decoding, engine serving behavior (slots, batch caps, layer split), layout search, reverse search. Runs in node too (`require('./engine.js')`) |
| `common.js` | Shared by the planner and the arcade: DOM helpers, formats, storage, catalogs with custom entries, the state (defaults, sanitizing, side effects of a change), engine parameters from a state, and the `#plan=` link between the two pages |
| `app.js` | The UI: inputs, KPI tiles, memory bar, timing table, capacity-frontier and speed charts, optimization ledger, reverse table, compatibility matrix, catalog editor |
| `arcade/index.html` | The arcade stage: markup and styles (night and day palettes) |
| `arcade/arcade.js` | The arcade's controller: state, control deck, keys, scoreboard, baseline, undo, scenes, the streaming chat, overlays |
| `arcade/model.js` | What the stage shows, derived from one engine result: the crowd (response time law), memory of one GPU, limits, sprites, layout, deltas and ticker lines. Pure, tested in node |
| `arcade/world.js` | The pixel world on a low-resolution canvas scaled up by whole device pixels: tanks, racks, people, tokens, effects |
| `arcade/sprites.js`, `arcade/sfx.js`, `arcade/scenes.js` | Pixel art and a 3×5 bitmap font; WebAudio blips; the built-in scenes |
| `arcade/fonts.css`, `arcade/fonts/` | Press Start 2P and Pixelify Sans (SIL Open Font License), embedded as data URLs by `tools/embed-fonts.js` so they load from `file://` |
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

The serving-engine dropdown (vLLM, SGLang, TensorRT-LLM, Ollama, or hardware capability only) applies each engine's
documented behavior on top of the silicon table:

- **Where it runs**: TensorRT-LLM on NVIDIA only, SGLang on NVIDIA, AMD and TPU, Ollama on NVIDIA, AMD and Apple,
  vLLM on everything in the catalog (some through plugins).
- **Formats**: which weight formats run natively or weight-only on each hardware family, and which KV-cache dtypes
  exist. Ollama loads GGUF (Q4_K_M 4.89 and Q8_0 8.5 bits per weight) and offers q8_0 and q4_0 KV caches; the other
  engines have FP8 KV caches but no INT8 or INT4 ones.
- **How requests are served**: vLLM, SGLang and TensorRT-LLM batch continuously over a paged KV cache, up to their
  default caps per replica (`max_num_seqs` 1024, or 256 below 70 GiB and on A100; `max_running_requests` 2048 to 4096;
  `max_batch_size` 2048). Ollama runs `OLLAMA_NUM_PARALLEL` slots (1 by default), each reserving its full context when
  the model loads, splits a larger model across the GPUs of one node by layers (memory adds up, speed does not), reuses
  prompt prefixes within a slot only, parks idle slots in llama-server's 8 GiB host prompt cache and queues 512 requests.
  An optimization an engine lacks counts as off. There are no per-engine speed factors: the same batch decodes at the
  same speed on every engine.

The data lives in `ENGINES` in `catalog.js` with the doc and source-code URLs and the date it was transcribed; the
Catalog view renders it. Re-transcribe when the docs change.

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

### Tests

All tests are plain node scripts without dependencies; each exits non-zero on a failure.

```bash
node tests/anchors.js
node tests/golden.js
node tests/engines.js
node tests/common.js
node tests/arcade-model.js
```

`tests/golden.js` compares the engine with a snapshot of about 1,900 cases. After an intended change to the model, read
the reported differences and rewrite the snapshot with `node tests/golden.js --update`.

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
