#!/usr/bin/env node
/* anchors.js — calibration anchors: published facts the model must reproduce.
 *   node tests/anchors.js
 * Two kinds: "fit" anchors (a vendor or engine statement that a model loads on N accelerators) and "measured" anchors
 * (numbers from engine logs or benchmarks, with a tolerance). Add your own measurements from `vllm bench serve` or
 * `sglang.bench_serving` to the MEASURED list; the test fails when a prediction drifts out of tolerance.
 */
const path = require('path');
const { HARDWARE, MODELS, NETWORKS } = require(path.join(__dirname, '..', 'catalog.js'));
const E = require(path.join(__dirname, '..', 'engine.js'));
const hw = (id) => HARDWARE.find((x) => x.id === id), md = (id) => MODELS.find((x) => x.id === id);
const net = NETWORKS.find((n) => n.id === 'ib3200');

const FIT = [
  { name: 'gpt-oss-120b (MXFP4) runs on a single 80 GB H100', src: 'OpenAI gpt-oss model card, Aug 2025', hw: 'h100-sxm', model: 'gpt-oss-120b', wPrec: 'fp4', kvPrec: 'bf16', ctx: 8192, expectG: 1 },
  { name: 'gpt-oss-20b (MXFP4) runs within 16 GB', src: 'OpenAI gpt-oss model card, Aug 2025', hw: { id: 'gpu-16gb', name: '16 GB GPU', mem: 16, bw: 900, tflops: { fp16: 100, fp8: 200 }, arch: 'sm89', nodeGpus: 1, link: 'none', linkBw: 0 }, model: 'gpt-oss-20b', wPrec: 'fp4', kvPrec: 'bf16', ctx: 8192, expectG: 1 },
  { name: 'Llama 3.1 405B in FP8 fits one 8× H100 80 GB node', src: 'Meta "Introducing Llama 3.1" blog, Jul 2024', hw: 'h100-sxm', model: 'llama-3.1-405b', wPrec: 'fp8', kvPrec: 'bf16', ctx: 8192, expectG: 8 },
  { name: 'DeepSeek-V3 (FP8) serves on 8× H200', src: 'DeepSeek-V3 README, Dec 2024', hw: 'h200-sxm', model: 'deepseek-v3', wPrec: 'fp8', kvPrec: 'bf16', ctx: 8192, expectG: 8 },
  { name: 'Llama 3.3 70B in BF16 does not fit 2× H100 80 GB (needs 4)', src: 'weights alone are 141 GB', hw: 'h100-sxm', model: 'llama-3.3-70b', wPrec: 'bf16', kvPrec: 'bf16', ctx: 8192, expectG: 4 },
  { name: 'Qwen3-235B-A22B in FP8 fits one 8× H100 node', src: 'Qwen3-235B-A22B-FP8 model card', hw: 'h100-sxm', model: 'qwen3-235b-a22b', wPrec: 'fp8', kvPrec: 'bf16', ctx: 8192, expectG: 8 },
];
const MEASURED = [
  { name: 'vLLM: Llama 3.1 70B BF16, TP 8 on 8× H100 SXM, gpu_memory_utilization 0.9 → KV cache of ~1.3M tokens', src: 'vLLM engine log "GPU KV cache size" as reported by users; e.g. vLLM GitHub discussions on 70B/H100 capacity',
    hw: 'h100-sxm', model: 'llama-3.3-70b', wPrec: 'bf16', kvPrec: 'bf16', tp: 8, pp: 1, metric: 'kvTokens', expect: 1.3e6, tol: 0.15 },
  // Add your own, e.g.:
  // { name: 'my cluster: Llama 3.3 70B FP8 on 4× H100, 64 concurrent, 4k ctx, per-user tok/s', src: 'vllm bench serve 2026-xx-xx', hw: 'h100-sxm', model: 'llama-3.3-70b', wPrec: 'fp8', kvPrec: 'fp8', tp: 4, pp: 1, metric: 'perUserTokS', b: 64, ctx: 4096, output: 256, expect: 41, tol: 0.25 },
];

let fail = 0;
const base = (o) => Object.assign({ hw: hw('h100-sxm'), model: md('llama-3.3-70b'), wPrec: 'bf16', kvPrec: 'bf16', engine: 'none', count: 8, nodeGpus: 8, link: 'nvlink', net, hostRamGB: 1024, dpAttention: false, allowCrossTp: false,
  wl: { users: 1, activity: 1, ctx: 8192, prefix: 0, newPrompt: 1000, output: 500, retention: 'none', target: 0.1, ttftMax: 1e9 }, opt: { prefixCache: false, spec: false, pd: false }, adv: {} }, o);
console.log('Fit anchors');
for (const a of FIT) {
  const h = typeof a.hw === 'string' ? hw(a.hw) : a.hw;
  const r = E.minGpus(h, md(a.model), a.wPrec, a.kvPrec, a.ctx);
  const got = r ? r.G : null;
  const ok = got != null && got <= a.expectG && (a.expectG === 1 ? got === 1 : true);
  if (!ok) fail++;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${a.name}: predicted ${got == null ? 'no layout' : got + ' (TP ' + r.tp + ' × PP ' + r.pp + ')'}, expected ≤ ${a.expectG}   [${a.src}]`);
}
console.log('Measured anchors');
for (const a of MEASURED) {
  const h = typeof a.hw === 'string' ? hw(a.hw) : a.hw;
  const p = base({ hw: h, model: md(a.model), wPrec: a.wPrec, kvPrec: a.kvPrec, count: a.tp * a.pp, nodeGpus: h.nodeGpus, link: h.link, tp: a.tp, pp: a.pp, replicas: 1 });
  if (a.ctx) p.wl = Object.assign({}, p.wl, { ctx: a.ctx, output: a.output || 500, users: a.b || 1 });
  const r = E.evaluate(p);
  let got;
  if (a.metric === 'kvTokens') got = r.kvAvail / E.kvPerTokenFull(E.norm(md(a.model)), a.kvPrec);
  else if (a.metric === 'perUserTokS') got = r.loadAt(a.b, false).perUser;
  else if (a.metric === 'aggTokS') got = r.loadAt(a.b, false).agg;
  else if (a.metric === 'ttft') got = r.ttft;
  const rel = Math.abs(got - a.expect) / a.expect;
  const ok = rel <= a.tol;
  if (!ok) fail++;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${a.name}: predicted ${got.toLocaleString('en-US', { maximumFractionDigits: 1 })}, expected ${a.expect.toLocaleString('en-US')} ±${Math.round(a.tol * 100)}% (off by ${(rel * 100).toFixed(0)}%)   [${a.src}]`);
}
console.log(`\n${fail} anchor${fail === 1 ? '' : 's'} failing.`);
process.exit(fail ? 1 : 0);
