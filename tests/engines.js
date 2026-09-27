#!/usr/bin/env node
/* engines.js - behavior of the serving-engine profiles in engine.js.
 *   node tests/engines.js
 * Hardware coverage, slot engines (Ollama), sequential layer split, gated optimizations, engine-specific format sizes,
 * default batch caps, and the reverse sizing search. A fixture engine exercises the mechanics independently of the
 * catalog data; the Ollama checks pin the documented behavior (see ENGINES.ollama.serving.sources).
 */
const path = require('path');
const catalog = require(path.join(__dirname, '..', 'catalog.js'));
const E = require(path.join(__dirname, '..', 'engine.js'));
const { HARDWARE, MODELS, NETWORKS, ENGINES, ENGINE_FAMILIES } = catalog;
const hw = (id) => HARDWARE.find((x) => x.id === id), md = (id) => MODELS.find((x) => x.id === id);
const net = NETWORKS.find((n) => n.id === 'ib3200');

let fail = 0, pass = 0;
function check(name, ok, detail) {
  if (ok) pass++; else fail++;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}${!ok && detail !== undefined ? '   (' + detail + ')' : ''}`);
}
const close = (a, b, rel = 1e-9) => Math.abs(a - b) <= rel * Math.max(1, Math.abs(a), Math.abs(b));

/* Engine parameters in the shape app.js builds them; `o` overrides top-level keys, `o.wl` / `o.opt` merge. */
function P(o = {}) {
  const h = typeof o.hw === 'object' ? o.hw : hw(o.hw || 'h100-sxm');
  const m = md(o.model || 'llama-3.3-70b');
  const engine = o.engine || 'vllm';
  const rest = Object.assign({}, o); delete rest.wl; delete rest.opt; delete rest.hw; delete rest.model;
  return Object.assign({
    hw: h, model: m, wPrec: 'bf16', kvPrec: 'fp8', engine, count: 8, nodeGpus: h.nodeGpus, link: h.link, net, hostRamGB: 1024,
    dpAttention: false, allowCrossTp: false, slots: 0, tp: 1, pp: 1, replicas: 1, adv: { engine },
  }, rest, {
    hw: h, model: m,
    wl: Object.assign({ users: 50, activity: 0.25, ctx: 131072, prefix: 2048, newPrompt: 1000, output: 500, retention: 'host', target: 20, ttftMax: 30 }, o.wl || {}),
    opt: Object.assign({ prefixCache: true, spec: false, pd: false }, o.opt || {}),
  });
}
const auto = (o) => E.autoConfig(P(o)).best;

console.log('Hardware coverage');
check('TensorRT-LLM does not load BF16 on AMD MI300X', E.formatSupport(hw('mi300x'), 'bf16', 'trtllm') === 'unsupported');
check('SGLang does not load BF16 on an Apple M3 Ultra', E.formatSupport(hw('m3-ultra-512'), 'bf16', 'sglang') === 'unsupported');
check('vLLM runs on Apple silicon through vllm-metal', E.formatSupport(hw('m3-ultra-512'), 'bf16', 'vllm') === 'native');
check('Ollama does not run on a TPU', E.formatSupport(hw('tpu-v5p'), 'bf16', 'ollama') === 'unsupported' && E.kvSupport(hw('tpu-v5p'), 'bf16', 'ollama') === 'unsupported');
check('hardware capability only covers everything', E.formatSupport(hw('mi300x'), 'bf16', 'none') === 'native');
const custom = { id: 'x', name: 'Custom X', mem: 64, bw: 2000, tflops: { fp16: 400 }, link: 'pcie5', linkBw: 64, nodeGpus: 8 };
check('custom hardware without a generation stays usable', E.formatSupport(custom, 'bf16', 'trtllm') === 'native' && E.engineCovers(custom, 'trtllm') === 'unknown');
{
  const r = auto({ hw: 'mi300x', engine: 'trtllm', kvPrec: 'bf16' });
  const crit = r.warnings.filter((w) => w.level === 'crit');
  check('TRT-LLM on 8x MI300X serves nobody, with one blocker naming the reason', !r.loadable && r.maxUsers === 0 && crit.length === 1 && /does not run on/.test(crit[0].text), crit.map((w) => w.text).join(' | '));
}
check('every engine family is named', ENGINES.vllm.families.every((f) => ENGINE_FAMILIES.some((x) => x.id === f)) && /NVIDIA Ampere, Ada Lovelace, Hopper/.test(E.familyNames(ENGINES.trtllm.families)));

console.log('Batch floor with sessions kept in GPU memory');
{
  const r = E.evaluate(P({ hw: 'rtx-4090', model: 'llama-3.1-8b', wPrec: 'int4', kvPrec: 'fp8', count: 1, wl: { retention: 'gpu', activity: 0.25 } }));
  check('one resident 128k session still serves one user', r.fits && r.maxSessions === 1 && r.maxUsers === 1, `maxSessions ${r.maxSessions}, maxUsers ${r.maxUsers}`);
}

console.log('Reverse sizing');
{
  const r = E.reverse(P({ wl: { users: 500 } }), [hw('h100-sxm')])[0];
  check('500 users of Llama 70B on H100 need at most 72 GPUs', !r.infeasible && r.gpusUsed <= 72, r.gpusUsed);
  let brute = null;
  for (let R = 1; R <= 64 && !brute; R++) { const q = E.evaluate(P({ wl: { users: 500 }, tp: 8, pp: 1, replicas: R, count: 8 * R })); if (q.memOK && q.speedOK) brute = 8 * R; }
  check('matches a brute-force search over TP 8 replicas', r.gpusUsed <= brute, `${r.gpusUsed} vs ${brute}`);
  for (const retention of ['host', 'none', 'gpu']) {
    let prev = 0, mono = true;
    for (let u = 50; u <= 1000; u += 50) { const q = E.reverse(P({ wl: { users: u, retention } }), [hw('h100-sxm')])[0]; const g = q.infeasible ? Infinity : q.gpusUsed; if (g < prev) mono = false; prev = g; }
    check(`GPU count never shrinks as users grow (retention ${retention})`, mono);
  }
  const x = E.reverse(P({ wl: { users: 100, ttftMax: 0.001 } }), [hw('h100-sxm')])[0];
  check('an impossible first-token limit reports why', x.infeasible && /over the .* limit/.test(x.reason), x.reason);
}

console.log('Engine mechanics (fixture engine)');
{
  const ALL = ENGINE_FAMILIES.map((f) => f.id);
  ENGINES.fixture = {
    name: 'Fixture', checked: 'test', families: ALL, weights: {}, kvCache: {},
    serving: { batching: 'slots', slots: { default: 4, env: 'FX_PARALLEL' }, paged: false, pp: 'sequential', maxTp: 1, multiNode: false,
      prefixCache: 'per-slot', spec: false, pd: false, dpAttention: false, retention: ['gpu', 'none'], queueMax: 512 },
  };
  const base = { hw: 'l40s', model: 'llama-3.1-8b', kvPrec: 'bf16', count: 1, wl: { ctx: 8192, retention: 'none' } };
  const fx = E.evaluate(P(Object.assign({}, base, { engine: 'fixture' })));
  const vl = E.evaluate(P(Object.assign({}, base, { engine: 'none' })));
  check('slots cap the sessions of a replica at the slot count', fx.fits && fx.maxSessions === 4 && fx.memSessions > 4 && vl.maxSessions > 20, `${fx.maxSessions} / ${fx.memSessions} / ${vl.maxSessions}`);
  check('slot engines never run more than slots x replicas at once', fx.maxConc <= 4 * fx.R && fx.bAt <= 4);
  check('every slot reserves its full context at load', fx.maxCtxAtLoad === fx.solveMaxCtx(4));
  const big = E.evaluate(P(Object.assign({}, base, { engine: 'fixture', slots: 64 })));
  check('more slots than memory holds does not load', !big.fits && big.warnings.some((w) => w.level === 'crit' && /CPU/.test(w.text)));
  const busy = E.evaluate(P(Object.assign({}, base, { engine: 'fixture', wl: { users: 100, activity: 1, ctx: 8192, retention: 'none' } })));
  check('requests beyond the slots wait in the queue', busy.fits && !busy.memOK && busy.slotLimited && busy.capLimit === 'slots');
  check('no paged-block fragmentation for slot reservations', close(fx.kvAvail, fx.kvAvailRaw / fx.kvRepl));
  // sequential layer split: a token passes every stage in turn
  const two = E.evaluate(P(Object.assign({}, base, { engine: 'fixture', model: 'llama-3.3-70b', hw: 'h100-sxm', wPrec: 'fp8', tp: 1, pp: 2, count: 2, wl: { ctx: 8192, retention: 'none' } })));
  const d = two.decodeStep(3, false);
  const expect = (two.denseBytes + two.expertBytes + 3 * E.kvAtCtx(E.norm(md('llama-3.3-70b')), 'bf16', 8192 - 250)) / (hw('h100-sxm').bw * 1e9 * E.DEFAULT_ADV.bwEff);
  check('layer split reads all weights and KV one stage after another', close(d.tBw, expect, 1e-9), `${d.tBw} vs ${expect}`);
  check('layer split has no pipeline bubble and computes on one GPU at a time', two.computeGpus === 1 && close(d.step, Math.max(d.tBw, d.tComp) + d.tComm));
  const tp2 = E.evaluate(P(Object.assign({}, base, { engine: 'none', model: 'llama-3.3-70b', hw: 'h100-sxm', wPrec: 'fp8', tp: 2, pp: 1, count: 2, wl: { ctx: 8192, retention: 'none' } })));
  check('layer split over 2 GPUs is slower per user than tensor parallel over 2', two.loadAt(1, false).perUser < tp2.loadAt(1, false).perUser);
  const ls = E.layouts(E.norm(md('llama-3.3-70b')), 16, 8, false, 'fixture');
  check('the layout search keeps TP 1 and one node', ls.length > 0 && ls.every((l) => l.tp === 1 && l.pp <= 8));
  const tp8 = E.evaluate(P(Object.assign({}, base, { engine: 'fixture', model: 'llama-3.3-70b', hw: 'h100-sxm', tp: 8, pp: 1, count: 8 })));
  check('tensor parallel on a layer-split engine does not load', !tp8.loadable && tp8.warnings.some((w) => w.level === 'crit' && /no tensor parallelism/.test(w.text)));
  // gated optimizations behave exactly like switched off
  const asked = E.evaluate(P(Object.assign({}, base, { engine: 'fixture', dpAttention: true, opt: { prefixCache: true, spec: true, pd: true }, wl: { ctx: 8192, retention: 'host' } })));
  const off = E.evaluate(P(Object.assign({}, base, { engine: 'fixture', opt: { prefixCache: false, spec: false, pd: false }, wl: { ctx: 8192, retention: 'none' } })));
  check('unavailable options count as off', !asked.effective.opt.spec && !asked.effective.opt.pd && !asked.effective.opt.prefixCache && asked.effective.retention === 'none' && asked.S === 0);
  check('results are identical to the options switched off', asked.maxConc === off.maxConc && close(asked.at.perUser, off.at.perUser) && close(asked.ttft, off.ttft) && asked.gpusUsed === off.gpusUsed);
  check('speculative decoding is gated inside the load model too', close(asked.loadAt(2, true).perUser, asked.loadAt(2, false).perUser));
  check('one note lists what the engine lacks', asked.warnings.filter((w) => /has no /.test(w.text)).length === 1);
  check('per-slot prefix reuse skips the prefill but stores the prefix per session', asked.S === 0 && asked.Scomp === 2048);
  delete ENGINES.fixture;
}

console.log('Ollama (documented defaults)');
{
  const s = ENGINES.ollama.serving;
  check('one parallel request by default (OLLAMA_NUM_PARALLEL=1)', s.batching === 'slots' && s.slots.default === 1);
  const r = auto({ engine: 'ollama', hw: 'rtx-4090', model: 'llama-3.1-8b', wPrec: 'int4', kvPrec: 'bf16', count: 1, wl: { users: 50, activity: 0.1, ctx: 8192, newPrompt: 300, output: 400 } });
  check('50 chat users on one 4090: one request at a time, the rest queue', r.fits && r.maxConc === 1 && r.slotLimited && r.capLimit === 'slots');
  const four = auto({ engine: 'ollama', hw: 'rtx-4090', model: 'llama-3.1-8b', wPrec: 'int4', kvPrec: 'bf16', count: 1, slots: 4, wl: { users: 50, activity: 0.1, ctx: 8192 } });
  check('OLLAMA_NUM_PARALLEL=4 serves four at once', four.fits && four.maxConc <= 4 && four.maxSessions === 4);
  const m8 = E.norm(md('llama-3.1-8b'));
  const ratio = E.weightBytes(m8, 'int4', 'ollama') / E.weightBytes(m8, 'int4', 'vllm');
  check('Q4_K_M GGUF is 4.89 bits per weight, about 15% above AWQ INT4', close(E.weightBytes(m8, 'int4', 'ollama'), 8.0e9 * 4.8944 / 8) && ratio > 1.1 && ratio < 1.2, ratio.toFixed(3));
  check('KV q8_0 is 8.5 bits per element', close(E.kvPerTokenFull(m8, 'int8', 'ollama') / E.kvPerTokenFull(m8, 'bf16'), 1.0625 / 2));
  check('INT8 KV exists in Ollama (q8_0) but not in vLLM', E.kvSupport(hw('h100-sxm'), 'int8', 'ollama') === 'supported' && E.kvSupport(hw('h100-sxm'), 'int8', 'vllm') === 'unsupported' && E.kvSupport(hw('h100-sxm'), 'int8', 'none') === 'supported');
  check('no FP8 weights in GGUF', E.formatSupport(hw('h100-sxm'), 'fp8', 'ollama') === 'unsupported');
  check('format labels follow the engine', E.precLabel('int4', 'ollama').startsWith('Q4_K_M') && E.kvLabel('int8', 'ollama') === 'q8_0' && E.precLabel('int4', 'vllm') === E.PREC_LABEL.int4);
  // no hidden speed multipliers: the same batch decodes at the same speed on every engine
  const b = { hw: 'l40s', model: 'llama-3.1-8b', kvPrec: 'bf16', count: 1, wl: { ctx: 8192, retention: 'none' } };
  const ol = E.evaluate(P(Object.assign({}, b, { engine: 'ollama' }))), vl = E.evaluate(P(Object.assign({}, b, { engine: 'vllm' })));
  check('Ollama and vLLM decode at the same speed for the same batch', close(ol.loadAt(2, false).perUser, vl.loadAt(2, false).perUser), `${ol.loadAt(2, false).perUser} vs ${vl.loadAt(2, false).perUser}`);
  check('Llama 70B BF16 needs 4 H100 by layer split', JSON.stringify(E.minGpus(hw('h100-sxm'), md('llama-3.3-70b'), 'bf16', 'bf16', 32768, {}, { engine: 'ollama' })) === '{"G":4,"tp":1,"pp":4}');
  const rv = E.reverse(P({ engine: 'ollama', model: 'llama-3.1-8b', wPrec: 'int4', kvPrec: 'bf16', slots: 4, wl: { users: 100, activity: 1, ctx: 8192, retention: 'none' } }), [hw('h100-sxm')])[0];
  check('reverse sizing adds Ollama servers of TP 1 until every request has a slot', !rv.infeasible && rv.tp === 1 && rv.R * 4 >= 100, `${rv.gpusUsed} = ${rv.tp}x${rv.pp}x${rv.R}`);
  const oneNode = E.reverse(P({ engine: 'ollama', model: 'kimi-k2', wPrec: 'int4', kvPrec: 'bf16', wl: { users: 10, ctx: 8192 } }), [hw('l40s')])[0];
  check('a model larger than one node is infeasible for Ollama', oneNode.infeasible && /one node/.test(oneNode.reason), oneNode.reason);
  const host = auto({ engine: 'ollama', hw: 'rtx-4090', model: 'llama-3.1-8b', wPrec: 'int4', kvPrec: 'bf16', count: 1, wl: { users: 50, activity: 0.1, ctx: 32768, retention: 'host' } });
  check('idle slots park in the 8 GiB llama-server prompt cache, not the whole RAM', host.hostSessions === Math.floor(8192 * 2 ** 20 / host.kv.perSession), `${host.hostSessions}`);
}

console.log('Default batch caps');
{
  const g = auto({ hw: 'b200', model: 'gpt-oss-20b', wPrec: 'fp4', count: 1, wl: { users: 20000, ctx: 4096, prefix: 512, newPrompt: 200, output: 200, retention: 'none' } });
  check('vLLM max_num_seqs caps a B200 replica at 1024 requests', g.maxBatch === 1024 && g.maxConc === 1024 && g.capLimit === 'max-batch' && g.memSessions > 1024, `${g.maxBatch} ${g.maxConc} ${g.capLimit}`);
  const a100 = E.evaluate(P({ hw: 'a100-sxm-80', model: 'llama-3.1-8b', count: 1, wl: { ctx: 4096 } }));
  check('vLLM max_num_seqs is 256 on an A100', a100.maxBatch === 256);
  const sg = E.evaluate(P({ engine: 'sglang', model: 'llama-3.1-8b', count: 1, wl: { ctx: 4096 } }));
  check('SGLang max_running_requests stays within 2048 to 4096', sg.maxBatch >= 2048 && sg.maxBatch <= 4096, sg.maxBatch);
  const tr = E.evaluate(P({ engine: 'trtllm', model: 'llama-3.1-8b', count: 1, wl: { ctx: 4096 } }));
  check('TensorRT-LLM max_batch_size is 2048', tr.maxBatch === 2048);
  check('hardware capability only has no cap', E.evaluate(P({ engine: 'none', model: 'llama-3.1-8b', count: 1 })).maxBatch === Infinity);
}

console.log('Planner default (regression)');
{
  const r = auto({});
  check('8x H100, Llama 70B BF16, 128k: TP 8, 18 concurrent, 72 users', r.tp === 8 && r.pp === 1 && r.R === 1 && r.maxConc === 18 && r.maxUsers === 72);
  check('37.77 tok/s per user, 0.264 s TTFT, $23.2/h', close(r.at.perUser, 37.769, 1e-3) && close(r.ttft, 0.2637, 1e-3) && close(r.price, 23.2, 1e-9), `${r.at.perUser} ${r.ttft} ${r.price}`);
}

console.log(`\n${pass} passed, ${fail} failed.`);
process.exit(fail ? 1 : 0);
