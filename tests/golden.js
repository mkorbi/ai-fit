#!/usr/bin/env node
/* golden.js - regression snapshot of the capacity engine.
 *   node tests/golden.js            compare the engine against tests/golden.json
 *   node tests/golden.js --update   rewrite the snapshot (only after reviewing the reported diff)
 * Covers autoConfig over an engine x hardware x model x precision x context x retention matrix, reverse sizing and the
 * minGpus compatibility matrix. Every engine change has to show up here as an intended, reviewed diff.
 */
const fs = require('fs');
const path = require('path');
const { HARDWARE, MODELS, NETWORKS, ENGINES } = require(path.join(__dirname, '..', 'catalog.js'));
const E = require(path.join(__dirname, '..', 'engine.js'));

const FILE = path.join(__dirname, 'golden.json');
const hw = (id) => HARDWARE.find((x) => x.id === id), md = (id) => MODELS.find((x) => x.id === id);
const net = NETWORKS.find((n) => n.id === 'ib3200');
const sig = (x) => (x == null || !isFinite(x) ? String(x) : Number(x.toPrecision(6)));

/* A stable 32-bit hash of the case key keeps the sampled subset fixed when the matrix grows. */
function hash(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }

const ENGINE_IDS = Object.keys(ENGINES);
const HW_IDS = ['h100-sxm', 'b200', 'mi300x', 'l40s', 'rtx-4090', 'a100-sxm-80', 'm3-ultra-512', 'tpu-v5p', 'gaudi3'];
const MODEL_IDS = ['llama-3.1-8b', 'llama-3.3-70b', 'qwen3-235b-a22b', 'deepseek-v3', 'gpt-oss-120b', 'gemma-3-27b', 'qwen3-next-80b'];
const PRECS = [['bf16', 'fp8'], ['int4', 'fp8'], ['bf16', 'bf16'], ['fp8', 'fp8']];
const CTXS = [8192, 131072];
const RETENTIONS = ['host', 'gpu', 'none'];

function params(o) {
  const h = hw(o.hw), m = md(o.model);
  return {
    hw: h, model: m, wPrec: o.w, kvPrec: o.kv, engine: o.engine, count: Math.min(8, h.nodeGpus), nodeGpus: h.nodeGpus, link: h.link, net,
    hostRamGB: 1024, dpAttention: E.isMla(E.norm(m)), allowCrossTp: false, slots: 0,
    wl: { users: o.users || 200, activity: o.activity || 0.25, ctx: o.ctx, prefix: 2048, newPrompt: 1000, output: 500, retention: o.retention, target: 20, ttftMax: 30 },
    opt: { prefixCache: true, spec: !!o.spec, pd: false }, adv: { engine: o.engine },
  };
}
const FIELDS = ['fits', 'memOK', 'speedOK', 'layout', 'maxSessions', 'bSpeed', 'maxConc', 'maxUsers', 'perUser', 'ttft', 'gpusUsed', 'kvAvail', 'warnings'];
function evalRecord(r) {
  const lv = { crit: 0, warn: 0, info: 0 };
  for (const w of r.warnings) lv[w.level]++;
  return [+r.fits, +r.memOK, +r.speedOK, `${r.tp}x${r.pp}x${r.R}`, r.maxSessions, r.bSpeed, r.maxConc, r.maxUsers,
    r.at ? sig(r.at.perUser) : null, sig(r.ttft), r.gpusUsed, sig(r.kvAvail), `c${lv.crit}w${lv.warn}i${lv.info}`];
}

function snapshot() {
  const out = { fields: FIELDS, evaluate: {}, reverse: {}, minGpus: {} };
  for (const engine of ENGINE_IDS) for (const h of HW_IDS) for (const m of MODEL_IDS) for (const [w, kv] of PRECS) for (const ctx of CTXS) for (const retention of RETENTIONS) {
    const key = [engine, h, m, w, kv, ctx, retention].join('|');
    if (hash(key) % 5 !== 0) continue;                       // a fixed fifth of the matrix keeps the file small
    out.evaluate[key] = evalRecord(E.autoConfig(params({ engine, hw: h, model: m, w, kv, ctx, retention })).best);
  }
  // speculative decoding and activity edge cases on the default planner shape
  for (const engine of ENGINE_IDS) for (const activity of [0.1, 0.6, 1]) {
    const key = ['spec', engine, activity].join('|');
    out.evaluate[key] = evalRecord(E.autoConfig(params({ engine, hw: 'h100-sxm', model: 'llama-3.3-70b', w: 'bf16', kv: 'fp8', ctx: 131072, retention: 'host', spec: true, activity })).best);
  }
  for (const engine of ['none', 'vllm']) for (const h of ['h100-sxm', 'b200', 'mi300x', 'l40s']) for (const m of ['llama-3.1-8b', 'llama-3.3-70b', 'deepseek-v3']) for (const users of [50, 200, 500, 1000]) for (const retention of ['host', 'none']) {
    const key = [engine, h, m, users, retention].join('|');
    const r = E.reverse(params({ engine, hw: h, model: m, w: 'fp8', kv: 'fp8', ctx: 32768, retention, users }), [hw(h)])[0];
    out.reverse[key] = r.infeasible ? 'infeasible' : `${r.gpusUsed} = ${r.tp}x${r.pp}x${r.R}`;
  }
  for (const engine of ['none', 'vllm']) for (const [w, kv] of [['bf16', 'bf16'], ['fp8', 'fp8']]) for (const m of MODELS) {
    const key = [engine, w, kv, m.id].join('|');
    out.minGpus[key] = HARDWARE.map((h) => { const g = E.minGpus(h, m, w, kv, 32768, { engine }); return g ? `${g.G}:${g.tp}x${g.pp}` : '-'; }).join(' ');
  }
  return out;
}

const now = snapshot();
if (process.argv.includes('--update')) {
  const lines = ['{"fields":' + JSON.stringify(FIELDS)];
  for (const section of ['evaluate', 'reverse', 'minGpus']) lines.push(`,"${section}":{\n` + Object.entries(now[section]).map(([k, v]) => JSON.stringify(k) + ':' + JSON.stringify(v)).join(',\n') + '\n}');
  fs.writeFileSync(FILE, lines.join('') + '}\n');
  const n = ['evaluate', 'reverse', 'minGpus'].reduce((a, s) => a + Object.keys(now[s]).length, 0);
  console.log(`Wrote ${n} golden cases to ${path.relative(process.cwd(), FILE)}.`);
  process.exit(0);
}
if (!fs.existsSync(FILE)) { console.error('No tests/golden.json yet: run node tests/golden.js --update'); process.exit(1); }
const was = JSON.parse(fs.readFileSync(FILE, 'utf8'));

const byField = {};
let total = 0, diff = 0;
const note = (field, line) => { (byField[field] = byField[field] || []).push(line); };
for (const section of Object.keys(now).filter((k) => k !== 'fields')) {
  const a = was[section] || {}, b = now[section];
  for (const key of new Set([...Object.keys(a), ...Object.keys(b)])) {
    total++;
    if (!(key in a)) { diff++; note(`${section}: new case`, key); continue; }
    if (!(key in b)) { diff++; note(`${section}: removed case`, key); continue; }
    const x = a[key], y = b[key];
    if (JSON.stringify(x) === JSON.stringify(y)) continue;
    diff++;
    if (!Array.isArray(x) || !Array.isArray(y)) { note(`${section}`, `${key}: ${JSON.stringify(x)} -> ${JSON.stringify(y)}`); continue; }
    FIELDS.forEach((f, i) => { if (JSON.stringify(x[i]) !== JSON.stringify(y[i])) note(`${section}.${f}`, `${key}: ${JSON.stringify(x[i])} -> ${JSON.stringify(y[i])}`); });
  }
}
const verbose = process.argv.includes('--all');
for (const [field, lines] of Object.entries(byField).sort()) {
  console.log(`${field}: ${lines.length} change${lines.length === 1 ? '' : 's'}`);
  for (const l of lines.slice(0, verbose ? Infinity : 6)) console.log('    ' + l);
  if (!verbose && lines.length > 6) console.log(`    ... ${lines.length - 6} more (--all lists every one)`);
}
console.log(`\n${diff} of ${total} golden cases differ.`);
process.exit(diff ? 1 : 0);
