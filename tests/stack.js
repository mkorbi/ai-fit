#!/usr/bin/env node
/* stack.js — the Engine Stack content must be complete: every layer states its need per purpose, links its options and sizes itself. */
const path = require('path');
const S = require(path.join(__dirname, '..', 'stack.js'));
const P = ['inference', 'finetune', 'training'], NEEDS = new Set(['required', 'recommended', 'optional', 'none']);
let fail = 0;
const bad = (msg) => { fail++; console.log('FAIL ' + msg); };
for (const k of P) { const p = S.data.purposes[k]; if (!p || !p.reference || !p.gpuMemoryRule || !p.totals) bad(`purpose ${k} incomplete`); }
for (const L of S.data.layers) {
  for (const k of P) {
    if (!NEEDS.has((L.need || {})[k])) bad(`${L.id}: need for ${k} is ${(L.need || {})[k]}`);
    const r = (L.resources || {})[k];
    if (L.need[k] !== 'none' && (!r || ![r.cpu, r.ramGB, r.storageGB].every((x) => Number.isFinite(+x) && +x >= 0))) bad(`${L.id}: resources for ${k} missing or not numeric`);
  }
  if (!(L.options || []).length) bad(`${L.id}: no options`);
  for (const o of L.options || []) if (!/^https:\/\//.test(o.url || '')) bad(`${L.id}: ${o.name} has no https link`);
}
for (const k of P) { const t = S.totals(k); console.log(`ok   ${k.padEnd(10)} beside the GPUs ${t.side.cpu} CPU, ${t.side.ramGB} GB RAM, ${t.side.storageGB} GB disk · on the GPU servers ${t.gpu.cpu} CPU, ${t.gpu.ramGB} GB RAM`); }
console.log(`\n${S.data.layers.length} layers, ${fail} problem${fail === 1 ? '' : 's'}.`);
process.exit(fail ? 1 : 0);
