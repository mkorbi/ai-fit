#!/usr/bin/env node
/* sim.js — the simulator's presets must load, and the one-knob sweeps must behave (people never grow with context). */
const path = require('path');
const S = require(path.join(__dirname, '..', 'sim.js'));
let fail = 0;
for (const pr of S.PRESETS) {
  const s = Object.assign({}, S.BASE, pr.s), ev = S.evaluateState(s);
  const ok = ev.r.fits && ev.ctxOK;
  if (!ok) fail++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} preset ${pr.id.padEnd(12)} ${ev.limit.padEnd(8)} people ${String(ev.r.maxUsers).padStart(5)} at once ${String(ev.r.maxConc).padStart(4)}  ${ev.r.at ? ev.r.at.perUser.toFixed(0) + ' tok/s' : ''}  | ${ev.explain.slice(0, 90)}`);
}
const base = Object.assign({}, S.BASE);
const byCtx = S.sweep(base, 'ctx', S.CTX_STEPS);
const mono = byCtx.every((p, i) => i === 0 || p.users <= byCtx[i - 1].users);
if (!mono) fail++;
console.log(`${mono ? 'ok  ' : 'FAIL'} people vs context is non-increasing: ${byCtx.map((p) => p.users).join(' ')}`);
const byHw = S.sweep(base, 'hw', S.HW_POOL); console.log(`info people vs GPU type: ${byHw.map((p) => p.v + '=' + p.users).join(' ')}`);
const byModel = S.sweep(base, 'model', S.MODEL_POOL); console.log(`info people vs model: ${byModel.map((p) => p.v + '=' + p.users).join(' ')}`);
// every GPU × every model must evaluate to finite numbers the stage can draw (both dropdowns list the whole catalog)
const { HARDWARE, MODELS } = require(path.join(__dirname, '..', 'catalog.js'));
let combos = 0, bad = [];
const LIMITS = new Set(['ok', 'memory', 'speed', 'weights', 'format', 'context', 'prefill']);
for (const hw of HARDWARE) for (const m of MODELS) {
  combos++;
  try {
    const ev = S.evaluateState(Object.assign({}, S.BASE, { hw: hw.id, model: m.id }));
    const r = ev.r, nums = [r.W, r.wPerGpu, r.overhead, r.kv.perSession, r.maxSessions, r.maxUsers, r.maxConc, r.kvAvailRaw];
    if (!nums.every(Number.isFinite) || !LIMITS.has(ev.limit) || !ev.explain) bad.push(`${hw.id} × ${m.id}: ${nums.join(', ')} ${ev.limit}`);
    else if (r.fits && (!r.at || !Number.isFinite(r.at.perUser) || !Number.isFinite(r.ttftWarm))) bad.push(`${hw.id} × ${m.id}: fits but timing is not finite`);
  } catch (e) { bad.push(`${hw.id} × ${m.id}: ${e.message}`); }
}
if (bad.length) fail++;
console.log(`${bad.length ? 'FAIL' : 'ok  '} ${combos} GPU × model combinations evaluate to finite numbers${bad.length ? ':\n  ' + bad.slice(0, 10).join('\n  ') : ''}`);
console.log(`\n${fail} problem${fail === 1 ? '' : 's'}.`);
process.exit(fail ? 1 : 0);
