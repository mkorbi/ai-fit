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
console.log(`\n${fail} problem${fail === 1 ? '' : 's'}.`);
process.exit(fail ? 1 : 0);
