#!/usr/bin/env node
/* prices.js — purchase and rental prices must point at known accelerators, be positive where given and carry a source. */
const path = require('path');
const { HARDWARE } = require(path.join(__dirname, '..', 'catalog.js'));
const { PROVIDERS, CLOUD } = require(path.join(__dirname, '..', 'providers.js'));
const ids = new Set(HARDWARE.map((h) => h.id));
let fail = 0;
const bad = (msg) => { fail++; console.log('FAIL ' + msg); };
for (const pv of PROVIDERS) {
  if (!CLOUD[pv.id]) bad(`${pv.id}: no offers table`);
  if (!['USD', 'EUR'].includes(pv.currency)) bad(`${pv.id}: currency ${pv.currency}`);
  let n = 0;
  for (const [id, o] of Object.entries(CLOUD[pv.id] || {})) {
    if (!ids.has(id)) bad(`${pv.id}: unknown accelerator ${id}`);
    if (o == null) continue;
    n++;
    if (!o.instance) bad(`${pv.id} ${id}: no instance type`);
    if (!o.source || !/^https:\/\//.test(o.source)) bad(`${pv.id} ${id}: no source URL`);
    if (o.gpuHour != null && !(o.gpuHour > 0 && o.gpuHour < 100)) bad(`${pv.id} ${id}: implausible ${o.gpuHour}/GPU-hour`);
    if (o.gpuHour != null && o.instanceHour != null && o.gpus && Math.abs(o.instanceHour / o.gpus - o.gpuHour) > 0.01 * o.gpuHour + 0.002) bad(`${pv.id} ${id}: instance price ÷ GPUs ≠ GPU price`);
  }
  console.log(`ok   ${pv.name.padEnd(16)} ${String(n).padStart(2)} accelerators offered (${pv.currency}, ${pv.region}, checked ${pv.checked})`);
}
let priced = 0;
for (const hw of HARDWARE) {
  if (!hw.buy) continue;
  if (hw.buy.usd != null) { priced++; if (!(hw.buy.usd > 50 && hw.buy.usd < 2e6)) bad(`${hw.id}: implausible purchase price ${hw.buy.usd}`); }
  if (hw.buy.usd != null && !/^https:\/\//.test(hw.buy.source || '')) bad(`${hw.id}: purchase price without source`);
}
console.log(`ok   ${priced} of ${HARDWARE.length} accelerators have a purchase price`);
console.log(`\n${fail} problem${fail === 1 ? '' : 's'}.`);
process.exit(fail ? 1 : 0);
