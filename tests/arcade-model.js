#!/usr/bin/env node
/* arcade-model.js - the arcade's view model: crowd, memory, sprites, layout, baseline deltas and ticker lines.
 *   node tests/arcade-model.js
 */
const path = require('path');
const C = require(path.join(__dirname, '..', 'common.js'));
const E = require(path.join(__dirname, '..', 'engine.js'));
const M = require(path.join(__dirname, '..', 'arcade', 'model.js'));

let fail = 0, pass = 0;
function check(name, ok, detail) {
  if (ok) pass++; else fail++;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}${!ok && detail !== undefined ? '   (' + detail + ')' : ''}`);
}
const close = (a, b, rel = 1e-9) => Math.abs(a - b) <= rel * Math.max(1, Math.abs(a), Math.abs(b));
/* state → { st, p, r, vm } the way the arcade computes it */
function view(o) {
  const st = C.freshState(o || {});
  const p = C.buildParams(st), r = C.runPlan(st).best;
  const need = E.reverse(p, [p.hw])[0];
  return { st, p, r, vm: M.derive({ r, p, need }) };
}
const sumGroups = (g) => M.GROUPS.reduce((a, k) => a + g[k], 0);

console.log('Crowd');
{
  const { vm } = view({});
  check('planner default: all 50 served, 12.5 active, the rest parked in host RAM', vm.status === 'ok' && close(vm.crowd.active, 12.5) && close(vm.crowd.parked, 37.5) && vm.crowd.waiting === 0, JSON.stringify(vm.crowd));
  check('capacity is the planner\'s number: 72 users at 20 tok/s', vm.capacity.users === 72 && vm.capacity.conc === 18);
  check('money: $23.2/h, 730 hours a month, per user served at target', close(vm.money.perMonth, 23.2 * 730) && close(vm.money.perUserMonth, 23.2 * 730 / 50));
  const kv = view({ kvPrec: 'bf16' }).vm;
  check('BF16 KV: memory runs out, a queue forms; at 20 tok/s bandwidth is the limit', kv.status === 'queue' && kv.crowd.queueBy === 'memory' && kv.limit.key === 'memory bandwidth' && kv.crowd.waiting > 0 && kv.capacity.users === 12, JSON.stringify(kv.crowd) + ' ' + kv.limit.key);
  check('a queue has a wait time and a headline that says why', kv.crowd.wait > 0 && /waiting \d+ min: KV cache full/.test(kv.headline), kv.headline);
  const ol = view({ engine: 'ollama', hw: 'rtx-4090', count: 1, model: 'llama-3.1-8b', wPrec: 'int4', kvPrec: 'bf16', users: 50, activity: 10, ctx: 8192, retention: 'none' }).vm;
  check('Ollama with one slot and 50 chat users: 1 active, 9 thinking, 40 waiting', close(ol.crowd.active, 1) && close(ol.crowd.evicted, 9) && close(ol.crowd.waiting, 40) && ol.limit.key === 'slots', JSON.stringify(ol.crowd));
  check('the response time law: W = N·S/C − Z − S', close(ol.crowd.wait, 50 * ol.crowd.S / 1 - ol.crowd.S * 0.9 / 0.1 - ol.crowd.S, 1e-9));
  check('... and Little\'s law: waiting = throughput × wait', close(ol.crowd.waiting, (ol.crowd.C / ol.crowd.S) * ol.crowd.wait, 1e-9));
  const flood = view({ engine: 'ollama', hw: 'rtx-4090', count: 1, model: 'llama-3.1-8b', wPrec: 'int4', kvPrec: 'bf16', users: 2000, activity: 100, ctx: 8192, retention: 'none' }).vm;
  check('beyond OLLAMA_MAX_QUEUE the rest are rejected', close(flood.crowd.waiting, 512) && flood.crowd.rejected > 1000 && close(sumGroups(flood.crowd), 2000), JSON.stringify(flood.crowd));
  const resident = view({ retention: 'gpu', users: 500, activity: 10 }).vm;
  check('sessions kept in GPU memory: whoever does not fit has no session', resident.crowd.blocked > 0 && close(sumGroups(resident.crowd), 500) && /without a session/.test(resident.headline), resident.headline);
  const oom = view({ model: 'llama-3.1-405b', count: 8 }).vm;
  check('a model that does not load leaves everyone outside', oom.status === 'oom' && oom.crowd.blocked === 50 && oom.capacity.users === 0);
  const nope = view({ engine: 'trtllm', hw: 'mi300x', kvPrec: 'bf16' }).vm;
  check('an engine that does not run here says so', nope.status === 'unsupported' && nope.limit.key === 'unsupported' && /cannot run/.test(nope.headline));
  // conservation and continuity over a sweep
  let conserved = true, count = 0;
  for (const users of [1, 7, 50, 333, 5000, 100000]) for (const activity of [5, 25, 60, 100]) for (const retention of ['host', 'gpu', 'none']) for (const engine of ['vllm', 'ollama']) {
    const v = view({ users, activity, retention, engine, kvPrec: engine === 'ollama' ? 'int8' : 'fp8', wPrec: engine === 'ollama' ? 'int4' : 'fp8' }).vm;
    count++;
    if (!close(sumGroups(v.crowd), users, 1e-9)) conserved = false;
  }
  check(`every user is somewhere (${count} states)`, conserved);
  const base = view({ kvPrec: 'bf16' });
  const Cc = base.vm.crowd.C, a = 0.25;
  const below = view({ kvPrec: 'bf16', users: Math.floor(Cc / a) }).vm, above = view({ kvPrec: 'bf16', users: Math.floor(Cc / a) + 1 }).vm;
  check('no jump at the saturation point', below.crowd.waiting === 0 && above.crowd.waiting > 0 && above.crowd.waiting <= 1 + 1e-9, `${below.crowd.waiting} ${above.crowd.waiting}`);
}

console.log('Memory of one accelerator');
{
  const sum = (fr) => fr.weights + fr.overhead + fr.kvUsed + fr.kvReserved + fr.kvFree + fr.headroom;
  let ok = true;
  for (const o of [{}, { kvPrec: 'bf16' }, { engine: 'ollama', kvPrec: 'int8', slots: 2 }, { model: 'deepseek-v3', hw: 'h200-sxm', wPrec: 'fp8', dpAttn: true }, { model: 'gpt-oss-20b', hw: 'rtx-4090', count: 1, wPrec: 'fp4' }]) {
    const v = view(o).vm;
    if (v.status !== 'oom' && v.status !== 'unsupported' && !close(sum(v.gpu.fr), 1, 1e-9)) { ok = false; console.log('    ', JSON.stringify(o), sum(v.gpu.fr)); }
  }
  check('the memory bar adds up to the whole accelerator', ok);
  const big = view({ model: 'llama-3.3-70b', count: 1, par: 'manual', tp: 1, pp: 1, reps: 1 }).vm;
  check('70B in BF16 on one 80 GB GPU overflows', big.status === 'oom' && big.gpu.overflow > 0.7 && big.gpu.cells.total === 0, big.gpu.overflow);
  const ol = view({ engine: 'ollama', hw: 'rtx-4090', count: 1, model: 'llama-3.1-8b', wPrec: 'int4', kvPrec: 'bf16', slots: 4, users: 2, activity: 50, ctx: 8192, retention: 'none' }).vm;
  check('Ollama slots: four cells, one in use, the rest reserved but empty', ol.gpu.cells.slots && ol.gpu.cells.total === 4 && ol.gpu.cells.used === 1 && ol.gpu.fr.kvReserved > 0, JSON.stringify(ol.gpu.cells));
  const many = view({ model: 'llama-3.1-8b', count: 1, users: 5000, ctx: 4096 }).vm;
  check('at most 64 cells per tank', many.gpu.cells.total <= 64 && many.gpu.cells.perCell > 1, JSON.stringify(many.gpu.cells));
}

console.log('Capacity matches the planner');
{
  let ok = true;
  const states = [{}, { kvPrec: 'bf16' }, { users: 900, activity: 60 }, { engine: 'sglang', hw: 'mi300x', count: 16 }, { engine: 'ollama', kvPrec: 'int8', wPrec: 'int4' },
    { model: 'qwen3-235b-a22b', wPrec: 'fp8' }, { retention: 'gpu' }, { ctx: 8192, users: 5000 }, { spec: true }, { hw: 'b200', model: 'gpt-oss-120b', wPrec: 'fp4' }];
  for (const o of states) { const x = view(o); if (x.vm.capacity.users !== (x.r.fits ? x.r.maxUsers : 0)) { ok = false; console.log('    ', JSON.stringify(o)); } }
  check('capacity equals the planner\'s maxUsers for 10 states', ok);
}

console.log('Sprites');
{
  const caps = { desks: 80, lounge: 40, street: 60 };
  const a = M.allocateSprites({ active: 12.5, parked: 37.5 }, caps);
  check('small crowds are one sprite per user', a.per === 1 && a.counts.active + a.counts.parked === 50, JSON.stringify(a));
  const b = M.allocateSprites({ active: 1000, evicted: 9000, waiting: 3 }, caps);
  const total = M.GROUPS.reduce((s, k) => s + b.counts[k], 0);
  check('large crowds use a 1-2-5 step and stay within every zone', [1, 2, 5].includes(b.per / Math.pow(10, Math.floor(Math.log10(b.per)))) && total <= 150 && b.counts.active <= caps.desks && b.counts.evicted <= caps.lounge, JSON.stringify(b));
  check('three waiting users among 10,000 still show', b.counts.waiting >= 1);
  const c = M.allocateSprites({ active: 1e6 }, caps);
  check('a million users still fit the stage', c.counts.active >= 1 && c.counts.active <= caps.desks);
  check('niceStep rounds up to 1, 2, 5 × 10^k', M.niceStep(1) === 1 && M.niceStep(3) === 5 && M.niceStep(11) === 20 && M.niceStep(50) === 50 && M.niceStep(51) === 100);
}

console.log('Layout');
{
  let ok = true;
  // the world picks an integer scale so the canvas is 150 to 299 art pixels tall, at any width a 16:9 or 16:10 stage gives it
  for (const [w, h] of [[240, 150], [260, 150], [295, 175], [329, 170], [340, 190], [360, 230], [440, 235], [520, 299]]) {
    const L = M.layout(w, h);
    const inside = (r) => r.x >= 0 && r.y >= 0 && r.x + r.w <= w && r.y + r.h <= h;
    const zones = [L.server, L.office, L.street];
    const apart = zones.every((z, i) => zones.every((o, j) => i === j || z.x + z.w <= o.x || o.x + o.w <= z.x));
    const slotsIn = (zone, cell) => zone.slots.every((s) => s.x >= zone.area.x && s.y >= zone.area.y && s.x + cell.w <= zone.area.x + zone.area.w + 1e-9 && s.y + cell.h <= zone.area.y + zone.area.h + 1e-9);
    const enough = L.desks.slots.length >= 20 && L.lounge.slots.length >= 10 && L.queue.slots.length >= 12;
    if (!(zones.every(inside) && apart && slotsIn(L.desks, M.DESK) && slotsIn(L.lounge, M.STAND) && slotsIn(L.queue, M.STAND) && enough)) { ok = false; console.log('    ', w, h, L.desks.slots.length, L.lounge.slots.length, L.queue.slots.length); }
  }
  check('zones and person slots stay inside the canvas without overlapping (240×150 to 520×299)', ok);
}

console.log('Baseline and ticker');
{
  const bf = view({ kvPrec: 'bf16' }).vm, f8 = view({ kvPrec: 'fp8' }).vm;
  const same = M.compare(f8, f8);
  check('a view compared with itself has no deltas', Object.values(same).every((d) => d.same && (d.pct === null || d.pct === 0)));
  const d = M.compare(f8, bf);
  check('FP8 KV against a BF16 baseline: capacity +500%, marked good', close(d.capacity.pct, 5) && d.capacity.good === true && d.waiting.good === true);
  const t = M.ticker({ what: 'KV cache', from: 'BF16', to: 'FP8' }, bf, f8);
  check('ticker: "KV cache BF16 → FP8: capacity 12 → 72 users (+500%)"', t.text === 'KV cache BF16 → FP8: capacity 12 → 72 users (+500%)' && t.tone === 'good' && t.banner === 'Queue cleared!', JSON.stringify(t));
  const oom = view({ model: 'llama-3.1-405b' }).vm;
  const t2 = M.ticker({ what: 'Model', from: '70B', to: '405B' }, f8, oom);
  check('ticker: a model that no longer fits', t2.banner === 'Out of memory!' && t2.tone === 'bad' && /out of memory/.test(t2.text));
  const t3 = M.ticker({ what: 'Model', from: '405B', to: '70B' }, oom, f8);
  check('ticker: back in memory', t3.banner === 'It fits!' && t3.tone === 'good');
  const big = view({ kvPrec: 'fp8', users: 50, ctx: 16384 }).vm;
  const t4 = M.ticker({ what: 'Context', from: '128K', to: '16K' }, f8, big);
  check('ticker: a big capacity jump gets a banner', /^Capacity ×\d/.test(t4.banner), JSON.stringify(t4));
}

console.log('Formats');
check('counts', M.fmtCount(950) === '950' && M.fmtCount(1500) === '1.5K' && M.fmtCount(25000) === '25K' && M.fmtCount(2.5e6) === '2.5M');
check('durations', M.fmtDuration(0.264) === '264 ms' && M.fmtDuration(4.2) === '4.2 s' && M.fmtDuration(248) === '4 min' && M.fmtDuration(9000) === '2.5 h');
check('money', M.fmtMoney(23.2) === '$23' && M.fmtMoney(16936) === '$17K' && M.fmtMoney(4.5) === '$4.50' && M.fmtMoney(2.4e6) === '$2.4M');

console.log(`\n${pass} passed, ${fail} failed.`);
process.exit(fail ? 1 : 0);
