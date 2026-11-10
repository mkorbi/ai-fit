#!/usr/bin/env node
/* ep.js — expert parallelism and data-parallel attention behave as the Method view describes.
 *   node tests/ep.js
 */
const path = require('path');
const { HARDWARE, MODELS, NETWORKS } = require(path.join(__dirname, '..', 'catalog.js'));
const E = require(path.join(__dirname, '..', 'engine.js'));
const hw = (id) => HARDWARE.find((x) => x.id === id), md = (id) => MODELS.find((x) => x.id === id);
const net = NETWORKS.find((n) => n.id === 'ib3200');

const base = (o) => {
  const h = hw(o.hw || 'h100-sxm');
  return Object.assign({ hw: h, model: md('deepseek-v3'), wPrec: 'fp8', kvPrec: 'fp8', engine: 'none', count: 8, nodeGpus: h.nodeGpus, link: h.link, net, hostRamGB: 2048,
    dpAttention: false, ep: 'off', allowCrossTp: false, tp: 8, pp: 1, replicas: 1,
    wl: { users: 500, activity: 0.25, peakZ: 0, ctx: 32768, prefix: 0, newPrompt: 1000, output: 1000, retention: 'host', target: 20, ttftMax: 1e9 },
    opt: { prefixCache: false, spec: false, pd: false }, adv: {} }, o, { hw: h, model: md(o.model || 'deepseek-v3') });
};

let fail = 0;
const check = (name, ok, detail) => { if (!ok) fail++; console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? ': ' + detail : ''}`); };

// 1. Dense models ignore DP attention and EP entirely.
{
  const a = E.evaluate(base({ model: 'llama-3.3-70b', hw: 'h100-sxm' }));
  const b = E.evaluate(base({ model: 'llama-3.3-70b', hw: 'h100-sxm', dpAttention: true, ep: 'on' }));
  check('dense model unchanged by DP attention / EP', a.kvAvail === b.kvAvail && a.at.step === b.at.step && !b.ep && !b.dpa);
}
// 2. DP attention: no KV replication, but each rank holds the dense weights of its stage.
{
  const tp = E.evaluate(base({ hw: 'h200-sxm' })), dpa = E.evaluate(base({ hw: 'h200-sxm', dpAttention: true }));
  check('DP attention removes MLA KV replication', tp.kvRepl === 8 && dpa.kvRepl === 1, `TP ${tp.kvRepl}×, DP ${dpa.kvRepl}×`);
  const extra = dpa.wPerGpu - tp.wPerGpu, expect = dpa.denseBytes * 7 / 8;
  check('DP attention copies the dense weights to every rank', Math.abs(extra - expect) < 1e6, `${(extra / 1e9).toFixed(1)} GB extra per GPU`);
}
// 3. EP over 8 ranks: imbalance makes expert reads and compute heavier than TP slices, never lighter.
{
  const dp = E.evaluate(base({ hw: 'h200-sxm', dpAttention: true })), ep = E.evaluate(base({ hw: 'h200-sxm', dpAttention: true, ep: 'on' }));
  check('EP keeps weight memory per rank', Math.abs(dp.wPerGpu - ep.wPerGpu) < 1);
  check('EP imbalance never speeds up the decode roofline', ep.at.tBw >= dp.at.tBw && ep.at.tComp >= dp.at.tComp);
  const flat = E.evaluate(base({ hw: 'h200-sxm', dpAttention: true, ep: 'on', adv: { epImbalance: 1 } }));
  check('with perfect balance EP reads no more than TP slices', flat.at.tBw <= dp.at.tBw + 1e-12);
}
// 4. Wide EP: allowed across nodes with DP attention, not as plain TP; and it beats replicas for a big MoE model.
{
  check('EP 32 across nodes allowed with DP attention', E.groupOK(md('deepseek-v3'), 32, 8, false, true, true));
  check('TP 32 across nodes refused without permission', !E.groupOK(md('deepseek-v3'), 32, 8, false, false, false));
  check('EP must divide the expert count', !E.groupOK(md('llama-4-scout'), 32, 8, false, true, true) && E.groupOK(md('llama-4-scout'), 16, 8, false, true, true));
  const p = base({ count: 32, dpAttention: true, ep: 'auto' });
  const wide = E.autoConfig(p).best, narrow = E.autoConfig(Object.assign({}, p, { ep: 'off' })).best;
  check('auto picks wide EP for DeepSeek-V3 on 32× H100', wide.ep && wide.tp > 8, `EP ${wide.tp}: ${wide.maxUsers} users vs ${narrow.maxUsers} without EP`);
  check('wide EP serves more users than replicas of TP 8 × PP 2', wide.maxUsers > narrow.maxUsers);
  const slow = E.evaluate(Object.assign({}, p, { tp: 32, ep: 'on', net: NETWORKS.find((n) => n.id === 'eth25') }));
  check('a slow network makes the cross-node all-to-all expensive', slow.at.tComm > 10 * wide.at.tComm, `${(slow.at.tComm * 1e3).toFixed(0)} ms vs ${(wide.at.tComm * 1e3).toFixed(1)} ms`);
}
// 5. DP attention prefill runs a request's attention on one rank: slower first token than TP.
{
  const tp = E.evaluate(base({ hw: 'h200-sxm' })), dpa = E.evaluate(base({ hw: 'h200-sxm', dpAttention: true }));
  check('DP attention lengthens single-request prefill', dpa.ttftCold > tp.ttftCold, `${tp.ttftCold.toFixed(2)} s → ${dpa.ttftCold.toFixed(2)} s`);
}

console.log(`\n${fail} problem${fail === 1 ? '' : 's'}.`);
process.exit(fail ? 1 : 0);
