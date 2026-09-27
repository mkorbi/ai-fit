#!/usr/bin/env node
/* common.js tests - the state and parameter helpers shared by the planner and the arcade.
 *   node tests/common.js
 */
const path = require('path');
const { HARDWARE, MODELS, NETWORKS } = require(path.join(__dirname, '..', 'catalog.js'));
const C = require(path.join(__dirname, '..', 'common.js'));

let fail = 0, pass = 0;
function check(name, ok, detail) {
  if (ok) pass++; else fail++;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}${!ok && detail !== undefined ? '   (' + detail + ')' : ''}`);
}
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/* The planner's params() before it moved to common.js (app.js, git 9ae5fa9), kept verbatim as the reference. */
function legacyParams(state, overrides) {
  const hwById = (id) => HARDWARE.find((x) => x.id === id) || HARDWARE.find((x) => x.id === 'h100-sxm');
  const modelById = (id) => MODELS.find((x) => x.id === id) || MODELS.find((x) => x.id === 'llama-3.3-70b');
  const st = Object.assign({}, state, overrides || {});
  st.adv = Object.assign({}, state.adv, (overrides && overrides.adv) || {});
  const hw = hwById(st.hw), model = modelById(st.model);
  const net = NETWORKS.find((n) => n.id === st.net) || NETWORKS[0];
  return {
    hw, model, wPrec: st.wPrec, kvPrec: st.kvPrec, engine: st.engine || 'none',
    count: Math.max(1, st.count | 0), nodeGpus: Math.max(1, st.nodeGpus | 0),
    link: st.link === 'auto' ? hw.link : st.link, net, hostRamGB: Math.max(0, +st.hostRam || 0),
    dpAttention: !!st.dpAttn, allowCrossTp: !!st.allowCrossTp,
    tp: Math.max(1, st.tp | 0), pp: Math.max(1, st.pp | 0), replicas: Math.max(1, st.reps | 0),
    wl: {
      users: Math.max(1, st.users | 0), activity: Math.min(1, Math.max(0.01, (+st.activity || 1) / 100)),
      ctx: Math.max(256, st.ctx | 0), prefix: Math.max(0, st.prefix | 0), newPrompt: Math.max(1, st.newPrompt | 0),
      output: Math.max(1, st.output | 0), retention: st.retention, target: Math.max(0.1, +st.target || 1), ttftMax: Math.max(0.1, +st.ttftMax || 30),
    },
    opt: { prefixCache: !!st.prefixCache, spec: !!st.spec, pd: !!st.pd },
    adv: {
      util: st.adv.util / 100, overheadGB: +st.adv.overheadGB, overheadFrac: st.adv.overheadFrac / 100, bwEff: st.adv.bwEff / 100,
      mfu: st.adv.mfu / 100, frag: st.adv.frag / 100, ppBubble: st.adv.ppBubble / 100, collEff: st.adv.collEff / 100,
      hostRestoreGBs: +st.adv.hostRestoreGBs, specK: Math.max(1, st.specK | 0), specAlpha: Math.min(0.99, Math.max(0.05, st.specAlpha / 100)), engine: st.engine || 'none',
    },
  };
}

console.log('Engine parameters');
const variants = [
  {},
  { hw: 'mi300x', count: 16, engine: 'sglang', wPrec: 'fp8', kvPrec: 'bf16', retention: 'gpu', users: 900, activity: 60 },
  { model: 'deepseek-v3', dpAttn: true, par: 'manual', tp: 8, pp: 2, reps: 3, spec: true, specK: 3, specAlpha: 55 },
  { hw: 'nope', model: 'missing', ctx: 100, users: 0, activity: 0, target: 0, ttftMax: 0, net: 'none' },
  { adv: Object.assign({}, C.DEFAULTS.adv, { util: 80, mfu: 40 }), link: 'pcie5', hostRam: 0, pd: true, prefixCache: false },
];
for (const v of variants) {
  const st = C.freshState(v);
  const a = C.buildParams(st, { users: 77 }), b = legacyParams(st, { users: 77 });
  const slots = a.slots; delete a.slots;
  check(`buildParams matches the planner's params() for ${JSON.stringify(v).slice(0, 60)}`, same(a, b) && slots === 0);
}
check('slots travel into the engine parameters', C.buildParams(C.freshState({ slots: 4 })).slots === 4);

console.log('State');
{
  const st = C.freshState({});
  st.adv.util = 50; st.candidates.push('x');
  check('fresh states never share nested objects with the defaults', C.DEFAULTS.adv.util === 90 && !C.DEFAULTS.candidates.includes('x'));
  const clean = C.sanitizeState({ hw: 'nope', model: 'llama-3.1-8b', users: -5, ctx: 1e12, engine: 'ollama', foo: 1, link: 'nvlink', wPrec: 'fp9', activity: NaN, adv: { util: 200, x: 1 }, candidates: ['b200', 'nope'] });
  check('the sanitizer drops unknown keys and ids and clamps numbers', same(clean, { model: 'llama-3.1-8b', users: 1, ctx: 1 << 26, engine: 'ollama', link: 'nvlink', adv: { util: 100 }, candidates: ['b200'] }), JSON.stringify(clean));
  const s1 = C.freshState({});
  s1.hw = 'gb200';
  check('changing the accelerator adopts its node size and fabric', same(C.applyChange(s1, 'hw', 'gb200'), ['nodeGpus']) && s1.nodeGpus === 72 && s1.link === 'auto');
  s1.model = 'deepseek-v3';
  check('an MLA model turns data-parallel attention on', same(C.applyChange(s1, 'model', 'deepseek-v3'), ['dpAttn']) && s1.dpAttn === true);
  s1.activityPreset = 'agents';
  C.applyChange(s1, 'activityPreset', 'agents');
  check('a usage pattern sets the activity', s1.activity === 60);
  s1.activity = 33;
  C.applyChange(s1, 'activity', 33);
  check('a hand-typed activity becomes a custom pattern', s1.activityPreset === 'custom');
  const s2 = C.freshState({ engine: 'ollama', wPrec: 'fp8', kvPrec: 'fp8' });
  const moved = C.applyChange(s2, 'engine', 'ollama', { snapFormats: true });
  check('switching to Ollama snaps FP8 to the nearest GGUF formats', same(moved, ['wPrec', 'kvPrec']) && s2.wPrec === 'int8' && s2.kvPrec === 'int8', `${s2.wPrec} ${s2.kvPrec}`);
  const s3 = C.freshState({});
  check('the planner does not snap formats', same(C.applyChange(s3, 'engine', 'ollama'), []) && s3.kvPrec === 'fp8');
}

console.log('Links between the planner and the arcade');
{
  const st = C.freshState({ hw: 'mi300x', users: 500, engine: 'ollama', slots: 4, adv: Object.assign({}, C.DEFAULTS.adv, { util: 85 }) });
  const enc = C.encodePlan(st);
  check('only changed keys travel', same(C.decodePlan('#' + enc), { hw: 'mi300x', engine: 'ollama', slots: 4, users: 500, adv: { util: 85 } }), JSON.stringify(C.decodePlan('#' + enc)));
  check('the link is URL-safe', /^plan=[A-Za-z0-9_-]+$/.test(enc) && enc.length < 200, enc);
  const back = C.freshState(Object.assign({}, C.decodePlan('#' + enc)));
  check('a decoded link rebuilds the same state', same(C.buildParams(back), C.buildParams(st)));
  const custom = { id: 'x-uni', name: 'Körbächer GPU 96 GB ✓', vendor: 'Custom', mem: 96, bw: 3000, tflops: { fp16: 500, fp8: 1000 } };
  C.store.addCustom('hardware', [custom]);
  const withCustom = C.encodePlan(C.freshState({ hw: 'x-uni' }));
  C.store.resetCustom();
  const dec = C.decodePlan('#' + withCustom);
  check('a custom accelerator travels with its definition, Unicode intact', dec && dec.hw === 'x-uni' && C.store.hwById('x-uni').name === custom.name, dec && JSON.stringify(dec));
  C.store.resetCustom();
  check('garbage links are ignored', C.decodePlan('#plan=%%%') === null && C.decodePlan('#plan=eyJ2IjoyfQ') === null && C.decodePlan('') === null);
}

console.log('Sliders');
check('the users slider walks a 1-2-5 ladder from 1 to 1M', C.USER_STEPS[0] === 1 && C.USER_STEPS[C.USER_STEPS.length - 1] === 1e6 && C.sliderToUsers(C.usersToSlider(500)) === 500);
check('the context slider round-trips 128k', C.sliderToCtx(C.ctxToSlider(131072)) === 131072);

console.log(`\n${pass} passed, ${fail} failed.`);
process.exit(fail ? 1 : 0);
