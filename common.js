/* common.js - shared by the planner (app.js) and the arcade (arcade/arcade.js).
 * DOM builders, number formats, safe storage, the catalogs including custom entries, the planner state (defaults,
 * sanitizing, the side effects of changing one input), the engine parameters built from a state, and the #plan= link
 * that carries a state between the two pages. Works in node too (no DOM access at load time).
 */
const Common = (() => {
  const E = typeof Engine !== 'undefined' ? Engine : require('./engine.js');
  const CAT = typeof HARDWARE !== 'undefined' ? { HARDWARE, MODELS, NETWORKS, ENGINES, LINKS } : require('./catalog.js');

  /* ---------- tiny DOM helpers ---------- */
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat(Infinity)) { if (kid == null || kid === false) continue; el.append(kid.nodeType ? kid : document.createTextNode(String(kid))); }
    return el;
  }
  const SVG = 'http://www.w3.org/2000/svg';
  function s(tag, attrs, ...kids) {
    const el = document.createElementNS(SVG, tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) { if (v != null) el.setAttribute(k, v); }
    for (const kid of kids.flat(Infinity)) { if (kid == null) continue; el.append(kid.nodeType ? kid : document.createTextNode(String(kid))); }
    return el;
  }

  /* ---------- number formats ---------- */
  const fmtNum = (n, d = 0) => (n == null || !isFinite(n)) ? '-' : n.toLocaleString('en-US', { maximumFractionDigits: d });
  const fmtMoney = (n, d = 2) => (n == null || !isFinite(n)) ? '-' : '$' + n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  const fmtTokS = (n) => (n == null || !isFinite(n)) ? '-' : (n >= 100 ? fmtNum(n) : n.toFixed(1));
  const pct = (x) => Math.round(x * 100) + '%';
  const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;

  /* ---------- storage that never throws (private mode, blocked storage, node) ---------- */
  const LS = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private mode etc. */ } },
    del(k) { try { localStorage.removeItem(k); } catch (e) { /* ignore */ } },
  };

  /* ---------- catalogs plus the custom entries kept in this browser ---------- */
  const store = (() => {
    let customHw = LS.get('cb.customHardware', []), customModels = LS.get('cb.customModels', []);
    const hardware = () => CAT.HARDWARE.concat(customHw);
    const models = () => CAT.MODELS.concat(customModels);
    return {
      hardware, models,
      hwById: (id) => hardware().find((x) => x.id === id) || CAT.HARDWARE.find((x) => x.id === 'h100-sxm'),
      modelById: (id) => models().find((x) => x.id === id) || CAT.MODELS.find((x) => x.id === 'llama-3.3-70b'),
      hasHw: (id) => hardware().some((x) => x.id === id),
      hasModel: (id) => models().some((x) => x.id === id),
      isCustom: (kind, id) => (kind === 'hardware' ? customHw : customModels).some((x) => x.id === id),
      /* Adds or replaces entries (same id); returns an error text or null. */
      addCustom(kind, list) {
        const need = kind === 'hardware' ? ['id', 'name', 'mem', 'bw'] : ['id', 'name', 'params', 'layers', 'dModel', 'nHeads', 'nKv', 'dHead', 'maxCtx'];
        for (const o of list) {
          const missing = need.filter((k) => o[k] == null);
          if (missing.length) return `Entry "${o.id || '?'}" is missing: ${missing.join(', ')}`;
          if (kind === 'hardware') { o.vendor = o.vendor || 'Custom'; o.tflops = o.tflops || { fp16: 100 }; o.link = o.link || 'pcie5'; o.linkBw = o.linkBw || 64; o.nodeGpus = o.nodeGpus || 8; o.custom = true; }
          else { o.family = o.family || 'Custom'; o.custom = true; }
        }
        if (kind === 'hardware') { customHw = customHw.filter((x) => !list.some((o) => o.id === x.id)).concat(list); LS.set('cb.customHardware', customHw); }
        else { customModels = customModels.filter((x) => !list.some((o) => o.id === x.id)).concat(list); LS.set('cb.customModels', customModels); }
        return null;
      },
      resetCustom() { customHw = []; customModels = []; LS.del('cb.customHardware'); LS.del('cb.customModels'); },
    };
  })();

  /* ---------- the planner state ---------- */
  const ACTIVITY = { chat: 10, assist: 25, agents: 60, batch: 100 };
  const DEFAULTS = {
    mode: 'forward', hw: 'h100-sxm', count: 8, nodeGpus: 8, link: 'auto', net: 'ib3200', hostRam: 1024,
    model: 'llama-3.3-70b', wPrec: 'bf16', kvPrec: 'fp8', engine: 'vllm', slots: 0, par: 'auto', tp: 8, pp: 1, reps: 1, dpAttn: false,
    users: 50, activityPreset: 'assist', activity: 25, ctx: 131072, prefix: 2048, newPrompt: 1000, output: 500, retention: 'host', target: 20, ttftMax: 30,
    prefixCache: true, spec: false, specK: 4, specAlpha: 70, pd: false, allowCrossTp: false,
    candidates: ['h100-sxm', 'h200-sxm', 'b200', 'b300', 'gb200', 'mi300x', 'mi325x', 'mi355x', 'l40s', 'rtx-pro-6000', 'a100-sxm-80', 'rtx-4090'],
    compatCtx: 32768,
    adv: { util: 90, overheadGB: 1.5, overheadFrac: 4, bwEff: 75, mfu: 50, frag: 4, ppBubble: 10, collEff: 70, hostRestoreGBs: 20 },
  };
  /* A full state from a saved or partial one; never shares the nested objects with DEFAULTS. */
  function freshState(saved) {
    const st = Object.assign({}, DEFAULTS, saved || {});
    st.adv = Object.assign({}, DEFAULTS.adv, (saved && saved.adv) || {});
    st.candidates = (saved && Array.isArray(saved.candidates) ? saved.candidates : DEFAULTS.candidates).slice();
    return st;
  }
  /* Typed whitelist for state that comes from outside (a link, a scene file, older storage): unknown keys are dropped,
   * numbers must be finite and are clamped, ids must exist in the catalogs. Returns a partial state. */
  const LIMITS = {
    count: [1, 100000], nodeGpus: [1, 1024], hostRam: [0, 1e6], slots: [0, 4096], tp: [1, 64], pp: [1, 16], reps: [1, 100000],
    users: [1, 1e7], activity: [1, 100], ctx: [256, 1 << 26], prefix: [0, 1 << 26], newPrompt: [1, 1 << 26], output: [1, 1 << 20],
    target: [0.1, 10000], ttftMax: [0.1, 1e6], specK: [1, 16], specAlpha: [5, 99], compatCtx: [1024, 1 << 26],
  };
  const ENUMS = {
    mode: ['forward', 'reverse', 'compat', 'catalog', 'method'], par: ['auto', 'manual'], retention: ['host', 'gpu', 'none'],
    activityPreset: ['chat', 'assist', 'agents', 'batch', 'custom'], wPrec: E.W_PRECS, kvPrec: E.KV_PRECS,
  };
  const ADV_LIMITS = { util: [50, 100], overheadGB: [0, 64], overheadFrac: [0, 100], bwEff: [10, 100], mfu: [10, 100], frag: [0, 90], ppBubble: [0, 100], collEff: [10, 100], hostRestoreGBs: [1, 1000] };
  const clamp = (v, [lo, hi]) => Math.min(hi, Math.max(lo, v));
  function sanitizeState(obj) {
    const out = {};
    if (!obj || typeof obj !== 'object') return out;
    for (const [k, v] of Object.entries(obj)) {
      if (!(k in DEFAULTS)) continue;
      const d = DEFAULTS[k];
      if (k === 'adv') {
        if (v && typeof v === 'object') { out.adv = {}; for (const [a, x] of Object.entries(v)) if (ADV_LIMITS[a] && typeof x === 'number' && isFinite(x)) out.adv[a] = clamp(x, ADV_LIMITS[a]); }
      } else if (k === 'candidates') {
        if (Array.isArray(v)) out.candidates = v.filter((id) => typeof id === 'string' && store.hasHw(id));
      } else if (k === 'hw') { if (typeof v === 'string' && store.hasHw(v)) out.hw = v; }
      else if (k === 'model') { if (typeof v === 'string' && store.hasModel(v)) out.model = v; }
      else if (k === 'engine') { if (typeof v === 'string' && (v === 'none' || CAT.ENGINES[v])) out.engine = v; }
      else if (k === 'net') { if (CAT.NETWORKS.some((n) => n.id === v)) out.net = v; }
      else if (k === 'link') { if (v === 'auto' || (typeof v === 'string' && CAT.LINKS[v])) out.link = v; }
      else if (ENUMS[k]) { if (ENUMS[k].includes(v)) out[k] = v; }
      else if (typeof d === 'boolean') { if (typeof v === 'boolean') out[k] = v; }
      else if (typeof d === 'number') { if (typeof v === 'number' && isFinite(v)) out[k] = LIMITS[k] ? clamp(v, LIMITS[k]) : v; }
    }
    for (const k of ['count', 'nodeGpus', 'slots', 'tp', 'pp', 'reps', 'users', 'ctx', 'prefix', 'newPrompt', 'output', 'specK', 'compatCtx']) if (k in out) out[k] = Math.round(out[k]);
    return out;
  }

  /* Which weight or KV format an engine switch lands on when the current one does not load: the nearest in bytes. */
  function nearestFormat(list, current, supported, bytes) {
    if (supported(current)) return current;
    const ok = list.filter(supported);
    if (!ok.length) return current;
    return ok.sort((a, b) => Math.abs(bytes(a) - bytes(current)) - Math.abs(bytes(b) - bytes(current)) || bytes(b) - bytes(a))[0];
  }
  /* The side effects of changing one input, shared by both pages. Mutates `state`; returns the other keys it changed.
   * `snapFormats` (the arcade) moves weight and KV formats the new engine or hardware cannot load to the nearest one. */
  function applyChange(state, key, value, opts = {}) {
    const changed = [];
    const set = (k, v) => { if (state[k] !== v) { state[k] = v; changed.push(k); } };
    if (key === 'hw') { const hw = store.hwById(value); set('nodeGpus', hw.nodeGpus); set('link', 'auto'); }
    if (key === 'model') set('dpAttn', E.isMla(E.norm(store.modelById(value))));
    if (key === 'activityPreset' && ACTIVITY[value] != null) set('activity', ACTIVITY[value]);
    if (key === 'activity') set('activityPreset', Object.keys(ACTIVITY).find((k) => ACTIVITY[k] === value) || 'custom');
    if (opts.snapFormats && ['engine', 'hw'].includes(key)) {
      const hw = store.hwById(state.hw), eng = state.engine;
      set('wPrec', nearestFormat(E.W_PRECS, state.wPrec, (f) => E.formatSupport(hw, f, eng) !== 'unsupported', (f) => E.bytesW(f, eng) ?? E.BYTES_W[f]));
      set('kvPrec', nearestFormat(E.KV_PRECS, state.kvPrec, (f) => E.kvSupport(hw, f, eng) === 'supported', (f) => E.bytesKv(f, eng)));
    }
    return changed;
  }

  /* ---------- engine parameters from a state (percent units become fractions) ---------- */
  function buildParams(state, overrides) {
    const st = Object.assign({}, state, overrides || {});
    st.adv = Object.assign({}, state.adv, (overrides && overrides.adv) || {});
    const hw = store.hwById(st.hw), model = store.modelById(st.model);
    const net = CAT.NETWORKS.find((n) => n.id === st.net) || CAT.NETWORKS[0];
    return {
      hw, model, wPrec: st.wPrec, kvPrec: st.kvPrec, engine: st.engine || 'none', slots: Math.max(0, st.slots | 0),
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
  /* The planner's evaluation: the layout set by hand, or the best one for the workload. */
  function runPlan(state, overrides) {
    const p = buildParams(state, overrides);
    const st = Object.assign({}, state, overrides || {});
    if (st.par === 'manual') return { best: E.evaluate(p), tried: [] };
    return E.autoConfig(p);
  }

  /* ---------- logarithmic sliders ---------- */
  const ctxToSlider = (c) => Math.round(Math.log2(Math.max(1024, c) / 1024) * 100);
  const sliderToCtx = (pos) => { const c = 1024 * Math.pow(2, pos / 100); const step = c >= 1e6 ? 65536 : c >= 131072 ? 16384 : c >= 32768 ? 4096 : 1024; return Math.max(1024, Math.round(c / step) * step); };
  const USER_STEPS = (() => { const out = []; for (let e = 0; e <= 6; e++) for (const m of [1, 2, 5]) if (m * 10 ** e <= 1e6) out.push(m * 10 ** e); return out; })();
  /* users on a 1-2-5 ladder from 1 to 1M; the slider position is the index */
  const usersToSlider = (u) => USER_STEPS.reduce((best, v, i) => (Math.abs(Math.log(v / u)) < Math.abs(Math.log(USER_STEPS[best] / u)) ? i : best), 0);
  const sliderToUsers = (i) => USER_STEPS[Math.min(USER_STEPS.length - 1, Math.max(0, Math.round(i)))];

  /* ---------- #plan= links between the planner and the arcade ---------- */
  const NOT_LINKED = ['mode', 'candidates', 'compatCtx'];
  function b64urlEncode(str) {
    const bytes = typeof TextEncoder !== 'undefined' ? new TextEncoder().encode(str) : Buffer.from(str, 'utf8');
    let bin = ''; for (const b of bytes) bin += String.fromCharCode(b);
    const b64 = typeof btoa !== 'undefined' ? btoa(bin) : Buffer.from(bin, 'binary').toString('base64');
    return b64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }
  function b64urlDecode(s) {
    const b64 = s.replace(/-/g, '+').replace(/_/g, '/') + '==='.slice((s.length + 3) % 4);
    const bin = typeof atob !== 'undefined' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
    const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
    return typeof TextDecoder !== 'undefined' ? new TextDecoder().decode(bytes) : Buffer.from(bytes).toString('utf8');
  }
  /* Only what differs from DEFAULTS travels; a custom accelerator or model travels with its definition. */
  function encodePlan(state) {
    const s = {};
    for (const [k, v] of Object.entries(state)) {
      if (NOT_LINKED.includes(k) || !(k in DEFAULTS)) continue;
      if (k === 'adv') { const a = {}; for (const [x, y] of Object.entries(v || {})) if (DEFAULTS.adv[x] !== y) a[x] = y; if (Object.keys(a).length) s.adv = a; continue; }
      if (DEFAULTS[k] !== v) s[k] = v;
    }
    const doc = { v: 1, s };
    if (store.isCustom('hardware', state.hw)) doc.hw = store.hwById(state.hw);
    if (store.isCustom('models', state.model)) doc.model = store.modelById(state.model);
    return 'plan=' + b64urlEncode(JSON.stringify(doc));
  }
  /* The partial state carried by a location hash, or null. Custom entries in it are added to this browser's catalog. */
  function decodePlan(hash) {
    const m = /(?:^#?|&)plan=([A-Za-z0-9_-]+)/.exec(hash || '');
    if (!m) return null;
    let doc;
    try { doc = JSON.parse(b64urlDecode(m[1])); } catch (e) { return null; }
    if (!doc || doc.v !== 1 || typeof doc.s !== 'object') return null;
    if (doc.hw && typeof doc.hw === 'object' && doc.hw.id && !store.hasHw(doc.hw.id)) store.addCustom('hardware', [doc.hw]);
    if (doc.model && typeof doc.model === 'object' && doc.model.id && !store.hasModel(doc.model.id)) store.addCustom('models', [doc.model]);
    return sanitizeState(doc.s);
  }

  return {
    h, s, fmtNum, fmtMoney, fmtTokS, pct, plural, LS, store,
    ACTIVITY, DEFAULTS, freshState, sanitizeState, applyChange, buildParams, runPlan,
    ctxToSlider, sliderToCtx, USER_STEPS, usersToSlider, sliderToUsers, encodePlan, decodePlan,
  };
})();

if (typeof module !== 'undefined') module.exports = Common;
