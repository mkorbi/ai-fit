/* app.js — UI for AI Fit. Depends on catalog.js and engine.js (classic scripts, loaded before this one). */
(() => {
  'use strict';
  const E = Engine;
  const { fmtTok, fmtGB, fmtTime } = E;
  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));

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
  const fmtNum = (n, d = 0) => (n == null || !isFinite(n)) ? '—' : n.toLocaleString('en-US', { maximumFractionDigits: d });
  const fmtMoney = (n, d = 2) => (n == null || !isFinite(n)) ? '—' : '$' + n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  const fmtTokS = (n) => (n == null || !isFinite(n)) ? '—' : (n >= 100 ? fmtNum(n) : n.toFixed(1));
  const pct = (x) => Math.round(x * 100) + '%';
  const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;

  /* ---------- persistence and catalogs ---------- */
  const LS = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private mode etc. */ } },
    del(k) { try { localStorage.removeItem(k); } catch (e) { /* ignore */ } },
  };
  let customHw = LS.get('cb.customHardware', []);
  let customModels = LS.get('cb.customModels', []);
  const allHardware = () => HARDWARE.concat(customHw);
  const allModels = () => MODELS.concat(customModels);
  const hwById = (id) => allHardware().find((x) => x.id === id) || HARDWARE.find((x) => x.id === 'h100-sxm');
  const modelById = (id) => allModels().find((x) => x.id === id) || MODELS.find((x) => x.id === 'llama-3.3-70b');

  const ACTIVITY = { chat: 10, assist: 25, agents: 60, batch: 100 };
  const DEFAULTS = {
    mode: 'forward', hw: 'h100-sxm', count: 8, nodeGpus: 8, link: 'auto', net: 'ib3200', hostRam: 1024,
    model: 'llama-3.3-70b', wPrec: 'bf16', kvPrec: 'fp8', engine: 'vllm', par: 'auto', tp: 8, pp: 1, reps: 1, dpAttn: false,
    users: 50, activityPreset: 'assist', activity: 25, ctx: 131072, prefix: 2048, newPrompt: 1000, output: 500, retention: 'host', target: 20, ttftMax: 30,
    prefixCache: true, spec: false, specK: 4, specAlpha: 70, pd: false, allowCrossTp: false,
    candidates: ['h100-sxm', 'h200-sxm', 'b200', 'b300', 'gb200', 'mi300x', 'mi325x', 'mi355x', 'l40s', 'rtx-pro-6000', 'a100-sxm-80', 'rtx-4090'],
    compatCtx: 32768,
    adv: { util: 90, overheadGB: 1.5, overheadFrac: 4, bwEff: 75, mfu: 50, frag: 4, ppBubble: 10, collEff: 70, hostRestoreGBs: 20 },
  };
  let state = Object.assign({}, DEFAULTS, LS.get('cb.state', {}));
  state.adv = Object.assign({}, DEFAULTS.adv, state.adv || {});

  /* ---------- parameters for the engine ---------- */
  function params(overrides) {
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
  function run(overrides) {
    const p = params(overrides);
    const st = Object.assign({}, state, overrides || {});
    if (st.par === 'manual') return { best: E.evaluate(p), tried: [] };
    return E.autoConfig(p);
  }

  /* ---------- controls ---------- */
  const BIND = [
    ['hw', 'hw', 'str'], ['count', 'count', 'int'], ['nodeGpus', 'nodeGpus', 'int'], ['link', 'link', 'str'], ['net', 'net', 'str'], ['hostRam', 'hostRam', 'num'],
    ['model', 'model', 'str'], ['engine', 'engine', 'str'], ['wPrec', 'wPrec', 'str'], ['kvPrec', 'kvPrec', 'str'], ['par', 'par', 'str'], ['tp', 'tp', 'int'], ['pp', 'pp', 'int'], ['reps', 'reps', 'int'], ['dpAttn', 'dpAttn', 'bool'],
    ['users', 'users', 'int'], ['activityPreset', 'activityPreset', 'str'], ['activity', 'activity', 'num'], ['ctx', 'ctx', 'int'], ['prefix', 'prefix', 'int'], ['newPrompt', 'newPrompt', 'int'], ['output', 'output', 'int'], ['retention', 'retention', 'str'], ['target', 'target', 'num'], ['ttftMax', 'ttftMax', 'num'],
    ['prefixCache', 'prefixCache', 'bool'], ['spec', 'spec', 'bool'], ['specK', 'specK', 'int'], ['specAlpha', 'specAlpha', 'num'], ['pd', 'pd', 'bool'], ['allowCrossTp', 'allowCrossTp', 'bool'],
    ['compatCtx', 'compatCtx', 'int'],
    ['advUtil', 'adv.util', 'num'], ['advOverheadGB', 'adv.overheadGB', 'num'], ['advOverheadFrac', 'adv.overheadFrac', 'num'], ['advBwEff', 'adv.bwEff', 'num'], ['advMfu', 'adv.mfu', 'num'],
    ['advFrag', 'adv.frag', 'num'], ['advPpBubble', 'adv.ppBubble', 'num'], ['advCollEff', 'adv.collEff', 'num'], ['advHostRestore', 'adv.hostRestoreGBs', 'num'],
  ];
  const getPath = (obj, path) => path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
  const setPath = (obj, path, v) => { const ks = path.split('.'); let o = obj; for (const k of ks.slice(0, -1)) { if (o[k] == null) o[k] = {}; o = o[k]; } o[ks[ks.length - 1]] = v; };
  const ctxToSlider = (c) => Math.round(Math.log2(Math.max(1024, c) / 1024) * 100);
  const sliderToCtx = (pos) => { const c = 1024 * Math.pow(2, pos / 100); const step = c >= 1e6 ? 65536 : c >= 131072 ? 16384 : c >= 32768 ? 4096 : 1024; return Math.max(1024, Math.round(c / step) * step); };

  function populateSelects() {
    const hwSel = $('#hw'); hwSel.replaceChildren();
    for (const v of [...new Set(allHardware().map((x) => x.vendor))]) {
      const og = h('optgroup', { label: v });
      for (const hw of allHardware().filter((x) => x.vendor === v)) og.append(h('option', { value: hw.id }, `${hw.name}${hw.approx ? ' ~' : ''}`));
      hwSel.append(og);
    }
    const mSel = $('#model'); mSel.replaceChildren();
    for (const f of [...new Set(allModels().map((x) => x.family))]) {
      const og = h('optgroup', { label: f });
      for (const m of allModels().filter((x) => x.family === f)) og.append(h('option', { value: m.id }, `${m.name}${m.approx ? ' ~' : ''}`));
      mSel.append(og);
    }
    const linkSel = $('#link'); linkSel.replaceChildren(h('option', { value: 'auto' }, 'Auto from catalog'));
    for (const [k, l] of Object.entries(LINKS)) if (k !== 'none') linkSel.append(h('option', { value: k }, l.name));
    const netSel = $('#net'); netSel.replaceChildren();
    for (const n of NETWORKS) netSel.append(h('option', { value: n.id }, n.name));
    const wSel = $('#wPrec'); wSel.replaceChildren(); for (const k of E.W_PRECS) wSel.append(h('option', { value: k }, E.PREC_LABEL[k]));
    const eSel = $('#engine'); eSel.replaceChildren(); for (const [k, e] of Object.entries(ENGINES)) eSel.append(h('option', { value: k }, e.name));
    labelPrecOptions(null);
    const kSel = $('#kvPrec'); kSel.replaceChildren(); for (const k of E.KV_PRECS) kSel.append(h('option', { value: k }, E.KV_LABEL[k]));
    const cand = $('#candidates'); cand.replaceChildren();
    for (const v of [...new Set(allHardware().map((x) => x.vendor))]) {
      cand.append(h('div', { class: 'cand-vendor' }, v));
      for (const hw of allHardware().filter((x) => x.vendor === v)) {
        const id = 'cand-' + hw.id;
        cand.append(h('label', { class: 'check', for: id }, h('input', { type: 'checkbox', id, 'data-id': hw.id, checked: state.candidates.includes(hw.id) }), h('span', null, hw.name)));
      }
    }
  }
  const SUPPORT_TXT = { native: 'native', 'weight-only': 'weight-only, compute in BF16', unsupported: 'not loadable' };
  function labelPrecOptions(hw) {
    const eng = state.engine || 'none';
    for (const opt of $$('#wPrec option')) opt.textContent = hw ? `${E.PREC_LABEL[opt.value]} — ${SUPPORT_TXT[E.formatSupport(hw, opt.value, eng)]}` : E.PREC_LABEL[opt.value];
    for (const opt of $$('#kvPrec option')) opt.textContent = hw ? `${E.KV_LABEL[opt.value]} — ${E.kvSupport(hw, opt.value, eng) === 'supported' ? 'supported' : 'not in ' + ENGINES[eng].name}` : (eng !== 'none' && !(ENGINES[eng].kvCache || {})[opt.value] && opt.value !== 'bf16' ? `${E.KV_LABEL[opt.value]} — not in ${ENGINES[eng].name}` : E.KV_LABEL[opt.value]);
    const e = ENGINES[eng];
    $('#engineHint').replaceChildren(e.docs ? h('span', null, `Per ${e.name} docs (${e.version}), checked ${e.checked}${e.unverified ? ', not re-verified' : ''}: `, h('a', { href: e.docs.weights, target: '_blank', rel: 'noopener' }, 'weights'), ', ', h('a', { href: e.docs.kv, target: '_blank', rel: 'noopener' }, 'KV cache'), '. The planner takes the weaker of engine support and silicon capability.') : h('span', null, e.note || ''));
  }
  const VENDOR_ORGS = ['RedHatAI', 'neuralmagic', 'nvidia', 'amd', 'Intel', 'hugging-quants', 'mistral-community'];
  function checkpointFor(model, prec) {
    const nk = (model.nativePrec || '').split(' ')[0];
    const official = model.hf ? model.hf.split('/')[0] : null;
    const repo = (model.variants && model.variants[prec]) || (nk === prec && model.hf ? model.hf : null);
    if (!repo) return null;
    const org = repo.split('/')[0];
    const kind = org === official ? 'official' : VENDOR_ORGS.includes(org) ? 'vendor' : 'community';
    return { repo, org, kind };
  }
  function checkpointNode(model, prec) {
    const c = checkpointFor(model, prec);
    if (!c) return h('span', null, `No known ${E.PREC_LABEL[prec]} checkpoint for this model: quantize it yourself (llm-compressor, NVIDIA ModelOpt, AutoAWQ) or pick another format.`);
    return h('span', null, `${E.PREC_LABEL[prec]} checkpoint: `, h('a', { href: 'https://huggingface.co/' + c.repo, target: '_blank', rel: 'noopener' }, c.repo), ' ', h('span', { class: 'chip ' + (c.kind === 'official' ? 'good' : c.kind === 'vendor' ? 'info' : 'warn') }, c.kind));
  }
  function archLine(hw) {
    const a = ARCHS[hw.arch];
    if (!a) return 'generation unknown: format support inferred from the TFLOPS columns';
    const up = (l) => l.map((x) => x.toUpperCase()).join(', ');
    return `${a.name} · native ${up(a.native)}${a.weightOnly.length ? ' · weight-only ' + up(a.weightOnly) : ''}`;
  }
  function catalogLint() {
    const issues = [];
    for (const hw of allHardware()) {
      const a = ARCHS[hw.arch], t = hw.tflops || {};
      if (!a) { issues.push(`${hw.name}: no chip generation (arch) set; format support is inferred from its TFLOPS columns.`); continue; }
      for (const f of ['fp8', 'fp4']) {
        if (a.native.includes(f) && !t[f]) issues.push(`${hw.name}: ${a.name} computes ${f.toUpperCase()} natively but the entry has no ${f.toUpperCase()} TFLOPS.`);
        if (!a.native.includes(f) && t[f]) issues.push(`${hw.name}: entry lists ${f.toUpperCase()} TFLOPS but ${a.name} has no native ${f.toUpperCase()} units.`);
      }
      if (t.fp8 && t.fp16 && (t.fp8 < t.fp16 || t.fp8 > 2.4 * t.fp16)) issues.push(`${hw.name}: FP8 is ${(t.fp8 / t.fp16).toFixed(1)}× BF16; dense tensor-core ratios are 1× to 2× (a sparsity-inflated figure?).`);
      if (t.fp4 && t.fp8 && (t.fp4 < t.fp8 || t.fp4 > 3.2 * t.fp8)) issues.push(`${hw.name}: FP4 is ${(t.fp4 / t.fp8).toFixed(1)}× FP8; expected 2× to 3× dense.`);
    }
    return issues;
  }
  function syncInputs() {
    for (const [id, key, type] of BIND) {
      const el = document.getElementById(id); if (!el) continue;
      const v = getPath(state, key);
      if (type === 'bool') el.checked = !!v; else el.value = v == null ? '' : v;
    }
    $('#ctxRange').value = ctxToSlider(state.ctx);
    $('#manualPar').hidden = state.par !== 'manual';
    $('#specOpts').hidden = !state.spec;
    $$('#candidates input').forEach((c) => { c.checked = state.candidates.includes(c.dataset.id); });
  }
  function afterChange(id, v) {
    if (id === 'hw') { const hw = hwById(v); state.nodeGpus = hw.nodeGpus; state.link = 'auto'; $('#nodeGpus').value = hw.nodeGpus; $('#link').value = 'auto'; }
    if (id === 'model') { const m = E.norm(modelById(v)); state.dpAttn = E.isMla(m); $('#dpAttn').checked = state.dpAttn; }
    if (id === 'activityPreset' && ACTIVITY[v] != null) { state.activity = ACTIVITY[v]; $('#activity').value = ACTIVITY[v]; }
    if (id === 'activity') { state.activityPreset = 'custom'; $('#activityPreset').value = 'custom'; }
    if (id === 'ctx') $('#ctxRange').value = ctxToSlider(v);
    if (id === 'par') $('#manualPar').hidden = v !== 'manual';
    if (id === 'spec') $('#specOpts').hidden = !v;
  }
  let raf = 0;
  function scheduleRender() { LS.set('cb.state', state); if (raf) return; raf = requestAnimationFrame(() => { raf = 0; render(); }); }
  function bindInputs() {
    for (const [id, key, type] of BIND) {
      const el = document.getElementById(id); if (!el) continue;
      const evt = el.type === 'checkbox' || el.tagName === 'SELECT' ? 'change' : 'input';
      el.addEventListener(evt, () => {
        let v = el.type === 'checkbox' ? el.checked : el.value;
        if (type === 'int') { v = parseInt(v, 10); if (!isFinite(v)) return; }
        else if (type === 'num') { v = parseFloat(v); if (!isFinite(v)) return; }
        setPath(state, key, v);
        afterChange(id, v);
        scheduleRender();
      });
    }
    $('#ctxRange').addEventListener('input', (e) => { state.ctx = sliderToCtx(+e.target.value); $('#ctx').value = state.ctx; scheduleRender(); });
    $$('.tab').forEach((t) => t.addEventListener('click', () => { state.mode = t.dataset.mode; scheduleRender(); }));
    $('#candidates').addEventListener('change', () => { state.candidates = $$('#candidates input').filter((c) => c.checked).map((c) => c.dataset.id); scheduleRender(); });
    $('#candAll').addEventListener('click', () => { state.candidates = allHardware().map((x) => x.id); syncInputs(); scheduleRender(); });
    $('#candNone').addEventListener('click', () => { state.candidates = []; syncInputs(); scheduleRender(); });
    $('#resetInputs').addEventListener('click', () => { state = Object.assign({}, DEFAULTS, { mode: state.mode }); state.adv = Object.assign({}, DEFAULTS.adv); syncInputs(); scheduleRender(); });
    $('#addHw').addEventListener('click', () => addCustom('hardware'));
    $('#addModel').addEventListener('click', () => addCustom('model'));
    $('#exportJson').addEventListener('click', () => { $('#customJson').value = JSON.stringify({ hardware: allHardware(), models: allModels() }, null, 1); setMsg('Catalog JSON is in the box; copy it from there.'); });
    $('#resetCustom').addEventListener('click', () => { customHw = []; customModels = []; LS.del('cb.customHardware'); LS.del('cb.customModels'); populateSelects(); syncInputs(); setMsg('Custom entries removed.'); scheduleRender(); });
  }
  function setMsg(t, bad) { const el = $('#customMsg'); el.textContent = t; el.className = 'hint ' + (bad ? 'bad' : 'good'); }
  function addCustom(kind) {
    let obj;
    try { obj = JSON.parse($('#customJson').value); } catch (e) { setMsg('That is not valid JSON: ' + e.message, true); return; }
    const list = Array.isArray(obj) ? obj : [obj];
    const need = kind === 'hardware' ? ['id', 'name', 'mem', 'bw'] : ['id', 'name', 'params', 'layers', 'dModel', 'nHeads', 'nKv', 'dHead', 'maxCtx'];
    for (const o of list) {
      const missing = need.filter((k) => o[k] == null);
      if (missing.length) { setMsg(`Entry "${o.id || '?'}" is missing: ${missing.join(', ')}`, true); return; }
      if (kind === 'hardware') { o.vendor = o.vendor || 'Custom'; o.tflops = o.tflops || { fp16: 100 }; o.link = o.link || 'pcie5'; o.linkBw = o.linkBw || 64; o.nodeGpus = o.nodeGpus || 8; o.custom = true; }
      else { o.family = o.family || 'Custom'; o.custom = true; }
    }
    if (kind === 'hardware') { customHw = customHw.filter((x) => !list.some((o) => o.id === x.id)).concat(list); LS.set('cb.customHardware', customHw); }
    else { customModels = customModels.filter((x) => !list.some((o) => o.id === x.id)).concat(list); LS.set('cb.customModels', customModels); }
    populateSelects(); syncInputs(); setMsg(`Added ${list.length} ${kind} ${list.length === 1 ? 'entry' : 'entries'}. They live in this browser's storage.`); scheduleRender();
  }

  /* ---------- rendering ---------- */
  function render() {
    $$('.tab').forEach((t) => t.setAttribute('aria-selected', String(t.dataset.mode === state.mode)));
    $$('.view').forEach((v) => { v.hidden = v.dataset.mode !== state.mode; });
    const planning = ['forward', 'reverse', 'compat'].includes(state.mode);
    $('#rail').hidden = !planning;
    $('#plan').classList.toggle('wide', !planning);
    $('#railForward').hidden = state.mode !== 'forward';
    $('#railReverse').hidden = state.mode !== 'reverse';
    $('#railWorkload').hidden = state.mode === 'compat';
    $('#railOpt').hidden = state.mode === 'compat';
    try {
      if (state.mode === 'forward') renderForward();
      else if (state.mode === 'reverse') renderReverse();
      else if (state.mode === 'compat') renderCompat();
      else if (state.mode === 'catalog') renderCatalog();
    } catch (err) {
      const v = $('#verdict'); if (v) { v.className = 'verdict bad'; v.replaceChildren(h('strong', null, 'Something went wrong while computing.'), h('p', null, String(err && err.message || err))); }
      console.error(err);
    }
  }

  /* ----- forward ----- */
  function renderForward() {
    const p = params();
    const { best: r, tried } = run();
    const model = E.norm(p.model), hw = p.hw, wl = p.wl;
    $('#hwHint').textContent = `${hw.mem} GB · ${fmtNum(hw.bw)} GB/s · ${hw.tflops.fp16 ? fmtNum(hw.tflops.fp16) + ' TFLOPS BF16' : ''}${hw.tflops.fp8 ? ', ' + fmtNum(hw.tflops.fp8) + ' FP8' : ''}${hw.tflops.fp4 ? ', ' + fmtNum(hw.tflops.fp4) + ' FP4' : ''} · ${hw.tdp ? hw.tdp + ' W' : ''}${hw.price != null ? ' · ~' + fmtMoney(hw.price) + '/h' : ''}${hw.approx ? ' · some specs approximate' : ''} · ${archLine(hw)}`;
    labelPrecOptions(hw);
    $('#modelHint').textContent = `${fmtNum(model.params, 1)}B params${model.moe ? `, ${fmtNum(model.active, 1)}B active (${model.moe.experts} experts, ${model.moe.active} per token)` : ''} · ${model.layers} layers · KV ${fmtGB(E.kvPerTokenFull(model, 'bf16'))}/token in BF16 · max context ${fmtTok(model.maxCtx)}${model.nativePrec ? ' · ships in ' + model.nativePrec.toUpperCase() : ''}${model.hf ? ' · verified against ' + model.hf : ''}${model.note ? ' · ' + model.note : ''}`;
    $('#precHint').replaceChildren(checkpointNode(model, p.wPrec));

    // verdict
    const v = $('#verdict');
    const ok = r.fits && r.memOK && r.speedOK;
    v.className = 'verdict ' + (ok ? 'ok' : (r.fits && r.memOK ? 'warn' : 'bad'));
    v.replaceChildren(...verdictContent(r, p));

    // KPIs
    const a = r.at;
    const tiles = [
      tile('Concurrent requests supported', r.fits ? fmtNum(r.maxConc) : '0', r.fits ? `≈ ${fmtNum(r.maxUsers)} users at ${pct(wl.activity)} active · you asked for ${fmtNum(r.B)}` : 'model does not load', r.fits && r.maxConc >= r.B ? 'good' : 'bad', true),
      tile('Context per session at your load', r.fits ? fmtTok(r.maxCtxAtLoad) : '—', `${fmtNum(r.sessionsPerRep)} sessions resident per replica · model max ${fmtTok(model.maxCtx)}`, r.maxCtxAtLoad >= wl.ctx ? 'good' : 'bad'),
      tile('Speed per user', a ? fmtTokS(a.perUser) + ' tok/s' : '—', a ? `target ≥ ${wl.target} · ${fmtTime(a.itl)} between tokens` : '', a && a.perUser >= wl.target && !a.saturated ? 'good' : 'bad'),
      tile('Aggregate throughput', a ? fmtNum(r.aggTotal) + ' tok/s' : '—', r.costPerMTok != null ? `${fmtMoney(r.costPerMTok)} per 1M output tokens at ~${fmtMoney(r.price, 0)}/h` : (r.kW ? `${fmtNum(r.kW, 1)} kW of accelerators` : '')),
      tile('Time to first token', a ? fmtTime(r.ttft) : '—', a ? `limit ${fmtTime(wl.ttftMax)} · first turn from cold: ${fmtTime(r.ttftCold)} for ${fmtTok(r.coldNew)} tokens` : '', a ? (r.ttftOK ? 'good' : 'bad') : null),
      tile('KV cache per session', fmtGB(r.kv.perSession), `${fmtGB(E.kvPerTokenFull(model, p.kvPrec))}/token · pool ${fmtGB(r.kvAvail)} per replica`),
    ];
    $('#kpis').replaceChildren(...tiles);

    $('#configBody').replaceChildren(...configBody(r, p, tried));
    $('#bottleneckBody').replaceChildren(...bottleneckBody(r, p));
    renderCapacityChart(r, p);
    renderSpeedChart(r, p);
    renderLedger(r, p);
    renderWarnings(r);
  }
  function tile(label, value, sub, status, hero) {
    return h('div', { class: 'kpi' + (hero ? ' hero' : '') + (status ? ' ' + status : '') }, h('div', { class: 'l' }, label), h('div', { class: 'v' }, value), h('div', { class: 's' }, sub));
  }
  function verdictContent(r, p) {
    const wl = p.wl, model = r.model, layout = `${r.total}× ${r.hw.name}`;
    const who = `${fmtNum(wl.users)} users (${fmtNum(r.B)} concurrent at ${pct(wl.activity)} active)`;
    if (!r.fits) {
      const mg = E.minGpus(r.hw, model, p.wPrec, p.kvPrec, Math.min(wl.ctx, 32768), p.adv);
      return [
        h('strong', null, `${model.name} in ${E.PREC_LABEL[p.wPrec]} does not load on ${layout}.`),
        h('p', null, `Weights take ${fmtGB(r.W)}; ${r.G} accelerators offer ${fmtGB(Math.max(0, r.G * r.capPerGpu))} after runtime overhead. `, mg ? `The smallest layout that loads it here is ${mg.G} accelerators (TP ${mg.tp} × PP ${mg.pp}) for one 32k session; lower the weight precision or pick bigger memory.` : 'No layout up to 64-way tensor × 16-stage pipeline parallelism loads it on this hardware.'),
      ];
    }
    if (!r.memOK) {
      return [
        h('strong', null, `Not enough KV cache for ${who} at ${fmtTok(wl.ctx)} context on ${layout}.`),
        h('p', null, `This layout keeps ${fmtNum(r.maxSessions * r.R)} sessions of ${fmtTok(wl.ctx)} tokens; you need ${fmtNum(r.residentUsers)} resident. At your load the context would have to drop to ${fmtTok(r.maxCtxAtLoad)}, or serve ${fmtNum(r.maxConc)} concurrent instead. The ledger below lists what buys more room.`),
      ];
    }
    if (!r.speedOK && !r.ttftOK) {
      return [
        h('strong', null, `Memory fits, but the first token of a turn takes ${fmtTime(r.ttft)} (limit ${fmtTime(p.wl.ttftMax)}).`),
        h('p', null, `Each turn prefills ${fmtTok(r.warmNew)} tokens on the ${r.tp} accelerators of one replica. Wider tensor parallelism, native FP8/FP4 compute, or keeping more of the prompt cached shortens it; or raise the limit if your users can wait.`),
      ];
    }
    if (!r.speedOK) {
      return [
        h('strong', null, r.at.saturated ? `Memory fits, but prefill saturates ${layout}.` : `Memory fits, but each user would get ${fmtTokS(r.at.perUser)} tok/s (target ${wl.target}).`),
        h('p', null, r.at.saturated
          ? `Each turn prefills ${fmtTok(r.warmNew)} new tokens; ${fmtNum(r.B)} concurrent requests arrive faster than the accelerators can compute those prompts. Cache more of the prompt, add compute, or disaggregate prefill.`
          : `Up to ${fmtNum(r.maxConc)} concurrent requests (${fmtNum(r.maxUsers)} users) hold ${wl.target} tok/s at ${fmtTok(wl.ctx)} context. Lower precision, speculative decoding or more accelerators raise the per-user speed.`),
      ];
    }
    return [
      h('strong', null, `${layout} serve ${model.name} to ${who} at ${fmtTok(wl.ctx)} context.`),
      h('p', null, `${fmtTokS(r.at.perUser)} tok/s per user, ${fmtTime(r.ttft)} to the first token of a turn. Headroom: ${fmtNum(r.maxConc)} concurrent requests (${fmtNum(r.maxUsers)} users) at this context, or ${fmtTok(r.maxCtxAtLoad)} of context at this load.`),
    ];
  }
  function configBody(r, p, tried) {
    const hw = r.hw, mem = hw.mem * 1e9;
    const wPer = r.W / r.G, kvPer = Math.max(0, r.kvAvailRaw / r.G), ovh = r.overhead, head = mem * (1 - p.adv.util);
    const segs = [['Weights', wPer, 'var(--s1)'], ['KV cache pool', kvPer, 'var(--s3)'], ['Runtime overhead', ovh, 'var(--muted)'], ['Headroom (unused by the engine)', head, 'var(--line)']];
    const bar = h('div', { class: 'membar', role: 'img', 'aria-label': `Memory per accelerator: weights ${fmtGB(wPer)}, KV cache ${fmtGB(kvPer)}, overhead ${fmtGB(ovh)}, headroom ${fmtGB(head)}` });
    for (const [name, val, color] of segs) if (val > 0) bar.append(h('div', { class: 'seg', style: `flex:${val};background:${color}`, title: `${name}: ${fmtGB(val)}` }));
    const key = h('ul', { class: 'key' }, segs.map(([name, val, color]) => h('li', null, h('i', { style: `background:${color}` }), h('span', { class: 't' }, name), h('b', null, fmtGB(Math.max(0, val))))));
    const layoutTxt = `TP ${r.tp} × PP ${r.pp} × ${plural(r.R, 'replica')} = ${r.total} accelerators on ${plural(r.nodes, 'node')}` + (r.count - r.total > 0 ? `, ${r.count - r.total} idle` : '') + (r.pdTotal ? `, plus ${r.pdTotal} in a prefill pool` : '');
    const out = [
      h('p', { class: 'lead' }, layoutTxt),
      h('p', { class: 'hint' }, `Per accelerator (${hw.mem} GB): ${fmtGB(wPer)} weights, ${fmtGB(kvPer)} KV cache = ${fmtTok(kvPer / Math.max(1, E.kvPerTokenFull(r.model, p.kvPrec) / r.kvRepl))} tokens${r.kvRepl > 1 ? ` (KV replicated ${r.kvRepl}×)` : ''}. ${r.S > 0 ? `Shared prefix of ${fmtTok(r.S)} tokens stored once per replica (${fmtGB(r.kv.shared)}).` : ''}`),
      bar, key,
    ];
    if (tried && tried.length > 1) {
      const rows = tried.slice().sort((x, y) => (y.maxUsers - x.maxUsers) || ((y.at ? y.at.perUser : 0) - (x.at ? x.at.perUser : 0))).slice(0, 5);
      out.push(h('h4', null, 'Layouts considered'), table(
        ['Layout', 'Concurrent', 'Users', 'tok/s per user', 'Context at load', 'Idle'],
        rows.map((t) => [`TP ${t.tp} × PP ${t.pp} × ${t.R}` + (t === r ? ' ◂' : ''), t.fits ? fmtNum(t.maxConc) : 'no fit', t.fits ? fmtNum(t.maxUsers) : '—', t.at ? fmtTokS(t.at.perUser) : '—', t.fits ? fmtTok(t.maxCtxAtLoad) : '—', fmtNum(t.count - t.total)]),
        { numeric: [1, 2, 3, 4, 5] }));
    }
    return out;
  }
  function bottleneckBody(r, p) {
    const a = r.at;
    if (!a) return [h('p', { class: 'hint' }, 'Nothing to time: the model does not load on this layout.')];
    const eff = r.tp * r.hw.bw * 1e9 * p.adv.bwEff;
    const linkName = r.tpCross ? p.net.name : (LINKS[p.link] || LINKS.nvlink).name;
    const rows = [
      ['Decode step', fmtTime(a.step), `${a.bound}-bound at ${fmtNum(r.bPerRep)} concurrent per replica`],
      ['Memory traffic', fmtTime(a.tBw), `${fmtGB(a.wRead)} of weights + ${fmtGB(a.kvRead)} of KV cache per step at ${fmtGB(eff)}/s effective`],
      ['Compute', fmtTime(a.tComp), `${r.pk.used} matmuls at ${pct(p.adv.mfu)} of ${fmtNum(r.pk.peak / 1e12)} TFLOPS × ${r.G}`],
      ['Collectives', fmtTime(a.tComm), r.tp > 1 ? `${2 * r.model.layers} all-reduces per token over ${linkName}` : 'no tensor parallelism'],
      ['Prefill share', pct(a.f) + (a.saturated ? ' (saturated)' : ''), `${a.lambda.toFixed(2)} requests/s per replica, ${fmtTok(r.warmNew)} new tokens each${p.opt.pd ? ' (in the prefill pool)' : ', interleaved with decode'}`],
      ['Warm turn TTFT', fmtTime(r.ttftWarm), p.wl.retention === 'host' ? `includes ${fmtTime(r.restoreS)} to restore ${fmtGB(r.kv.perSession)} from host memory` : `${fmtTok(r.warmNew)} tokens prefilled`],
      ['Cold TTFT', fmtTime(r.ttftCold), `${fmtTok(r.coldNew)} tokens prefilled on ${r.tp} accelerators`],
    ];
    return [table(['What', 'Time', 'Why'], rows, { numeric: [1] })];
  }
  function table(cols, rows, opts) {
    const numeric = new Set((opts && opts.numeric) || []);
    return h('div', { class: 'tw' }, h('table', { class: 'data' },
      h('thead', null, h('tr', null, cols.map((c, i) => h('th', { class: numeric.has(i) ? 'n' : null }, c)))),
      h('tbody', null, rows.map((row) => h('tr', { class: row.cls || null }, (row.cells || row).map((c, i) => h('td', { class: numeric.has(i) ? 'n' : 't' }, c)))))));
  }

  /* ----- charts ----- */
  function evalAt(p, r, overrides) {
    const q = Object.assign({}, p, overrides || {});
    if (overrides && overrides.wl) q.wl = Object.assign({}, p.wl, overrides.wl);
    return E.evaluate(Object.assign(q, { tp: r.tp, pp: r.pp, replicas: r.R }));
  }
  function variantLabel(p) {
    const hw = p.hw;
    const list = [];
    if (p.kvPrec === 'bf16') list.push({ name: 'KV cache FP8', o: { kvPrec: 'fp8' } });
    if (p.kvPrec !== 'int4') list.push({ name: 'KV cache INT4', o: { kvPrec: 'int4' } });
    let w = null;
    if (p.wPrec === 'bf16') w = hw.tflops.fp8 ? 'fp8' : 'int4';
    else if (['fp8', 'int8'].includes(p.wPrec)) w = hw.tflops.fp4 ? 'fp4' : 'int4';
    if (w) list.push({ name: `Weights ${E.PREC_LABEL[w].split(' ')[0]} + KV FP8`, o: { wPrec: w, kvPrec: p.kvPrec === 'bf16' ? 'fp8' : p.kvPrec } });
    return list.slice(0, 3);
  }
  function renderCapacityChart(r, p) {
    const el = $('#chartCapacity');
    if (!r.fits) { el.replaceChildren(h('p', { class: 'hint' }, 'The frontier needs a layout that loads the model.')); return; }
    const model = r.model, gpu = p.wl.retention === 'gpu';
    const xMax = Math.min(1 << 24, Math.max(model.maxCtx, p.wl.ctx * 2, 65536));
    const xs = []; for (let x = 1024; x <= xMax * 1.0001; x *= Math.SQRT2) xs.push(Math.round(x));
    if (xs[xs.length - 1] !== xMax) xs.push(xMax);
    const colors = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)'];
    const variants = [{ name: 'As configured', o: {} }].concat(variantLabel(p));
    const series = variants.map((vr, i) => ({ name: vr.name, color: colors[i], points: xs.map((x) => { const q = evalAt(p, r, Object.assign({}, vr.o, { wl: { ctx: x } })); return [x, gpu ? q.maxUsers : q.maxConc]; }) }));
    const yUnit = gpu ? 'users (sessions kept in GPU memory)' : 'concurrent requests';
    lineChart({
      el, title: `${gpu ? 'Users' : 'Concurrent requests'} you can serve at ≥ ${p.wl.target} tok/s each, by context length`,
      x: { label: 'context per session (tokens)', log: true, fmt: fmtTok, ticks: tokenTicks }, y: { label: yUnit, log: true, fmt: (v) => fmtNum(v) },
      series, refX: [{ x: model.maxCtx, label: 'model max' }], marker: { x: p.wl.ctx, y: gpu ? p.wl.users : r.B, ok: (gpu ? r.maxUsers >= p.wl.users : r.maxConc >= r.B), label: 'your workload' },
    });
  }
  function renderSpeedChart(r, p) {
    const el = $('#chartSpeed');
    if (!r.fits) { el.replaceChildren(); return; }
    const bMax = Math.max(2, Math.min(r.maxSessions, 2048));
    const xs = [1]; for (let x = 1; x < bMax; x *= Math.SQRT2) { const v = Math.round(x * Math.SQRT2); if (v > xs[xs.length - 1] && v < bMax) xs.push(v); } xs.push(bMax);
    const series = [{ name: p.opt.spec ? 'With speculative decoding' : 'As configured', color: 'var(--s1)', points: xs.map((b) => [b, r.loadAt(b, p.opt.spec).perUser]) }];
    series.push(p.opt.spec ? { name: 'Without speculative decoding', color: 'var(--s2)', points: xs.map((b) => [b, r.loadAt(b, false).perUser]) } : { name: 'With speculative decoding', color: 'var(--s2)', points: xs.map((b) => [b, r.loadAt(b, true).perUser]) });
    lineChart({
      el, title: `Speed per user as one replica (${r.G} accelerators) takes more concurrent requests at ${fmtTok(p.wl.ctx)} context`,
      x: { label: 'concurrent requests per replica', log: true, fmt: (v) => fmtNum(v), ticks: countTicks }, y: { label: 'tokens per second per user', log: true, fmt: (v) => fmtTokS(v) },
      series, refY: [{ y: p.wl.target, label: 'target' }], marker: { x: Math.min(r.bPerRep, r.maxSessions), y: r.at.perUser, ok: r.speedOK, label: 'your load' },
    });
  }
  function tokenTicks(min, max) { const out = []; const step = Math.log2(max / min) > 8 ? 4 : 2; for (let v = 1024; v <= max * 1.0001; v *= step) if (v >= min) out.push(v); return out; }
  function countTicks(min, max) { const out = []; const dec = Math.log10(max / min); for (let e = Math.floor(Math.log10(min)); Math.pow(10, e) <= max; e++) for (const m of (dec > 2.5 ? [1] : [1, 2, 5])) { const v = m * Math.pow(10, e); if (v >= min && v <= max) out.push(v); } return out; }
  function logTicks(min, max) { const out = []; const dec = Math.log10(max / min); for (let e = Math.floor(Math.log10(min)); Math.pow(10, e) <= max * 1.0001; e++) for (const m of (dec > 3 ? [1] : dec > 1.5 ? [1, 3] : [1, 2, 5])) { const v = m * Math.pow(10, e); if (v >= min * 0.999 && v <= max * 1.001) out.push(v); } return out; }

  function lineChart(o) {
    const W = 720, H = 330, m = { l: 62, r: 22, t: 20, b: 48 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const xs = o.series[0].points.map(([x]) => x);
    const valid = o.series.flatMap((sr) => sr.points).filter(([x, y]) => x > 0 && y > 0 && isFinite(y));
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    let yMax = Math.max(1, ...valid.map(([, y]) => y), ...(o.refY || []).map((r) => r.y), o.marker && o.marker.y > 0 ? o.marker.y : 0);
    let yMin = Math.min(1, ...valid.map(([, y]) => y), o.marker && o.marker.y > 0 ? o.marker.y : 1);
    yMin = Math.pow(10, Math.floor(Math.log10(Math.max(1e-2, yMin)))); yMax = Math.pow(10, Math.ceil(Math.log10(yMax * 1.05)));
    if (yMax <= yMin) yMax = yMin * 10;
    const lx = Math.log(xMin), ux = Math.log(xMax), ly = Math.log(yMin), uy = Math.log(yMax);
    const sx = (x) => m.l + (Math.log(x) - lx) / (ux - lx) * pw;
    const sy = (y) => m.t + ph - (Math.log(y) - ly) / (uy - ly) * ph;
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': o.title });
    svg.append(s('rect', { x: 0, y: 0, width: W, height: H, fill: 'var(--chart-surface)' }));
    for (const t of logTicks(yMin, yMax)) { svg.append(s('line', { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), stroke: 'var(--grid)', 'stroke-width': 1 })); svg.append(s('text', { x: m.l - 8, y: sy(t), 'text-anchor': 'end', 'dominant-baseline': 'middle' }, o.y.fmt(t))); }
    svg.append(s('line', { x1: m.l, x2: W - m.r, y1: m.t + ph, y2: m.t + ph, stroke: 'var(--axis)', 'stroke-width': 1 }));
    for (const t of o.x.ticks(xMin, xMax)) { svg.append(s('line', { x1: sx(t), x2: sx(t), y1: m.t + ph, y2: m.t + ph + 4, stroke: 'var(--axis)' })); svg.append(s('text', { x: sx(t), y: m.t + ph + 16, 'text-anchor': 'middle' }, o.x.fmt(t))); }
    svg.append(s('text', { x: m.l + pw / 2, y: H - 8, 'text-anchor': 'middle', class: 'axis-title' }, o.x.label));
    svg.append(s('text', { x: 14, y: m.t + ph / 2, 'text-anchor': 'middle', transform: `rotate(-90 14 ${m.t + ph / 2})`, class: 'axis-title' }, o.y.label));
    for (const rx of o.refX || []) { if (rx.x < xMin || rx.x > xMax) continue; svg.append(s('line', { x1: sx(rx.x), x2: sx(rx.x), y1: m.t, y2: m.t + ph, stroke: 'var(--axis)', 'stroke-width': 1 })); svg.append(s('text', { x: sx(rx.x) - 4, y: m.t + 10, 'text-anchor': 'end', class: 'ref' }, rx.label)); }
    for (const ry of o.refY || []) { if (ry.y < yMin || ry.y > yMax) continue; svg.append(s('line', { x1: m.l, x2: W - m.r, y1: sy(ry.y), y2: sy(ry.y), stroke: 'var(--axis)', 'stroke-width': 1 })); svg.append(s('text', { x: W - m.r - 4, y: sy(ry.y) - 5, 'text-anchor': 'end', class: 'ref' }, `${ry.label} ${o.y.fmt(ry.y)}`)); }
    o.series.forEach((sr) => {
      let d = '', pen = false, last = null;
      for (const [x, y] of sr.points) { if (!(y > 0) || !isFinite(y)) { pen = false; continue; } d += `${pen ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(Math.min(Math.max(y, yMin), yMax)).toFixed(1)} `; pen = true; last = [x, y]; }
      if (d) svg.append(s('path', { d, fill: 'none', stroke: sr.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
      if (last) { svg.append(s('circle', { cx: sx(last[0]), cy: sy(Math.min(Math.max(last[1], yMin), yMax)), r: 6, fill: 'var(--chart-surface)' })); svg.append(s('circle', { cx: sx(last[0]), cy: sy(Math.min(Math.max(last[1], yMin), yMax)), r: 4, fill: sr.color })); }
    });
    if (o.marker && o.marker.x >= xMin && o.marker.x <= xMax) {
      const my = Math.min(Math.max(o.marker.y > 0 ? o.marker.y : yMin, yMin), yMax);
      const col = o.marker.ok ? 'var(--good)' : 'var(--crit)';
      svg.append(s('circle', { cx: sx(o.marker.x), cy: sy(my), r: 8, fill: 'var(--chart-surface)' }));
      svg.append(s('circle', { cx: sx(o.marker.x), cy: sy(my), r: 5.5, fill: col }));
      svg.append(s('text', { x: sx(o.marker.x) + 10, y: sy(my) + 4, class: 'ref' }, o.marker.label));
    }
    const cross = s('line', { x1: 0, x2: 0, y1: m.t, y2: m.t + ph, stroke: 'var(--axis)', 'stroke-width': 1, visibility: 'hidden' });
    svg.append(cross);
    const hit = s('rect', { x: m.l, y: m.t, width: pw, height: ph, fill: 'transparent', tabindex: 0, style: 'outline:none;cursor:crosshair' });
    svg.append(hit);
    const tip = h('div', { class: 'tip', hidden: true });
    const wrap = h('div', { class: 'chart-wrap' }, h('p', { class: 'chart-title' }, o.title), h('div', { class: 'chart-box' }, svg, tip));
    const legend = h('ul', { class: 'legend' }, o.series.map((sr) => h('li', null, h('i', { style: `background:${sr.color}` }), sr.name)));
    if (o.marker) legend.append(h('li', null, h('i', { class: 'dot', style: `background:${o.marker.ok ? 'var(--good)' : 'var(--crit)'}` }), o.marker.label + (o.marker.ok ? ' (met)' : ' (not met)')));
    wrap.append(legend);
    const tbl = table([o.x.label].concat(o.series.map((sr) => sr.name)), xs.map((x, i) => [o.x.fmt(x)].concat(o.series.map((sr) => { const y = sr.points[i][1]; return y > 0 ? o.y.fmt(y) : '0'; }))), { numeric: o.series.map((_, i) => i + 1) });
    tbl.hidden = true;
    const btn = h('button', { class: 'ghost', type: 'button', onclick: () => { tbl.hidden = !tbl.hidden; btn.textContent = tbl.hidden ? 'Show as table' : 'Hide table'; } }, 'Show as table');
    wrap.append(h('div', { class: 'chart-foot' }, btn), tbl);
    let idx = -1;
    const show = (i) => {
      if (i < 0 || i >= xs.length) return; idx = i;
      const x = xs[i]; cross.setAttribute('x1', sx(x)); cross.setAttribute('x2', sx(x)); cross.setAttribute('visibility', 'visible');
      tip.replaceChildren(h('div', { class: 'tip-h' }, `${o.x.fmt(x)} ${o.x.label.split(' ')[0]}`), ...o.series.map((sr) => { const y = sr.points[i][1]; return h('div', { class: 'tip-r' }, h('i', { style: `background:${sr.color}` }), h('b', null, y > 0 ? o.y.fmt(y) : '0'), h('span', null, sr.name)); }));
      tip.hidden = false;
      const box = wrap.querySelector('.chart-box').getBoundingClientRect(), r = svg.getBoundingClientRect();
      const px = (sx(x) / W) * r.width + (r.left - box.left);
      tip.style.left = Math.min(box.width - tip.offsetWidth - 4, Math.max(0, px + 12 > box.width - tip.offsetWidth ? px - tip.offsetWidth - 12 : px + 12)) + 'px';
      tip.style.top = '28px';
    };
    hit.addEventListener('pointermove', (ev) => { const r = svg.getBoundingClientRect(); const px = (ev.clientX - r.left) * (W / r.width); const lxv = lx + (px - m.l) / pw * (ux - lx); let bi = 0, bd = Infinity; xs.forEach((x, i) => { const d = Math.abs(Math.log(x) - lxv); if (d < bd) { bd = d; bi = i; } }); show(bi); });
    hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); tip.hidden = true; });
    hit.addEventListener('focus', () => show(idx < 0 ? Math.floor(xs.length / 2) : idx));
    hit.addEventListener('blur', () => { cross.setAttribute('visibility', 'hidden'); tip.hidden = true; });
    hit.addEventListener('keydown', (ev) => { if (ev.key === 'ArrowRight') { ev.preventDefault(); show(Math.min(xs.length - 1, idx + 1)); } if (ev.key === 'ArrowLeft') { ev.preventDefault(); show(Math.max(0, idx - 1)); } });
    o.el.replaceChildren(wrap);
  }

  /* ----- ledger ----- */
  function renderLedger(base, p) {
    const el = $('#ledger');
    if (!base.fits) { el.replaceChildren(h('p', { class: 'hint' }, 'Once the model loads, this table compares the options that buy more concurrency or context.')); return; }
    const hw = p.hw, st = state;
    const rows = [{ name: 'As configured', r: base, o: null, note: `${E.PREC_LABEL[st.wPrec]} weights, ${E.KV_LABEL[st.kvPrec]} KV${st.spec ? ', speculative decoding' : ''}${st.pd ? ', prefill/decode split' : ''}` }];
    const add = (name, o, note) => rows.push({ name, r: run(o).best, o, note });
    if (st.kvPrec === 'bf16') add('KV cache in FP8', { kvPrec: 'fp8' }, 'Half the KV bytes per token; usually no measurable quality loss' + (E.kvSupport(hw, 'fp8', st.engine || 'none') === 'supported' ? '' : `; not offered by ${ENGINES[st.engine || 'none'].name} here`));
    if (st.kvPrec !== 'int4') add('KV cache in INT4', { kvPrec: 'int4' }, 'A quarter of the KV bytes; quality risk grows with context' + (E.kvSupport(hw, 'int4', st.engine || 'none') === 'supported' ? '' : `; not offered by ${ENGINES[st.engine || 'none'].name} (LMDeploy has it)`));
    const model = E.norm(p.model);
    const ckpt = (prec) => { const c = checkpointFor(model, prec); return c ? `; checkpoint: ${c.org} (${c.kind})` : '; no known checkpoint'; };
    if (st.wPrec === 'bf16') add('Weights in FP8', { wPrec: 'fp8' }, (E.formatSupport(hw, 'fp8', st.engine || 'none') === 'native' ? 'Native FP8 here: halves weight bytes and speeds up decode' : 'Weight-only on this accelerator: saves memory, compute stays BF16') + ckpt('fp8'));
    if (!['int4', 'fp4'].includes(st.wPrec)) { const f4 = E.formatSupport(hw, 'fp4', st.engine || 'none') === 'native'; add(f4 ? 'Weights in FP4 (NVFP4)' : 'Weights in INT4 (AWQ / GPTQ)', { wPrec: f4 ? 'fp4' : 'int4' }, (f4 ? 'Native FP4: a quarter of the weight bytes and faster prefill' : 'Weight-only 4-bit: a quarter of the weight bytes, compute stays BF16') + ckpt(f4 ? 'fp4' : 'int4')); }
    if (!st.prefixCache && st.prefix > 0) add('Cache the shared prefix', { prefixCache: true }, `The ${fmtTok(st.prefix)}-token shared prefix is stored once per replica`);
    if (!st.spec) add('Speculative decoding', { spec: true }, `Draft ${st.specK} tokens per step at ${st.specAlpha}% acceptance; helps when memory-bound`);
    if (st.retention === 'gpu') add('Park idle sessions in host memory', { retention: 'host' }, 'GPU memory holds only in-flight requests; idle KV restores over PCIe');
    if (st.retention === 'none') add('Keep idle sessions in host memory', { retention: 'host' }, 'Skips re-prefilling the whole conversation on every turn');
    if (!st.pd) add('Disaggregate prefill from decode', { pd: true }, 'A separate prefill pool keeps token speed steady; needs KV transfer over the network');
    const combo = {};
    if (st.kvPrec === 'bf16') combo.kvPrec = 'fp8';
    if (st.wPrec === 'bf16') combo.wPrec = E.formatSupport(hw, 'fp8', st.engine || 'none') === 'native' ? 'fp8' : 'int4';
    if (!st.spec) combo.spec = true;
    if (!st.prefixCache && st.prefix > 0) combo.prefixCache = true;
    if (Object.keys(combo).length > 1) add('All of the above (memory + speed)', combo, 'The combination most production stacks run');
    const b = base;
    const cells = rows.map((row) => {
      const r = row.r, a = r.at;
      const verdict = !r.fits ? ['out of memory', 'bad'] : !r.memOK ? ['not enough KV', 'bad'] : !r.speedOK ? [!r.ttftOK ? 'slow first token' : a && a.saturated ? 'prefill-bound' : 'too slow', 'warn'] : ['meets target', 'good'];
      const delta = (v, bv) => bv > 0 && v !== bv ? h('small', { class: v > bv ? 'up' : 'down' }, ` ${v > bv ? '+' : ''}${Math.round((v / bv - 1) * 100)}%`) : null;
      return {
        cls: row.r === base ? 'base' : null,
        cells: [
          h('div', null, h('b', null, row.name), h('div', { class: 'note' }, row.note)),
          r.fits ? h('span', null, fmtNum(r.maxConc), delta(r.maxConc, b.maxConc)) : '0',
          r.fits ? fmtNum(r.maxUsers) : '0',
          r.fits ? h('span', null, fmtTok(r.maxCtxAtLoad), delta(r.maxCtxAtLoad, b.maxCtxAtLoad)) : '—',
          a ? h('span', null, fmtTokS(a.perUser), delta(a.perUser, b.at ? b.at.perUser : 0)) : '—',
          a ? fmtTime(r.ttft) : '—',
          r.costPerMTok != null ? fmtMoney(r.costPerMTok) : '—',
          h('span', { class: 'chip ' + verdict[1] }, verdict[0]),
          row.o ? h('button', { class: 'ghost', type: 'button', onclick: () => { Object.assign(state, row.o); syncInputs(); scheduleRender(); } }, 'Apply') : '',
        ],
      };
    });
    el.replaceChildren(table(['Option', 'Concurrent', 'Users', 'Context at load', 'tok/s per user', 'TTFT', '$ / 1M tok', 'Verdict', ''], cells, { numeric: [1, 2, 3, 4, 5, 6] }));
  }
  function renderWarnings(r) {
    const el = $('#warnings');
    if (!r.warnings.length) { el.replaceChildren(); return; }
    el.replaceChildren(h('h3', null, 'Notes'), h('ul', { class: 'warnings' }, r.warnings.map((w) => h('li', { class: w.level }, h('span', { class: 'chip ' + (w.level === 'crit' ? 'bad' : w.level === 'warn' ? 'warn' : 'info') }, w.level === 'crit' ? 'blocker' : w.level === 'warn' ? 'caution' : 'note'), ' ', w.text))));
  }

  /* ----- reverse ----- */
  function renderReverse() {
    const p = params();
    const model = E.norm(p.model);
    $('#modelHint').textContent = `${fmtNum(model.params, 1)}B params${model.moe ? `, ${fmtNum(model.active, 1)}B active` : ''} · KV ${fmtGB(E.kvPerTokenFull(model, p.kvPrec))}/token in ${E.KV_LABEL[p.kvPrec]} · max context ${fmtTok(model.maxCtx)}${model.nativePrec ? ' · ships in ' + model.nativePrec.toUpperCase() : ''}`;
    labelPrecOptions(null);
    $('#precHint').replaceChildren(checkpointNode(model, p.wPrec));
    const list = allHardware().filter((hw) => state.candidates.includes(hw.id));
    const B = Math.max(1, Math.round(p.wl.users * p.wl.activity));
    $('#revIntro').textContent = `Smallest layout per accelerator type that serves ${fmtNum(p.wl.users)} users (${fmtNum(B)} concurrent at ${pct(p.wl.activity)} active) of ${model.name} at ${fmtTok(p.wl.ctx)} context with at least ${p.wl.target} tok/s each and a first token within ${fmtTime(p.wl.ttftMax)}. Session KV: ${fmtGB(E.kvAtCtx(model, p.kvPrec, p.wl.ctx))} in ${E.KV_LABEL[p.kvPrec]}; all sessions: ${fmtGB(E.kvAtCtx(model, p.kvPrec, p.wl.ctx) * (p.wl.retention === 'gpu' ? p.wl.users : B))}.`;
    if (!list.length) { $('#revSummary').replaceChildren(); $('#revTable').replaceChildren(h('p', { class: 'hint' }, 'Pick at least one candidate accelerator in the left rail.')); return; }
    const results = E.reverse(p, list);
    const feasible = results.filter((r) => !r.infeasible);
    const fewest = feasible[0];
    const cheapest = feasible.filter((r) => r.price != null).sort((a, b) => a.price - b.price)[0];
    const perTok = feasible.filter((r) => r.costPerMTok != null).sort((a, b) => a.costPerMTok - b.costPerMTok)[0];
    const sum = [];
    if (fewest) sum.push(tile('Fewest accelerators', `${fewest.gpusUsed}× ${fewest.hw.name}`, `${plural(fewest.nodes, 'node')} · ${fmtTokS(fewest.at.perUser)} tok/s per user`, 'good', true));
    if (cheapest) sum.push(tile('Lowest hourly cost', `${fmtMoney(cheapest.price, 0)}/h`, `${cheapest.gpusUsed}× ${cheapest.hw.name}`));
    if (perTok) sum.push(tile('Lowest cost per token', `${fmtMoney(perTok.costPerMTok)} / 1M`, `${perTok.gpusUsed}× ${perTok.hw.name} at ${fmtNum(perTok.aggTotal)} tok/s`));
    if (!feasible.length) sum.push(tile('No candidate works', '—', 'see the reasons below', 'bad', true));
    $('#revSummary').replaceChildren(...sum);
    const rows = results.map((r) => r.infeasible
      ? { cls: 'muted', cells: [h('b', null, r.hw.name), '—', '—', h('span', { class: 'note' }, r.reason), '', '', '', '', '', '', ''] }
      : { cells: [
        h('div', null, h('b', null, r.hw.name), r.warnings.some((w) => w.level === 'warn') ? h('div', { class: 'note' }, r.warnings.filter((w) => w.level === 'warn').map((w) => w.text).join(' ')) : null),
        h('b', null, fmtNum(r.gpusUsed)), fmtNum(r.nodes), `TP ${r.tp} × PP ${r.pp} × ${r.R}${r.pdTotal ? ` + ${r.pdTotal} prefill` : ''}`,
        fmtTokS(r.at.perUser), fmtTime(r.ttft), fmtGB(r.kvAvail * r.R), r.price != null ? fmtMoney(r.price, 0) : '—', r.costPerMTok != null ? fmtMoney(r.costPerMTok) : '—', r.kW != null ? fmtNum(r.kW, 1) : '—',
        h('button', { class: 'ghost', type: 'button', onclick: () => openInPlanner(r) }, 'Open'),
      ] });
    $('#revTable').replaceChildren(table(['Accelerator', 'Count', 'Nodes', 'Layout', 'tok/s per user', 'TTFT', 'KV pool', '$ / hour', '$ / 1M tok', 'kW', ''], rows, { numeric: [1, 2, 4, 5, 6, 7, 8, 9] }));
  }
  function openInPlanner(r) {
    Object.assign(state, { mode: 'forward', hw: r.hw.id, count: r.total, nodeGpus: r.hw.nodeGpus, link: 'auto', par: 'manual', tp: r.tp, pp: r.pp, reps: r.R });
    syncInputs(); scheduleRender(); window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  /* ----- compatibility ----- */
  function renderCompat() {
    const p = params();
    const hws = allHardware(), models = allModels();
    const ctx = Math.max(1024, state.compatCtx | 0);
    $('#compatIntro').textContent = `Each cell is the fewest accelerators of that type on which the model loads with ${E.PREC_LABEL[p.wPrec]} weights and still has room for one ${fmtTok(ctx)}-token session (${E.KV_LABEL[p.kvPrec]} KV cache), using the best tensor × pipeline layout. Change the precisions in the left rail.`;
    labelPrecOptions(null);
    $('#precHint').replaceChildren(checkpointNode(E.norm(p.model), p.wPrec));
    $('#compatLegend').replaceChildren(
      h('li', null, h('b', { class: 'sample' }, '4'), h('span', null, 'number = accelerators needed: the weights split across them plus one session of KV cache. Hover a cell for the TP × PP layout.')),
      h('li', null, h('b', { class: 'sample one' }, '1'), h('span', null, 'one accelerator is enough')),
      h('li', null, h('b', { class: 'sample node' }, '4'), h('span', null, 'several accelerators, but within one node (tensor parallel over the node fabric)')),
      h('li', null, h('b', { class: 'sample multi' }, '16'), h('span', null, 'more accelerators than one node holds (pipeline stages across the inter-node network)')),
      h('li', null, h('b', { class: 'sample none' }, '—'), h('span', null, 'no layout up to TP 64 × PP 16 loads it on this hardware')),
      h('li', null, h('b', { class: 'sample none' }, '✕'), h('span', null, `${E.PREC_LABEL[p.wPrec]} weights are not loadable on that chip generation (no kernel)`)),
      h('li', null, h('b', { class: 'sample' }, 'KV/token'), h('span', null, `bytes of KV cache one token costs in ${E.KV_LABEL[p.kvPrec]}; multiply by the context to size a session`)),
    );
    const head = h('tr', null, h('th', { class: 'sticky' }, 'Model'), h('th', { class: 'n' }, 'KV/token'), hws.map((hw) => h('th', { class: 'n rot' }, h('span', null, hw.name))));
    const body = models.map((m0) => {
      const m = E.norm(m0);
      return h('tr', null, h('td', { class: 't sticky' }, m.name), h('td', { class: 'n' }, fmtGB(E.kvPerTokenFull(m, p.kvPrec))), hws.map((hw) => {
        const unsupported = E.formatSupport(hw, p.wPrec, p.engine) === 'unsupported' || E.kvSupport(hw, p.kvPrec, p.engine) !== 'supported';
        const mg = unsupported ? null : E.minGpus(hw, m, p.wPrec, p.kvPrec, ctx, p.adv);
        const cls = !mg ? 'none' : mg.G === 1 ? 'one' : mg.G <= hw.nodeGpus ? 'node' : 'multi';
        return h('td', { class: 'n cell ' + cls, title: unsupported ? `${hw.name}: ${E.PREC_LABEL[p.wPrec]} weights are not loadable on ${(ARCHS[hw.arch] || {}).name || 'this generation'}` : mg ? `${hw.name}: ${mg.G} (TP ${mg.tp} × PP ${mg.pp})` : `${hw.name}: no layout up to 64 × 16 loads it` }, unsupported ? '✕' : mg ? String(mg.G) : '—');
      }));
    });
    $('#compatTable').replaceChildren(h('div', { class: 'tw tall' }, h('table', { class: 'data compat' }, h('thead', null, head), h('tbody', null, body))));
  }

  /* ----- catalog ----- */
  function renderCatalog() {
    const hwRows = allHardware().map((hw) => [
      h('b', null, hw.name + (hw.approx ? ' ~' : '')), hw.vendor, fmtNum(hw.mem), fmtNum(hw.bw), hw.tflops.fp16 ? fmtNum(hw.tflops.fp16) : '—', hw.tflops.fp8 ? fmtNum(hw.tflops.fp8) : '—', hw.tflops.fp4 ? fmtNum(hw.tflops.fp4) : '—',
      `${(LINKS[hw.link] || LINKS.none).name}${hw.linkBw ? ' ' + fmtNum(hw.linkBw) + ' GB/s' : ''}`, fmtNum(hw.nodeGpus), hw.tdp != null ? fmtNum(hw.tdp) : '—', hw.price != null ? fmtMoney(hw.price) : '—',
      ARCHS[hw.arch] ? ARCHS[hw.arch].name : '—', ARCHS[hw.arch] ? ARCHS[hw.arch].native.map((x) => x.toUpperCase()).join(' ') : '—', ARCHS[hw.arch] ? ARCHS[hw.arch].weightOnly.map((x) => x.toUpperCase()).join(' ') : '—',
    ]);
    $('#catHardware').replaceChildren(table(['Accelerator', 'Vendor', 'Memory GB', 'GB/s', 'BF16 TFLOPS', 'FP8', 'FP4', 'Fabric', 'Per node', 'W', '$/h', 'Generation', 'Native formats', 'Weight-only'], hwRows, { numeric: [2, 3, 4, 5, 6, 8, 9, 10] }));
    const fams = ['ampere', 'ada', 'hopper', 'blackwell', 'blackwell-sm120', 'cdna3', 'cdna4', 'gaudi', 'tpu', 'other'];
    const engRows = [];
    for (const [k, e] of Object.entries(ENGINES)) {
      if (!e.weights) continue;
      for (const f of ['fp8', 'int8', 'int4', 'fp4']) engRows.push([h('b', null, e.name), E.PREC_LABEL[f]].concat(fams.map((fam) => { const v = (e.weights[f] || {})[fam]; return v ? h('span', { class: 'chip ' + (v === 'native' ? 'good' : 'info') }, v) : h('span', { class: 'chip bad' }, 'no'); })));
      engRows.push([h('b', null, e.name), 'KV cache'].concat(fams.map((fam) => { const ks = Object.entries(e.kvCache || {}).filter(([, l]) => l.includes(fam)).map(([d]) => d.toUpperCase()); return ks.length ? h('span', { class: 'chip good' }, ks.join(' ')) : h('span', { class: 'chip bad' }, 'BF16 only'); })));
    }
    $('#catEngines').replaceChildren(table(['Engine', 'Format'].concat(fams.map((f) => f.toUpperCase())), engRows), h('ul', { class: 'warnings', style: 'margin-top:10px' }, Object.values(ENGINES).filter((e) => e.docs).map((e) => h('li', null, h('b', null, `${e.name} (${e.version}, checked ${e.checked}${e.unverified ? ', not re-verified' : ''}): `), e.notes.join(' '), ' ', h('a', { href: e.docs.weights, target: '_blank', rel: 'noopener' }, 'source')))));
    const lint = catalogLint();
    $('#catLint').replaceChildren(...(lint.length ? [h('b', null, 'Consistency checks: '), h('ul', { class: 'warnings' }, lint.map((t) => h('li', null, h('span', { class: 'chip warn' }, 'check'), ' ', t)))] : [h('span', { class: 'chip good' }, 'ok'), " Every accelerator's TFLOPS columns agree with its chip generation."]));
    const mRows = allModels().map((m0) => {
      const m = E.norm(m0);
      const arch = m.attn.map((l) => `${l.n}× ${l.type}${l.window ? ' ' + fmtTok(l.window) : ''}`).join(' + ') + (m.sparse ? ` · sparse top-${m.sparse.topk}` : '');
      return [h('b', null, m.name + (m.approx ? ' ~' : '')), m.family, fmtNum(m.params, 1), m.moe ? fmtNum(m.active, 1) : '—', fmtNum(m.layers), `${m.nHeads} / ${m.attn.some((l) => l.type === 'mla') ? 'MLA' : m.nKv}`, fmtGB(E.kvPerTokenFull(m, 'bf16')), fmtGB(E.kvAtCtx(m, 'bf16', 1e6)), fmtTok(m.maxCtx) + (m.nativeCtx && m.nativeCtx < m.maxCtx ? ` (${fmtTok(m.nativeCtx)} native)` : ''), arch, m.nativePrec ? m.nativePrec.toUpperCase() : '—', m.hf ? h('a', { href: 'https://huggingface.co/' + m.hf, target: '_blank', rel: 'noopener' }, m.hf) : '—', m.variants ? h('span', null, ...Object.entries(m.variants).map(([k, r]) => h('span', { class: 'vlink' }, h('a', { href: 'https://huggingface.co/' + r, target: '_blank', rel: 'noopener', title: r }, k.toUpperCase()), ' '))) : '—'];
    });
    $('#catModels').replaceChildren(table(['Model', 'Family', 'Params B', 'Active B', 'Layers', 'Heads / KV heads', 'KV per token (BF16)', 'KV per 1M-token session', 'Max context', 'Attention layers', 'Ships in', 'Verified against', 'Known checkpoints'], mRows, { numeric: [2, 3, 4, 5, 6, 7, 8] }));
  }

  /* ---------- boot ---------- */
  function init() {
    populateSelects();
    bindInputs();
    syncInputs();
    render();
  }
  const start = (data) => {
    try { if (data && data.state) { state = Object.assign({}, DEFAULTS, data.state); state.adv = Object.assign({}, DEFAULTS.adv, state.adv || {}); } } catch (e) { /* ignore */ }
    init();
  };
  try { if (window.claude && window.claude.hot && window.claude.hot.snapshot) window.claude.hot.snapshot(() => ({ state })); } catch (e) { /* ignore */ }
  try {
    if (window.claude && window.claude.hot && window.claude.hot.ready) window.claude.hot.ready(start);
    else start((window.claude && window.claude.hot && window.claude.hot.data) || {});
  } catch (e) { init(); }
})();
