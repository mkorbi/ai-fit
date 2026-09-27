/* app.js - UI for AI Fit. Depends on catalog.js, glossary.js, engine.js and common.js (classic scripts, loaded before this one). */
(() => {
  'use strict';
  const E = Engine;
  const { fmtTok, fmtGB, fmtTime } = E;
  const { h, s, fmtNum, fmtMoney, fmtTokS, pct, plural, LS, store, ctxToSlider, sliderToCtx } = Common;
  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));

  /* ---------- glossary tooltips ---------- */
  const T = (key, text) => (GLOSSARY[key] ? h('span', { class: 'term', 'data-term': key, tabindex: 0 }, text) : document.createTextNode(text));
  function initTooltips() {
    const box = h('div', { id: 'tipbox', role: 'tooltip', hidden: true });
    document.body.append(box);
    let current = null, pinned = false;
    const fill = (key) => {
      const g = GLOSSARY[key]; if (!g) return false;
      box.replaceChildren(h('b', null, g.t), h('p', null, g.d), h('div', { class: 'tip-meta' },
        g.typ ? h('span', null, h('i', null, 'Typical'), ' ', g.typ) : null,
        g.lo ? h('span', null, h('i', null, 'Low'), ' ', g.lo) : null,
        g.hi ? h('span', null, h('i', null, 'High'), ' ', g.hi) : null));
      return true;
    };
    const place = (el) => {
      const r = el.getBoundingClientRect(); box.hidden = false;
      const bw = box.offsetWidth, bh = box.offsetHeight, vw = window.innerWidth, vh = window.innerHeight;
      let left = Math.min(Math.max(8, r.left), vw - bw - 8);
      let top = r.bottom + 8; if (top + bh > vh - 8) top = Math.max(8, r.top - bh - 8);
      box.style.left = left + 'px'; box.style.top = top + 'px';
    };
    const show = (el) => { if (!fill(el.dataset.term)) return; current = el; el.setAttribute('aria-describedby', 'tipbox'); place(el); };
    const hide = () => { if (current) current.removeAttribute('aria-describedby'); current = null; pinned = false; box.hidden = true; };
    document.addEventListener('mouseover', (e) => { const el = e.target.closest && e.target.closest('.term'); if (el && !pinned) show(el); });
    document.addEventListener('mouseout', (e) => { const el = e.target.closest && e.target.closest('.term'); if (el && !pinned && el === current) hide(); });
    document.addEventListener('focusin', (e) => { const el = e.target.closest && e.target.closest('.term'); if (el) { pinned = false; show(el); } });
    document.addEventListener('focusout', (e) => { const el = e.target.closest && e.target.closest('.term'); if (el && !pinned) hide(); });
    document.addEventListener('click', (e) => { const el = e.target.closest && e.target.closest('.term'); if (el) { if (pinned && el === current) hide(); else { show(el); pinned = true; } } else if (pinned) hide(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hide(); });
    window.addEventListener('scroll', () => { if (current && !pinned) hide(); }, true);
  }
  function renderGlossary() {
    const el = $('#glossary'); if (!el) return;
    const entries = Object.values(GLOSSARY).sort((a, b) => a.t.localeCompare(b.t));
    el.replaceChildren(...entries.map((g) => [h('dt', null, g.t), h('dd', null, g.d, g.typ ? h('div', { class: 'gmeta' }, h('i', null, 'Typical '), g.typ) : null, g.lo ? h('div', { class: 'gmeta' }, h('i', null, 'Low '), g.lo) : null, g.hi ? h('div', { class: 'gmeta' }, h('i', null, 'High '), g.hi) : null)]).flat());
  }

  /* ---------- state and catalogs ---------- */
  const allHardware = store.hardware, allModels = store.models;
  let state = Common.freshState(LS.get('cb.state', {}));
  const params = (overrides) => Common.buildParams(state, overrides);
  const run = (overrides) => Common.runPlan(state, overrides);

  /* ---------- controls ---------- */
  const BIND = [
    ['hw', 'hw', 'str'], ['count', 'count', 'int'], ['nodeGpus', 'nodeGpus', 'int'], ['link', 'link', 'str'], ['net', 'net', 'str'], ['hostRam', 'hostRam', 'num'],
    ['model', 'model', 'str'], ['engine', 'engine', 'str'], ['slots', 'slots', 'int'], ['wPrec', 'wPrec', 'str'], ['kvPrec', 'kvPrec', 'str'], ['par', 'par', 'str'], ['tp', 'tp', 'int'], ['pp', 'pp', 'int'], ['reps', 'reps', 'int'], ['dpAttn', 'dpAttn', 'bool'],
    ['users', 'users', 'int'], ['activityPreset', 'activityPreset', 'str'], ['activity', 'activity', 'num'], ['ctx', 'ctx', 'int'], ['prefix', 'prefix', 'int'], ['newPrompt', 'newPrompt', 'int'], ['output', 'output', 'int'], ['retention', 'retention', 'str'], ['target', 'target', 'num'], ['ttftMax', 'ttftMax', 'num'],
    ['prefixCache', 'prefixCache', 'bool'], ['spec', 'spec', 'bool'], ['specK', 'specK', 'int'], ['specAlpha', 'specAlpha', 'num'], ['pd', 'pd', 'bool'], ['allowCrossTp', 'allowCrossTp', 'bool'],
    ['compatCtx', 'compatCtx', 'int'],
    ['advUtil', 'adv.util', 'num'], ['advOverheadGB', 'adv.overheadGB', 'num'], ['advOverheadFrac', 'adv.overheadFrac', 'num'], ['advBwEff', 'adv.bwEff', 'num'], ['advMfu', 'adv.mfu', 'num'],
    ['advFrag', 'adv.frag', 'num'], ['advPpBubble', 'adv.ppBubble', 'num'], ['advCollEff', 'adv.collEff', 'num'], ['advHostRestore', 'adv.hostRestoreGBs', 'num'],
  ];
  const getPath = (obj, path) => path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
  const setPath = (obj, path, v) => { const ks = path.split('.'); let o = obj; for (const k of ks.slice(0, -1)) { if (o[k] == null) o[k] = {}; o = o[k]; } o[ks[ks.length - 1]] = v; };

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
  const SUPPORT_TXT = { native: 'native', 'weight-only': 'weight-only', unsupported: 'not loadable' };
  const engineOf = (id) => ENGINES[id] || ENGINES.none;
  function labelPrecOptions(hw) {
    const eng = ENGINES[state.engine] ? state.engine : 'none', e = engineOf(eng);
    for (const opt of $$('#engine option')) {
      const other = engineOf(opt.value);
      opt.textContent = other.name + (hw && E.engineCovers(hw, opt.value) === 'no' ? ` - not on ${((ARCHS[hw.arch] || {}).name || hw.name).replace(/ \(.*\)$/, '')}` : '');
    }
    for (const opt of $$('#wPrec option')) opt.textContent = hw ? `${E.precLabel(opt.value, eng)} - ${SUPPORT_TXT[E.formatSupport(hw, opt.value, eng)]}` : E.precLabel(opt.value, eng);
    for (const opt of $$('#kvPrec option')) opt.textContent = hw ? `${E.kvLabel(opt.value, eng)} - ${E.kvSupport(hw, opt.value, eng) === 'supported' ? 'supported' : 'not in ' + e.name}` : (eng !== 'none' && !(e.kvCache || {})[opt.value] && opt.value !== 'bf16' ? `${E.kvLabel(opt.value, eng)} - not in ${e.name}` : E.kvLabel(opt.value, eng));
    $('#engineHint').replaceChildren(...engineHint(eng, hw));
    syncEngineControls(eng);
  }
  const extLink = (url, text) => (url ? h('a', { href: url, target: '_blank', rel: 'noopener' }, text) : text);
  /* What the engine's docs say it supports, and how it serves requests, with the sources. */
  function engineHint(eng, hw) {
    const e = engineOf(eng);
    if (!e.docs) return [h('span', null, e.note || '')];
    const sv = E.servingOf(eng), src = sv.sources || {};
    const out = [h('span', null, `Per ${e.name} docs (${e.version}), checked ${e.checked}${e.unverified ? ', not re-verified' : ''}: `, extLink(e.docs.weights, 'weights'), ', ', extLink(e.docs.kv, 'KV cache'), '. The planner takes the weaker of engine support and silicon capability. ')];
    if (e.families) out.push(h('span', null, e.families.length === ENGINE_FAMILIES.length ? 'Runs on every accelerator family in the catalog' : `Runs on ${E.familyNames(e.families)}`, e.familiesSource ? [' (', extLink(e.familiesSource.url, 'source'), ')'] : null, '. '));
    if (sv.batching === 'slots') out.push(h('span', null, `${sv.slots.default} parallel request${sv.slots.default === 1 ? '' : 's'} per server by default (`, h('code', null, sv.slots.env), '), every slot reserving its full context when the model loads (', extLink(src.slots, 'config'), ', ', extLink(src.memory, 'FAQ'), '). '));
    else if (sv.maxBatch) {
      const mb = sv.maxBatch;
      const n = mb.value ? fmtNum(mb.value) : mb.tiers ? (hw ? fmtNum(E.maxBatchOf(sv, hw, 0, null)) : `${fmtNum(mb.tiers[0].value)} or ${fmtNum(mb.default)}`) : `${fmtNum(mb.min)} to ${fmtNum(mb.max)}`;
      out.push(h('span', null, `Continuous batching over a paged KV cache, at most ${n} concurrent requests per replica by default (`, h('code', null, mb.name), ', ', extLink(src.maxBatch, 'source'), '). '));
    }
    if (sv.pp === 'sequential') out.push(h('span', null, 'Spreads a model over the GPUs of one node by layers, one after another: no tensor parallelism (', extLink(src.split, 'source'), '). '));
    if (sv.prefixCache === 'per-slot') out.push(h('span', null, 'Reuses prompt prefixes within a slot, not across sessions. '));
    if (sv.hostCacheGB != null) out.push(h('span', null, `Parks idle sessions in a ${fmtGB(sv.hostCacheGB * 1e9)} host prompt cache per server (`, extLink(src.host, 'source'), '). '));
    if (sv.queueMax) out.push(h('span', null, `Queues ${sv.queueMax} requests, then rejects more (`, h('code', null, sv.queueEnv), ').'));
    return out;
  }
  const GATE_NAMES = { prefixCache: 'prefix sharing across sessions', spec: 'speculative decoding', pd: 'prefill/decode disaggregation', dpAttn: 'data-parallel attention' };
  /* Inputs for optimizations the engine lacks are disabled (the saved state keeps its value); the parallel-slots field
   * shows for slot engines only. */
  function syncEngineControls(eng) {
    const sv = E.servingOf(eng), name = engineOf(eng).name;
    $('#slotsField').hidden = sv.batching !== 'slots';
    const off = [];
    for (const [id, ok] of [['prefixCache', sv.prefixCache === true], ['spec', !!sv.spec], ['pd', !!sv.pd], ['dpAttn', !!sv.dpAttention]]) {
      const el = document.getElementById(id);
      el.disabled = !ok;
      if (!ok) off.push(GATE_NAMES[id]);
    }
    for (const opt of $$('#retention option')) opt.disabled = !sv.retention.includes(opt.value);
    const hint = $('#optHint');
    hint.hidden = !off.length;
    hint.textContent = off.length ? `${name} has no ${off.join(', ').replace(/, ([^,]*)$/, ' or $1')}; those switches count as off.${sv.prefixCache === 'per-slot' ? ' It reuses prompt prefixes within each slot on its own.' : ''}` : '';
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
    if (state.engine === 'ollama') return h('span', null, `Ollama loads GGUF files: pull the model from the Ollama library or a GGUF repository on Hugging Face in ${E.precLabel(prec, 'ollama').replace(' (GGUF)', '')}; AWQ, GPTQ and FP8 checkpoints do not apply.`);
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
  function afterChange(id, key, v) {
    for (const k of Common.applyChange(state, key, v)) {       // shared side effects (hardware → node size, model → DP attention, presets)
      const b = BIND.find(([, bk]) => bk === k), el = b && document.getElementById(b[0]);
      if (el) { if (el.type === 'checkbox') el.checked = !!state[k]; else el.value = state[k]; }
    }
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
        afterChange(id, key, v);
        scheduleRender();
      });
    }
    $('#ctxRange').addEventListener('input', (e) => { state.ctx = sliderToCtx(+e.target.value); $('#ctx').value = state.ctx; scheduleRender(); });
    $$('.tab').forEach((t) => t.addEventListener('click', () => { state.mode = t.dataset.mode; scheduleRender(); }));
    $('.tabs').addEventListener('keydown', (e) => {                // arrow keys move between tabs (WAI-ARIA tabs pattern)
      const tabs = $$('.tab'), i = tabs.indexOf(document.activeElement);
      if (i < 0) return;
      const j = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
      if (j == null) return;
      e.preventDefault();
      const t = tabs[(j + tabs.length) % tabs.length];
      t.focus(); state.mode = t.dataset.mode; scheduleRender();
    });
    $('#candidates').addEventListener('change', () => { state.candidates = $$('#candidates input').filter((c) => c.checked).map((c) => c.dataset.id); scheduleRender(); });
    $('#candAll').addEventListener('click', () => { state.candidates = allHardware().map((x) => x.id); syncInputs(); scheduleRender(); });
    $('#candNone').addEventListener('click', () => { state.candidates = []; syncInputs(); scheduleRender(); });
    $('#resetInputs').addEventListener('click', () => { state = Common.freshState({ mode: state.mode }); syncInputs(); scheduleRender(); });
    $('#addHw').addEventListener('click', () => addCustom('hardware'));
    $('#addModel').addEventListener('click', () => addCustom('model'));
    $('#exportJson').addEventListener('click', () => { $('#customJson').value = JSON.stringify({ hardware: allHardware(), models: allModels() }, null, 1); setMsg('Catalog JSON is in the box; copy it from there.'); });
    $('#resetCustom').addEventListener('click', () => { store.resetCustom(); populateSelects(); syncInputs(); setMsg('Custom entries removed.'); scheduleRender(); });
  }
  function setMsg(t, bad) { const el = $('#customMsg'); el.textContent = t; el.className = 'hint ' + (bad ? 'bad' : 'good'); }
  function addCustom(kind) {
    let obj;
    try { obj = JSON.parse($('#customJson').value); } catch (e) { setMsg('That is not valid JSON: ' + e.message, true); return; }
    const list = Array.isArray(obj) ? obj : [obj];
    const err = store.addCustom(kind, list);
    if (err) { setMsg(err, true); return; }
    populateSelects(); syncInputs(); setMsg(`Added ${list.length} ${kind} ${list.length === 1 ? 'entry' : 'entries'}. They live in this browser's storage.`); scheduleRender();
  }

  /* ---------- rendering ---------- */
  function render() {
    $$('.tab').forEach((t) => { const on = t.dataset.mode === state.mode; t.setAttribute('aria-selected', String(on)); t.tabIndex = on ? 0 : -1; });
    const arcade = $('#arcadeLink');
    arcade.hidden = !!window.claude;                               // the Claude artifact publishes the planner page only
    arcade.href = 'arcade/index.html#' + Common.encodePlan(state);
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
      tile('Concurrent requests supported', r.fits ? fmtNum(r.maxConc) : '0', r.fits ? `≈ ${fmtNum(r.maxUsers)} users at ${pct(wl.activity)} active · you asked for ${fmtNum(r.B)}${r.capLimit === 'slots' ? ` · capped by ${plural(r.slots.n, 'parallel slot')} per server` : r.capLimit === 'max-batch' ? ` · capped by ${r.sv.maxBatch.name} = ${fmtNum(r.maxBatch)} per replica` : ''}` : 'model does not load', r.fits && r.maxConc >= r.B ? 'good' : 'bad', true, 'concurrent'),
      tile('Context per session at your load', r.fits ? fmtTok(r.maxCtxAtLoad) : '-', `${fmtNum(r.sessionsPerRep)} sessions resident per replica · model max ${fmtTok(model.maxCtx)}`, r.maxCtxAtLoad >= wl.ctx ? 'good' : 'bad', false, 'ctx-at-load'),
      tile('Speed per user', a ? fmtTokS(a.perUser) + ' tok/s' : '-', a ? `target ≥ ${wl.target} · ${fmtTime(a.itl)} between tokens` : '', a && a.perUser >= wl.target && !a.saturated ? 'good' : 'bad', false, 'speed-per-user'),
      tile('Aggregate throughput', a ? fmtNum(r.aggTotal) + ' tok/s' : '-', r.costPerMTok != null ? `${fmtMoney(r.costPerMTok)} per 1M output tokens at ~${fmtMoney(r.price, 0)}/h` : (r.kW ? `${fmtNum(r.kW, 1)} kW of accelerators` : ''), null, false, 'aggregate'),
      tile('Time to first token', a ? fmtTime(r.ttft) : '-', a ? `limit ${fmtTime(wl.ttftMax)} · first turn from cold: ${fmtTime(r.ttftCold)} for ${fmtTok(r.coldNew)} tokens` : '', a ? (r.ttftOK ? 'good' : 'bad') : null, false, 'ttft'),
      tile('KV cache per session', fmtGB(r.kv.perSession), `${fmtGB(E.kvPerTokenFull(model, p.kvPrec, p.engine))}/token · pool ${fmtGB(r.kvAvail)} per replica`, null, false, 'kv-per-session'),
    ];
    $('#kpis').replaceChildren(...tiles);

    $('#configBody').replaceChildren(...configBody(r, p, tried));
    $('#bottleneckBody').replaceChildren(...bottleneckBody(r, p));
    renderCapacityChart(r, p);
    renderSpeedChart(r, p);
    renderLedger(r, p);
    renderWarnings(r);
  }
  function tile(label, value, sub, status, hero, term) {
    return h('div', { class: 'kpi' + (hero ? ' hero' : '') + (status ? ' ' + status : '') }, h('div', { class: 'l' }, term ? T(term, label) : label), h('div', { class: 'v' }, value), h('div', { class: 's' }, sub));
  }
  function verdictContent(r, p) {
    const wl = p.wl, model = r.model, layout = `${r.total}× ${r.hw.name}`;
    const who = `${fmtNum(wl.users)} users (${fmtNum(r.B)} concurrent at ${pct(wl.activity)} active)`;
    const eng = engineOf(p.engine);
    if (!r.loadable) {
      const why = r.warnings.find((w) => w.level === 'crit' && w.text !== (r.warnings.find((x) => /exceeds the model's maximum/.test(x.text)) || {}).text);
      return [
        h('strong', null, `${model.name} in ${E.precLabel(p.wPrec, p.engine)} cannot run on ${layout}${p.engine !== 'none' ? ' with ' + eng.name : ''}.`),
        h('p', null, why ? why.text : 'The engine has no kernels for this format on this hardware.'),
      ];
    }
    if (r.slots && !r.slots.fit && r.memSessions >= 1) {
      return [
        h('strong', null, `${r.slots.n} parallel slots of ${fmtTok(wl.ctx)} tokens do not fit next to ${model.name} on ${layout}.`),
        h('p', null, `${eng.name} reserves every slot's full context when the model loads: ${fmtGB(r.slots.reserved)} of KV cache, but ${fmtGB(r.kvAvail)} is free after the weights. Lower `, h('code', null, r.slots.env), ` to ${r.memSessions} or shorten the context.`),
      ];
    }
    if (!r.fits) {
      const mg = E.minGpus(r.hw, model, p.wPrec, p.kvPrec, Math.min(wl.ctx, 32768), p.adv, { engine: p.engine, slots: p.slots });
      return [
        h('strong', null, `${model.name} in ${E.precLabel(p.wPrec, p.engine)} does not load on ${layout}.`),
        h('p', null, `Weights take ${fmtGB(r.W)}; ${r.G} accelerators offer ${fmtGB(Math.max(0, r.G * r.capPerGpu))} after runtime overhead. `, mg ? `The smallest layout that loads it here is ${mg.G} accelerators (TP ${mg.tp} × PP ${mg.pp}) for one 32k session; lower the weight precision or pick bigger memory.` : 'No layout up to 64-way tensor × 16-stage pipeline parallelism loads it on this hardware.'),
      ];
    }
    if (r.slotLimited) {
      return [
        h('strong', null, `${eng.name} serves ${r.slots.n * r.R} request${r.slots.n * r.R === 1 ? '' : 's'} at a time on ${layout}; ${who} need ${fmtNum(r.B)}.`),
        h('p', null, `Each ${eng.name} server runs ${r.slots.n} parallel slot${r.slots.n === 1 ? '' : 's'} (`, h('code', null, r.slots.env), `); the other requests wait in its queue. Every slot reserves ${fmtTok(wl.ctx)} tokens of KV cache, so more slots need memory for their full context. A continuous-batching engine such as vLLM shares the same memory between all requests.`,
          !r.ttftOK ? ` Even the requests in a slot wait ${fmtTime(r.ttft)} for their first token (limit ${fmtTime(wl.ttftMax)}): ${fmtTok(r.warmNew)} tokens are prefilled on one accelerator per turn.` : r.at && r.at.perUser < wl.target ? ` Even the requests in a slot get ${fmtTokS(r.at.perUser)} tok/s (target ${wl.target}).` : ''),
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
    const segs = [['Weights', wPer, 'var(--s1)', 'weights-mem'], ['KV cache pool', kvPer, 'var(--s3)', 'kv-pool'], ['Runtime overhead', ovh, 'var(--muted)', 'overhead'], ['Headroom (unused by the engine)', head, 'var(--line)', 'headroom']];
    const bar = h('div', { class: 'membar', role: 'img', 'aria-label': `Memory per accelerator: weights ${fmtGB(wPer)}, KV cache ${fmtGB(kvPer)}, overhead ${fmtGB(ovh)}, headroom ${fmtGB(head)}` });
    for (const [name, val, color] of segs) if (val > 0) bar.append(h('div', { class: 'seg', style: `flex:${val};background:${color}`, title: `${name}: ${fmtGB(val)}` }));
    const key = h('ul', { class: 'key' }, segs.map(([name, val, color, term]) => h('li', null, h('i', { style: `background:${color}` }), h('span', { class: 't' }, T(term, name)), h('b', null, fmtGB(Math.max(0, val))))));
    const tail = ` = ${r.total} accelerators on ${plural(r.nodes, 'node')}` + (r.count - r.total > 0 ? `, ${r.count - r.total} idle` : '') + (r.pdTotal ? `, plus ${r.pdTotal} in a prefill pool` : '');
    const unit = r.slots ? `${engineOf(p.engine).name} server` : 'replica';
    const lead = r.seqPP
      ? (r.pp > 1 ? [T('layer-split', 'Layer split'), ` over ${plural(r.pp, 'accelerator')} × `] : [`1 accelerator per ${unit} × `]).concat([T('replicas', plural(r.R, unit)), tail])
      : [T('tp', 'TP'), ` ${r.tp} × `, T('pp', 'PP'), ` ${r.pp} × `, T('replicas', plural(r.R, unit)), tail];
    const tokens = r.kvAvail / Math.max(1, E.kvPerTokenFull(r.model, p.kvPrec, p.engine));
    const out = [
      h('p', { class: 'lead' }, ...lead),
      h('p', { class: 'hint' }, `Per accelerator (${hw.mem} GB): ${fmtGB(wPer)} weights, ${fmtGB(kvPer)} KV cache. The ${unit}'s KV pool holds ${fmtTok(tokens)} tokens${r.kvRepl > 1 ? ` (KV replicated ${r.kvRepl}×)` : ''}. `,
        r.slots && r.fits ? `${plural(r.slots.n, 'slot')} of ${fmtTok(r.C)} tokens ${r.slots.n === 1 ? 'reserves' : 'reserve'} ${fmtGB(r.slots.reserved)} of it. ` : '',
        r.S > 0 ? `Shared prefix of ${fmtTok(r.S)} tokens stored once per replica (${fmtGB(r.kv.shared)}).` : ''),
      bar, key,
    ];
    if (tried && tried.length > 1) {
      const rows = tried.slice().sort((x, y) => (y.maxUsers - x.maxUsers) || ((y.at ? y.at.perUser : 0) - (x.at ? x.at.perUser : 0))).slice(0, 5);
      out.push(h('h4', null, 'Layouts considered'), table(
        [{ t: 'Layout', k: 'layout-col' }, { t: 'Concurrent', k: 'concurrent' }, 'Users', { t: 'tok/s per user', k: 'speed-per-user' }, { t: 'Context at load', k: 'ctx-at-load' }, 'Idle'],
        rows.map((t) => [h('span', { class: 'nowrap' }, `TP ${t.tp} × PP ${t.pp} × ${t.R}` + (t === r ? ' ◂' : '')), t.fits ? fmtNum(t.maxConc) : 'no fit', t.fits ? fmtNum(t.maxUsers) : '-', t.at ? fmtTokS(t.at.perUser) : '-', t.fits ? fmtTok(t.maxCtxAtLoad) : '-', fmtNum(t.count - t.total)]),
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
      [T('decode-step', 'Decode step'), fmtTime(a.step), `${a.bound}-bound at ${fmtNum(r.bAt)} concurrent per replica`],
      [T('mem-traffic', 'Memory traffic'), fmtTime(a.tBw), `${fmtGB(a.wRead)} of weights + ${fmtGB(a.kvRead)} of KV cache per step at ${fmtGB(eff)}/s effective`],
      [T('compute', 'Compute'), fmtTime(a.tComp), `${r.pk.used} matmuls at ${pct(p.adv.mfu)} of ${fmtNum(r.pk.peak / 1e12)} TFLOPS × ${r.computeGpus}${r.seqPP && r.pp > 1 ? ' (layer split: one accelerator at a time)' : ''}`],
      [T('collectives', 'Collectives'), fmtTime(a.tComm), r.tp > 1 ? `${2 * r.model.layers} all-reduces per token over ${linkName}` : 'no tensor parallelism'],
      [T('prefill-share', 'Prefill share'), pct(a.f) + (a.saturated ? ' (saturated)' : ''), `${a.lambda.toFixed(2)} requests/s per replica, ${fmtTok(r.warmNew)} new tokens each${r.effective.opt.pd ? ' (in the prefill pool)' : ', interleaved with decode'}`],
      [T('ttft', 'Warm turn TTFT'), fmtTime(r.ttftWarm), r.effective.retention === 'host' ? `includes ${fmtTime(r.restoreS)} to restore ${fmtGB(r.kv.perSession)} from host memory` : `${fmtTok(r.warmNew)} tokens prefilled`],
      [T('ttft', 'Cold TTFT'), fmtTime(r.ttftCold), `${fmtTok(r.coldNew)} tokens prefilled on ${r.tp} accelerators`],
    ];
    return [table(['What', 'Time', 'Why'], rows, { numeric: [1] })];
  }
  /* rows: arrays of cells, or { cells, cls, sub } where `sub` is a note shown in a full-width row below; { group } starts a
   * new row group under a full-width heading. */
  function table(cols, rows, opts) {
    const numeric = new Set((opts && opts.numeric) || []);
    const bodies = [];
    for (const row of rows) {
      if (row.group || !bodies.length) bodies.push(h('tbody'));
      const body = bodies[bodies.length - 1];
      if (row.group) { body.append(h('tr', { class: 'group' }, h('th', { colspan: cols.length, scope: 'rowgroup' }, row.group))); continue; }
      body.append(h('tr', { class: [row.cls, row.sub ? 'with-note' : null].filter(Boolean).join(' ') || null }, (row.cells || row).map((c, i) => h('td', { class: numeric.has(i) ? 'n' : 't' }, c))));
      if (row.sub) body.append(h('tr', { class: [row.cls, 'note-row'].filter(Boolean).join(' ') }, h('td', { class: 't', colspan: cols.length }, h('div', { class: 'note' }, row.sub))));
    }
    return h('div', { class: 'tw' }, h('table', { class: ['data', opts && opts.cls].filter(Boolean).join(' ') },
      h('thead', null, h('tr', null, cols.map((c, i) => h('th', { class: numeric.has(i) ? 'n' : null }, typeof c === 'string' || c.nodeType ? c : T(c.k, c.t))))),
      bodies));
  }

  /* ----- charts ----- */
  function evalAt(p, r, overrides) {
    const q = Object.assign({}, p, overrides || {});
    if (overrides && overrides.wl) q.wl = Object.assign({}, p.wl, overrides.wl);
    return E.evaluate(Object.assign(q, { tp: r.tp, pp: r.pp, replicas: r.R }));
  }
  /* Up to three smaller-format variants the engine can run on this hardware: a smaller KV cache, the smallest KV cache,
   * and smaller weights with the smaller KV cache. */
  function variantLabel(p) {
    const hw = p.hw, eng = p.engine;
    const kvOk = (k) => E.kvSupport(hw, k, eng) === 'supported', wOk = (w) => E.formatSupport(hw, w, eng) !== 'unsupported';
    const kvBytes = (k) => E.bytesKv(k, eng), wBytes = (w) => E.bytesW(w, eng) ?? E.BYTES_W[w];
    const smallerKv = E.KV_PRECS.filter((k) => kvOk(k) && kvBytes(k) < kvBytes(p.kvPrec)).sort((a, b) => kvBytes(b) - kvBytes(a));
    const list = [];
    if (smallerKv[0]) list.push({ name: `KV cache ${E.kvLabel(smallerKv[0], eng)}`, o: { kvPrec: smallerKv[0] } });
    if (smallerKv.length > 1) list.push({ name: `KV cache ${E.kvLabel(smallerKv[smallerKv.length - 1], eng)}`, o: { kvPrec: smallerKv[smallerKv.length - 1] } });
    // prefer formats the hardware computes natively, then the smallest weight-only one
    const smallerW = E.W_PRECS.filter((w) => wOk(w) && wBytes(w) < wBytes(p.wPrec)).sort((a, b) => (E.formatSupport(hw, b, eng) === 'native') - (E.formatSupport(hw, a, eng) === 'native') || wBytes(b) - wBytes(a));
    if (smallerW[0]) { const kv = smallerKv[0] || p.kvPrec; list.push({ name: `Weights ${E.precLabel(smallerW[0], eng).split(' ')[0]} + KV ${E.kvLabel(kv, eng)}`, o: { wPrec: smallerW[0], kvPrec: kv } }); }
    return list.slice(0, 3);
  }
  function renderCapacityChart(r, p) {
    const el = $('#chartCapacity');
    if (!r.fits) { el.replaceChildren(h('p', { class: 'hint' }, 'The frontier needs a layout that loads the model.')); return; }
    const model = r.model, gpu = r.effective.retention === 'gpu';
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
    const spec = r.effective.opt.spec;
    const series = [{ name: spec ? 'With speculative decoding' : 'As configured', color: 'var(--s1)', points: xs.map((b) => [b, r.loadAt(b, spec).perUser]) }];
    if (r.sv.spec) series.push(spec ? { name: 'Without speculative decoding', color: 'var(--s2)', points: xs.map((b) => [b, r.loadAt(b, false).perUser]) } : { name: 'With speculative decoding', color: 'var(--s2)', points: xs.map((b) => [b, r.loadAt(b, true).perUser]) });
    lineChart({
      el, title: `Speed per user as one ${r.slots ? 'server' : 'replica'} (${plural(r.G, 'accelerator')}) takes more concurrent requests at ${fmtTok(p.wl.ctx)} context`,
      x: { label: 'concurrent requests per replica', log: true, fmt: (v) => fmtNum(v), ticks: countTicks }, y: { label: 'tokens per second per user', log: true, fmt: (v) => fmtTokS(v) },
      series, refY: [{ y: p.wl.target, label: 'target' }], marker: { x: r.bAt, y: r.at.perUser, ok: r.speedOK, label: 'your load' },
    });
  }
  function tokenTicks(min, max) { const out = []; const step = Math.log2(max / min) > 8 ? 4 : 2; for (let v = 1024; v <= max * 1.0001; v *= step) if (v >= min) out.push(v); return out; }
  function countTicks(min, max) { const out = []; const dec = Math.log10(max / min); for (let e = Math.floor(Math.log10(min)); Math.pow(10, e) <= max; e++) for (const m of (dec > 2.5 ? [1] : [1, 2, 5])) { const v = m * Math.pow(10, e); if (v >= min && v <= max) out.push(v); } return out; }
  function logTicks(min, max) { const out = []; const dec = Math.log10(max / min); for (let e = Math.floor(Math.log10(min)); Math.pow(10, e) <= max * 1.0001; e++) for (const m of (dec > 3 ? [1] : dec > 1.5 ? [1, 3] : [1, 2, 5])) { const v = m * Math.pow(10, e); if (v >= min * 0.999 && v <= max * 1.001) out.push(v); } return out; }

  function lineChart(o) {
    // drawn at the container's own pixel width (and redrawn when the page width changes), so text keeps its size on any screen
    const W = Math.max(300, Math.round(o.el.clientWidth || 720)), H = Math.round(Math.min(380, Math.max(300, W * 0.42)));
    const xs = o.series[0].points.map(([x]) => x);
    const valid = o.series.flatMap((sr) => sr.points).filter(([x, y]) => x > 0 && y > 0 && isFinite(y));
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    let yMax = Math.max(1, ...valid.map(([, y]) => y), ...(o.refY || []).map((r) => r.y), o.marker && o.marker.y > 0 ? o.marker.y : 0);
    let yMin = Math.min(1, ...valid.map(([, y]) => y), o.marker && o.marker.y > 0 ? o.marker.y : 1);
    yMin = Math.pow(10, Math.floor(Math.log10(Math.max(1e-2, yMin)))); yMax = Math.pow(10, Math.ceil(Math.log10(yMax * 1.05)));
    if (yMax <= yMin) yMax = yMin * 10;
    const yTicks = logTicks(yMin, yMax);
    const m = { l: 34 + Math.ceil(7.2 * Math.max(1, ...yTicks.map((t) => String(o.y.fmt(t)).length))), r: 22, t: 20, b: 48 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const lx = Math.log(xMin), ux = Math.log(xMax), ly = Math.log(yMin), uy = Math.log(yMax);
    const sx = (x) => m.l + (Math.log(x) - lx) / (ux - lx) * pw;
    const sy = (y) => m.t + ph - (Math.log(y) - ly) / (uy - ly) * ph;
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': o.title });
    svg.append(s('rect', { x: 0, y: 0, width: W, height: H, fill: 'var(--chart-surface)' }));
    for (const t of yTicks) { svg.append(s('line', { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), stroke: 'var(--grid)', 'stroke-width': 1 })); svg.append(s('text', { x: m.l - 8, y: sy(t), 'text-anchor': 'end', 'dominant-baseline': 'middle' }, o.y.fmt(t))); }
    svg.append(s('line', { x1: m.l, x2: W - m.r, y1: m.t + ph, y2: m.t + ph, stroke: 'var(--axis)', 'stroke-width': 1 }));
    let labelEnd = -Infinity;                     // on narrow charts some tick labels give way so none overlap
    for (const t of o.x.ticks(xMin, xMax)) {
      svg.append(s('line', { x1: sx(t), x2: sx(t), y1: m.t + ph, y2: m.t + ph + 4, stroke: 'var(--axis)' }));
      const label = String(o.x.fmt(t)), half = label.length * 3.6;
      if (sx(t) - half < labelEnd + 8) continue;
      svg.append(s('text', { x: sx(t), y: m.t + ph + 16, 'text-anchor': 'middle' }, label));
      labelEnd = sx(t) + half;
    }
    svg.append(s('text', { x: m.l + pw / 2, y: H - 8, 'text-anchor': 'middle', class: 'axis-title' }, o.x.label));
    svg.append(s('text', { x: 14, y: m.t + ph / 2, 'text-anchor': 'middle', transform: `rotate(-90 14 ${m.t + ph / 2})`, class: 'axis-title' }, o.y.label));
    const labels = [], markerRight = !!o.marker && o.marker.x > Math.sqrt(xMin * xMax);
    for (const rx of o.refX || []) { if (rx.x < xMin || rx.x > xMax) continue; svg.append(s('line', { x1: sx(rx.x), x2: sx(rx.x), y1: m.t, y2: m.t + ph, stroke: 'var(--axis)', 'stroke-width': 1 })); labels.push(s('text', { x: sx(rx.x) - 4, y: m.t + 10, 'text-anchor': 'end', class: 'ref' }, rx.label)); }
    for (const ry of o.refY || []) { if (ry.y < yMin || ry.y > yMax) continue; svg.append(s('line', { x1: m.l, x2: W - m.r, y1: sy(ry.y), y2: sy(ry.y), stroke: 'var(--axis)', 'stroke-width': 1 })); labels.push(s('text', { x: markerRight ? m.l + 4 : W - m.r - 4, y: sy(ry.y) - 5, 'text-anchor': markerRight ? 'start' : 'end', class: 'ref' }, `${ry.label} ${o.y.fmt(ry.y)}`)); }
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
      // the label sits right of the dot, left of it near the right edge, and below it when a reference line (labelled above) is close
      const mx = sx(o.marker.x), left = mx > m.l + pw * 0.75;
      const ref = (o.refY || []).find((ry) => ry.y >= yMin && ry.y <= yMax && Math.abs(sy(ry.y) - sy(my)) < 16);
      labels.push(s('text', { x: left ? mx - 12 : mx + 10, y: ref ? Math.max(sy(my), sy(ref.y)) + 18 : sy(my) + 4, 'text-anchor': left ? 'end' : 'start', class: 'ref' }, o.marker.label));
    }
    svg.append(...labels);
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
    const hw = p.hw, st = state, eng = ENGINES[st.engine] ? st.engine : 'none', en = engineOf(eng), sv = E.servingOf(eng);
    const kvOk = (k) => E.kvSupport(hw, k, eng) === 'supported', wSup = (w, e = eng) => E.formatSupport(hw, w, e);
    const kvBytes = (k, e = eng) => E.bytesKv(k, e), wBytes = (w, e = eng) => E.bytesW(w, e) ?? E.BYTES_W[w];
    const nearest = (list, cur, ok, bytes) => list.filter(ok).sort((a, b) => Math.abs(bytes(a) - bytes(cur)) - Math.abs(bytes(b) - bytes(cur)))[0] || cur;
    const rows = [{ name: 'As configured', term: null, r: base, o: null, note: `${E.precLabel(st.wPrec, eng)} weights, ${E.kvLabel(st.kvPrec, eng)} KV${base.effective.opt.spec ? ', speculative decoding' : ''}${base.effective.opt.pd ? ', prefill/decode split' : ''}${base.slots ? `, ${plural(base.slots.n, 'parallel slot')}` : ''}` }];
    const add = (name, o, note, term) => rows.push({ name, term, r: run(o).best, o, note });
    const eight = ['fp8', 'int8'].find(kvOk) || 'fp8';
    if (kvBytes(st.kvPrec) > kvBytes(eight)) add(`KV cache in ${E.kvLabel(eight, eng)}`, { kvPrec: eight }, 'Half the KV bytes per token; usually no measurable quality loss' + (kvOk(eight) ? '' : `; not offered by ${en.name} here`), 'opt-kv-fp8');
    if (st.kvPrec !== 'int4') add(`KV cache in ${E.kvLabel('int4', eng)}`, { kvPrec: 'int4' }, 'A quarter of the KV bytes; quality risk grows with context' + (kvOk('int4') ? '' : `; not offered by ${en.name} here (Ollama q4_0 and LMDeploy have it)`), 'opt-kv-int4');
    const model = E.norm(p.model);
    const ckpt = (prec) => { if (eng === 'ollama') return '; GGUF from the Ollama library'; const c = checkpointFor(model, prec); return c ? `; checkpoint: ${c.org} (${c.kind})` : '; no known checkpoint'; };
    const w8 = ['fp8', 'int8'].find((w) => wSup(w) !== 'unsupported');
    if (st.wPrec === 'bf16' && w8) add(`Weights in ${E.precLabel(w8, eng)}`, { wPrec: w8 }, (wSup(w8) === 'native' ? `Native ${w8.toUpperCase()} here: halves weight bytes and speeds up decode` : 'Weight-only on this accelerator: saves memory, compute stays BF16') + ckpt(w8), 'opt-w-fp8');
    const w4 = wSup('fp4') === 'native' ? 'fp4' : wSup('int4') !== 'unsupported' ? 'int4' : null;
    if (w4 && !['int4', 'fp4'].includes(st.wPrec)) add(`Weights in ${E.precLabel(w4, eng)}`, { wPrec: w4 }, (w4 === 'fp4' ? 'Native FP4: a quarter of the weight bytes and faster prefill' : 'Weight-only 4-bit: about a quarter of the weight bytes, compute stays BF16') + ckpt(w4), 'opt-w-4bit');
    if (!st.prefixCache && st.prefix > 0 && sv.prefixCache === true) add('Cache the shared prefix', { prefixCache: true }, `The ${fmtTok(st.prefix)}-token shared prefix is stored once per replica`, 'prefix-caching');
    if (!st.spec && sv.spec) add('Speculative decoding', { spec: true }, `Draft ${st.specK} tokens per step at ${st.specAlpha}% acceptance; helps when memory-bound${eng === 'ollama' ? '; needs a draft model in the Modelfile' : ''}`, 'spec');
    const hostNote = sv.hostCacheGB != null ? `; ${en.name} caches up to ${fmtGB(sv.hostCacheGB * 1e9)} per server` : '';
    if (base.effective.retention === 'gpu' && sv.retention.includes('host')) add('Park idle sessions in host memory', { retention: 'host' }, 'GPU memory holds only in-flight requests; idle KV restores over PCIe' + hostNote, 'opt-host');
    if (base.effective.retention === 'none' && sv.retention.includes('host')) add('Keep idle sessions in host memory', { retention: 'host' }, 'Skips re-prefilling the whole conversation on every turn' + hostNote, 'opt-host');
    if (!st.pd && sv.pd) add('Disaggregate prefill from decode', { pd: true }, 'A separate prefill pool keeps token speed steady; needs KV transfer over the network', 'pd');
    if (base.slots) {
      add(`${base.slots.n * 2} parallel slots`, { slots: base.slots.n * 2 }, `Doubles ${sv.slots.env}; every slot reserves its own ${fmtTok(st.ctx)}-token context`, 'slots');
      const vw = nearest(E.W_PRECS, st.wPrec, (w) => wSup(w, 'vllm') !== 'unsupported', (w) => wBytes(w, 'vllm'));
      const vk = nearest(E.KV_PRECS, st.kvPrec, (k) => E.kvSupport(hw, k, 'vllm') === 'supported', (k) => kvBytes(k, 'vllm'));
      add('Same hardware with vLLM', { engine: 'vllm', wPrec: vw, kvPrec: vk }, `Continuous batching over a paged KV cache: every request shares the memory (${E.precLabel(vw, 'vllm')} weights, ${E.kvLabel(vk, 'vllm')} KV)`, 'continuous-batching');
    }
    const combo = {};
    if (kvBytes(st.kvPrec) > kvBytes(eight) && kvOk(eight)) combo.kvPrec = eight;
    if (st.wPrec === 'bf16' && w8) combo.wPrec = wSup(w8) === 'native' ? w8 : (wSup('int4') !== 'unsupported' ? 'int4' : w8);
    if (!st.spec && sv.spec) combo.spec = true;
    if (!st.prefixCache && st.prefix > 0 && sv.prefixCache === true) combo.prefixCache = true;
    if (Object.keys(combo).length > 1) add('All of the above (memory + speed)', combo, 'The combination most production stacks run');
    const b = base;
    const cells = rows.map((row) => {
      const r = row.r, a = r.at;
      const verdict = !r.loadable ? ['not supported', 'bad'] : !r.fits ? ['out of memory', 'bad'] : r.slotLimited ? ['slots full', 'bad'] : !r.memOK ? ['not enough KV', 'bad'] : !r.speedOK ? [!r.ttftOK ? 'slow first token' : a && a.saturated ? 'prefill-bound' : 'too slow', 'warn'] : ['meets target', 'good'];
      const delta = (v, bv) => bv > 0 && v !== bv ? h('small', { class: v > bv ? 'up' : 'down' }, ` ${v > bv ? '+' : ''}${Math.round((v / bv - 1) * 100)}%`) : null;
      return {
        cls: row.r === base ? 'base' : null,
        sub: row.note,
        cells: [
          h('b', null, row.term ? T(row.term, row.name) : row.name),
          r.fits ? h('span', null, fmtNum(r.maxConc), delta(r.maxConc, b.maxConc)) : '0',
          r.fits ? fmtNum(r.maxUsers) : '0',
          r.fits ? h('span', null, fmtTok(r.maxCtxAtLoad), delta(r.maxCtxAtLoad, b.maxCtxAtLoad)) : '-',
          a ? h('span', null, fmtTokS(a.perUser), delta(a.perUser, b.at ? b.at.perUser : 0)) : '-',
          a ? fmtTime(r.ttft) : '-',
          r.costPerMTok != null ? fmtMoney(r.costPerMTok) : '-',
          h('span', { class: 'chip ' + verdict[1] }, verdict[0]),
          row.o ? h('button', { class: 'ghost', type: 'button', onclick: () => { Object.assign(state, row.o); syncInputs(); scheduleRender(); } }, 'Apply') : '',
        ],
      };
    });
    el.replaceChildren(table(['Option', { t: 'Concurrent', k: 'concurrent' }, 'Users', { t: 'Context at load', k: 'ctx-at-load' }, { t: 'tok/s per user', k: 'speed-per-user' }, { t: 'TTFT', k: 'ttft' }, { t: '$ / 1M tok', k: 'cost-token' }, 'Verdict', ''], cells, { numeric: [1, 2, 3, 4, 5, 6] }));
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
    $('#modelHint').textContent = `${fmtNum(model.params, 1)}B params${model.moe ? `, ${fmtNum(model.active, 1)}B active` : ''} · KV ${fmtGB(E.kvPerTokenFull(model, p.kvPrec, p.engine))}/token in ${E.kvLabel(p.kvPrec, p.engine)} · max context ${fmtTok(model.maxCtx)}${model.nativePrec ? ' · ships in ' + model.nativePrec.toUpperCase() : ''}`;
    labelPrecOptions(null);
    $('#precHint').replaceChildren(checkpointNode(model, p.wPrec));
    const list = allHardware().filter((hw) => state.candidates.includes(hw.id));
    const B = Math.max(1, Math.round(p.wl.users * p.wl.activity));
    const sessionKv = E.kvAtCtx(model, p.kvPrec, p.wl.ctx, p.engine);
    $('#revIntro').textContent = `Smallest layout per accelerator type that serves ${fmtNum(p.wl.users)} users (${fmtNum(B)} concurrent at ${pct(p.wl.activity)} active) of ${model.name} at ${fmtTok(p.wl.ctx)} context with at least ${p.wl.target} tok/s each and a first token within ${fmtTime(p.wl.ttftMax)}${p.engine !== 'none' ? `, served by ${engineOf(p.engine).name}` : ''}. Session KV: ${fmtGB(sessionKv)} in ${E.kvLabel(p.kvPrec, p.engine)}; all sessions: ${fmtGB(sessionKv * (E.effective(p).retention === 'gpu' ? p.wl.users : B))}.`;
    if (!list.length) { $('#revSummary').replaceChildren(); $('#revTable').replaceChildren(h('p', { class: 'hint' }, 'Pick at least one candidate accelerator in the left rail.')); return; }
    const results = E.reverse(p, list);
    const feasible = results.filter((r) => !r.infeasible);
    const fewest = feasible[0];
    const cheapest = feasible.filter((r) => r.price != null).sort((a, b) => a.price - b.price)[0];
    const perTok = feasible.filter((r) => r.costPerMTok != null).sort((a, b) => a.costPerMTok - b.costPerMTok)[0];
    const sum = [];
    if (fewest) sum.push(tile('Fewest accelerators', `${fewest.gpusUsed}× ${fewest.hw.name}`, `${plural(fewest.nodes, 'node')} · ${fmtTokS(fewest.at.perUser)} tok/s per user`, 'good', true));
    if (cheapest) sum.push(tile('Lowest hourly cost', `${fmtMoney(cheapest.price, 0)}/h`, `${cheapest.gpusUsed}× ${cheapest.hw.name}`, null, false, 'cost-hour'));
    if (perTok) sum.push(tile('Lowest cost per token', `${fmtMoney(perTok.costPerMTok)} / 1M`, `${perTok.gpusUsed}× ${perTok.hw.name} at ${fmtNum(perTok.aggTotal)} tok/s`, null, false, 'cost-token'));
    if (!feasible.length) sum.push(tile('No candidate works', '-', 'see the reasons below', 'bad', true));
    $('#revSummary').replaceChildren(...sum);
    const rows = results.map((r) => r.infeasible
      ? { cls: 'muted', cells: [h('b', null, r.hw.name), '-', '-', '-', '', '', '', '', '', '', ''], sub: r.reason }
      : { sub: r.warnings.filter((w) => w.level === 'warn').map((w) => w.text).join(' ') || null, cells: [
        h('b', null, r.hw.name),
        h('b', null, fmtNum(r.gpusUsed)), fmtNum(r.nodes), [h('span', { class: 'nowrap' }, `TP ${r.tp} × PP ${r.pp} × ${r.R}`), r.pdTotal ? ` + ${r.pdTotal} prefill` : ''],
        fmtTokS(r.at.perUser), fmtTime(r.ttft), fmtGB(r.kvAvail * r.R), r.price != null ? fmtMoney(r.price, 0) : '-', r.costPerMTok != null ? fmtMoney(r.costPerMTok) : '-', r.kW != null ? fmtNum(r.kW, 1) : '-',
        h('button', { class: 'ghost', type: 'button', onclick: () => openInPlanner(r) }, 'Open'),
      ] });
    $('#revTable').replaceChildren(table([{ t: 'Accelerator', k: 'accelerator' }, 'Count', { t: 'Nodes', k: 'nodes' }, { t: 'Layout', k: 'layout-col' }, { t: 'tok/s per user', k: 'speed-per-user' }, { t: 'TTFT', k: 'ttft' }, { t: 'KV pool', k: 'kv-pool' }, { t: '$ / hour', k: 'cost-hour' }, { t: '$ / 1M tok', k: 'cost-token' }, { t: 'kW', k: 'kw' }, ''], rows, { numeric: [1, 2, 4, 5, 6, 7, 8, 9] }));
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
    const sv = E.servingOf(p.engine), room = sv.batching === 'slots' ? `${plural(p.slots > 0 ? p.slots : sv.slots.default, 'parallel slot')} of ${fmtTok(ctx)} tokens` : `one ${fmtTok(ctx)}-token session`;
    $('#compatIntro').textContent = `Each cell is the fewest accelerators of that type on which the model loads with ${E.precLabel(p.wPrec, p.engine)} weights and still has room for ${room} (${E.kvLabel(p.kvPrec, p.engine)} KV cache), using the best ${sv.pp === 'sequential' ? 'layer split' : 'tensor × pipeline layout'}${p.engine !== 'none' ? ` ${engineOf(p.engine).name} supports` : ''}. Change the engine and precisions in the left rail.`;
    labelPrecOptions(null);
    $('#precHint').replaceChildren(checkpointNode(E.norm(p.model), p.wPrec));
    $('#compatLegend').replaceChildren(
      h('li', null, h('b', { class: 'sample' }, '4'), h('span', null, 'number = accelerators needed: the weights split across them plus one session of KV cache. Hover a cell for the TP × PP layout.')),
      h('li', null, h('b', { class: 'sample one' }, '1'), h('span', null, 'one accelerator is enough')),
      h('li', null, h('b', { class: 'sample node' }, '4'), h('span', null, 'several accelerators, but within one node (tensor parallel over the node fabric)')),
      h('li', null, h('b', { class: 'sample multi' }, '16'), h('span', null, 'more accelerators than one node holds (pipeline stages across the inter-node network)')),
      h('li', null, h('b', { class: 'sample none' }, '-'), h('span', null, sv.multiNode ? 'no layout up to TP 64 × PP 16 loads it on this hardware' : `does not fit in one node, and ${engineOf(p.engine).name} serves a model within one node`)),
      h('li', null, h('b', { class: 'sample none' }, '✕'), h('span', null, `${E.precLabel(p.wPrec, p.engine)} weights are not loadable on that chip generation${p.engine !== 'none' ? `, or ${engineOf(p.engine).name} does not run there` : ' (no kernel)'}`)),
      h('li', null, h('b', { class: 'sample' }, 'KV/token'), h('span', null, `bytes of KV cache one token costs in ${E.kvLabel(p.kvPrec, p.engine)}; multiply by the context to size a session`)),
    );
    const head = h('tr', null, h('th', { class: 'sticky' }, 'Model'), h('th', { class: 'n' }, T('kv-per-token', 'KV/token')), hws.map((hw) => h('th', { class: 'n rot' }, h('span', null, hw.name))));
    const body = models.map((m0) => {
      const m = E.norm(m0);
      return h('tr', null, h('td', { class: 't sticky' }, m.name), h('td', { class: 'n' }, fmtGB(E.kvPerTokenFull(m, p.kvPrec, p.engine))), hws.map((hw) => {
        const notHere = E.engineCovers(hw, p.engine) === 'no';
        const unsupported = notHere || E.formatSupport(hw, p.wPrec, p.engine) === 'unsupported' || E.kvSupport(hw, p.kvPrec, p.engine) !== 'supported';
        const mg = unsupported ? null : E.minGpus(hw, m, p.wPrec, p.kvPrec, ctx, p.adv, { engine: p.engine, slots: p.slots });
        const cls = !mg ? 'none' : mg.G === 1 ? 'one' : mg.G <= hw.nodeGpus ? 'node' : 'multi';
        const why = notHere ? `${engineOf(p.engine).name} does not run on ${(ARCHS[hw.arch] || {}).name || 'this generation'}` : `${E.precLabel(p.wPrec, p.engine)} weights are not loadable on ${(ARCHS[hw.arch] || {}).name || 'this generation'}`;
        return h('td', { class: 'n cell ' + cls, title: unsupported ? `${hw.name}: ${why}` : mg ? `${hw.name}: ${mg.G} (${sv.pp === 'sequential' ? `layer split over ${mg.pp}` : `TP ${mg.tp} × PP ${mg.pp}`})` : `${hw.name}: no layout ${sv.multiNode ? 'up to 64 × 16' : 'within one node'} loads it` }, unsupported ? '✕' : mg ? String(mg.G) : '-');
      }));
    });
    $('#compatTable').replaceChildren(h('div', { class: 'tw tall' }, h('table', { class: 'data compat' }, h('thead', null, head), h('tbody', null, body))));
  }

  /* ----- catalog ----- */
  function renderCatalog() {
    const hwRows = allHardware().map((hw) => [
      h('b', null, hw.name + (hw.approx ? ' ~' : '')), hw.vendor, fmtNum(hw.mem), fmtNum(hw.bw), hw.tflops.fp16 ? fmtNum(hw.tflops.fp16) : '-', hw.tflops.fp8 ? fmtNum(hw.tflops.fp8) : '-', hw.tflops.fp4 ? fmtNum(hw.tflops.fp4) : '-',
      `${(LINKS[hw.link] || LINKS.none).name}${hw.linkBw ? ' ' + fmtNum(hw.linkBw) + ' GB/s' : ''}`, fmtNum(hw.nodeGpus), hw.tdp != null ? fmtNum(hw.tdp) : '-', hw.price != null ? fmtMoney(hw.price) : '-',
      ARCHS[hw.arch] ? ARCHS[hw.arch].name : '-', ARCHS[hw.arch] ? ARCHS[hw.arch].native.map((x) => x.toUpperCase()).join(' ') : '-', ARCHS[hw.arch] ? ARCHS[hw.arch].weightOnly.map((x) => x.toUpperCase()).join(' ') : '-',
    ]);
    $('#catHardware').replaceChildren(table([{ t: 'Accelerator', k: 'accelerator' }, 'Vendor', { t: 'Memory GB', k: 'mem-gb' }, { t: 'GB/s', k: 'bw' }, { t: 'BF16 TFLOPS', k: 'tflops-bf16' }, { t: 'FP8', k: 'tflops-fp8' }, { t: 'FP4', k: 'tflops-fp4' }, { t: 'Fabric', k: 'fabric' }, { t: 'Per node', k: 'per-node' }, { t: 'W', k: 'tdp' }, { t: '$/h', k: 'price' }, { t: 'Generation', k: 'generation' }, { t: 'Native formats', k: 'native-fmt' }, { t: 'Weight-only', k: 'weight-only' }], hwRows, { numeric: [2, 3, 4, 5, 6, 8, 9, 10] }));
    const fams = ENGINE_FAMILIES;
    const yes = (t) => h('span', { class: 'chip good' }, t), no = (t) => h('span', { class: 'chip bad' }, t);
    const engRows = [];
    for (const [k, e] of Object.entries(ENGINES)) {
      if (!e.weights) continue;
      const runs = (fam) => !e.families || e.families.includes(fam.id);
      engRows.push({ group: [h('b', null, e.name), ` · ${e.version}, checked ${e.checked}`] });
      engRows.push({ cls: 'base', cells: ['Runs on'].concat(fams.map((fam) => (runs(fam) ? yes('yes') : no('no')))) });
      for (const f of ['fp8', 'int8', 'int4', 'fp4']) engRows.push([E.precLabel(f, k)].concat(fams.map((fam) => { const v = (e.weights[f] || {})[fam.id]; return !runs(fam) ? '' : v ? h('span', { class: 'chip ' + (v === 'native' ? 'good' : 'info') }, v) : no('no'); })));
      engRows.push(['KV cache'].concat(fams.map((fam) => { if (!runs(fam)) return ''; const ks = Object.entries(e.kvCache || {}).filter(([, l]) => l.includes(fam.id)).map(([d]) => E.kvLabel(d, k)); return ks.length ? yes(ks.join(' ')) : no(`${E.kvLabel('bf16', k)} only`); })));
    }
    const servingRows = Object.entries(ENGINES).filter(([, e]) => e.docs).map(([k, e]) => {
      const sv = E.servingOf(k), src = sv.sources || {}, mb = sv.maxBatch;
      const cap = sv.batching === 'slots' ? [`${sv.slots.default} slot${sv.slots.default === 1 ? '' : 's'} by default (`, h('code', null, sv.slots.env), '), full context reserved per slot']
        : mb ? ['continuous, paged KV; ', h('code', null, mb.name), ' ', mb.value ? fmtNum(mb.value) : mb.tiers ? `${fmtNum(mb.tiers[0].value)} (${fmtNum(mb.default)} below ${mb.tiers[1].minGiB} GiB or on A100)` : `${fmtNum(mb.min)} to ${fmtNum(mb.max)} from the KV pool`] : 'continuous, paged KV';
      const links = Object.entries(src).map(([what, url]) => h('a', { href: url, target: '_blank', rel: 'noopener' }, what));
      return [h('b', null, e.name), h('span', null, cap), sv.pp === 'sequential' ? 'layer split, no TP' : 'tensor + pipeline', sv.multiNode ? 'many' : 'one',
        sv.prefixCache === true ? 'shared' : sv.prefixCache === 'per-slot' ? 'per slot only' : 'no',
        sv.hostCacheGB != null ? `host RAM, ${fmtGB(sv.hostCacheGB * 1e9)} cache` : 'host RAM', sv.spec ? (k === 'ollama' ? 'with a draft model' : 'yes') : 'no', sv.pd ? 'yes' : 'no', sv.dpAttention ? 'yes' : 'no',
        sv.queueMax ? fmtNum(sv.queueMax) : 'unbounded', h('span', { class: 'vlink' }, ...links.flatMap((a) => [a, ' ']))];
    });
    $('#catEngines').replaceChildren(
      table([''].concat(fams.map((f) => h('span', null, f.vendor, h('br'), f.name.replace(f.vendor + ' ', '').replace('Blackwell GB20x / GB10', 'GB20x, GB10')))), engRows, { cls: 'matrix' }),
      h('h4', null, 'How each engine serves requests'),
      table(['Engine', { t: 'Batching and cap', k: 'max-batch' }, { t: 'Multi-GPU', k: 'layer-split' }, 'Nodes', { t: 'Prefix cache', k: 'prefix-caching' }, { t: 'Idle sessions', k: 'retention' }, { t: 'Spec. decoding', k: 'spec' }, { t: 'PD split', k: 'pd' }, { t: 'DP attention', k: 'dp-attn' }, { t: 'Queue', k: 'queue-limit' }, 'Sources'], servingRows),
      h('ul', { class: 'warnings', style: 'margin-top:10px' }, Object.values(ENGINES).filter((e) => e.docs).map((e) => h('li', null, h('b', null, `${e.name} (${e.version}, checked ${e.checked}${e.unverified ? ', not re-verified' : ''}): `), e.notes.join(' '), ' ', h('a', { href: e.docs.weights, target: '_blank', rel: 'noopener' }, 'source')))));
    const lint = catalogLint();
    $('#catLint').replaceChildren(...(lint.length ? [h('b', null, 'Consistency checks: '), h('ul', { class: 'warnings' }, lint.map((t) => h('li', null, h('span', { class: 'chip warn' }, 'check'), ' ', t)))] : [h('span', { class: 'chip good' }, 'ok'), " Every accelerator's TFLOPS columns agree with its chip generation."]));
    const mRows = allModels().map((m0) => {
      const m = E.norm(m0);
      const arch = m.attn.map((l) => `${l.n}× ${l.type}${l.window ? ' ' + fmtTok(l.window) : ''}`).join(' + ') + (m.sparse ? ` · sparse top-${m.sparse.topk}` : '');
      return [h('b', null, m.name + (m.approx ? ' ~' : '')), m.family, fmtNum(m.params, 1), m.moe ? fmtNum(m.active, 1) : '-', fmtNum(m.layers), `${m.nHeads} / ${m.attn.some((l) => l.type === 'mla') ? 'MLA' : m.nKv}`, fmtGB(E.kvPerTokenFull(m, 'bf16')), fmtGB(E.kvAtCtx(m, 'bf16', 1e6)), fmtTok(m.maxCtx) + (m.nativeCtx && m.nativeCtx < m.maxCtx ? ` (${fmtTok(m.nativeCtx)} native)` : ''), arch, m.nativePrec ? m.nativePrec.toUpperCase() : '-', m.hf ? h('a', { href: 'https://huggingface.co/' + m.hf, target: '_blank', rel: 'noopener' }, m.hf) : '-', m.variants ? h('span', null, ...Object.entries(m.variants).map(([k, r]) => h('span', { class: 'vlink' }, h('a', { href: 'https://huggingface.co/' + r, target: '_blank', rel: 'noopener', title: r }, k.toUpperCase()), ' '))) : '-'];
    });
    $('#catModels').replaceChildren(table([{ t: 'Model', k: 'model' }, 'Family', { t: 'Params B', k: 'params' }, { t: 'Active B', k: 'active' }, { t: 'Layers', k: 'layers' }, { t: 'Heads / KV heads', k: 'heads' }, { t: 'KV per token (BF16)', k: 'kv-per-token' }, { t: 'KV per 1M-token session', k: 'kv-1m' }, { t: 'Max context', k: 'max-ctx' }, { t: 'Attention layers', k: 'attn-layers' }, { t: 'Ships in', k: 'ships-in' }, { t: 'Verified against', k: 'verified' }, { t: 'Known checkpoints', k: 'checkpoints' }], mRows, { numeric: [2, 3, 4, 5, 6, 7, 8] }));
  }

  /* ---------- boot ---------- */
  function init() {
    populateSelects();
    bindInputs();
    syncInputs();
    initTooltips();
    renderGlossary();
    render();
    let width = document.documentElement.clientWidth;
    window.addEventListener('resize', () => { const w = document.documentElement.clientWidth; if (w !== width) { width = w; scheduleRender(); } });
  }
  const start = (data) => {
    try { if (data && data.state) state = Common.freshState(data.state); } catch (e) { /* ignore */ }
    // a link from the arcade (#plan=...) opens that setup on top of the defaults; the planner keeps its own candidate list
    const linked = Common.decodePlan(location.hash);
    if (linked) {
      state = Common.freshState(Object.assign({ candidates: state.candidates, compatCtx: state.compatCtx }, linked, { mode: 'forward' }));
      LS.set('cb.state', state);
      try { history.replaceState(null, '', location.pathname + location.search); } catch (e) { /* file:// in some browsers */ }
    }
    init();
  };
  try { if (window.claude && window.claude.hot && window.claude.hot.snapshot) window.claude.hot.snapshot(() => ({ state })); } catch (e) { /* ignore */ }
  try {
    if (window.claude && window.claude.hot && window.claude.hot.ready) window.claude.hot.ready(start);
    else start((window.claude && window.claude.hot && window.claude.hot.data) || {});
  } catch (e) { init(); }
})();
