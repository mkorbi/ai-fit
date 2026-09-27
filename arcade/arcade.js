/* arcade/arcade.js - the arcade stage: state, control deck, keyboard, scoreboard, baseline, undo, scenes, the chat
 * that streams at the real speed, overlays, and the handoff to and from the planner. Uses the same state keys and
 * units as the planner (common.js), plus `autoBuild`. */
(() => {
  'use strict';
  const E = Engine, M = ArcadeModel, C = Common;
  const { h, LS } = C;
  const $ = (id) => document.getElementById(id);
  const reducedQuery = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false, addEventListener() {} };
  const reduced = () => reducedQuery.matches || /[?&]motion=reduce/.test(location.search);

  /* ---------- pixel icons as inline SVG (the pixel fonts have no arrows) ---------- */
  const ART = {
    left: ['..#', '.##', '###', '.##', '..#'], right: ['#..', '##.', '###', '##.', '#..'],
    up: ['..#..', '.###.', '#####'], down: ['#####', '.###.', '..#..'], arrow: ['...#.', '....#', '#####', '....#', '...#.'],
    sound: ['...#.', '..##.', '####.', '####.', '..##.', '...#.'], mute: ['...#..#', '..##..#', '####...', '####...', '..##..#', '...#..#'],
    sun: ['#.#.#', '.###.', '##.##', '.###.', '#.#.#'], full: ['##.##', '#...#', '.....', '#...#', '##.##'],
  };
  function pix(name, color) {
    const rows = ART[name], w = rows[0].length, hgt = rows.length;
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${w} ${hgt}`); svg.setAttribute('shape-rendering', 'crispEdges'); svg.setAttribute('aria-hidden', 'true');
    for (let y = 0; y < hgt; y++) for (let x = 0; x < w; x++) if (rows[y][x] === '#') {
      const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      r.setAttribute('x', x); r.setAttribute('y', y); r.setAttribute('width', 1); r.setAttribute('height', 1); r.setAttribute('fill', color || 'currentColor');
      svg.append(r);
    }
    return svg;
  }
  const keycap = (k) => h('span', { class: 'key' }, k);

  /* ---------- catalog orderings and short names ---------- */
  const models = () => C.store.models().slice().sort((a, b) => a.params - b.params || a.name.localeCompare(b.name));
  const VENDOR_ORDER = ['NVIDIA', 'AMD', 'Intel', 'Google', 'AWS', 'Apple'];
  const hardware = () => C.store.hardware().slice().sort((a, b) => ((VENDOR_ORDER.indexOf(a.vendor) + 1 || 99) - (VENDOR_ORDER.indexOf(b.vendor) + 1 || 99)) || a.mem - b.mem || a.name.localeCompare(b.name));
  const ENGINE_ORDER = ['ollama', 'vllm', 'sglang', 'trtllm', 'none'];
  const USAGE = [['chat', 10, 'Chat'], ['assist', 25, 'Assist'], ['agents', 60, 'Agents'], ['batch', 100, 'Batch']];
  const TARGETS = [5, 10, 20, 30, 50];
  const IDLE = [['host', 'Park in RAM', 'RAM'], ['gpu', 'Keep on GPU', 'GPU'], ['none', 'Evict', 'Evict']];
  function modelShort(m) {
    let base = m.name.split(' / ')[0].replace(/\s*\(.*?\)/g, '');
    if (!/\d+(\.\d+)?B\b/i.test(base)) { const x = m.name.match(/(\d+(?:\.\d+)?B(?:-A\d+B)?)\b/); base += ' ' + (x ? x[1] : m.params >= 1000 ? (m.params / 1000).toFixed(1).replace(/\.0$/, '') + 'T' : Math.round(m.params) + 'B'); }
    return base;
  }
  const hwShort = (hw) => hw.name.replace(/^(GeForce|Instinct|Mac Studio)\s+/i, '').replace(/\s*\(.*?\)/g, '').replace(/, per GPU,?/i, '').replace(/\s+unified$/i, '');
  const engName = (id) => (id === 'none' ? 'Ideal' : (ENGINES[id] || {}).name || id);
  const shortPrec = (label) => label.split(' ')[0];
  const fmtTok = E.fmtTok;

  /* ---------- state ---------- */
  const PREFS0 = { palette: 'night', sound: false, deck: true };
  let prefs = Object.assign({}, PREFS0, LS.get('cb.arcade.prefs', {}));
  let userScenes = LS.get('cb.arcade.scenes', {});
  let baseline = LS.get('cb.arcade.baseline', null);
  let sceneIndex = LS.get('cb.arcade.scene', null);
  let st = null;
  const undo = [], redo = [];
  let p, r, need, vm;

  const scenes = () => SCENES.map((s, i) => userScenes[i] || s).concat([]).slice(0, 9);
  function sceneState(sc) { const s = C.freshState(Object.assign({}, C.sanitizeState(sc.state || {}), { mode: 'forward' })); s.autoBuild = !!sc.autoBuild; return s; }
  function loadState() {
    const linked = C.decodePlan(location.hash);
    if (linked) {
      try { history.replaceState(null, '', location.pathname + location.search); } catch (e) { /* file:// */ }
      const s = C.freshState(Object.assign({}, linked, { mode: 'forward' })); s.autoBuild = false; sceneIndex = null; return { s, linked: true };
    }
    const saved = LS.get('cb.arcade.state', null);
    if (saved) { const s = C.freshState(C.sanitizeState(saved)); s.autoBuild = !!saved.autoBuild; return { s, linked: false }; }
    sceneIndex = 0;
    return { s: sceneState(scenes()[0]), linked: false };
  }
  const snapshot = () => JSON.parse(JSON.stringify(st));
  function persist() { LS.set('cb.arcade.state', st); LS.set('cb.arcade.scene', sceneIndex); }

  function compute() {
    p = C.buildParams(st);
    need = E.reverse(p, [p.hw])[0];
    r = st.autoBuild && !need.infeasible ? need : C.runPlan(st).best;
    vm = M.derive({ r, p, need });
  }

  /* ---------- changes: every knob goes through here ---------- */
  const IDLE_NAME = Object.fromEntries(IDLE);
  function describe(key, value, s) {
    const eng = s.engine;
    switch (key) {
      case 'model': return modelShort(C.store.modelById(value));
      case 'hw': return hwShort(C.store.hwById(value));
      case 'engine': return engName(value);
      case 'wPrec': return shortPrec(E.precLabel(value, eng));
      case 'kvPrec': return shortPrec(E.kvLabel(value, eng));
      case 'ctx': return fmtTok(value);
      case 'users': return M.fmtCount(value);
      case 'activity': return `${value}%`;
      case 'target': return `${value} tok/s`;
      case 'retention': return IDLE_NAME[value];
      case 'slots': return String(value > 0 ? value : (E.servingOf(eng).slots || {}).default || 1);
      case 'count': return String(value);
      case 'prefixCache': case 'spec': case 'pd': case 'autoBuild': return value ? 'on' : 'off';
      default: return String(value);
    }
  }
  const WHAT = { model: 'Model', hw: 'GPU', count: 'GPUs', engine: 'Engine', slots: 'Parallel slots', wPrec: 'Weights', kvPrec: 'KV cache', ctx: 'Context', users: 'Users', activity: 'Usage', target: 'Speed target', retention: 'Idle sessions', prefixCache: 'Prefix cache', spec: 'Speculative decoding', pd: 'PD split', autoBuild: 'Auto-build' };
  /* Set one knob: shared side effects (common.js), format snapping, recompute, then ticker, banner, sound and world. */
  function set(key, value, opts = {}) {
    if (st[key] === value && !opts.force) return;
    const before = vm, from = describe(key, st[key], st);
    if (!opts.noUndo) { undo.push(snapshot()); if (undo.length > 60) undo.shift(); redo.length = 0; }
    if (key === 'count' && st.autoBuild) st.autoBuild = false;      // stepping the count by hand leaves auto-build
    st[key] = value;
    C.applyChange(st, key, value, { snapFormats: true });
    if (key !== 'autoBuild' && !opts.keepScene) sceneIndex = null;
    refresh({ what: WHAT[key] || key, from, to: describe(key, value, st) }, before);
  }
  function refresh(change, before, label) {
    try { compute(); } catch (e) { console.error(e); $('corner-err') || document.body.append(h('div', { class: 'corner-err', id: 'corner-err', title: String(e) })); return; }
    persist();
    world.setView(vm, change);
    render();
    if (before && (change || label)) {
      const t = M.ticker(label ? null : change, before, vm);
      let text = label ? `${label} · ${t.text}` : t.text, tone = t.tone;
      if (st.ctx > C.store.modelById(st.model).maxCtx) { text += ` (beyond the model's ${fmtTok(C.store.modelById(st.model).maxCtx)} maximum)`; tone = 'bad'; }
      tick(text, tone);
      if (t.banner) banner(t.banner, t.banner === 'Queue cleared!' || t.banner === 'It fits!' || /×/.test(t.banner) ? 'good' : /÷|Queue|slow|memory|supported/i.test(t.banner) ? 'bad' : 'neutral');
      Sfx.play(vm.status === 'oom' || vm.status === 'unsupported' ? 'error' : t.tone === 'good' ? 'good' : t.tone === 'bad' ? 'bad' : 'blip');
      if (before.speed.perUser !== vm.speed.perUser || before.status !== vm.status) chat.update();
    }
  }

  /* ---------- the world ---------- */
  const world = ArcadeWorld.create($('world'), { reducedMotion: reduced(), onError: () => { if (!$('corner-err')) document.body.append(h('div', { class: 'corner-err', id: 'corner-err', title: 'A frame failed to draw; the numbers are still correct.' })); } });
  reducedQuery.addEventListener && reducedQuery.addEventListener('change', () => world.setReducedMotion(reduced()));
  let bannerTimer = 0;
  function banner(text, tone) {
    const el = $('banner');
    el.className = 'banner';
    void el.offsetWidth;                                  // restart the animation
    el.textContent = text.toUpperCase();
    el.className = `banner show ${tone}`;
    clearTimeout(bannerTimer);
    bannerTimer = setTimeout(() => { el.className = 'banner'; }, 1900);   // also ends the static banner of reduced motion
    if (tone === 'bad' && /memory|supported/i.test(text) && !reduced()) { const w = $('worldBox'); w.classList.remove('shake'); void w.offsetWidth; w.classList.add('shake'); }
  }
  function tick(text, tone) {
    const msg = $('tickerMsg');
    msg.className = 'msg ' + (tone || 'neutral');
    msg.replaceChildren(...text.split('→').flatMap((part, i) => (i ? [pix('arrow'), part] : [part])));
  }

  /* ---------- scoreboard ---------- */
  const HUD = {};
  function buildHud() {
    const tile = (id, label, cls) => {
      const el = h('section', { class: 'tile frame' + (cls ? ' ' + cls : ''), id },
        h('div', { class: 'lbl' }, h('span', null, label), h('span', { class: 'delta', id: id + 'D' })),
        h('div', { class: 'num' }, h('span', { id: id + 'V' }), h('small', { id: id + 'U' })),
        h('div', { class: 'sub', id: id + 'S' }));
      HUD[id] = el;
      return el;
    };
    const hero = tile('tCap', 'Capacity', 'hero');
    hero.append(h('div', { class: 'headline', id: 'headline' }));
    $('hud').replaceChildren(
      hero,
      h('div', { class: 'facts frame' }, h('div', { class: 'fact', id: 'need' }, h('span', { id: 'needL' }), h('b', { id: 'needV' })), h('div', { class: 'fact limit' }, h('span', null, 'Limit'), h('b', { id: 'limitV' }))),
      h('div', { class: 'tiles' }, tile('tSpeed', 'Speed per user'), tile('tTtft', 'First token'), tile('tGpus', 'GPUs'), tile('tCost', 'Cost per month')),
      h('section', { class: 'chat frame' }, h('div', { class: 'lbl' }, h('span', null, "A user's view"), h('span', { id: 'chatTag' })), h('div', { class: 'q', id: 'chatQ' }), h('div', { class: 'bubble', id: 'bubble' }), h('div', { class: 'meter', id: 'meter' })));
  }
  const tweens = new Map();
  function countTo(el, value, fmt) {
    const from = tweens.has(el) ? tweens.get(el).cur : value;
    if (reduced() || !isFinite(from) || !isFinite(value) || from === value) { tweens.set(el, { cur: value }); el.textContent = fmt(value); return; }
    const t0 = performance.now(), tw = { cur: from };
    tweens.set(el, tw);
    const step = (now) => {
      if (tweens.get(el) !== tw) return;
      const k = Math.max(0, Math.min(1, (now - t0) / 350)), e = 1 - Math.pow(1 - k, 3);
      tw.cur = from + (value - from) * e;
      el.textContent = fmt(k < 1 ? tw.cur : value);
      if (k < 1) requestAnimationFrame(step); else tw.cur = value;
    };
    requestAnimationFrame(step);
  }
  function delta(id, d, fmtPct) {
    const el = $(id + 'D');
    el.replaceChildren();
    el.className = 'delta';
    if (!baseline || !d || d.base == null) return;
    if (d.same) { el.className = 'delta same'; el.textContent = '='; return; }
    if (d.pct == null) return;
    el.className = 'delta ' + (d.good ? 'good' : 'bad');
    el.append(pix(d.value > d.base ? 'up' : 'down'), fmtPct ? fmtPct(d) : `${d.pct > 0 ? '+' : '−'}${Math.abs(Math.round(d.pct * 100))}%`);
  }
  function render() {
    const d = M.compare(vm, baseline && baseline.vm);
    const cap = vm.capacity.users;
    countTo($('tCapV'), cap, (v) => { const n = Math.round(v); $('tCapU').textContent = n === 1 ? ' user' : ' users'; return M.fmtCount(n); });
    $('tCapS').textContent = `at ${vm.speed.target}+ tok/s, first token under ${M.fmtDuration(vm.speed.ttftMax)}`;
    const hl = $('headline');
    hl.textContent = vm.headline;
    hl.className = 'headline ' + ({ ok: 'good', queue: 'bad', oom: 'bad', unsupported: 'bad', slow: 'warn', prefill: 'warn', ttft: 'warn' }[vm.status] || '');
    delta('tCap', d.capacity);
    // speed
    if (vm.speed.perUser > 0) countTo($('tSpeedV'), vm.speed.perUser, M.fmtSpeed); else { tweens.delete($('tSpeedV')); $('tSpeedV').textContent = '-'; }
    $('tSpeedU').textContent = ' tok/s';
    $('tSpeedS').textContent = `target ${vm.speed.target}, reading 6`;
    $('tSpeedV').style.color = vm.speed.perUser > 0 && vm.speed.slow ? 'var(--yellow)' : '';
    delta('tSpeed', d.speed);
    // first token
    $('tTtftV').textContent = vm.speed.ttft != null ? M.fmtDuration(vm.speed.ttft) : '-';
    $('tTtftU').textContent = '';
    $('tTtftS').textContent = vm.speed.cold != null ? `cold turn ${M.fmtDuration(vm.speed.cold)}` : 'model does not load';
    $('tTtftV').style.color = vm.speed.ttftSlow ? 'var(--yellow)' : '';
    delta('tTtft', d.ttft);
    // GPUs
    const c = vm.cluster;
    countTo($('tGpusV'), c.gpusUsed, (v) => M.fmtCount(Math.round(v)));
    $('tGpusU').textContent = ' × ' + hwShort(C.store.hwById(st.hw)).replace(/\s+\d+\s*GB$/i, '');
    const split = c.split === 'layer' ? (c.pp > 1 ? `layer split ${c.pp}` : '') : c.tp > 1 || c.pp > 1 ? `TP${c.tp}${c.pp > 1 ? ' PP' + c.pp : ''}` : '';
    const units = c.servers ? `${c.R} server${c.R === 1 ? '' : 's'}` : split ? `${split} × ${c.R}` : `${c.R} replica${c.R === 1 ? '' : 's'}`;
    const kw = vm.money.kW != null ? `${vm.money.kW < 10 ? vm.money.kW.toFixed(1) : Math.round(vm.money.kW)} kW` : '';
    $('tGpusS').textContent = [c.servers && split ? split : '', units, kw].filter(Boolean).join(' · ');
    delta('tGpus', d.gpus);
    // money
    if (vm.money.perMonth != null) countTo($('tCostV'), vm.money.perMonth, M.fmtMoney); else $('tCostV').textContent = '-';
    $('tCostU').textContent = '';
    $('tCostS').textContent = vm.money.perHour != null ? `${M.fmtMoney(vm.money.perHour)}/h · ${vm.money.perUserMonth != null ? M.fmtMoney(vm.money.perUserMonth) + '/user' : 'no users'}` : 'no cloud price in the catalog';
    delta('tCost', d.perMonth);
    // what would it take
    const n = vm.need;
    const who = `${M.fmtCount(st.users)} user${st.users === 1 ? '' : 's'}`;
    $('needL').textContent = st.autoBuild ? (n && n.feasible ? 'Auto-built for ' : 'Cannot build for ') + who : `Need for ${who}`;
    $('needV').textContent = n && n.feasible ? `${M.fmtCount(n.gpus)} × ${hwShort(C.store.hwById(st.hw)).replace(/\s+\d+\s*GB$/i, '').toUpperCase()}` : 'NO FIT';
    $('need').title = n && !n.feasible ? n.reason : '';
    $('limitV').textContent = vm.limit.label.toUpperCase();
    renderTop();
    renderDeck();
    $('worldBox').setAttribute('aria-label', `${vm.headline}. ${vm.cluster.gpusUsed} ${hwShort(C.store.hwById(st.hw))}, capacity ${cap} users.`);
  }
  function renderTop() {
    const m = C.store.modelById(st.model), hw = C.store.hwById(st.hw);
    const count = st.autoBuild && vm.need && vm.need.feasible ? vm.need.gpus : st.count;
    $('cfg').replaceChildren(h('em', null, modelShort(m)), ` · ${shortPrec(E.precLabel(st.wPrec, st.engine))} / KV ${shortPrec(E.kvLabel(st.kvPrec, st.engine))} · `, h('em', null, `${count}× ${hwShort(hw)}`), ` · ${engName(st.engine)} · ${fmtTok(st.ctx)} ctx · ${M.fmtCount(st.users)} user${st.users === 1 ? '' : 's'}`);
    const chip = $('sceneChip');
    chip.hidden = sceneIndex == null;
    if (sceneIndex != null) { chip.textContent = `Scene ${sceneIndex + 1}/9`; chip.title = scenes()[sceneIndex].name; }
    const bc = $('baseChip');
    bc.hidden = !baseline;
    if (baseline) { bc.textContent = `Baseline: ${baseline.short || baseline.name}`; bc.title = baseline.name; }
  }

  /* ---------- a user's view: one answer streamed at the real tokens per second ---------- */
  const QA = [
    ['How many GPUs do we need for the launch?', 'It depends on three numbers: how many people use it at the same time, how long their conversations get, and how fast each answer has to stream. Memory holds the conversations, bandwidth sets the speed.'],
    ['Why is the model slower today?', 'More people are asking at once. Every extra request in the batch reads its own KV cache on every step, so each answer gets a smaller share of the memory bandwidth.'],
    ['Write a haiku about GPUs.', 'Tokens trickle out,\nthe queue grows longer each hour.\nOrder one more node.'],
    ['Can we run it on a laptop?', 'For one person, yes. For a team, a laptop tool serves one request at a time and everyone else waits in line, so pick a serving engine that batches requests.'],
  ];
  const chat = (() => {
    let qi = 0, raf = 0, phase = 'idle', last = 0, waited = 0, pos = 0, pause = 0, drawn = '';
    function start() { waited = 0; pos = 0; pause = 0; phase = 'ttft'; drawn = ''; last = performance.now(); cancelAnimationFrame(raf); raf = requestAnimationFrame(frame); }
    function next() { qi = (qi + 1) % QA.length; start(); }
    function update() { if (phase === 'error' || phase === 'idle') start(); }
    function paint(key, ...nodes) { if (key === drawn) return; drawn = key; $('bubble').replaceChildren(...nodes); $('chatQ').textContent = '> ' + QA[qi][0]; }
    function frame(now) {
      raf = requestAnimationFrame(frame);
      const dt = Math.max(0, Math.min(0.25, (now - last) / 1000)); last = Math.max(last, now);   // a frame can start before start() ran
      const [q, a] = QA[qi];
      if (!vm || vm.status === 'oom' || vm.status === 'unsupported') {
        phase = 'error';
        paint('err' + vm.status, h('div', { class: 'err' }, vm && vm.status === 'unsupported' ? `ERROR: ${engName(st.engine)} does not run on this hardware.` : 'ERROR 503: the model does not fit on these GPUs.'));
        $('meter').replaceChildren(h('span', null, 'no answer'), h('span', null, ''));
        $('chatTag').textContent = '';
        return;
      }
      if (phase === 'error') { phase = 'ttft'; waited = 0; pos = 0; }
      const ttft = vm.speed.ttft || 0, speed = vm.speed.perUser;
      // long first-token times play fast-forward so the room does not wait with you
      const ff = ttft > 5 ? Math.ceil(ttft / 3) : 1;
      if (phase === 'ttft') {
        waited += dt * ff;
        if (waited >= ttft) phase = 'stream';
        else paint(`t${Math.floor(waited * 4 / ff)}`, h('div', { class: 'wait' }, 'thinking' + '.'.repeat(1 + (Math.floor(waited * 4 / ff) % 3)), ff > 1 ? `  ${M.fmtDuration(Math.max(0, ttft - waited))} left (×${ff})` : ''));
      }
      if (phase === 'stream') {
        pos = Math.min(a.length, pos + dt * speed * 4);        // about four characters per token
        const n = Math.floor(pos);
        paint(`s${n}`, h('div', { class: 'a' }, a.slice(0, n), n < a.length ? h('span', { class: 'caret' }) : null));
        if (n >= a.length) phase = 'done';
      }
      if (phase === 'done') { pause += dt; if (pause > 3) { next(); return; } }
      const others = vm.crowd.waiting + vm.crowd.rejected + vm.crowd.blocked;
      $('chatTag').textContent = speed < 6 ? 'slower than reading' : '';
      const meter = `${M.fmtSpeed(speed)} tok/s|${others >= 0.5 ? `${M.fmtCount(others)} others waiting${isFinite(vm.crowd.wait) && vm.crowd.wait > 0 ? ' ' + M.fmtDuration(vm.crowd.wait) : ''}` : 'no queue'}`;
      if ($('meter').dataset.v !== meter) { $('meter').dataset.v = meter; const [l, r2] = meter.split('|'); $('meter').replaceChildren(h('span', null, l), h('span', null, r2)); }
    }
    return { start, next, update, replay: start };
  })();

  /* ---------- the control deck ---------- */
  const D = {};
  function ctl(cls, label, keys, ...body) {
    return h('div', { class: 'ctl ' + cls }, h('div', { class: 'head' }, h('span', null, label), h('span', { class: 'keys' }, keys.map(keycap))), h('div', { class: 'row' }, ...body));
  }
  function arrowBtn(dir, title, onclick) { return h('button', { class: 'btn', type: 'button', title, 'aria-label': title, onclick }, pix(dir)); }
  function cycle(list, cur, dir) { const i = list.indexOf(cur); return list[((i < 0 ? 0 : i) + dir + list.length) % list.length]; }
  function buildDeck() {
    D.model = h('select', { class: 'val', 'aria-label': 'Model', onchange: (e) => set('model', e.target.value) });
    D.hw = h('select', { class: 'val', 'aria-label': 'GPU', onchange: (e) => set('hw', e.target.value) });
    D.count = h('div', { class: 'val' });
    D.auto = h('button', { class: 'btn', type: 'button', title: 'Size the cluster to the users (A)', onclick: () => set('autoBuild', !st.autoBuild) }, 'Auto');
    D.engine = h('div', { class: 'val' });
    D.slots = h('div', { class: 'val' });
    D.idle = h('div', { class: 'seg' }, IDLE.map(([v, t, s]) => h('button', { class: 'btn', type: 'button', 'data-v': v, title: t, onclick: () => set('retention', v) }, s)));
    D.w = h('div', { class: 'seg' }, E.W_PRECS.map((v) => h('button', { class: 'btn', type: 'button', 'data-v': v, onclick: () => set('wPrec', v) })));
    D.kv = h('div', { class: 'seg' }, E.KV_PRECS.map((v) => h('button', { class: 'btn', type: 'button', 'data-v': v, onclick: () => set('kvPrec', v) })));
    D.ctx = h('input', { type: 'range', min: 0, max: 1000, step: 1, 'aria-label': 'Context per session', oninput: (e) => set('ctx', C.sliderToCtx(+e.target.value)) });
    D.ctxOut = h('output');
    D.users = h('input', { type: 'range', min: 0, max: C.USER_STEPS.length - 1, step: 1, 'aria-label': 'Users', oninput: (e) => set('users', C.sliderToUsers(+e.target.value)) });
    D.usersOut = h('output');
    D.usage = h('div', { class: 'seg' }, USAGE.map(([k, v, t]) => h('button', { class: 'btn', type: 'button', 'data-v': v, title: `${v}% of the time a request in flight`, onclick: () => { set('activity', v); } }, t)));
    D.target = h('div', { class: 'seg' }, TARGETS.map((v) => h('button', { class: 'btn', type: 'button', 'data-v': v, onclick: () => set('target', v) }, String(v))));
    const power = (key, label, name) => h('button', { class: 'btn gate', type: 'button', 'data-k': key, 'data-name': name, onclick: () => set(key, !st[key]) }, label);
    D.power = h('div', { class: 'seg' }, power('prefixCache', 'Prefix', 'Prefix caching'), power('spec', 'Spec', 'Speculative decoding'), power('pd', 'PD', 'Prefill/decode disaggregation'));
    const step = (dir) => () => set('count', Math.max(1, (st.autoBuild && vm.need && vm.need.feasible ? vm.need.gpus : st.count) + dir));
    const slotStep = (dir) => () => set('slots', Math.max(1, (st.slots > 0 ? st.slots : 1) + dir));
    $('deck').replaceChildren(
      ctl('c-model', 'Model', ['M'], arrowBtn('left', 'Previous model', () => set('model', cycle(models().map((x) => x.id), st.model, -1))), D.model, arrowBtn('right', 'Next model', () => set('model', cycle(models().map((x) => x.id), st.model, 1)))),
      ctl('c-hw', 'GPU', ['G'], arrowBtn('left', 'Previous GPU', () => set('hw', cycle(hardware().map((x) => x.id), st.hw, -1))), D.hw, arrowBtn('right', 'Next GPU', () => set('hw', cycle(hardware().map((x) => x.id), st.hw, 1)))),
      ctl('c-count', 'How many', ['-', '+', 'A'], h('button', { class: 'btn', type: 'button', title: 'One GPU less', 'aria-label': 'One GPU less', onclick: step(-1) }, '−'), D.count, h('button', { class: 'btn', type: 'button', title: 'One GPU more', 'aria-label': 'One GPU more', onclick: step(1) }, '+'), D.auto),
      ctl('c-engine', 'Engine', ['E'], arrowBtn('left', 'Previous engine', () => set('engine', cycle(ENGINE_ORDER, st.engine, -1))), D.engine, arrowBtn('right', 'Next engine', () => set('engine', cycle(ENGINE_ORDER, st.engine, 1)))),
      ctl('c-slots', 'Slots', ['N'], h('button', { class: 'btn', type: 'button', title: 'One slot less', 'aria-label': 'One slot less', onclick: slotStep(-1) }, '−'), D.slots, h('button', { class: 'btn', type: 'button', title: 'One slot more', 'aria-label': 'One slot more', onclick: slotStep(1) }, '+')),
      ctl('c-idle', 'Idle sessions', ['I'], D.idle),
      ctl('c-w', 'Weights', ['W'], D.w),
      ctl('c-kv', 'KV cache', ['K'], D.kv),
      ctl('c-ctx', 'Context', ['C'], h('div', { class: 'range' }, D.ctx, D.ctxOut)),
      ctl('c-users', 'Users', ['U'], h('div', { class: 'range' }, D.users, D.usersOut)),
      ctl('c-usage', 'Usage', ['Y'], D.usage),
      ctl('c-target', 'Tok/s target', ['T'], D.target),
      ctl('c-power', 'Power-ups', ['Q', 'J', 'V'], D.power));
    fillSelects();
  }
  function fillSelects() {
    D.model.replaceChildren(...models().map((m) => h('option', { value: m.id }, modelShort(m).toUpperCase())));
    D.hw.replaceChildren(...hardware().map((x) => h('option', { value: x.id }, hwShort(x).toUpperCase())));
  }
  function renderDeck() {
    const hw = C.store.hwById(st.hw), eng = st.engine, sv = E.servingOf(eng), ename = engName(eng);
    D.model.value = st.model; D.hw.value = st.hw;
    const autoCount = st.autoBuild && vm.need && vm.need.feasible;
    D.count.textContent = String(autoCount ? vm.need.gpus : st.count);
    D.auto.setAttribute('aria-pressed', String(!!st.autoBuild));
    D.engine.textContent = ename.toUpperCase();
    const slotEngine = sv.batching === 'slots';
    D.slots.textContent = slotEngine ? String(st.slots > 0 ? st.slots : sv.slots.default) : '-';
    for (const b of D.slots.parentElement.querySelectorAll('.btn')) b.disabled = !slotEngine;
    D.slots.parentElement.parentElement.title = slotEngine ? `${sv.slots.env}: requests one server runs at once, each reserving its full context` : `${ename} batches requests continuously; no fixed slots`;
    for (const b of D.idle.children) { b.setAttribute('aria-pressed', String(b.dataset.v === st.retention)); const ok = sv.retention.includes(b.dataset.v); b.classList.toggle('na', !ok); b.title = ok ? '' : `Not in ${ename}`; }
    for (const b of D.w.children) {
      const v = b.dataset.v, sup = E.formatSupport(hw, v, eng);
      b.textContent = shortPrec(E.precLabel(v, eng));
      b.setAttribute('aria-pressed', String(v === st.wPrec));
      b.classList.toggle('na', sup === 'unsupported');
      b.title = sup === 'unsupported' ? `${E.precLabel(v, eng)} does not load on ${hw.name}${eng !== 'none' ? ' with ' + ename : ''}` : sup === 'weight-only' ? `${E.precLabel(v, eng)}: weights stay small, compute runs in BF16` : `${E.precLabel(v, eng)}: native compute`;
    }
    for (const b of D.kv.children) {
      const v = b.dataset.v, ok = E.kvSupport(hw, v, eng) === 'supported';
      b.textContent = shortPrec(E.kvLabel(v, eng)).toUpperCase();
      b.setAttribute('aria-pressed', String(v === st.kvPrec));
      b.classList.toggle('na', !ok);
      b.title = ok ? '' : `${ename} has no ${E.kvLabel(v, eng)} KV cache on ${hw.name}`;
    }
    const maxCtx = C.store.modelById(st.model).maxCtx;
    if (document.activeElement !== D.ctx) D.ctx.value = C.ctxToSlider(st.ctx);
    D.ctxOut.textContent = fmtTok(st.ctx);
    D.ctxOut.style.color = st.ctx > maxCtx ? 'var(--red)' : '';
    D.ctxOut.title = st.ctx > maxCtx ? `The model supports ${fmtTok(maxCtx)} tokens` : '';
    if (document.activeElement !== D.users) D.users.value = C.usersToSlider(st.users);
    D.usersOut.textContent = M.fmtCount(st.users);
    for (const b of D.usage.children) b.setAttribute('aria-pressed', String(+b.dataset.v === +st.activity));
    for (const b of D.target.children) b.setAttribute('aria-pressed', String(+b.dataset.v === +st.target));
    for (const b of D.power.children) {
      const k = b.dataset.k, ok = k === 'prefixCache' ? sv.prefixCache === true : !!sv[k];
      b.setAttribute('aria-pressed', String(!!st[k]));
      b.classList.toggle('na', !ok);
      b.title = b.dataset.name + (ok ? '' : k === 'prefixCache' && sv.prefixCache === 'per-slot' ? `: ${ename} reuses a prompt prefix within each slot only` : `: not in ${ename}`);
    }
  }

  /* ---------- top bar tools ---------- */
  function buildTools() {
    const tool = (id, label, key, onclick, icon) => h('button', { class: 'tool', type: 'button', id, onclick, title: `${label} (${key})`, 'aria-label': label }, icon ? pix(icon) : label, keycap(key));
    $('tools').replaceChildren(
      tool('tSound', 'Sound', 'S', toggleSound, prefs.sound ? 'sound' : 'mute'),
      tool('tPalette', prefs.palette === 'day' ? 'Night' : 'Day', 'L', togglePalette, 'sun'),
      tool('tFull', 'Full', 'F', toggleFull, 'full'),
      tool('tHelp', 'Keys', '?', () => showHelp(true)),
      tool('tPlanner', 'Planner', 'O', openPlanner));
  }

  /* ---------- baseline, undo, scenes ---------- */
  function baselineName() { return sceneIndex != null ? scenes()[sceneIndex].name : `${modelShort(C.store.modelById(st.model))} · ${vm.cluster.gpusUsed}× ${hwShort(C.store.hwById(st.hw))} · ${engName(st.engine)}`; }
  const baselineShort = () => (sceneIndex != null ? `Scene ${sceneIndex + 1}` : `${vm.cluster.gpusUsed}× ${hwShort(C.store.hwById(st.hw)).replace(/\s+\d+\s*GB$/i, '')}, ${engName(st.engine)}`);
  function pin() { baseline = { state: snapshot(), vm: JSON.parse(JSON.stringify(vm)), name: baselineName(), short: baselineShort() }; LS.set('cb.arcade.baseline', baseline); render(); tick(`Baseline pinned: ${baseline.name}. Every number now shows its change.`, 'neutral'); Sfx.play('coin'); }
  function clearBaseline() { baseline = null; LS.del('cb.arcade.baseline'); render(); tick('Baseline cleared.', 'neutral'); }
  function swap() {
    if (!baseline) { tick('Pin a baseline first (P).', 'neutral'); return; }
    const before = vm, cur = { state: snapshot(), vm: JSON.parse(JSON.stringify(vm)), name: baselineName(), short: baselineShort() };
    undo.push(snapshot());
    st = C.freshState(C.sanitizeState(baseline.state)); st.autoBuild = !!baseline.state.autoBuild;
    baseline = cur; LS.set('cb.arcade.baseline', baseline); sceneIndex = null;
    refresh(null, before, 'Swapped with the baseline');
  }
  function restore(snap, label) { const before = vm; st = C.freshState(C.sanitizeState(snap)); st.autoBuild = !!snap.autoBuild; sceneIndex = null; refresh(null, before, label); }
  function doUndo() { if (!undo.length) { tick('Nothing to undo.', 'neutral'); return; } redo.push(snapshot()); restore(undo.pop(), 'Undo'); }
  function doRedo() { if (!redo.length) { tick('Nothing to redo.', 'neutral'); return; } undo.push(snapshot()); restore(redo.pop(), 'Redo'); }
  function recall(i) {
    const list = scenes();
    if (i < 0 || i >= list.length) return;
    const sc = list[i], before = vm;
    if (before) undo.push(snapshot());
    st = sceneState(sc); sceneIndex = i;
    if (sc.baseline === 'clear') { baseline = null; LS.del('cb.arcade.baseline'); }
    refresh(null, before, `Scene ${i + 1}: ${sc.name}`);
    if (sc.baseline === 'pin') { baseline = { state: snapshot(), vm: JSON.parse(JSON.stringify(vm)), name: sc.name, short: `Scene ${i + 1}` }; LS.set('cb.arcade.baseline', baseline); render(); }
    if (!before) tick(`Scene ${i + 1}: ${sc.name}`, 'neutral');
    chat.start();
  }
  function saveScene(i) {
    const base = scenes()[i];
    userScenes[i] = { name: userScenes[i] ? userScenes[i].name : `My scene ${i + 1}`, baseline: 'keep', autoBuild: !!st.autoBuild, state: Object.fromEntries(Object.entries(snapshot()).filter(([k]) => k in C.DEFAULTS && !['mode', 'candidates', 'compatCtx'].includes(k))) };
    LS.set('cb.arcade.scenes', userScenes);
    sceneIndex = i;
    renderTop();
    tick(`Saved as scene ${i + 1}${base ? ` (was: ${base.name})` : ''}.`, 'good');
    Sfx.play('coin');
  }
  function stepScene(dir) { recall(sceneIndex == null ? (dir > 0 ? 0 : scenes().length - 1) : Math.min(scenes().length - 1, Math.max(0, sceneIndex + dir))); }
  function showScenes(v) {
    $('scenesScreen').hidden = !v;
    if (v) { $('scenesText').value = JSON.stringify(scenes(), null, 1); $('scenesMsg').textContent = ''; }
  }
  function importScenes() {
    try {
      const list = JSON.parse($('scenesText').value);
      if (!Array.isArray(list)) throw new Error('expected a list of scenes');
      const next = {};
      list.slice(0, 9).forEach((sc, i) => { if (sc && typeof sc === 'object' && sc.state) next[i] = { name: String(sc.name || `Scene ${i + 1}`).slice(0, 60), baseline: ['pin', 'clear', 'keep'].includes(sc.baseline) ? sc.baseline : 'keep', autoBuild: !!sc.autoBuild, state: C.sanitizeState(sc.state) }; });
      userScenes = next; LS.set('cb.arcade.scenes', userScenes);
      $('scenesMsg').textContent = `Imported ${Object.keys(next).length} scenes.`;
      renderTop();
    } catch (e) { $('scenesMsg').textContent = 'Not a scene list: ' + e.message; }
  }

  /* ---------- tools ---------- */
  function toggleSound() { prefs.sound = Sfx.enable(!prefs.sound); LS.set('cb.arcade.prefs', prefs); buildTools(); tick(`Sound ${prefs.sound ? 'on' : 'off'}.`, 'neutral'); Sfx.play('coin'); }
  function applyPalette() { document.documentElement.dataset.palette = prefs.palette; world.setPalette(); }
  function togglePalette() { prefs.palette = prefs.palette === 'day' ? 'night' : 'day'; LS.set('cb.arcade.prefs', prefs); applyPalette(); buildTools(); }
  function toggleFull() { if (document.fullscreenElement) document.exitFullscreen(); else if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {}); }
  function toggleDeck() { prefs.deck = !prefs.deck; LS.set('cb.arcade.prefs', prefs); $('stage').classList.toggle('deck-off', !prefs.deck); }
  function openPlanner() { window.open('../index.html#' + C.encodePlan(st), '_blank', 'noopener'); }
  function showHelp(v) { $('helpScreen').hidden = !v; }
  function blank(v) { $('blankScreen').hidden = !v; }
  function startGame() { $('titleScreen').hidden = true; Sfx.play('start'); chat.start(); }

  /* ---------- keyboard: one table drives the handler and the help overlay ---------- */
  const usersStep = (dir) => { const i = C.usersToSlider(st.users); set('users', C.sliderToUsers(i + dir)); };
  const KEYMAP = [
    { k: ['Space'], d: 'Start; replay the answer', run: () => ($('titleScreen').hidden ? chat.replay() : startGame()) },
    { k: ['1', '…', '9'], d: 'Recall scene (Shift saves)' },
    { k: ['PgDn', 'Right'], d: 'Next scene (clicker)' }, { k: ['PgUp', 'Left'], d: 'Previous scene' },
    { k: ['B', '.'], d: 'Blank screen; any key returns' }, { k: ['F5'], d: 'Replay the scene' },
    { k: ['P'], d: 'Pin baseline (Shift clears)' }, { k: ['X'], d: 'Swap current and baseline' },
    { k: ['Z'], d: 'Undo (Shift redoes)' }, { k: ['M'], d: 'Next model (Shift: previous)' },
    { k: ['G'], d: 'Next GPU (Shift: previous)' }, { k: ['+', '−'], d: 'Twice / half the GPUs' },
    { k: ['A'], d: 'Auto-build for the users' }, { k: ['E'], d: 'Next engine (Shift: previous)' },
    { k: ['N'], d: 'Twice the slots (Shift: half)' }, { k: ['W', 'K'], d: 'Weights / KV cache format' },
    { k: ['C'], d: 'Twice the context (Shift: half)' }, { k: ['U'], d: 'More users (Shift: fewer)' },
    { k: ['Y'], d: 'Usage pattern' }, { k: ['T'], d: 'Speed target' },
    { k: ['I'], d: 'Idle sessions' }, { k: ['Q', 'J', 'V'], d: 'Prefix cache, speculation, PD split' },
    { k: ['D'], d: 'Show or hide the deck' }, { k: ['F'], d: 'Fullscreen' },
    { k: ['L'], d: 'Day or night palette' }, { k: ['S'], d: 'Sound' },
    { k: ['R'], d: 'Scenes: export or import' }, { k: ['O'], d: 'Open in the planner' },
    { k: ['?', 'H'], d: 'This help' }, { k: ['Esc'], d: 'Close' },
  ];
  function buildHelp() { $('helpKeys').replaceChildren(...KEYMAP.flatMap((e) => [h('div', { class: 'k' }, e.k.map(keycap)), h('div', { class: 'd' }, e.d)])); }
  function onKey(e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = (e.target && e.target.tagName) || '';
    const typing = tag === 'TEXTAREA' || (tag === 'INPUT' && e.target.type !== 'range') || tag === 'SELECT';
    if (!$('blankScreen').hidden) { e.preventDefault(); blank(false); return; }
    if (e.key === 'Escape') { showHelp(false); showScenes(false); return; }
    if (typing) return;
    if (tag === 'INPUT' && ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown'].includes(e.key)) return;   // the slider's own keys
    const key = e.key, lower = key.length === 1 ? key.toLowerCase() : key, shift = e.shiftKey;
    const digit = /^(Digit|Numpad)([1-9])$/.exec(e.code || '');
    const title = !$('titleScreen').hidden;
    if (title) {
      if (key === ' ' || key === 'Enter') { e.preventDefault(); startGame(); }
      else if (key === '?' || lower === 'h') showHelp(true);
      return;
    }
    let handled = true;
    if (digit) { const i = +digit[2] - 1; if (shift) saveScene(i); else recall(i); }
    else if (key === 'F5') { if (sceneIndex != null) recall(sceneIndex); else refresh(null, null); }
    else if (key === 'PageDown' || key === 'ArrowRight') stepScene(1);
    else if (key === 'PageUp' || key === 'ArrowLeft') stepScene(-1);
    else if (key === ' ' || key === 'Enter') chat.replay();
    else if (lower === 'b' || key === '.') blank(true);
    else if (lower === 'p') (shift ? clearBaseline() : pin());
    else if (lower === 'x') swap();
    else if (lower === 'z') (shift ? doRedo() : doUndo());
    else if (lower === 'm') set('model', cycle(models().map((x) => x.id), st.model, shift ? -1 : 1));
    else if (lower === 'g') set('hw', cycle(hardware().map((x) => x.id), st.hw, shift ? -1 : 1));
    else if (key === '+' || key === '=' || key === '*') set('count', Math.min(100000, (st.autoBuild && vm.need && vm.need.feasible ? vm.need.gpus : st.count) * 2));
    else if (key === '-' || key === '_' || key === '/') set('count', Math.max(1, Math.floor((st.autoBuild && vm.need && vm.need.feasible ? vm.need.gpus : st.count) / 2)));
    else if (lower === 'a') set('autoBuild', !st.autoBuild);
    else if (lower === 'e') set('engine', cycle(ENGINE_ORDER, st.engine, shift ? -1 : 1));
    else if (lower === 'n') { if (E.servingOf(st.engine).batching === 'slots') { const cur = st.slots > 0 ? st.slots : 1; set('slots', shift ? Math.max(1, Math.floor(cur / 2)) : Math.min(1024, cur * 2)); } else tick(`${engName(st.engine)} has no fixed slots.`, 'neutral'); }
    else if (lower === 'w') set('wPrec', cycle(E.W_PRECS.filter((v) => E.formatSupport(C.store.hwById(st.hw), v, st.engine) !== 'unsupported'), st.wPrec, shift ? -1 : 1));
    else if (lower === 'k') set('kvPrec', cycle(E.KV_PRECS.filter((v) => E.kvSupport(C.store.hwById(st.hw), v, st.engine) === 'supported'), st.kvPrec, shift ? -1 : 1));
    else if (lower === 'c') set('ctx', C.sliderToCtx(Math.max(0, Math.min(1000, C.ctxToSlider(st.ctx) + (shift ? -100 : 100)))));
    else if (lower === 'u') usersStep(shift ? -1 : 1);
    else if (lower === 'y') set('activity', cycle(USAGE.map((x) => x[1]), st.activity, shift ? -1 : 1));
    else if (lower === 't') set('target', cycle(TARGETS, st.target, shift ? -1 : 1));
    else if (lower === 'i') set('retention', cycle(IDLE.map((x) => x[0]), st.retention, shift ? -1 : 1));
    else if (lower === 'q') set('prefixCache', !st.prefixCache);
    else if (lower === 'j') set('spec', !st.spec);
    else if (lower === 'v') set('pd', !st.pd);
    else if (lower === 'd') toggleDeck();
    else if (lower === 'f') toggleFull();
    else if (lower === 'l') togglePalette();
    else if (lower === 's') toggleSound();
    else if (lower === 'r') showScenes($('scenesScreen').hidden);
    else if (lower === 'o') openPlanner();
    else if (key === '?' || lower === 'h') showHelp($('helpScreen').hidden);
    else handled = false;
    if (handled) e.preventDefault();
  }

  /* ---------- presenter comfort: hide the idle cursor in fullscreen, keep the screen awake ---------- */
  let cursorTimer = 0;
  function wakeCursor() {
    $('stage').classList.remove('hide-cursor');
    clearTimeout(cursorTimer);
    cursorTimer = setTimeout(() => { if (document.fullscreenElement) $('stage').classList.add('hide-cursor'); }, 2000);
  }
  let wakeLock = null;
  async function keepAwake() { try { if (navigator.wakeLock && document.visibilityState === 'visible') wakeLock = await navigator.wakeLock.request('screen'); } catch (e) { /* not allowed here */ } }

  /* ---------- boot ---------- */
  const loaded = loadState();
  st = loaded.s;
  applyPalette();
  $('stage').classList.toggle('deck-off', !prefs.deck);
  if (prefs.sound) prefs.sound = false;                     // audio needs a key press to start
  buildHud(); buildDeck(); buildTools(); buildHelp();
  compute();
  world.setView(vm, null);
  render();
  world.start();
  tick(loaded.linked ? 'Opened from the planner.' : sceneIndex != null ? `Scene ${sceneIndex + 1}: ${scenes()[sceneIndex].name}` : 'Ready.', 'neutral');
  $('titleScreen').hidden = loaded.linked || /[?&]skip/.test(location.search);
  if ($('titleScreen').hidden) chat.start();
  document.addEventListener('keydown', onKey);
  $('titleScreen').addEventListener('click', startGame);
  $('helpScreen').addEventListener('click', () => showHelp(false));
  $('blankScreen').addEventListener('click', () => blank(false));
  $('scenesImport').addEventListener('click', importScenes);
  $('scenesReset').addEventListener('click', () => { userScenes = {}; LS.del('cb.arcade.scenes'); showScenes(true); renderTop(); $('scenesMsg').textContent = 'Back to the built-in scenes.'; });
  $('scenesClose').addEventListener('click', () => showScenes(false));
  document.addEventListener('mousemove', wakeCursor);
  window.addEventListener('resize', () => world.resize());
  document.addEventListener('fullscreenchange', () => { wakeCursor(); const b = $('tFull'); if (b) b.setAttribute('aria-pressed', String(!!document.fullscreenElement)); });
  document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') keepAwake(); });
  keepAwake();
  window.addEventListener('hashchange', () => { const linked = C.decodePlan(location.hash); if (linked) { const before = vm; st = C.freshState(Object.assign({}, linked, { mode: 'forward' })); st.autoBuild = false; sceneIndex = null; try { history.replaceState(null, '', location.pathname + location.search); } catch (e) { /* file:// */ } refresh(null, before, 'Opened from the planner'); } });
  // test hook: the state and view model, read-only copies
  window.__arcade = { state: () => snapshot(), vm: () => JSON.parse(JSON.stringify(vm)), world: () => world.stats(), advance: (s) => world.advance(s) };
})();
