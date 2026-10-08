/* sim.js — AI Fit Simulator: a pixel-art serving simulation on the planner's own engine.
 * People think, ask, wait for a free slot, wait for their first token, stream tokens and go back to thinking;
 * their KV cache occupies the rack while they are active. Every number comes from Engine.evaluate. */
const Sim = (() => {
  'use strict';
  const E = typeof Engine !== 'undefined' ? Engine : require('./engine.js');
  const CAT = typeof HARDWARE !== 'undefined' ? { HARDWARE, MODELS, NETWORKS } : require('./catalog.js');
  const { fmtTok, fmtGB, fmtTime } = E;
  const hwById = (id) => CAT.HARDWARE.find((x) => x.id === id);
  const modelById = (id) => CAT.MODELS.find((x) => x.id === id);
  const NET = CAT.NETWORKS.find((n) => n.id === 'ib3200');
  const fmtNum = (n, d = 0) => (n == null || !isFinite(n)) ? '—' : n.toLocaleString('en-US', { maximumFractionDigits: d });
  const fmtMoney = (n) => n == null ? '—' : '$' + n.toLocaleString('en-US', { maximumFractionDigits: n < 100 ? 1 : 0 });

  /* ---------- knobs and presets ---------- */
  const HW_POOL = ['l40s', 'rtx-pro-6000', 'h100-sxm', 'h200-sxm', 'mi300x', 'b200'];
  const MODEL_POOL = ['llama-3.1-8b', 'qwen3-32b', 'llama-3.3-70b', 'gpt-oss-120b', 'qwen3-next-80b', 'qwen3-235b-a22b', 'llama-4-maverick', 'deepseek-v3'];
  const CTX_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1000000];
  const USER_STEPS = [1, 2, 4, 8, 16, 25, 50, 100, 200, 500];
  const PREC = { bf16: 'BF16', fp8: 'FP8', int4: 'INT4', fp4: 'FP4' };
  const KNOBS = [
    ['hw',          { label: 'GPU type',       values: HW_POOL, show: (v) => hwById(v).name, term: 'accelerator' }],
    ['count',       { label: 'GPUs',           values: [1, 2, 4, 8, 16, 32], show: String, term: 'count' }],
    ['model',       { label: 'Model',          values: MODEL_POOL, show: (v) => modelById(v).name, term: 'model' }],
    ['wPrec',       { label: 'Weights',        values: ['bf16', 'fp8', 'int4', 'fp4'], show: (v) => PREC[v], term: 'w-prec' }],
    ['kvPrec',      { label: 'KV cache',       values: ['bf16', 'fp8', 'int4'], show: (v) => PREC[v], term: 'kv-prec' }],
    ['ctx',         { label: 'Context',        values: CTX_STEPS, show: (v) => fmtTok(v) + ' tokens', term: 'ctx' }],
    ['users',       { label: 'People',         values: USER_STEPS, show: String, term: 'users' }],
    ['activity',    { label: 'Active share',   values: [0.1, 0.25, 0.5, 1], show: (v) => Math.round(v * 100) + '%', term: 'activity' }],
    ['target',      { label: 'Wanted tok/s',   values: [5, 10, 20, 30, 50], show: String, term: 'target' }],
    ['retention',   { label: 'Between turns',  values: ['none', 'host', 'gpu'], show: (v) => ({ none: 'evict, prefill again', host: 'park in host RAM', gpu: 'keep in GPU memory' })[v], term: 'retention' }],
    ['prefixCache', { label: 'Prefix caching', values: [false, true], show: (v) => (v ? 'on' : 'off'), term: 'prefix-caching' }],
    ['spec',        { label: 'Speculative',    values: [false, true], show: (v) => (v ? 'on' : 'off'), term: 'spec' }],
  ];
  const PRESETS = [
    { id: 'agents', name: 'Coding agents, 128k', s: { hw: 'h100-sxm', count: 8, model: 'llama-3.3-70b', wPrec: 'fp8', kvPrec: 'fp8', ctx: 131072, users: 50, activity: 0.5, target: 20, retention: 'host' } },
    { id: 'chat', name: 'Team chat on two L40S', s: { hw: 'l40s', count: 2, model: 'llama-3.1-8b', wPrec: 'fp8', kvPrec: 'fp8', ctx: 8192, users: 50, activity: 0.1, target: 20, retention: 'host' } },
    { id: 'docs', name: 'Million-token documents', s: { hw: 'h200-sxm', count: 8, model: 'qwen3-next-80b', wPrec: 'fp8', kvPrec: 'fp8', ctx: 1000000, users: 25, activity: 0.25, target: 15, retention: 'host' } },
    { id: 'frontier', name: 'Frontier MoE, 64k', s: { hw: 'h200-sxm', count: 8, model: 'deepseek-v3', wPrec: 'fp8', kvPrec: 'fp8', ctx: 65536, users: 100, activity: 0.25, target: 20, retention: 'host' } },
    { id: 'workstation', name: 'One workstation GPU', s: { hw: 'rtx-pro-6000', count: 1, model: 'gpt-oss-120b', wPrec: 'fp4', kvPrec: 'fp8', ctx: 32768, users: 8, activity: 0.25, target: 20, retention: 'host' } },
  ];
  const BASE = { hw: 'h100-sxm', count: 8, model: 'llama-3.3-70b', wPrec: 'fp8', kvPrec: 'fp8', ctx: 131072, users: 50, activity: 0.5, target: 20, retention: 'host', prefix: 0, prefixCache: true, spec: false };

  /* ---------- evaluation ---------- */
  function evaluateState(s) {
    const hw = hwById(s.hw), model = modelById(s.model), m = E.norm(model);
    const p = {
      hw, model, wPrec: s.wPrec, kvPrec: s.kvPrec, engine: 'none', count: s.count, nodeGpus: hw.nodeGpus, link: hw.link, net: NET, hostRamGB: 1024,
      dpAttention: true, allowCrossTp: false,
      wl: { users: s.users, activity: s.activity, ctx: s.ctx, prefix: s.prefix || 0, newPrompt: 500, output: 300, retention: s.retention, target: s.target, ttftMax: 1e9 },
      opt: { prefixCache: !!s.prefixCache, spec: !!s.spec, pd: false }, adv: {},
    };
    const r = E.autoConfig(p).best;
    const ctxOK = s.ctx <= m.maxCtx;
    let limit, explain;
    if (!r.fits) {
      limit = 'weights';
      explain = r.kvAvailRaw <= 0 ? `The weights alone (${fmtGB(r.W)}) do not fit on ${r.G} × ${hw.mem} GB. Add GPUs or lower the weight precision.` : `The weights fit (${fmtGB(r.W)}) but not a single ${fmtTok(s.ctx)} session next to them (${fmtGB(r.kv.perSession)}). Shorter context, FP8 KV cache or more GPUs.`;
    } else if (!ctxOK) { limit = 'context'; explain = `${model.name} cannot address ${fmtTok(s.ctx)} tokens; its maximum is ${fmtTok(m.maxCtx)}. Pick a long-context model or shorten the sessions.`; }
    else {
      const sessions = r.maxSessions * r.R, memUsers = s.retention === 'gpu' ? sessions : Math.floor(sessions / s.activity);
      const speedUsers = s.retention === 'gpu' ? Math.floor(r.bSpeed * r.R / s.activity) : Math.floor(r.bSpeed * r.R / s.activity);
      if (r.maxUsers >= s.users) { limit = 'ok'; explain = `Fits with headroom: this rack serves up to ${fmtNum(r.maxUsers)} people (${fmtNum(r.maxConc)} at once) at ${fmtTok(s.ctx)} with at least ${s.target} tok/s each. At your load each person gets about ${r.at.perUser.toFixed(0)} tok/s and the first token takes ${fmtTime(r.ttft)}.`; }
      else if (r.at && r.at.saturated) { limit = 'prefill'; explain = `Prefill is the limit: people send ${fmtTok(r.warmNew)}-token turns faster than the GPUs can compute them. Keep sessions warm (prefix cache, host RAM) or add compute.`; }
      else if (memUsers < s.users && memUsers <= speedUsers) { limit = 'memory'; explain = `Memory is the limit: each ${fmtTok(s.ctx)} session takes ${fmtGB(r.kv.perSession)} and the ${fmtGB(r.kvAvail * r.R)} pool holds ${fmtNum(sessions)} at once, so ${fmtNum(memUsers)} people at ${Math.round(s.activity * 100)}% activity. Levers: FP8 or INT4 KV cache, lower weight precision, bigger GPU memory, shorter context, a model with cheaper KV.`; }
      else { limit = 'speed'; explain = `Bandwidth is the limit: the pool holds ${fmtNum(sessions)} sessions, but past ${fmtNum(r.bSpeed * r.R)} active people each one drops under ${s.target} tok/s, because every step reads all weights and every active KV cache. Levers: speculative decoding, faster memory, FP8 weights, fewer active people, a lower target.`; }
    }
    return { r, hw, model: m, ctxOK, limit, explain };
  }
  function sweep(s, key, values) {
    return values.map((v) => { const ev = evaluateState(Object.assign({}, s, { [key]: v })); const ok = ev.r.fits && ev.ctxOK; return { v, users: ok ? ev.r.maxUsers : 0, conc: ok ? ev.r.maxConc : 0, perUser: ok && ev.r.at ? ev.r.at.perUser : 0 }; });
  }

  /* ---------- the simulation ---------- */
  const expRand = (mean) => (mean <= 0 ? 0 : -Math.log(1 - Math.random()) * mean);
  const SIM = { t: 0, speed: 10, paused: false, people: [], queue: [], active: 0, served: 0, waitSum: 0, waitN: 0, lastHud: 0 };
  const rateCache = new Map();
  let s = Object.assign({}, BASE), ev = null, sweeps = null, sweepTimer = 0;
  const TOKENS_PER_TURN = 300;
  function perUserAt(n) {
    const k = Math.max(1, n);
    if (!rateCache.has(k)) rateCache.set(k, ev.r.fits ? ev.r.loadAt(Math.min(k, Math.max(1, ev.r.maxSessions)), !!s.spec).perUser : 0);
    return rateCache.get(k);
  }
  function slots() { return ev.r.fits && ev.ctxOK ? ev.r.maxSessions * ev.r.R : 0; }
  function thinkMean() { if (s.activity >= 1) return 0; const dur = ev.r.at ? ev.r.at.dur : 10; return dur * (1 - s.activity) / s.activity; }
  function resetSim() {
    rateCache.clear();
    SIM.t = 0; SIM.queue = []; SIM.active = 0; SIM.served = 0; SIM.waitSum = 0; SIM.waitN = 0;
    const n = Math.min(s.users, 240);
    SIM.people = Array.from({ length: n }, (_, i) => ({ id: i, state: 'idle', timer: Math.random() * Math.max(2, thinkMean()), progress: 0, tokens: 0, firstTurn: true, emit: 0 }));
  }
  function startRequest(p) {
    if (SIM.active < slots()) {
      SIM.active++;
      const cold = p.firstTurn || s.retention === 'none';
      p.state = 'prefill'; p.progress = 0; p.ttft = Math.max(0.05, cold ? ev.r.ttftCold : ev.r.ttftWarm);
      if (p.waitStart != null) { SIM.waitSum += SIM.t - p.waitStart; SIM.waitN++; p.waitStart = null; }
    } else { p.state = 'waiting'; if (p.waitStart == null) p.waitStart = SIM.t; if (!SIM.queue.includes(p)) SIM.queue.push(p); }
  }
  function step(dt) {
    SIM.t += dt;
    for (const p of SIM.people) {
      if (p.state === 'idle') { p.timer -= dt; if (p.timer <= 0) startRequest(p); }
      else if (p.state === 'prefill') { p.progress += dt / p.ttft; if (p.progress >= 1) { p.state = 'decode'; p.tokens = 0; } }
      else if (p.state === 'decode') {
        const rate = perUserAt(SIM.active);
        p.tokens += rate * dt;
        if (p.tokens >= TOKENS_PER_TURN) { p.state = 'idle'; p.firstTurn = false; p.timer = expRand(thinkMean()); SIM.active--; SIM.served++; const next = SIM.queue.shift(); if (next) startRequest(next); }
      }
    }
  }

  /* ---------- pixel drawing ---------- */
  const W = 320, H = 180;
  const hueOf = (u) => (u * 47) % 360;
  function drawGpu(c, x, y, w, hgt, g, t) {
    c.fillStyle = '#9aa3ad'; c.fillRect(x, y, w, hgt);
    c.fillStyle = '#161b24'; c.fillRect(x + 1, y + 1, w - 2, hgt - 2);
    const cols = Math.floor((w - 2) / 2), rows = Math.floor((hgt - 2) / 2), cells = cols * rows;
    const segs = [];
    if (g.overflow) segs.push({ n: cells, color: '#b43a3a' });
    else {
      segs.push({ n: Math.round(g.headroom * cells), color: '#232a36', hatch: true });
      segs.push({ n: Math.round(g.overhead * cells), color: '#4b5563' });
      segs.push({ n: Math.round(g.weights * cells), color: '#2f5fd1' });
      if (g.sharedFrac > 0) segs.push({ n: Math.max(1, Math.round(g.sharedFrac * cells)), color: '#e0b84a' });
      for (const sess of g.sessions) segs.push({ n: Math.max(1, Math.round(sess.frac * cells)), color: sess.active ? `hsl(${hueOf(sess.user)} 70% 58%)` : `hsl(${hueOf(sess.user)} 30% 30%)` });
    }
    let k = 0;
    for (const seg of segs) for (let i = 0; i < seg.n && k < cells; i++, k++) {
      const col = k % cols, row = Math.floor(k / cols);
      if (seg.hatch && ((col + row) % 2)) continue;
      c.fillStyle = seg.color; c.fillRect(x + 1 + col * 2, y + hgt - 3 - row * 2, 2, 2);
    }
    c.fillStyle = g.sessions.some((z) => z.active) && Math.floor(t * 4) % 2 ? '#3ddc84' : '#1e5e3a'; c.fillRect(x + w - 4, y - 3, 2, 2);
  }
  function drawPerson(c, x, y, p, t) {
    const hue = hueOf(p.id);
    const idle = p.state === 'idle';
    const skin = '#f1c27d', hair = `hsl(${hue} 40% 25%)`, body = idle ? `hsl(${hue} 30% 36%)` : `hsl(${hue} 65% 52%)`, legs = '#2b2f3a';
    const bob = p.state === 'decode' ? Math.round(Math.sin(t * 6 + p.id) * 0.5 + 0.5) : 0;
    y += bob;
    c.fillStyle = hair; c.fillRect(x + 1, y, 4, 1); c.fillRect(x, y + 1, 1, 1); c.fillRect(x + 5, y + 1, 1, 1);
    c.fillStyle = skin; c.fillRect(x + 1, y + 1, 4, 3);
    c.fillStyle = '#1a1a1a'; c.fillRect(x + 2, y + 2, 1, 1); c.fillRect(x + 4, y + 2, 1, 1);
    c.fillStyle = body; c.fillRect(x + 1, y + 4, 4, 3); c.fillRect(x, y + 5, 6, 1);
    const stepf = Math.floor(t * 4 + p.id) % 2;
    c.fillStyle = legs; c.fillRect(x + 1, y + 7, 1, 2); c.fillRect(x + 4, y + 7 + (idle ? 0 : stepf), 1, 2 - (idle ? 0 : stepf));
    if (p.state === 'waiting') { c.fillStyle = '#f2c14e'; c.fillRect(x + 2, y - 4, 2, 3); c.fillRect(x + 1, y - 5, 4, 1); }                 // hourglass-ish
    if (p.state === 'prefill') { c.fillStyle = '#263041'; c.fillRect(x - 1, y - 4, 8, 2); c.fillStyle = '#7fb0ea'; c.fillRect(x - 1, y - 4, Math.max(1, Math.round(8 * Math.min(1, p.progress))), 2); }
  }
  function drawScene(c, S, dots, t) {
    c.fillStyle = '#0d1117'; c.fillRect(0, 0, W, H);
    c.fillStyle = '#131a24'; c.fillRect(4, 12, 204, 140); c.fillRect(210, 12, 106, 156);
    c.fillStyle = '#263041'; c.fillRect(4, 150, 204, 2); c.fillRect(210, 166, 106, 2);
    const n = S.gpus.length;
    if (n) {
      const perRow = n <= 8 ? n : 8, rows = n <= 8 ? 1 : 2;
      const boxW = Math.max(8, Math.min(22, Math.floor(196 / perRow) - 2)), boxH = rows === 1 ? 124 : 60;
      const rowW = perRow * (boxW + 2) - 2, x0 = 6 + Math.floor((200 - rowW) / 2);
      S.gpus.forEach((g, i) => { const col = i % perRow, row = Math.floor(i / perRow); drawGpu(c, x0 + col * (boxW + 2), 22 + row * (boxH + 8), boxW, boxH, g, t); });
    }
    // host RAM strip
    c.fillStyle = '#1a2130'; c.fillRect(4, 156, 204, 12);
    c.fillStyle = '#9aa3ad'; c.fillRect(4, 156, 204, 1);
    if (S.host.capacity > 0) {
      const cellW = Math.max(1, Math.min(4, Math.floor(200 / Math.max(1, S.host.capacity))));
      S.host.ids.slice(0, Math.floor(200 / cellW)).forEach((id, i) => { c.fillStyle = `hsl(${hueOf(id)} 45% 42%)`; c.fillRect(6 + i * cellW, 159, Math.max(1, cellW - 1), 6); });
    }
    S.people.forEach((p, i) => drawPerson(c, 216 + (i % 8) * 12, 28 + Math.floor(i / 8) * 23, p, t));
    for (const d of dots) { c.fillStyle = `hsl(${d.hue} 80% 70%)`; c.fillRect(Math.round(d.x), Math.round(d.y), 1, 1); }
  }
  function sceneModel() {
    const { r, hw } = ev, util = 0.9, mem = hw.mem * 1e9, G = r.G, R = r.R, total = r.total;
    const activeIds = SIM.people.filter((p) => p.state === 'prefill' || p.state === 'decode').map((p) => p.id);
    const retainedIds = s.retention !== 'none' ? SIM.people.filter((p) => p.state === 'idle' && !p.firstTurn).map((p) => p.id) : [];
    const gpuResident = s.retention === 'gpu' ? activeIds.concat(retainedIds) : activeIds;
    const gpus = [];
    for (let i = 0; i < Math.min(total, 16); i++) {
      const rep = Math.floor(i / G);
      const here = gpuResident.filter((id, k) => k % R === rep).slice(0, Math.max(1, Math.ceil(r.maxSessions)));
      gpus.push({ headroom: 1 - util, overhead: Math.min(1, r.overhead / mem), weights: Math.min(1, (r.W / G) / mem), overflow: r.kvAvailRaw <= 0,
        sessions: here.map((id) => ({ user: id, frac: (r.kv.perSession / G) / mem, active: activeIds.includes(id) })), sharedFrac: r.S > 0 ? (r.kv.shared / G) / mem : 0 });
    }
    const host = { capacity: s.retention === 'host' ? r.hostSessions : 0, ids: s.retention === 'host' ? retainedIds : [] };
    return { gpus, people: SIM.people.slice(0, 48), host, hiddenGpus: Math.max(0, total - 16), hiddenPeople: Math.max(0, s.users - 48) };
  }

  /* ---------- DOM ---------- */
  const el = (tag, attrs, ...kids) => {
    const n = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) { if (v == null || v === false) continue; if (k === 'class') n.className = v; else if (k.startsWith('on')) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v === true ? '' : v); }
    for (const kid of kids.flat(Infinity)) { if (kid == null || kid === false) continue; n.append(kid.nodeType ? kid : document.createTextNode(String(kid))); }
    return n;
  };
  const LS = { get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } }, set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* ignore */ } } };
  const term = (key, text) => el('span', { class: 'term', 'data-term': key, tabindex: 0 }, text);
  let root = null, canvas = null, ctx2d = null, raf = 0, dots = [], lastT = 0, scene = null;

  function recompute(resetPeople) {
    ev = evaluateState(s);
    rateCache.clear();
    if (resetPeople) resetSim(); else { for (const p of SIM.people) if (p.state === 'prefill') p.ttft = Math.max(0.05, p.firstTurn || s.retention === 'none' ? ev.r.ttftCold : ev.r.ttftWarm); }
    LS.set('cb.sim', s);
    renderHud(); renderExplain();
    clearTimeout(sweepTimer); sweepTimer = setTimeout(renderSweeps, 120);
  }
  function knobRow([key, K]) {
    let idx = K.values.findIndex((v) => v === s[key]); if (idx < 0) idx = 0;
    const show = el('span', { class: 'sim-val' }, K.show(K.values[idx]));
    const set = (d) => { idx = (idx + d + K.values.length) % K.values.length; s[key] = K.values[idx]; show.textContent = K.show(K.values[idx]); recompute(key === 'users' || key === 'activity' || key === 'retention'); };
    return el('div', { class: 'sim-knob' }, term(K.term, K.label),
      el('div', { class: 'sim-kctl' }, el('button', { class: 'pxbtn', type: 'button', 'aria-label': `${K.label}: previous`, onclick: () => set(-1) }, '◀'), show, el('button', { class: 'pxbtn', type: 'button', 'aria-label': `${K.label}: next`, onclick: () => set(1) }, '▶')));
  }
  function renderPanel() {
    root.replaceChildren(
      el('div', { class: 'sim-head' }, el('h2', { class: 'sim-title' }, 'Serving simulator'),
        el('p', { class: 'sim-sub' }, 'Turn any knob and watch the rack and the crowd react. The rack fills with weights and one block per active person\'s KV cache; people wait for a free slot, wait for their first token, stream tokens, then think. All numbers are the planner\'s.')),
      el('div', { class: 'sim-presets' }, el('span', { class: 'sim-plabel' }, 'Start from'), PRESETS.map((pr) => el('button', { class: 'pxbtn ghost', type: 'button', onclick: () => { Object.assign(s, BASE, pr.s); renderPanel(); recompute(true); } }, pr.name))),
      el('div', { class: 'sim-stage' }, canvas, el('div', { class: 'sim-legend' },
        el('span', null, el('i', { style: 'background:#2f5fd1' }), 'weights'), el('span', null, el('i', { style: 'background:#4b5563' }), 'runtime overhead'), el('span', null, el('i', { style: 'background:#232a36' }), 'headroom'),
        el('span', null, el('i', { style: 'background:hsl(94 70% 58%)' }), 'active person\'s KV cache'), el('span', null, el('i', { style: 'background:hsl(94 30% 30%)' }), 'parked in GPU memory'), el('span', null, el('i', { style: 'background:#e0b84a' }), 'shared prefix'),
        el('span', null, el('i', { style: 'background:#b43a3a' }), 'weights overflow'), el('span', null, el('i', { style: 'background:#1a2130;border:1px solid #9aa3ad' }), 'host RAM strip'),
        el('span', null, el('i', { style: 'background:#f2c14e' }), 'waiting for a slot'), el('span', null, el('i', { style: 'background:#7fb0ea' }), 'waiting for the first token'),
        el('span', { id: 'simHidden' }))),
      el('div', { class: 'sim-controls' },
        el('span', { class: 'sim-plabel' }, 'Simulation speed'),
        [1, 10, 60].map((v) => el('button', { class: 'pxbtn' + (SIM.speed === v ? ' on' : ''), type: 'button', 'data-speed': v, onclick: (e) => { SIM.speed = v; root.querySelectorAll('[data-speed]').forEach((b) => b.classList.toggle('on', +b.dataset.speed === v)); } }, v + '×')),
        el('button', { class: 'pxbtn ghost', type: 'button', onclick: (e) => { SIM.paused = !SIM.paused; e.target.textContent = SIM.paused ? 'Resume' : 'Pause'; } }, 'Pause'),
        el('button', { class: 'pxbtn ghost', type: 'button', onclick: () => { resetSim(); dots = []; } }, 'Restart people'),
        el('span', { class: 'sim-clock', id: 'simClock' })),
      el('div', { class: 'sim-hud', id: 'simHud' }),
      el('div', { class: 'sim-explain', id: 'simExplain' }),
      el('div', { class: 'sim-knobs' }, KNOBS.map(knobRow)),
      el('div', { class: 'sim-sweeps', id: 'simSweeps' }),
      el('div', { class: 'sim-nav' }, el('button', { class: 'pxbtn go', type: 'button', onclick: () => window.openSimInPlanner && window.openSimInPlanner(s) }, 'Open this setup in the planner ▶'), el('button', { class: 'pxbtn ghost', type: 'button', onclick: () => { s = Object.assign({}, BASE); renderPanel(); recompute(true); } }, 'Reset')));
  }
  function renderHud() {
    const hud = root.querySelector('#simHud'); if (!hud) return;
    const r = ev.r;
    const active = SIM.active, waiting = SIM.queue.length;
    const chip = (label, value, cls, key) => el('div', { class: 'sim-chip' + (cls ? ' ' + cls : '') }, key ? term(key, label) : el('span', null, label), el('b', null, value));
    hud.replaceChildren(
      chip('active now', fmtNum(active), '', 'concurrent'), chip('waiting for a slot', fmtNum(waiting), waiting ? 'bad' : '', 'concurrent'),
      chip('tok/s each right now', r.fits ? fmtNum(perUserAt(active)) : '0', r.fits && perUserAt(active) < s.target ? 'bad' : 'good', 'speed-per-user'),
      chip('first token', r.fits ? fmtTime(r.ttftWarm) + (s.retention === 'none' ? '' : ` (cold ${fmtTime(r.ttftCold)})`) : '—', '', 'ttft'),
      chip('capacity: people', r.fits && ev.ctxOK ? fmtNum(r.maxUsers) : '0', r.fits && r.maxUsers >= s.users ? 'good' : 'bad', 'users'),
      chip('capacity: at once', r.fits && ev.ctxOK ? fmtNum(r.maxConc) : '0', '', 'concurrent'),
      chip('KV per person', fmtGB(r.kv.perSession), '', 'kv-per-session'), chip('KV pool', r.fits ? fmtGB(r.kvAvail * r.R) : '0', '', 'kv-pool'), chip('weights', fmtGB(r.W), '', 'weights-mem'),
      chip('layout', `TP ${r.tp} × PP ${r.pp} × ${r.R}`, '', 'layout-col'), chip('price', r.price != null ? fmtMoney(r.price) + '/h' : '—', '', 'cost-hour'), chip('served turns', fmtNum(SIM.served)));
    const hid = root.querySelector('#simHidden'); if (hid) hid.textContent = (scene && scene.hiddenGpus ? `+${scene.hiddenGpus} GPUs not drawn` : '') + (s.users > 48 ? ` · ${Math.min(s.users, 240) - 48} more people simulated but not drawn` : '');
  }
  function renderExplain() {
    const box = root.querySelector('#simExplain'); if (!box) return;
    const cls = { ok: 'good', memory: 'bad', speed: 'bad', weights: 'bad', context: 'bad', prefill: 'bad' }[ev.limit];
    const label = { ok: 'ROOM TO SPARE', memory: 'MEMORY-BOUND', speed: 'BANDWIDTH-BOUND', weights: 'DOES NOT LOAD', context: 'CONTEXT TOO LONG', prefill: 'PREFILL-BOUND' }[ev.limit];
    box.replaceChildren(el('div', { class: 'sim-limit ' + cls }, label), el('p', null, ev.explain));
  }
  function barChart(title, items, current, fmtLabel, termKey) {
    const max = Math.max(1, ...items.map((i) => i.users));
    const bars = items.map((i) => {
      const hpx = i.users > 0 ? 8 + Math.round(52 * Math.log10(1 + i.users) / Math.log10(1 + max) / 4) * 4 : 2;
      return el('div', { class: 'sim-bar' + (i.v === current ? ' on' : ''), title: `${fmtLabel(i.v)}: ${fmtNum(i.users)} people (${fmtNum(i.conc)} at once)` },
        el('span', { class: 'sim-barv' }, fmtNum(i.users)), el('i', { style: `height:${hpx}px` }), el('span', { class: 'sim-barl' }, fmtLabel(i.v)));
    });
    return el('div', { class: 'sim-chart' }, el('div', { class: 'sim-ctitle' }, termKey ? term(termKey, title) : title), el('div', { class: 'sim-bars' }, bars));
  }
  function renderSweeps() {
    const box = root.querySelector('#simSweeps'); if (!box) return;
    const byCtx = sweep(s, 'ctx', CTX_STEPS), byHw = sweep(s, 'hw', HW_POOL), byModel = sweep(s, 'model', MODEL_POOL), byKv = sweep(s, 'kvPrec', ['bf16', 'fp8', 'int4']), byW = sweep(s, 'wPrec', ['bf16', 'fp8', 'int4', 'fp4']);
    box.replaceChildren(
      el('h3', null, 'How many people this setup serves if you change one thing'),
      barChart('context per person', byCtx, s.ctx, (v) => fmtTok(v), 'ctx'),
      barChart(`GPU type (${s.count} of them)`, byHw, s.hw, (v) => hwById(v).name.replace(/ \d+ GB.*$/, '').replace('Instinct ', '').replace('GeForce ', '').replace(' Blackwell', ''), 'accelerator'),
      barChart('model', byModel, s.model, (v) => modelById(v).name.replace(/ \/.*$/, '').replace(/\(.*\)/, '').trim(), 'model'),
      barChart('KV cache precision', byKv, s.kvPrec, (v) => PREC[v], 'kv-prec'),
      barChart('weight precision', byW, s.wPrec, (v) => PREC[v], 'w-prec'));
  }

  /* ---------- animation ---------- */
  const reduced = typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function frame(ts) {
    const t = ts / 1000, dt = Math.min(0.1, lastT ? t - lastT : 0.016); lastT = t;
    if (!SIM.paused) { const simDt = dt * SIM.speed; const sub = Math.ceil(simDt / 0.25); for (let i = 0; i < sub; i++) step(simDt / sub); }
    scene = sceneModel();
    if (!reduced && !SIM.paused) {
      scene.people.forEach((p, i) => {
        if (p.state !== 'decode') return;
        p.emit += dt * Math.min(12, Math.max(0.5, perUserAt(SIM.active) / 6));
        while (p.emit >= 1 && dots.length < 400) { p.emit -= 1; dots.push({ x: 216 + (i % 8) * 12 + 2 + Math.random() * 3, y: 28 + Math.floor(i / 8) * 23 - 2, vy: -(8 + Math.random() * 6), life: 1.4, hue: hueOf(p.id) }); }
      });
      for (const d of dots) { d.y += d.vy * dt; d.life -= dt; }
      dots = dots.filter((d) => d.life > 0 && d.y > 14);
    }
    drawScene(ctx2d, scene, dots, reduced ? 0 : t);
    if (t - SIM.lastHud > 0.4) { SIM.lastHud = t; renderHud(); const clk = root.querySelector('#simClock'); if (clk) clk.textContent = `simulated ${fmtTime(SIM.t)}`; }
    raf = requestAnimationFrame(frame);
  }
  function render(container) {
    if (!root) {
      s = Object.assign({}, BASE, LS.get('cb.sim', {}));
      root = el('div', { class: 'sim' });
      canvas = el('canvas', { width: W, height: H, class: 'sim-canvas', role: 'img', 'aria-label': 'Pixel rack of GPUs on the left filling with weights and the KV cache of each active person; a host-RAM strip below it; a crowd of people on the right who wait, prefill and stream tokens' });
      ctx2d = canvas.getContext('2d'); ctx2d.imageSmoothingEnabled = false;
      container.replaceChildren(root);
      renderPanel(); recompute(true);
    }
    if (!raf) raf = requestAnimationFrame(frame);
  }
  function pause() { if (raf) cancelAnimationFrame(raf); raf = 0; lastT = 0; }
  return { render, pause, BASE, PRESETS, KNOBS, CTX_STEPS, HW_POOL, MODEL_POOL, evaluateState, sweep };
})();
if (typeof module !== 'undefined') module.exports = Sim;
