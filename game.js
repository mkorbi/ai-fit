/* game.js — AI Fit Arcade: a pixel-art game that shows how model size, GPU type, context, precision and the number of
 * people trade off. Every number on screen comes from the same engine as the planner. */
const Arcade = (() => {
  'use strict';
  const E = typeof Engine !== 'undefined' ? Engine : require('./engine.js');
  const CAT = typeof HARDWARE !== 'undefined' ? { HARDWARE, MODELS, NETWORKS } : require('./catalog.js');
  const { fmtTok, fmtGB, fmtTime } = E;
  const hwById = (id) => CAT.HARDWARE.find((x) => x.id === id);
  const modelById = (id) => CAT.MODELS.find((x) => x.id === id);
  const NET = CAT.NETWORKS.find((n) => n.id === 'ib3200');
  const fmtNum = (n, d = 0) => (n == null || !isFinite(n)) ? '—' : n.toLocaleString('en-US', { maximumFractionDigits: d });
  const fmtMoney = (n) => n == null ? '—' : '$' + n.toLocaleString('en-US', { maximumFractionDigits: n < 100 ? 1 : 0 });

  /* ---------- knobs ---------- */
  const HW_POOL = ['l40s', 'rtx-pro-6000', 'h100-sxm', 'h200-sxm', 'mi300x', 'b200'];
  const MODEL_POOL = ['llama-3.1-8b', 'qwen3-32b', 'llama-3.3-70b', 'gpt-oss-120b', 'qwen3-next-80b', 'qwen3-235b-a22b', 'llama-4-maverick', 'deepseek-v3'];
  const CTX_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1000000];
  const USER_STEPS = [1, 2, 4, 8, 16, 25, 50, 100, 200, 500];
  const KNOBS = {
    count:       { label: 'GPUs',           values: (lv) => Array.from({ length: lv.maxGpus || 16 }, (_, i) => i + 1), show: (v) => String(v), term: 'count' },
    hw:          { label: 'GPU type',       values: (lv) => lv.hwPool || HW_POOL, show: (v) => hwById(v).name, term: 'accelerator' },
    model:       { label: 'Model',          values: (lv) => lv.modelPool || MODEL_POOL, show: (v) => modelById(v).name, term: 'model' },
    wPrec:       { label: 'Weights',        values: () => ['bf16', 'fp8', 'int4'], show: (v) => ({ bf16: 'BF16', fp8: 'FP8', int4: 'INT4' })[v], term: 'w-prec' },
    kvPrec:      { label: 'KV cache',       values: () => ['bf16', 'fp8', 'int4'], show: (v) => ({ bf16: 'BF16', fp8: 'FP8', int4: 'INT4' })[v], term: 'kv-prec' },
    ctx:         { label: 'Context',        values: (lv) => CTX_STEPS.filter((c) => c <= (lv.maxCtx || 1000000)), show: (v) => fmtTok(v) + ' tokens', term: 'ctx' },
    users:       { label: 'Users',          values: () => USER_STEPS, show: (v) => String(v), term: 'users' },
    activity:    { label: 'Active share',   values: () => [0.1, 0.25, 0.5, 1], show: (v) => Math.round(v * 100) + '%', term: 'activity' },
    prefixCache: { label: 'Prefix caching', values: () => [false, true], show: (v) => (v ? 'on' : 'off'), term: 'prefix-caching' },
    spec:        { label: 'Speculative',    values: () => [false, true], show: (v) => (v ? 'on' : 'off'), term: 'spec' },
    retention:   { label: 'Between turns',  values: () => ['none', 'host'], show: (v) => (v === 'host' ? 'keep KV in host RAM' : 'evict, prefill again'), term: 'retention' },
  };
  const BASE = { hw: 'h100-sxm', count: 8, model: 'llama-3.3-70b', wPrec: 'bf16', kvPrec: 'bf16', ctx: 8192, users: 1, prefix: 0, prefixCache: true, spec: false, retention: 'host', activity: 1 };

  /* ---------- missions ---------- */
  const LEVELS = [
    { id: 'weights', title: 'Fit the weights', mission: 'Load Llama 3.3 70B in BF16 so that one person can chat with 8k context.',
      lesson: 'A model is a block of [[params|parameters]]: 70 billion of them at 2 bytes each in [[w-prec|BF16]] is 141 GB before a single user shows up. One H100 holds 80 GB. [[tp|Tensor parallelism]] splits the block across the GPUs of a node, and every GPU keeps a slice.',
      start: { count: 1 }, knobs: ['count'], maxGpus: 8, goal: { users: 1, ctx: 8192, target: 1, ttftMax: 1e9 }, solve: { count: 4 } },
    { id: 'boxes', title: 'People are boxes', mission: 'Serve 60 people at 8k context with at least 10 tokens per second each.',
      lesson: 'Each session is a box of [[kv-per-session|KV cache]]: [[ctx|context]] × [[kv-per-token|bytes per token]]. 8k × 328 KB is about 2.7 GB. The shelf is whatever memory is left after the weights, so more GPUs means a longer shelf.',
      start: { count: 4 }, knobs: ['count'], maxGpus: 8, goal: { users: 60, ctx: 8192, target: 10, ttftMax: 1e9 }, solve: { count: 8 } },
    { id: 'context', title: 'Context is the multiplier', mission: 'The same 60 people now open 128k-token sessions. Serve them at 20 tokens per second each.',
      lesson: 'The same rack holds about 148 sessions at 8k and 9 at 128k: context multiplies the size of every box. Capping the context is the cheapest lever there is.',
      start: { count: 8, ctx: 131072 }, knobs: ['ctx'], maxCtx: 131072, goal: { users: 60, target: 20, ttftMax: 1e9 }, solve: { ctx: 16384 } },
    { id: 'kv', title: 'Shrink the boxes', mission: 'Serve 16 people at 128k context on 8× H100 with at least 15 tokens per second.',
      lesson: '[[kv-prec|KV cache precision]] sets the bytes per token. FP8 halves every box with rarely measurable quality loss; INT4 halves it again but few engines offer it and long contexts suffer.',
      start: { count: 8, ctx: 131072, kvPrec: 'bf16' }, knobs: ['kvPrec'], goal: { users: 16, ctx: 131072, target: 15, ttftMax: 1e9 }, solve: { kvPrec: 'fp8' } },
    { id: 'w', title: 'Shrink the block', mission: 'Serve 21 people at 128k on the same rack. The KV cache is already FP8.',
      lesson: '[[w-prec|Weight precision]]: FP8 halves the block and frees 70 GB, three more 128k sessions on this rack. Hopper computes FP8 natively, so decode gets faster too. INT4 frees even more but computes in BF16.',
      start: { count: 8, ctx: 131072, kvPrec: 'fp8', wPrec: 'bf16' }, knobs: ['wPrec'], goal: { users: 21, ctx: 131072, target: 15, ttftMax: 1e9 }, solve: { wPrec: 'fp8' } },
    { id: 'gpu', title: 'A bigger box', mission: 'Serve 40 people at 128k with FP8 weights and FP8 KV cache, eight GPUs at most.',
      lesson: '[[accelerator|GPU memory]] is the lever for context: H100 80 GB, H200 141 GB, B200 180 GB with the same number of GPUs. Bandwidth rises with the newer parts too, so tokens per second follow.',
      start: { hw: 'h100-sxm', count: 8, ctx: 131072, wPrec: 'fp8', kvPrec: 'fp8' }, knobs: ['hw', 'count'], hwPool: ['h100-sxm', 'h200-sxm', 'b200'], maxGpus: 8, goal: { users: 40, ctx: 131072, target: 20, ttftMax: 1e9 }, solve: { hw: 'h200-sxm' } },
    { id: 'model', title: 'Choose the model', mission: 'Twenty people want 1M-token sessions on 8× H200. Pick a model that can.',
      lesson: '[[kv-per-token|KV per token]] is architecture: grouped-query attention in a 70B model costs 328 KB, multi-head latent attention (DeepSeek) 70 KB, hybrid linear attention (Qwen3-Next) 25 KB, chunked attention (Llama 4) caps most layers at 8k. A model that cannot address a million tokens fails whatever the memory.',
      start: { hw: 'h200-sxm', count: 8, ctx: 1000000, wPrec: 'fp8', kvPrec: 'fp8', model: 'llama-3.3-70b' }, knobs: ['model'], modelPool: ['llama-3.3-70b', 'deepseek-v3', 'llama-4-maverick', 'qwen3-next-80b'], goal: { users: 20, ctx: 1000000, target: 15, ttftMax: 1e9 }, solve: { model: 'qwen3-next-80b' } },
    { id: 'speed', title: 'Speed is bandwidth', mission: 'Eighty people at 32k fit on 8× H100 with FP8 everywhere, but they want 50 tokens per second each.',
      lesson: '[[speed-per-user|Tokens per second]] is [[bw|memory bandwidth]] divided by the bytes read per step, and every step reads all the weights plus every active session\'s KV cache. [[spec|Speculative decoding]] gets several tokens out of one read; faster memory reads faster.',
      start: { hw: 'h100-sxm', count: 8, ctx: 32768, wPrec: 'fp8', kvPrec: 'fp8', spec: false }, knobs: ['hw', 'spec'], hwPool: ['h100-sxm', 'h200-sxm', 'b200'], goal: { users: 80, ctx: 32768, target: 50, ttftMax: 1e9 }, solve: { spec: true } },
    { id: 'prefix', title: 'Share the prefix', mission: 'Fifty people each load the same 60k-token knowledge pack plus 8k of their own context on 8× H100 (FP8/FP8).',
      lesson: '[[prefix-caching|Prefix caching]]: tokens that are identical for everyone, a system prompt or a shared document pack, are stored once per replica. Each person then pays only for their own part, and their first token arrives without recomputing the pack.',
      start: { hw: 'h100-sxm', count: 8, ctx: 69632, wPrec: 'fp8', kvPrec: 'fp8', prefix: 61440, prefixCache: false }, knobs: ['prefixCache'], goal: { users: 50, ctx: 69632, target: 15, ttftMax: 1e9 }, solve: { prefixCache: true } },
    { id: 'ttft', title: 'The first token', mission: 'One person opens a 1M-token document with Qwen3-Next on H100s and wants the first token within 30 seconds, turn after turn.',
      lesson: '[[ttft|Time to first token]] is prefill: compute that grows with the square of the context. A million tokens on two H100s takes most of a minute. Keep the session\'s KV cache between turns and the next turn prefills only the new words; more GPUs in tensor parallel also split the work.',
      start: { hw: 'h100-sxm', count: 2, model: 'qwen3-next-80b', ctx: 1000000, wPrec: 'fp8', kvPrec: 'fp8', retention: 'none' }, knobs: ['count', 'retention'], maxGpus: 8, goal: { users: 1, ctx: 1000000, target: 5, ttftMax: 30 }, solve: { retention: 'host' } },
    { id: 'budget', title: 'Mind the budget', mission: 'A hundred people, a quarter of them active at any moment, 64k context, 20 tokens per second, first token within 5 s, for at most $30 an hour.',
      lesson: '[[cost-token|Cost per token]] is the hourly price divided by the tokens per hour you actually get. The cheapest rack is rarely the one with the most GPUs: precision, memory per GPU and the price per GPU-hour all pull in different directions.',
      start: { hw: 'h100-sxm', count: 8, model: 'llama-3.3-70b', ctx: 65536, wPrec: 'bf16', kvPrec: 'bf16', activity: 0.25 }, knobs: ['hw', 'count', 'wPrec', 'kvPrec', 'spec'], hwPool: HW_POOL, maxGpus: 16, goal: { users: 100, activity: 0.25, ctx: 65536, target: 20, ttftMax: 5, budget: 30 }, solve: { wPrec: 'fp8', kvPrec: 'fp8' }, stars: (r) => (r.price <= 12 ? 3 : r.price <= 20 ? 2 : 1) },
    { id: 'free', title: 'Free play', mission: 'No goal. Turn every knob and watch the rack and the crowd react; open the planner for the full picture.',
      lesson: 'Everything here is the planner\'s own arithmetic: weights = parameters × bytes, boxes = context × bytes per token, speed = bandwidth ÷ bytes read per step, first token = prefill compute. The planner tab adds engines, networks, costs and the reverse search.',
      start: { hw: 'h100-sxm', count: 8, model: 'llama-3.3-70b', ctx: 32768, users: 25, wPrec: 'fp8', kvPrec: 'fp8', activity: 1 }, knobs: ['hw', 'count', 'model', 'wPrec', 'kvPrec', 'ctx', 'users', 'activity', 'prefixCache', 'spec', 'retention'], maxGpus: 16, goal: null },
  ];

  /* ---------- evaluation (shared with tests) ---------- */
  function stateFor(level) {
    const s = Object.assign({}, BASE, level.start);
    if (level.goal) { if (level.goal.users != null) s.users = level.goal.users; if (level.goal.ctx != null) s.ctx = level.goal.ctx; if (level.goal.activity != null) s.activity = level.goal.activity; }
    return s;
  }
  function evaluateState(s, level) {
    const hw = hwById(s.hw), model = modelById(s.model);
    const goal = level.goal || {};
    const p = {
      hw, model, wPrec: s.wPrec, kvPrec: s.kvPrec, engine: 'none', count: s.count, nodeGpus: hw.nodeGpus, link: hw.link, net: NET, hostRamGB: 1024,
      dpAttention: true, allowCrossTp: false,
      wl: { users: s.users, activity: s.activity, ctx: s.ctx, prefix: s.prefix || 0, newPrompt: 500, output: 300, retention: s.retention, target: goal.target || 1, ttftMax: goal.ttftMax || 1e9 },
      opt: { prefixCache: !!s.prefixCache, spec: !!s.spec, pd: false }, adv: {},
    };
    const r = E.autoConfig(p).best;
    const m = E.norm(model);
    const reasons = [];
    if (!r.fits) reasons.push(r.kvAvailRaw <= 0 ? `the weights (${fmtGB(r.W)}) do not fit on ${r.G} × ${hw.mem} GB` : `one ${fmtTok(s.ctx)} session (${fmtGB(r.kv.perSession)}) does not fit next to the weights`);
    else if (s.ctx > m.maxCtx) reasons.push(`${model.name} cannot address ${fmtTok(s.ctx)} tokens (its maximum is ${fmtTok(m.maxCtx)})`);
    else if (!r.memOK) reasons.push(`only ${fmtNum(r.maxSessions * r.R)} sessions of ${fmtTok(s.ctx)} fit (${fmtGB(r.kv.perSession)} each in a ${fmtGB(r.kvAvail * r.R)} pool); ${fmtNum(r.residentUsers)} needed`);
    else if (!r.ttftOK) reasons.push(`the first token takes ${fmtTime(r.ttft)} (limit ${fmtTime(goal.ttftMax)})`);
    else if (r.at && r.at.saturated) reasons.push('prefill saturates the GPUs: prompts arrive faster than they can be computed');
    else if (!r.speedOK) reasons.push(`each person gets ${r.at.perUser.toFixed(1)} tokens per second (goal ${goal.target})`);
    if (goal.budget != null && r.price != null && r.price > goal.budget) reasons.push(`${fmtMoney(r.price)}/h is over the ${fmtMoney(goal.budget)}/h budget`);
    if (level.maxGpus && s.count > level.maxGpus) reasons.push(`at most ${level.maxGpus} GPUs`);
    const ok = level.goal ? reasons.length === 0 : true;
    return { r, hw, model: m, ok, reasons, stars: ok && level.stars ? level.stars(r) : ok ? 1 : 0 };
  }

  /* ---------- scene model ---------- */
  function sceneFor(s, ev) {
    const { r, hw } = ev;
    const util = 0.9;
    const mem = hw.mem * 1e9;
    const G = r.G, R = r.R, total = r.total;
    const served = r.fits ? Math.min(r.B, r.maxSessions * R) : 0;
    const perGpuWeights = r.W / G;
    const gpus = [];
    for (let i = 0; i < Math.min(total, 16); i++) {
      const rep = Math.floor(i / G);
      const sessionsHere = [];
      for (let u = rep; u < served; u += R) sessionsHere.push(u);           // round-robin across replicas
      gpus.push({ headroom: mem * (1 - util) / mem, overhead: Math.min(1, r.overhead / mem), weights: Math.min(1, perGpuWeights / mem), overflow: r.kvAvailRaw <= 0,
        sessions: sessionsHere.map((u) => ({ user: u, frac: (r.kv.perSession / G) / mem })), sharedFrac: r.S > 0 ? (r.kv.shared / G) / mem : 0 });
    }
    const slow = r.fits && r.at && r.at.perUser < (s.goalTarget || 0);
    const people = [];
    const shown = Math.min(s.users, 48);
    for (let u = 0; u < shown; u++) people.push({ user: u, status: u < served ? (ev.ok || r.speedOK ? 'served' : 'slow') : 'rejected', speed: r.at ? r.at.perUser : 0 });
    return { gpus, people, hiddenGpus: Math.max(0, total - 16), hiddenPeople: Math.max(0, s.users - shown), served, slow };
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
      for (const sess of g.sessions) segs.push({ n: Math.max(1, Math.round(sess.frac * cells)), color: `hsl(${hueOf(sess.user)} 70% 58%)` });
    }
    let k = 0;
    for (const seg of segs) {
      for (let i = 0; i < seg.n && k < cells; i++, k++) {
        const col = k % cols, row = Math.floor(k / cols);
        const cx = x + 1 + col * 2, cy = y + hgt - 3 - row * 2;
        if (seg.hatch && ((col + row) % 2)) continue;
        c.fillStyle = seg.color; c.fillRect(cx, cy, 2, 2);
      }
    }
    // activity LED
    c.fillStyle = g.sessions.length && Math.floor(t * 4) % 2 ? '#3ddc84' : '#1e5e3a'; c.fillRect(x + w - 4, y - 3, 2, 2);
  }
  function drawPerson(c, x, y, p, t) {
    const hue = hueOf(p.user);
    const dim = p.status === 'rejected';
    const skin = dim ? '#6b6b6b' : '#f1c27d', hair = dim ? '#4a4a4a' : `hsl(${hue} 40% 25%)`, body = dim ? '#555' : `hsl(${hue} 65% 52%)`, legs = dim ? '#444' : '#2b2f3a';
    const bob = dim ? 0 : Math.round(Math.sin(t * 3 + p.user) * 0.5 + 0.5);
    y += bob;
    c.fillStyle = hair; c.fillRect(x + 1, y, 4, 1); c.fillRect(x, y + 1, 1, 1); c.fillRect(x + 5, y + 1, 1, 1);
    c.fillStyle = skin; c.fillRect(x + 1, y + 1, 4, 3);
    c.fillStyle = '#1a1a1a'; c.fillRect(x + 2, y + 2, 1, 1); c.fillRect(x + 4, y + 2, 1, 1);
    c.fillStyle = body; c.fillRect(x + 1, y + 4, 4, 3); c.fillRect(x, y + 5, 6, 1);
    const step = Math.floor(t * 4 + p.user) % 2;
    c.fillStyle = legs; c.fillRect(x + 1, y + 7, 1, 2); c.fillRect(x + 4, y + 7 + (dim ? 0 : step), 1, 2 - (dim ? 0 : step));
    if (p.status === 'rejected') { c.fillStyle = '#e05252'; c.fillRect(x + 1, y - 4, 1, 1); c.fillRect(x + 3, y - 4, 1, 1); c.fillRect(x + 2, y - 3, 1, 1); c.fillRect(x + 1, y - 2, 1, 1); c.fillRect(x + 3, y - 2, 1, 1); }
    if (p.status === 'slow') { c.fillStyle = '#f2c14e'; c.fillRect(x + 2, y - 4, 2, 1); c.fillRect(x + 2, y - 2, 2, 1); }
  }
  function drawScene(c, S, dots, t) {
    c.fillStyle = '#0d1117'; c.fillRect(0, 0, W, H);
    c.fillStyle = '#131a24'; c.fillRect(4, 12, 204, 156); c.fillRect(210, 12, 106, 156);
    c.fillStyle = '#263041'; c.fillRect(4, 166, 204, 2); c.fillRect(210, 166, 106, 2);
    const n = S.gpus.length;
    if (n) {
      const perRow = n <= 8 ? n : 8, rows = n <= 8 ? 1 : 2;
      const boxW = Math.max(8, Math.min(22, Math.floor(196 / perRow) - 2)), boxH = rows === 1 ? 140 : 68;
      const rowW = perRow * (boxW + 2) - 2, x0 = 6 + Math.floor((200 - rowW) / 2);
      S.gpus.forEach((g, i) => { const col = i % perRow, row = Math.floor(i / perRow); drawGpu(c, x0 + col * (boxW + 2), 22 + row * (boxH + 8), boxW, boxH, g, t); });
    }
    S.people.forEach((p, i) => drawPerson(c, 216 + (i % 8) * 12, 28 + Math.floor(i / 8) * 23, p, t));
    for (const d of dots) { c.fillStyle = `hsl(${d.hue} 80% 70%)`; c.fillRect(Math.round(d.x), Math.round(d.y), 1, 1); }
  }

  /* ---------- DOM ---------- */
  const el = (tag, attrs, ...kids) => {
    const n = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) { if (v == null || v === false) continue; if (k === 'class') n.className = v; else if (k.startsWith('on')) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v === true ? '' : v); }
    for (const kid of kids.flat(Infinity)) { if (kid == null || kid === false) continue; n.append(kid.nodeType ? kid : document.createTextNode(String(kid))); }
    return n;
  };
  const LS = { get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } }, set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* ignore */ } } };
  function lessonNodes(text) {
    const out = []; let last = 0; const re = /\[\[([a-z0-9-]+)\|([^\]]+)\]\]/g; let m;
    while ((m = re.exec(text))) { if (m.index > last) out.push(text.slice(last, m.index)); out.push(el('span', { class: 'term', 'data-term': m[1], tabindex: 0 }, m[2])); last = m.index + m[0].length; }
    if (last < text.length) out.push(text.slice(last));
    return out;
  }

  let root = null, canvas = null, ctx2d = null, raf = 0, dots = [], emit = [], lastT = 0;
  let progress = LS.get('cb.arcade', { unlocked: 0, stars: {} });
  let levelIdx = Math.min(progress.unlocked, LEVELS.length - 1), s = null, ev = null, scene = null;

  function setLevel(i) {
    levelIdx = i; const lv = LEVELS[i]; s = stateFor(lv); recompute(); renderPanel();
  }
  function recompute() {
    const lv = LEVELS[levelIdx];
    ev = evaluateState(s, lv);
    scene = sceneFor(Object.assign({}, s, { goalTarget: lv.goal ? lv.goal.target : 0 }), ev);
    emit = scene.people.map(() => 0);
    if (ev.ok && lv.goal && progress.unlocked < levelIdx + 1) { progress.unlocked = levelIdx + 1; }
    if (ev.ok && lv.goal) { progress.stars[lv.id] = Math.max(progress.stars[lv.id] || 0, ev.stars); }
    LS.set('cb.arcade', progress);
  }
  function knobRow(key, lv) {
    const K = KNOBS[key]; const values = K.values(lv); let idx = values.findIndex((v) => v === s[key]); if (idx < 0) idx = 0;
    const show = el('span', { class: 'arc-val' }, K.show(values[idx]));
    const set = (d) => { idx = (idx + d + values.length) % values.length; s[key] = values[idx]; show.textContent = K.show(values[idx]); recompute(); renderHud(); };
    return el('div', { class: 'arc-knob' }, el('span', { class: 'arc-klabel term', 'data-term': K.term, tabindex: 0 }, K.label),
      el('div', { class: 'arc-kctl' }, el('button', { class: 'pxbtn', type: 'button', 'aria-label': `${K.label}: previous`, onclick: () => set(-1) }, '◀'), show, el('button', { class: 'pxbtn', type: 'button', 'aria-label': `${K.label}: next`, onclick: () => set(1) }, '▶')));
  }
  function renderPanel() {
    const lv = LEVELS[levelIdx];
    root.replaceChildren(
      el('div', { class: 'arc-head' },
        el('div', { class: 'arc-level' }, lv.goal ? `MISSION ${levelIdx + 1} / ${LEVELS.length - 1}` : 'FREE PLAY'),
        el('h2', { class: 'arc-title' }, lv.title),
        el('div', { class: 'arc-stars', id: 'arcStars' })),
      el('p', { class: 'arc-mission' }, lv.mission),
      el('div', { class: 'arc-stage' }, canvas, el('div', { class: 'arc-legend' },
        el('span', null, el('i', { style: 'background:#2f5fd1' }), 'weights'), el('span', null, el('i', { style: 'background:#4b5563' }), 'runtime overhead'), el('span', null, el('i', { style: 'background:#232a36' }), 'headroom'),
        el('span', null, el('i', { style: 'background:hsl(94 70% 58%)' }), 'one person\'s KV cache'), el('span', null, el('i', { style: 'background:#e0b84a' }), 'shared prefix'), el('span', null, el('i', { style: 'background:#b43a3a' }), 'weights overflow'),
        el('span', { id: 'arcHidden' }))),
      el('div', { class: 'arc-hud', id: 'arcHud' }),
      el('div', { class: 'arc-knobs' }, lv.knobs.map((k) => knobRow(k, lv))),
      el('div', { class: 'arc-feedback', id: 'arcFeedback' }),
      el('div', { class: 'arc-nav' }, LEVELS.map((L, i) => el('button', { class: 'pxbtn' + (i === levelIdx ? ' on' : ''), type: 'button', disabled: i > progress.unlocked, title: L.title, onclick: () => setLevel(i) }, L.goal ? String(i + 1) : '∞')),
        el('button', { class: 'pxbtn ghost', type: 'button', onclick: () => { s = stateFor(lv); recompute(); renderPanel(); } }, 'Reset mission'),
        el('button', { class: 'pxbtn ghost', type: 'button', onclick: () => { progress = { unlocked: 0, stars: {} }; LS.set('cb.arcade', progress); setLevel(0); } }, 'Start over')));
    renderHud();
  }
  function renderHud() {
    const lv = LEVELS[levelIdx], r = ev.r, hud = root.querySelector('#arcHud'), fb = root.querySelector('#arcFeedback');
    const chip = (label, value, cls) => el('div', { class: 'arc-chip' + (cls ? ' ' + cls : '') }, el('span', null, label), el('b', null, value));
    const servedTxt = lv.goal ? `${fmtNum(scene.served)} / ${fmtNum(ev.r.B)}` : fmtNum(scene.served);
    hud.replaceChildren(
      chip('people served', servedTxt, scene.served >= r.B && r.fits ? 'good' : 'bad'),
      chip('tokens/s each', r.at ? r.at.perUser.toFixed(0) : '—', r.at && lv.goal && r.at.perUser < lv.goal.target ? 'bad' : ''),
      chip('first token', r.at ? fmtTime(r.ttft) : '—', lv.goal && !r.ttftOK ? 'bad' : ''),
      chip('weights', fmtGB(r.W)), chip('KV per person', fmtGB(r.kv.perSession)), chip('KV pool', r.fits ? fmtGB(r.kvAvail * r.R) : '0'),
      chip('layout', `${r.total} GPU${r.total > 1 ? 's' : ''}: TP ${r.tp} × PP ${r.pp} × ${r.R}`), chip('price', r.price != null ? fmtMoney(r.price) + '/h' : '—', lv.goal && lv.goal.budget != null && r.price > lv.goal.budget ? 'bad' : ''));
    root.querySelector('#arcHidden').textContent = (scene.hiddenGpus ? `+${scene.hiddenGpus} GPUs not drawn` : '') + (scene.hiddenPeople ? ` · +${scene.hiddenPeople} people not drawn` : '');
    const stars = root.querySelector('#arcStars'); const got = progress.stars[lv.id] || 0;
    stars.textContent = lv.goal ? '★'.repeat(got) + '☆'.repeat((lv.stars ? 3 : 1) - got) : '';
    if (!lv.goal) { fb.replaceChildren(el('p', { class: 'arc-lesson' }, lessonNodes(lv.lesson)), el('button', { class: 'pxbtn', type: 'button', onclick: () => window.openArcadeInPlanner && window.openArcadeInPlanner(s) }, 'Open this setup in the planner')); return; }
    if (ev.ok) {
      const next = levelIdx + 1 < LEVELS.length ? el('button', { class: 'pxbtn go', type: 'button', onclick: () => setLevel(levelIdx + 1) }, levelIdx + 2 < LEVELS.length ? 'Next mission ▶' : 'Free play ▶') : null;
      fb.replaceChildren(el('div', { class: 'arc-win' }, `MISSION COMPLETE ${'★'.repeat(ev.stars)}`), el('p', { class: 'arc-lesson' }, lessonNodes(lv.lesson)), next);
    } else {
      fb.replaceChildren(el('div', { class: 'arc-fail' }, 'Not yet: ' + ev.reasons[0] + '.'), el('p', { class: 'arc-lesson' }, lessonNodes(lv.lesson)));
    }
  }

  /* ---------- animation ---------- */
  const reduced = typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function frame(ts) {
    const t = ts / 1000, dt = Math.min(0.1, lastT ? t - lastT : 0.016); lastT = t;
    if (!reduced && scene) {
      scene.people.forEach((p, i) => {
        if (p.status === 'rejected') return;
        emit[i] += dt * Math.min(12, Math.max(0.3, p.speed / 8));
        while (emit[i] >= 1 && dots.length < 400) { emit[i] -= 1; dots.push({ x: 216 + (i % 8) * 12 + 2 + Math.random() * 3, y: 28 + Math.floor(i / 8) * 23 - 2, vy: -(8 + Math.random() * 6), life: 1.4, hue: hueOf(p.user) }); }
      });
      for (const d of dots) { d.y += d.vy * dt; d.life -= dt; }
      dots = dots.filter((d) => d.life > 0 && d.y > 14);
    }
    if (scene) drawScene(ctx2d, scene, dots, reduced ? 0 : t);
    raf = requestAnimationFrame(frame);
  }

  function render(container) {
    if (!root) {
      root = el('div', { class: 'arcade' });
      canvas = el('canvas', { width: W, height: H, class: 'arc-canvas', role: 'img', 'aria-label': 'Pixel rack of GPUs on the left filling with weights and each person\'s KV cache; crowd of users on the right streaming tokens' });
      ctx2d = canvas.getContext('2d'); ctx2d.imageSmoothingEnabled = false;
      container.replaceChildren(root);
      setLevel(levelIdx);
    }
    if (!raf) raf = requestAnimationFrame(frame);
  }
  function pause() { if (raf) cancelAnimationFrame(raf); raf = 0; lastT = 0; }

  return { render, pause, LEVELS, BASE, stateFor, evaluateState };
})();
if (typeof module !== 'undefined') module.exports = Arcade;
