/* arcade/model.js - what the arcade stage shows, derived from one engine result.
 * Pure functions, no DOM: a classic script in the browser (global ArcadeModel) and a module for tests/arcade-model.js.
 */
const ArcadeModel = (() => {
  const E = typeof Engine !== 'undefined' ? Engine : require('../engine.js');
  const ENGINES_ = typeof ENGINES !== 'undefined' ? ENGINES : require('../catalog.js').ENGINES;

  /* ---------- short arcade formats ---------- */
  function fmtCount(n) {
    if (n == null || !isFinite(n)) return '-';
    const a = Math.abs(n);
    if (a >= 1e6) return (n / 1e6).toFixed(a >= 1e7 ? 0 : 1).replace(/\.0$/, '') + 'M';
    if (a >= 1e4) return (n / 1e3).toFixed(0) + 'K';
    if (a >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'K';
    return String(Math.round(n));
  }
  function fmtDuration(s) {
    if (s == null || !isFinite(s)) return '-';
    if (s < 1) return Math.max(1, Math.round(s * 1000)) + ' ms';
    if (s < 10) return s.toFixed(1).replace(/\.0$/, '') + ' s';
    if (s < 90) return Math.round(s) + ' s';
    if (s < 5400) return Math.round(s / 60) + ' min';
    return (s / 3600).toFixed(s < 36000 ? 1 : 0).replace(/\.0$/, '') + ' h';
  }
  function fmtMoney(n) {
    if (n == null || !isFinite(n)) return '-';
    if (n >= 1e6) return '$' + (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
    if (n >= 1e4) return '$' + (n / 1e3).toFixed(0) + 'K';
    if (n >= 1e3) return '$' + (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'K';
    if (n >= 10) return '$' + Math.round(n);
    return '$' + n.toFixed(2);
  }
  const fmtSpeed = (x) => (x == null || !isFinite(x) ? '-' : x >= 100 ? String(Math.round(x)) : x >= 10 ? x.toFixed(0) : x.toFixed(1));
  const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;

  /* ---------- the crowd ----------
   * Users form a closed interactive system: each one thinks, sends a request, waits for the answer, thinks again.
   * U users (N of them with a session), a = the share of time a promptly served user has a request in flight,
   * S = seconds one request takes (first token plus output), C = requests the cluster runs at once.
   * If N·a ≤ C everyone is served: N·a active, the rest thinking. Otherwise throughput is C/S and the response time
   * law gives the response time R = N·S/C − Z with think time Z = S(1 − a)/a: C users active, C(1 − a)/a thinking and
   * N − C/a waiting, each for W = (S/C)(N − C/a). */
  function crowd(r, p) {
    const U = p.wl.users, a = p.wl.activity;
    const g = { active: 0, sleeping: 0, parked: 0, evicted: 0, waiting: 0, rejected: 0, blocked: 0 };
    if (!r.fits || !r.at) { g.blocked = U; return { groups: g, served: 0, wait: Infinity, S: 0, C: 0, N: 0, queueBy: null, deskCap: 0 }; }
    const retention = r.effective.retention, R = r.R;
    const N = retention === 'gpu' ? Math.min(U, r.maxSessions * R) : U;   // with every session kept on the GPU, memory caps the sessions
    g.blocked = U - N;
    const perReplica = retention === 'gpu' ? Math.min(r.maxBatch, Math.ceil(N / R)) : Math.min(r.maxSessions, r.maxBatch);
    const C = perReplica * R, S = r.at.dur;
    let active, thinking, waiting = 0, wait = 0;
    if (N * a <= C) { active = N * a; thinking = N - active; }
    else { active = C; thinking = C * (1 - a) / a; waiting = N - C / a; wait = S / C * waiting; }
    if (r.sv.queueMax != null && waiting > r.sv.queueMax * R) { g.rejected = waiting - r.sv.queueMax * R; waiting -= g.rejected; }
    g.active = active; g.waiting = waiting;
    if (retention === 'gpu') g.sleeping = thinking;
    else if (retention === 'host') { g.parked = Math.min(thinking, r.hostSessions); g.evicted = thinking - g.parked; }
    else g.evicted = thinking;
    // what makes the queue: the engine's slots or batch cap, or the KV cache memory
    const queueBy = r.slots ? 'slots' : retention !== 'gpu' && r.maxBatch < r.maxSessions ? 'max-batch' : retention === 'gpu' && g.blocked > 0 ? 'memory' : r.maxBatch < Math.ceil(N / R) ? 'max-batch' : 'memory';
    // desks on stage: requests the cluster runs at once, or with every session on the GPU, the sessions it holds
    const deskCap = retention === 'gpu' ? r.maxSessions * R : C;
    return { groups: g, served: active + thinking, wait, S, C, N, queueBy, deskCap };
  }

  /* ---------- memory of one accelerator in the replica on stage ----------
   * Fractions of the accelerator's memory from the bottom: weights, runtime overhead, KV cache in use, KV cache reserved
   * but empty (slot engines), KV cache free, and the headroom the engine leaves unused. `overflow` is how far the weights
   * and overhead reach beyond the usable memory. `cells` split the KV pool into at most 64 drawable blocks. */
  function memory(r, p, c) {
    const mem = r.hw.mem * 1e9, util = p.adv && p.adv.util != null ? p.adv.util : E.DEFAULT_ADV.util;
    const perGpu = (x) => Math.max(0, x) / r.G;
    const weights = r.W / r.G, overhead = r.overhead, head = mem * (1 - util);
    const kvRaw = perGpu(r.kvAvailRaw);
    const overflowBytes = Math.max(0, weights + overhead - mem * util);
    const sessions = r.slots ? r.slots.n : r.maxSessions;
    const resident = !r.fits ? 0 : Math.min(sessions, r.effective.retention === 'gpu' ? Math.ceil(c.N / r.R) : Math.ceil(c.groups.active / r.R - 1e-9));
    const usable = r.kvAvail > 0 ? r.kvAvail : 1;
    const usedShare = r.fits ? Math.min(1, (r.kv.shared + resident * r.kv.perSession) / usable) : 0;
    const reservedShare = r.slots && r.fits ? Math.min(1, r.slots.reserved / usable) : usedShare;
    const perCell = Math.max(1, Math.ceil(sessions / 64));
    const fr = {
      weights: Math.min(1, weights / mem), overhead: Math.min(1, overhead / mem),
      kvUsed: kvRaw * usedShare / mem, kvReserved: kvRaw * Math.max(0, reservedShare - usedShare) / mem,
      kvFree: kvRaw * (1 - reservedShare) / mem, headroom: head / mem,
    };
    if (overflowBytes > 0) { fr.overhead = Math.max(0, Math.min(fr.overhead, util - fr.weights)); fr.kvUsed = fr.kvReserved = fr.kvFree = 0; }
    return {
      fr, overflow: overflowBytes / mem,
      cells: { total: r.fits ? Math.ceil(sessions / perCell) : 0, used: r.fits ? Math.ceil(resident / perCell) : 0, perCell, slots: !!r.slots },
    };
  }

  const LIMIT_LABEL = {
    unsupported: 'Not supported', oom: 'Model too big', memory: 'KV cache memory', slots: 'Parallel slots', 'max-batch': 'Batch limit',
    'memory bandwidth': 'Memory bandwidth', compute: 'Compute', communication: 'Network', ttft: 'First token', prefill: 'Prefill', none: '-',
  };
  /* What bounds the cluster's capacity (the LIMIT badge). */
  function limitOf(r) {
    if (!r.loadable) return { key: 'unsupported', label: LIMIT_LABEL.unsupported };
    if (!r.fits) return r.slots && !r.slots.fit && r.memSessions >= 1 ? { key: 'slots', label: LIMIT_LABEL.slots } : { key: 'oom', label: LIMIT_LABEL.oom };
    if (r.capLimit === 'speed') { const k = r.at ? r.at.bound : 'memory bandwidth'; return { key: k, label: LIMIT_LABEL[k] || k }; }
    if (r.capLimit === 'max-batch') return { key: 'max-batch', label: (r.sv.maxBatch && r.sv.maxBatch.name) || LIMIT_LABEL['max-batch'] };
    return { key: r.capLimit, label: LIMIT_LABEL[r.capLimit] || r.capLimit };
  }

  /* ---------- everything the stage shows ---------- */
  function derive({ r, p, need }) {
    const c = crowd(r, p), g = c.groups;
    const U = p.wl.users, target = p.wl.target;
    const stuck = g.waiting + g.rejected + g.blocked;
    const perUser = r.at ? r.at.perUser : 0;
    let status;
    if (!r.loadable) status = 'unsupported';
    else if (!r.fits) status = 'oom';
    else if (stuck >= 0.5) status = 'queue';
    else if (!r.ttftOK) status = 'ttft';
    else if (r.at.saturated) status = 'prefill';
    else if (perUser < target) status = 'slow';
    else status = 'ok';
    const limit = limitOf(r);
    const engName = r.engine !== 'none' && ENGINES_[r.engine] ? ENGINES_[r.engine].name : 'Ideal engine';
    const why = { memory: 'KV cache full', slots: r.slots ? plural(r.slots.n, 'parallel slot') : 'slots full', 'max-batch': `${(r.sv.maxBatch && r.sv.maxBatch.name) || 'batch limit'} reached` }[c.queueBy] || 'cluster full';
    let headline;
    if (status === 'unsupported') headline = `${engName} cannot run this here`;
    else if (status === 'oom') headline = limit.key === 'slots' ? `${plural(r.slots.n, 'slot')} of context do not fit` : 'The model does not fit';
    else if (g.blocked >= 0.5 && g.waiting + g.rejected < 0.5) headline = `${fmtCount(g.blocked)} without a session: GPU memory full`;
    else if (stuck >= 0.5) headline = `${fmtCount(g.waiting + g.rejected + g.blocked)} waiting${isFinite(c.wait) && c.wait > 0 ? ' ' + fmtDuration(c.wait) : ''}: ${why}`;
    else if (status === 'ttft') headline = `Everyone in, first token after ${fmtDuration(r.ttft)}`;
    else if (status === 'slow' || status === 'prefill') headline = `Everyone in, but ${fmtSpeed(perUser)} tok/s`;
    else headline = `All ${fmtCount(U)} served at ${fmtSpeed(perUser)} tok/s`;
    const served = Math.min(U, r.maxUsers);
    const perHour = r.price;
    const perMonth = perHour != null ? perHour * 730 : null;
    return {
      status, limit, headline, users: U, activity: p.wl.activity,
      crowd: Object.assign({}, g, { served: c.served, wait: c.wait, S: c.S, C: c.C, queueBy: c.queueBy, deskCap: c.deskCap }),
      capacity: { users: r.fits ? r.maxUsers : 0, conc: r.fits ? r.maxConc : 0 },
      need: need ? { feasible: !need.infeasible, gpus: need.infeasible ? null : need.gpusUsed, tp: need.tp, pp: need.pp, R: need.R, reason: need.reason || '' } : null,
      speed: { perUser, target, slow: status === 'slow' || status === 'prefill' || (r.at ? r.at.perUser < target : false), ttft: r.fits ? r.ttft : null, ttftMax: r.ttftMax, ttftSlow: r.fits && !r.ttftOK, cold: r.fits ? r.ttftCold : null },
      cluster: { hwId: r.hw.id, hwName: r.hw.name, count: r.count, total: r.total, gpusUsed: r.gpusUsed, idle: Math.max(0, (r.count || r.total) - r.total), tp: r.tp, pp: r.pp, R: r.R, G: r.G, split: r.seqPP ? 'layer' : 'tensor', pd: r.pdTotal, servers: !!r.slots },
      gpu: memory(r, p, c),
      engine: { name: engName },
      money: { perHour, perMonth, perUserMonth: perMonth != null && served > 0 ? perMonth / served : null, kW: r.kW },
      tokens: { total: r.aggTotal || 0 },
      model: { name: r.model.name, params: r.model.params },
    };
  }

  /* ---------- sprites: how many users one pixel person stands for ---------- */
  const ZONE_OF = { active: 'desks', sleeping: 'desks', parked: 'lounge', evicted: 'lounge', waiting: 'street', rejected: 'street', blocked: 'street' };
  const GROUPS = Object.keys(ZONE_OF);
  function niceStep(x) {
    if (!(x > 1)) return 1;
    const e = Math.floor(Math.log10(x)), b = Math.pow(10, e);
    for (const m of [1, 2, 5, 10]) if (m * b >= x - 1e-9) return m * b;
    return 10 * b;
  }
  const nextStep = (s) => { const e = Math.floor(Math.log10(s) + 1e-9), m = Math.round(s / Math.pow(10, e)); return (m === 1 ? 2 : m === 2 ? 5 : 10) * Math.pow(10, e); };
  /* Sprites per group for a users-per-sprite step that fits every zone: largest remainder, and one sprite at least for a
   * group of half a user or more (three waiting users among 10,000 must still show). */
  function allocateSprites(groups, caps, maxTotal = 150) {
    const total = GROUPS.reduce((a, k) => a + (groups[k] || 0), 0);
    let per = niceStep(total / maxTotal);
    for (let guard = 0; guard < 40; guard++, per = nextStep(per)) {
      const want = {}, counts = {};
      let assigned = 0;
      for (const k of GROUPS) { want[k] = (groups[k] || 0) / per; counts[k] = Math.floor(want[k]); assigned += counts[k]; }
      const target = Math.round(total / per);
      const order = GROUPS.slice().sort((a, b) => (want[b] - counts[b]) - (want[a] - counts[a]));
      for (const k of order) { if (assigned >= target) break; if (want[k] - counts[k] > 0) { counts[k]++; assigned++; } }
      for (const k of GROUPS) if (counts[k] === 0 && (groups[k] || 0) >= 0.5) counts[k] = 1;
      const zone = { desks: 0, lounge: 0, street: 0 };
      for (const k of GROUPS) zone[ZONE_OF[k]] += counts[k];
      if (zone.desks <= caps.desks && zone.lounge <= caps.lounge && zone.street <= caps.street) return { per, counts };
    }
    return { per, counts: Object.fromEntries(GROUPS.map((k) => [k, 0])) };
  }

  /* ---------- where things go on the canvas (logical pixels) ---------- */
  const DESK = { w: 9, h: 12 }, STAND = { w: 6, h: 12 };
  function grid(area, cell) {
    const cols = Math.max(0, Math.floor(area.w / cell.w)), rows = Math.max(0, Math.floor(area.h / cell.h));
    const offX = area.x + Math.floor((area.w - cols * cell.w) / 2);
    const out = [];
    for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) out.push({ x: offX + col * cell.w, y: area.y + row * cell.h });
    return out;
  }
  function layout(lw, lh) {
    const signH = 13, labelH = 8, floorY = signH + 2, floorH = lh - floorY - 1;
    const serverW = Math.round(lw * 0.43), streetW = Math.max(42, Math.round(lw * 0.19));
    const server = { x: 1, y: floorY, w: serverW - 2, h: floorH };
    const street = { x: lw - streetW, y: floorY, w: streetW - 1, h: floorH };
    const office = { x: serverW + 2, y: floorY, w: lw - serverW - streetW - 4, h: floorH };
    // below each zone label, room for the icons drawn above a person's head (zzz, hourglass, !)
    const deskArea = { x: office.x + 1, y: office.y + labelH + 2, w: office.w - 2, h: Math.floor((office.h - labelH) * 0.6) - 2 };
    const lounge = { x: office.x + 1, y: deskArea.y + deskArea.h + labelH, w: office.w - 2, h: office.y + office.h - (deskArea.y + deskArea.h + labelH) };
    const queue = { x: street.x + 2, y: street.y + labelH + 5, w: street.w - 4, h: street.h - labelH - 6 };
    const racks = { x: server.x + 2, y: server.y + labelH, w: server.w - 4, h: server.h - labelH - 1 };
    return {
      lw, lh, sign: { x: 0, y: 0, w: lw, h: signH }, server, office, street, racks,
      desks: { area: deskArea, slots: grid(deskArea, DESK) }, lounge: { area: lounge, slots: grid(lounge, STAND) }, queue: { area: queue, slots: grid(queue, STAND) },
    };
  }

  /* ---------- baseline deltas ---------- */
  const METRICS = [
    ['capacity', (v) => v.capacity.users, 'up'], ['speed', (v) => v.speed.perUser, 'up'], ['ttft', (v) => v.speed.ttft, 'down'],
    ['gpus', (v) => v.cluster.gpusUsed, 'down'], ['perMonth', (v) => v.money.perMonth, 'down'], ['perUserMonth', (v) => v.money.perUserMonth, 'down'],
    ['kW', (v) => v.money.kW, 'down'], ['wait', (v) => v.crowd.wait, 'down'], ['waiting', (v) => v.crowd.waiting + v.crowd.rejected + v.crowd.blocked, 'down'],
  ];
  function compare(vm, base) {
    const out = {};
    for (const [id, get, better] of METRICS) {
      const a = get(vm), b = base ? get(base) : null;
      const ok = a != null && b != null && isFinite(a) && isFinite(b);
      const pct = ok && b !== 0 ? a / b - 1 : null;
      out[id] = { value: a, base: b, pct, same: ok ? Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b)) : a === b, good: ok ? (better === 'up' ? a >= b : a <= b) : null };
    }
    return out;
  }

  /* ---------- the ticker line and the big banner for one change ----------
   * change: { what, from, to } in display words. Picks the metric that moved most visibly. */
  function ticker(change, before, after) {
    const head = change ? `${change.what} ${change.from} → ${change.to}: ` : '';
    const bad = (s) => s === 'oom' || s === 'unsupported';
    if (bad(after.status) && !bad(before.status)) return { text: head + (after.status === 'oom' ? 'out of memory' : 'not supported here'), tone: 'bad', banner: after.status === 'oom' ? 'Out of memory!' : 'Not supported!' };
    if (bad(before.status) && !bad(after.status)) return { text: head + `it fits: capacity ${fmtCount(after.capacity.users)} users`, tone: 'good', banner: 'It fits!' };
    if (bad(after.status)) return { text: head + (after.status === 'oom' ? 'still out of memory' : 'still not supported'), tone: 'bad', banner: null };
    const pct = (a, b) => (b > 0 ? ` (${a >= b ? '+' : '−'}${Math.abs(Math.round((a / b - 1) * 100))}%)` : '');
    const cb = before.capacity.users, ca = after.capacity.users;
    const wb = before.crowd.waiting + before.crowd.rejected + before.crowd.blocked, wa = after.crowd.waiting + after.crowd.rejected + after.crowd.blocked;
    let banner = null;
    if (wa >= 0.5 && wb < 0.5) banner = 'Queue!';
    else if (wa < 0.5 && wb >= 0.5) banner = 'Queue cleared!';
    else if (after.speed.slow && !before.speed.slow) banner = 'Too slow!';
    if (cb !== ca) {
      const ratio = cb > 0 ? ca / cb : Infinity;
      if (!banner && ratio >= 2) banner = `Capacity ×${ratio >= 10 ? Math.round(ratio) : ratio.toFixed(1).replace(/\.0$/, '')}!`;
      if (!banner && ratio <= 0.5 && ca > 0) banner = `Capacity ÷${(1 / ratio) >= 10 ? Math.round(1 / ratio) : (1 / ratio).toFixed(1).replace(/\.0$/, '')}!`;
      return { text: head + `capacity ${fmtCount(cb)} → ${fmtCount(ca)} users${pct(ca, cb)}`, tone: ca > cb ? 'good' : 'bad', banner };
    }
    const sb = before.speed.perUser, sa = after.speed.perUser;
    if (sb > 0 && Math.abs(sa / sb - 1) >= 0.03) return { text: head + `${fmtSpeed(sb)} → ${fmtSpeed(sa)} tok/s per user${pct(sa, sb)}`, tone: sa > sb ? 'good' : 'bad', banner };
    const tb = before.speed.ttft, ta = after.speed.ttft;
    if (tb > 0 && ta > 0 && Math.abs(ta / tb - 1) >= 0.05) return { text: head + `first token ${fmtDuration(tb)} → ${fmtDuration(ta)}`, tone: ta < tb ? 'good' : 'bad', banner };
    if (before.cluster.gpusUsed !== after.cluster.gpusUsed) return { text: head + `${before.cluster.gpusUsed} → ${after.cluster.gpusUsed} GPUs`, tone: after.cluster.gpusUsed < before.cluster.gpusUsed ? 'good' : 'neutral', banner };
    if (wb >= 0.5 || wa >= 0.5) return { text: head + `${fmtCount(wb)} → ${fmtCount(wa)} waiting`, tone: wa < wb ? 'good' : wa > wb ? 'bad' : 'neutral', banner };
    return { text: head + 'no change in capacity or speed', tone: 'neutral', banner };
  }

  return { crowd, memory, limitOf, derive, niceStep, allocateSprites, layout, compare, ticker, fmtCount, fmtDuration, fmtMoney, fmtSpeed, ZONE_OF, GROUPS, DESK, STAND };
})();

if (typeof module !== 'undefined') module.exports = ArcadeModel;
