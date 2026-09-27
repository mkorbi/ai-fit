/* arcade/world.js - the pixel world: GPUs in the server room, served users in the office, the queue in the street.
 * The canvas backing store has the logical (art) resolution and CSS scales it up by an integer factor in device pixels,
 * so every art pixel is a crisp block. setView(vm) sets targets; the frame loop animates toward them: memory levels
 * ease, people keep their identity and walk between zones, tokens fly from the GPUs to the desks.
 */
const ArcadeWorld = (() => {
  const S = Sprites, M = ArcadeModel;
  const WALK = 42;                 // art pixels per second

  function create(canvas, opts = {}) {
    const ctx = canvas.getContext('2d', { alpha: false });
    const box = canvas.parentElement;
    let lw = 300, lh = 170, scale = 4, L = M.layout(lw, lh);
    let pal = null;
    const cache = new Map();       // baked sprites per palette
    let vm = null, sprites = { per: 1, counts: {} };
    let reduced = !!opts.reducedMotion;
    const people = [];
    const particles = [];
    let nextId = 1, raf = 0, last = 0, time = 0, spawnAcc = 0;
    let shake = 0, flash = 0, flashColor = 'red', drop = 0, errors = 0, frameMs = 0;
    const lvl = { weights: 0, overhead: 0, kvUsed: 0, kvReserved: 0, kvFree: 0, headroom: 0.1, overflow: 0 };
    let seed = 20260927;
    const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);

    /* ---------- palette from the page's CSS variables ---------- */
    function readPalette() {
      const cs = getComputedStyle(document.documentElement);
      const v = (n) => cs.getPropertyValue('--w-' + n).trim();
      const list = (n) => v(n).split(/\s+/).filter(Boolean);
      pal = {
        bg: v('bg'), floor: v('floor'), grid: v('grid'), wall: v('wall'), wallEdge: v('wall-edge'), sign: v('sign'), signGlow: v('sign-glow'), sign2: v('sign2'),
        ink: v('ink'), ink2: v('ink2'), rack: v('rack'), rackIn: v('rack-in'), led: v('led'), led2: v('led2'), bus: v('bus'),
        weights: v('weights'), mortar: v('mortar'), overhead: v('overhead'), kv: v('kv'), kvEdge: v('kv-edge'), free: v('free'), head: v('head'),
        red: v('red'), red2: v('red2'), desk: v('desk'), desk2: v('desk2'), monitor: v('monitor'), glow: v('glow'), glowOff: v('glow-off'),
        token: v('token'), token2: v('token2'), rug: v('rug'), door: v('door'), sweat: v('sweat'), yellow: v('yellow'), case: v('case'),
        skin: list('skin'), hair: list('hair'), shirt: list('shirt'), trousers: v('trousers'), shoes: v('shoes'),
      };
      cache.clear();
    }
    function baked(kind, frame, vr) {
      const key = `${kind}|${frame}|${vr.skin}|${vr.hair}|${vr.shirt}`;
      let c = cache.get(key);
      if (!c) {
        const map = { H: vr.hair, F: vr.skin, S: vr.shirt, P: pal.trousers, B: pal.shoes, D: pal.desk, d: pal.desk2, M: pal.monitor, G: pal.glow, g: pal.glowOff };
        const art = kind === 'stand' ? S.PERSON[frame] : kind === 'seated' ? S.SEATED[frame] : kind === 'sleep' ? S.SLEEPING : S.DESK;
        c = S.bake(art, map);
        cache.set(key, c);
      }
      return c;
    }
    function icon(name, color) {
      const key = `icon|${name}|${color}`;
      let c = cache.get(key);
      if (!c) { const map = { Z: color, W: color, Y: color, K: color, R: color }; c = S.bake(S.ICONS[name], map); cache.set(key, c); }
      return c;
    }

    /* ---------- size: an integer number of device pixels per art pixel ---------- */
    function resize(entry) {
      const dpr = window.devicePixelRatio || 1;
      let W, H;
      const dev = entry && entry.devicePixelContentBoxSize && entry.devicePixelContentBoxSize[0];
      if (dev) { W = dev.inlineSize; H = dev.blockSize; }
      else { W = Math.round(box.clientWidth * dpr); H = Math.round(box.clientHeight * dpr); }   // inside the frame's border
      if (W < 50 || H < 50) return;
      scale = Math.max(2, Math.min(8, Math.floor(Math.min(H / 150, W / 280))));   // at least 280 logical pixels wide, also on 4:3
      lw = Math.floor(W / scale); lh = Math.floor(H / scale);
      canvas.width = lw; canvas.height = lh;
      canvas.style.width = (lw * scale / dpr) + 'px';
      canvas.style.height = (lh * scale / dpr) + 'px';
      ctx.imageSmoothingEnabled = false;
      L = M.layout(lw, lh);
      if (vm) assign(true);
    }
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver((es) => resize(es[es.length - 1])) : null;
    if (ro) { try { ro.observe(box, { box: 'device-pixel-content-box' }); } catch (e) { ro.observe(box); } }

    /* ---------- people: stable identities, reassigned to groups when the view changes ---------- */
    const ZONE = M.ZONE_OF;
    const ORDER = ['active', 'sleeping', 'parked', 'evicted', 'waiting', 'blocked', 'rejected'];
    function slotFor(group, i, counts) {
      const zone = ZONE[group];
      if (zone === 'desks') { const k = group === 'active' ? i : counts.active + i; const s = L.desks.slots[k]; return s ? { x: s.x, y: s.y + 3, seat: k } : null; }
      if (zone === 'lounge') { const k = group === 'parked' ? i : counts.parked + i; const s = L.lounge.slots[k]; return s ? { x: s.x, y: s.y + 2 } : null; }
      // the queue fills from the door outward; the rejected stand at the far end
      const byCol = L.queue.slots.slice().sort((a, b) => a.x - b.x || a.y - b.y);
      const k = group === 'waiting' ? i : group === 'blocked' ? counts.waiting + i : byCol.length - 1 - i;
      const s = byCol[Math.max(0, Math.min(byCol.length - 1, k))];
      return s ? { x: s.x, y: s.y + 2 } : null;
    }
    const door = () => ({ x: L.street.x - 2, y: L.street.y + Math.floor(L.street.h * 0.55) });
    function assign(instant) {
      if (!vm) return;
      const caps = { desks: L.desks.slots.length, lounge: L.lounge.slots.length, street: L.queue.slots.length };
      sprites = M.allocateSprites(vm.crowd, caps);
      const want = sprites.counts;
      const pool = [];
      const byGroup = {};
      for (const g of ORDER) byGroup[g] = [];
      for (const p of people) if (!p.leaving) (byGroup[p.group] || (byGroup[p.group] = [])).push(p);
      for (const g of ORDER) {
        byGroup[g].sort((a, b) => a.order - b.order);
        while (byGroup[g].length > (want[g] || 0)) pool.push(byGroup[g].pop());
      }
      // people move between groups along plausible paths: a served user who loses the seat walks out to the queue
      const takeFrom = (g) => {
        const pref = { active: ['waiting', 'blocked', 'parked', 'evicted', 'sleeping', 'rejected'], sleeping: ['active', 'parked', 'evicted'], parked: ['evicted', 'active', 'sleeping'],
          evicted: ['parked', 'active', 'sleeping'], waiting: ['active', 'blocked', 'evicted', 'parked', 'rejected'], blocked: ['waiting', 'sleeping', 'active'], rejected: ['waiting', 'blocked'] }[g];
        for (const from of pref) { const i = pool.findIndex((p) => p.group === from); if (i >= 0) return pool.splice(i, 1)[0]; }
        return pool.length ? pool.shift() : null;
      };
      for (const g of ORDER) {
        while (byGroup[g].length < (want[g] || 0)) {
          let p = takeFrom(g);
          if (!p) { p = spawn(); }
          p.group = g;
          byGroup[g].push(p);
        }
      }
      for (const p of pool) { p.leaving = true; p.tx = lw + 8; p.ty = p.y; p.path = []; }
      // slots in a stable order
      let order = 0;
      for (const g of ORDER) byGroup[g].forEach((p, i) => {
        p.order = order++;
        const s = slotFor(g, i, want);
        if (!s) { p.hidden = true; return; }
        p.hidden = false;
        const zoneChange = p.zone && p.zone !== ZONE[g] && (p.zone === 'street' || ZONE[g] === 'street');
        p.zone = ZONE[g];
        p.tx = s.x; p.ty = s.y; p.seat = s.seat;
        p.path = zoneChange ? [door()] : [];
        if (instant || reduced) { p.x = p.tx; p.y = p.ty; p.path = []; }
      });
    }
    function spawn() {
      const vr = S.variant(nextId, pal);
      const p = { id: nextId++, x: lw + 4, y: L.street.y + Math.floor(rnd() * Math.max(1, L.street.h - 12)), tx: 0, ty: 0, path: [], vr, phase: rnd() * 10, order: 1e9 };
      people.push(p);
      return p;
    }

    /* ---------- view ---------- */
    function setView(next, change) {
      const prev = vm;
      vm = next;
      if (prev && !reduced) {
        const bad = (s) => s === 'oom' || s === 'unsupported';
        if (bad(vm.status) && !bad(prev.status)) { shake = 0.45; flash = 0.35; flashColor = pal.red; }
        else if (!bad(vm.status) && bad(prev.status)) { flash = 0.3; flashColor = pal.kv; }
        if (prev.cluster.G !== vm.cluster.G || prev.cluster.R !== vm.cluster.R || prev.cluster.hwId !== vm.cluster.hwId) drop = 1;
      }
      assign(!prev);
    }

    /* ---------- animation ---------- */
    function ease(cur, tgt, dt, tau) { return reduced ? tgt : cur + (tgt - cur) * (1 - Math.exp(-dt / tau)); }
    function update(dt) {
      time += dt;
      if (vm) {
        const fr = vm.gpu.fr;
        for (const k of ['weights', 'overhead', 'kvUsed', 'kvReserved', 'kvFree', 'headroom']) lvl[k] = ease(lvl[k], fr[k], dt, 0.22);
        lvl.overflow = ease(lvl.overflow, vm.gpu.overflow, dt, 0.3);
      }
      shake = Math.max(0, shake - dt); flash = Math.max(0, flash - dt); drop = Math.max(0, drop - dt * 2.5);
      for (let i = people.length - 1; i >= 0; i--) {
        const p = people[i];
        const target = p.path.length ? p.path[0] : { x: p.tx, y: p.ty };
        const dx = target.x - p.x, dy = target.y - p.y, d = Math.hypot(dx, dy);
        const step = WALK * dt;
        if (d <= step || reduced) { p.x = target.x; p.y = target.y; if (p.path.length) p.path.shift(); }
        else { p.x += dx / d * step; p.y += dy / d * step; }
        p.moving = d > 0.5 && !reduced;
        if (p.leaving && p.x >= lw + 6) people.splice(i, 1);
      }
      // tokens fly from the GPUs to the people at a desk; more tokens per second, more particles
      if (vm && !reduced && vm.status !== 'oom' && vm.status !== 'unsupported') {
        const seated = people.filter((p) => p.group === 'active' && !p.moving && !p.hidden);
        const rate = seated.length ? Math.min(40, 4 * Math.log10(1 + vm.tokens.total)) : 0;
        spawnAcc += rate * dt;
        const travel = Math.min(2.4, Math.max(0.5, 12 / Math.max(1, vm.speed.perUser)));
        while (spawnAcc >= 1 && particles.length < 200) {
          spawnAcc -= 1;
          const t = seated[Math.floor(rnd() * seated.length)];
          const src = tankTop(Math.floor(rnd() * Math.max(1, Math.min(8, vm.cluster.G))));
          particles.push({ x0: src.x, y0: src.y, x1: t.x + 7, y1: t.y + 2, t: 0, dur: travel * (0.8 + rnd() * 0.4), c: rnd() < 0.5 ? pal.token : pal.token2 });
        }
        if (spawnAcc > 5) spawnAcc = 0;
      } else spawnAcc = 0;
      for (let i = particles.length - 1; i >= 0; i--) { const q = particles[i]; q.t += dt / q.dur; if (q.t >= 1) particles.splice(i, 1); }
    }

    /* ---------- geometry of the server room ---------- */
    function tanks() {
      const G = vm ? vm.cluster.G : 1, n = Math.max(1, Math.min(8, G));
      const area = L.racks;
      const miniH = vm && (vm.cluster.R > 1 || vm.cluster.idle > 0 || vm.cluster.pd > 0) ? 14 : 0;
      const gap = n > 1 ? Math.max(2, Math.min(5, Math.floor(area.w / n / 5))) : 0;
      const labelTop = 7;
      const w = Math.max(6, Math.min(30, Math.floor((area.w - 6 - (n - 1) * gap) / n)));
      const h = Math.max(30, area.h - labelTop - miniH - 9);
      const total = n * w + (n - 1) * gap;
      const x0 = area.x + Math.floor((area.w - total) / 2);
      return { n, w, h, gap, x0, y0: area.y + labelTop + 2, miniY: area.y + area.h - miniH, more: G - n };
    }
    function tankTop(i) { const t = tanks(); return { x: t.x0 + i * (t.w + t.gap) + Math.floor(t.w / 2), y: t.y0 }; }

    /* ---------- drawing ---------- */
    const rect = (x, y, w, h, c) => { if (w <= 0 || h <= 0) return; ctx.fillStyle = c; ctx.fillRect(Math.round(x), Math.round(y), Math.round(w), Math.round(h)); };
    function text2x(str, x, y, color) {
      ctx.fillStyle = color;
      let cx = Math.round(x);
      for (const ch of String(str).toUpperCase()) {
        const g = S.GLYPHS[ch];
        if (g) for (let i = 0; i < 15; i++) if (g[i] === '1') ctx.fillRect(cx + (i % 3) * 2, Math.round(y) + Math.floor(i / 3) * 2, 2, 2);
        cx += 8;
      }
    }
    function drawBackground() {
      rect(0, 0, lw, lh, pal.bg);
      rect(0, L.sign.h, lw, 2, pal.wallEdge);
      rect(0, L.server.y, lw, lh - L.server.y, pal.floor);
      ctx.fillStyle = pal.grid;
      for (let x = 0; x < lw; x += 8) ctx.fillRect(x, L.server.y, 1, lh - L.server.y);
      for (let y = L.server.y + 4; y < lh; y += 8) ctx.fillRect(0, y, lw, 1);
      // walls between the zones, with a door into the street
      rect(L.office.x - 2, L.office.y, 1, L.office.h, pal.wallEdge);
      const d = door();
      rect(L.street.x - 2, L.street.y, 1, d.y - 7 - L.street.y, pal.wallEdge);
      rect(L.street.x - 2, d.y + 7, 1, L.street.y + L.street.h - d.y - 7, pal.wallEdge);
      rect(L.street.x - 3, d.y - 7, 3, 1, pal.door); rect(L.street.x - 3, d.y + 6, 3, 1, pal.door);
      // lounge rug
      const la = L.lounge.area;
      rect(la.x, la.y - 1, la.w, la.h + 1, pal.rug);
    }
    function modelSign(name, params) {
      let base = name.split(' / ')[0];
      if (!/\d+(\.\d+)?B\b/i.test(base)) { const m = name.match(/(\d+(?:\.\d+)?B(?:-A\d+B)?)\b/); base += ' ' + (m ? m[1] : params >= 1000 ? (params / 1000).toFixed(1).replace(/\.0$/, '') + 'T' : Math.round(params) + 'B'); }
      return base.replace(/\s*\(.*?\)/g, '');
    }
    // the longest candidate that fits in maxW pixels, else the last one cut short
    function fitText(candidates, maxW) {
      const t = candidates.find((c) => S.textWidth(c) <= maxW);
      if (t != null) return t;
      const last = candidates[candidates.length - 1];
      return last.slice(0, Math.max(0, Math.floor((maxW + 1) / S.ADVANCE)));
    }
    function drawSign() {
      const name = modelSign(vm.model.name, vm.model.params).toUpperCase();
      const eng = vm.engine.name.toUpperCase(), engX = lw - S.textWidth(eng) - 3;
      S.text(ctx, eng, engX, 5, pal.sign2);
      let x = 3, w = name.length * 8 - 2;
      if (w <= engX - 9) {                                 // the big neon sign, centred over the server room when there is room
        x = Math.min(engX - 6 - w, Math.max(3, Math.floor((L.server.w - w) / 2) + L.server.x));
        for (const [ox, oy] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) text2x(name, x + ox, 2 + oy, pal.signGlow);
        text2x(name, x, 2, pal.sign);
      } else {                                             // a narrow stage: the name in small letters
        const small = fitText([name], engX - 9);
        w = S.textWidth(small);
        S.text(ctx, small, x, 5, pal.sign);
      }
      // how many users one pixel person stands for, left of the engine name (shorter when the model name is long)
      if (sprites.per > 1) {
        const n = M.fmtCount(sprites.per), right = engX - 9;
        const t = [`1 PERSON = ${n} USERS`, `1 = ${n} USERS`].find((c) => right - S.textWidth(c) > x + w + 6);
        if (t) S.text(ctx, t, right - S.textWidth(t), 5, pal.ink2);
      }
    }
    function drawZoneLabels() {
      const c = vm.cluster;
      const gpus = `${c.total > 0 ? c.total : c.count} × ${shortHw(c.hwName)}`;
      const units = c.servers ? `${c.R} SERVER${c.R === 1 ? '' : 'S'}` : c.R > 1 ? `${c.R} REPLICAS` : '';
      S.text(ctx, fitText(units ? [`${gpus}  ${units}`, gpus] : [gpus], L.server.w - 4), L.server.x + 2, L.server.y + 1, pal.ink2);
      S.text(ctx, 'SERVED', L.office.x + 1, L.office.y + 1, pal.ink2);
      S.text(ctx, 'IDLE', L.lounge.area.x, L.lounge.area.y - 7, pal.ink2);
      S.text(ctx, 'WAITING', L.street.x + 2, L.street.y + 1, pal.ink2);
    }
    function shortHw(name) {
      return name.replace(/^(GeForce|Instinct|Mac Studio|Google|AWS)\s+/i, '').replace(/\s*\(.*?\)/g, '').replace(/,.*$/, '').replace(/\s+\d+\s*GB.*$/i, '')
        .replace(/\s+(SXM|PCIe|NVL|HGX)\b.*$/i, '').replace(/Blackwell\s*/i, '').replace(/\s+unified$/i, '').trim().toUpperCase().slice(0, 12);
    }
    function drawTank(x, y, w, h, alive) {
      const unsupported = vm.status === 'unsupported';
      const frameC = unsupported ? pal.ink2 : vm.status === 'oom' && Math.floor(time * 4) % 2 === 0 ? pal.red : pal.rack;
      rect(x, y, w, h, frameC);
      const ix = x + 1, iy = y + 1, iw = w - 2, ih = h - 2;
      rect(ix, iy, iw, ih, pal.free);
      if (!alive) return;
      // from the bottom: weights, overhead, KV cache, headroom on top
      let yb = iy + ih;
      const seg = (fr) => Math.round(fr * ih);
      const hW = Math.min(ih, seg(lvl.weights)), hO = Math.min(ih - hW, seg(lvl.overhead));
      const hHead = Math.min(ih - hW - hO, seg(lvl.headroom));
      const hKv = Math.max(0, ih - hW - hO - hHead);
      // weights as bricks
      rect(ix, yb - hW, iw, hW, unsupported ? pal.overhead : pal.weights);
      if (!unsupported) {
        ctx.fillStyle = pal.mortar;
        for (let r = 0; r * 3 < hW; r++) {
          const ry = yb - 1 - r * 3;
          if (ry >= yb - hW) ctx.fillRect(ix, ry, iw, 1);
          for (let bx = (r % 2) * 2; bx < iw; bx += 4) if (ry - 1 >= yb - hW) ctx.fillRect(ix + bx, ry - 2, 1, 2);
        }
      }
      yb -= hW;
      rect(ix, yb - hO, iw, hO, pal.overhead);
      yb -= hO;
      // KV cache: one cell per session (or group of sessions); slot engines show reserved slots and the unused rest
      const cells = vm.gpu.cells;
      const kvShare = lvl.kvUsed + lvl.kvReserved + lvl.kvFree;
      if (hKv > 0 && cells.total > 0 && kvShare > 0) {
        const hCells = cells.slots ? Math.round(hKv * (lvl.kvUsed + lvl.kvReserved) / kvShare) : hKv;
        const ch = hCells / cells.total;
        const gapPx = ch >= 3 ? 1 : 0;
        for (let i = 0; i < cells.total; i++) {
          const top = Math.round(yb - (i + 1) * ch), bot = Math.round(yb - i * ch);
          const hh = bot - top - gapPx;
          if (hh <= 0) continue;
          if (i < cells.used) { rect(ix, top + gapPx, iw, hh, pal.kv); if (hh >= 3 && iw >= 4) rect(ix, top + gapPx, iw, 1, pal.kvEdge); }
          else if (cells.slots) { rect(ix, top + gapPx, iw, 1, pal.kvEdge); rect(ix, bot - 1, iw, 1, pal.kvEdge); rect(ix, top + gapPx, 1, hh, pal.kvEdge); rect(ix + iw - 1, top + gapPx, 1, hh, pal.kvEdge); }
          else if (gapPx) rect(ix, bot - 1, iw, 1, pal.grid);
        }
        if (cells.slots && hKv - hCells > 0) {           // pool memory no slot can use
          ctx.fillStyle = pal.head;
          for (let yy = yb - hKv; yy < yb - hCells; yy++) for (let xx = 0; xx < iw; xx++) if ((xx + yy) % 4 === 0) ctx.fillRect(ix + xx, yy, 1, 1);
        }
      }
      yb -= hKv;
      // headroom the engine leaves unused, hatched
      ctx.fillStyle = pal.head;
      for (let yy = yb - hHead; yy < yb; yy++) for (let xx = 0; xx < iw; xx++) if ((xx - yy) % 3 === 0) ctx.fillRect(ix + xx, yy, 1, 1);
    }
    function drawServerRoom() {
      const t = tanks(), c = vm.cluster;
      const dy = Math.round(drop * -t.h * 0.5);
      const alive = vm.status !== 'unsupported';
      for (let i = 0; i < t.n; i++) {
        const x = t.x0 + i * (t.w + t.gap);
        drawTank(x, t.y0 + dy, t.w, t.h, alive);
        // status LED under each accelerator
        const led = !alive ? pal.red : vm.status === 'oom' ? pal.red : Math.floor(time * 3 + i) % 4 === 0 ? pal.led2 : pal.led;
        rect(x + Math.floor(t.w / 2) - 1, t.y0 + t.h + 2 + dy, 2, 1, led);
      }
      // how the accelerators of one replica work together: a bus for tensor parallel, a chain for a layer split
      if (t.n > 1 && alive) {
        if (c.split === 'tensor' && c.tp > 1) {
          const n = Math.min(t.n, c.tp);
          rect(t.x0 + Math.floor(t.w / 2), t.y0 - 3 + dy, (n - 1) * (t.w + t.gap), 1, pal.bus);
          for (let i = 0; i < n; i++) rect(t.x0 + i * (t.w + t.gap) + Math.floor(t.w / 2), t.y0 - 3 + dy, 1, 3, pal.bus);
        } else if (c.pp > 1) {
          for (let i = 0; i < Math.min(t.n, c.pp) - 1; i++) {
            const ax = t.x0 + (i + 1) * (t.w + t.gap) - t.gap, ay = t.y0 + Math.floor(t.h / 2) + dy;
            S.text(ctx, '>', ax + Math.floor((t.gap - 3) / 2), ay - 2, pal.bus);
          }
        }
      }
      if (t.more > 0) S.text(ctx, `+${t.more}`, t.x0 + t.n * (t.w + t.gap), t.y0 + 2 + dy, pal.ink2);
      // weights that do not fit pile up above the accelerators
      if (lvl.overflow > 0.002 && alive) {
        const hOver = Math.min(t.y0 - L.server.y - 2, Math.round(lvl.overflow * (t.h - 2)));
        const blink = Math.floor(time * 5) % 2 === 0;
        for (let i = 0; i < t.n; i++) {
          const x = t.x0 + i * (t.w + t.gap) + 1;
          rect(x, t.y0 - hOver + dy, t.w - 2, hOver, blink ? pal.red : pal.red2);
          ctx.fillStyle = pal.mortar;
          for (let r = 0; r * 3 < hOver; r++) ctx.fillRect(x, t.y0 - 1 - r * 3 + dy, t.w - 2, 1);
        }
      }
      if (!alive) {                                       // a big red cross: this engine does not run on this hardware
        const x0 = t.x0 - 2, y0 = t.y0 - 2, x1 = t.x0 + t.n * (t.w + t.gap) - t.gap + 2, y1 = t.y0 + t.h + 2;
        ctx.fillStyle = pal.red;
        const n = Math.max(x1 - x0, y1 - y0);
        for (let i = 0; i <= n; i++) {
          const x = x0 + (x1 - x0) * i / n, y = y0 + (y1 - y0) * i / n, y2 = y1 - (y1 - y0) * i / n;
          ctx.fillRect(Math.round(x), Math.round(y), 2, 2); ctx.fillRect(Math.round(x), Math.round(y2), 2, 2);
        }
      }
      // the other replicas, idle accelerators and a prefill pool as small racks
      if (t.miniY < L.racks.y + L.racks.h) {
        let x = L.racks.x + 1;
        const y = t.miniY + 2;
        const mini = (c1, fill) => { rect(x, y, 5, 9, pal.rack); rect(x + 1, y + 1, 3, 7, pal.rackIn); if (fill > 0) rect(x + 1, y + 8 - Math.max(1, Math.round(fill * 7)), 3, Math.max(1, Math.round(fill * 7)), c1); x += 7; };
        const used = Math.min(1, lvl.weights + lvl.overhead + lvl.kvUsed);
        const shown = Math.min(c.R, 20);
        for (let i = 0; i < shown; i++) mini(i === 0 ? pal.token : pal.kv, used);
        if (c.R > shown) { S.text(ctx, `+${M.fmtCount(c.R - shown)}`, x, y + 2, pal.ink2); x += S.textWidth(`+${M.fmtCount(c.R - shown)}`) + 3; }
        if (c.pd > 0) { S.text(ctx, `+${c.pd} PREFILL`, x, y + 2, pal.ink2); x += S.textWidth(`+${c.pd} PREFILL`) + 3; }
        if (c.idle > 0) S.text(ctx, `${c.idle} IDLE`, x, y + 2, pal.head);
      }
    }
    function drawPeople() {
      const occupied = people.filter((p) => !p.hidden && !p.leaving && ZONE[p.group] === 'desks').length;
      const deskCount = Math.min(L.desks.slots.length, Math.max(occupied, Math.ceil((vm.crowd.deskCap || 0) / sprites.per - 1e-9)));
      const taken = new Set(people.filter((p) => ZONE[p.group] === 'desks' && !p.moving && !p.hidden).map((p) => p.seat));
      const emptyDesk = baked('desk', 0, { skin: '', hair: '', shirt: '' });
      for (let i = 0; i < deskCount; i++) if (!taken.has(i)) { const s = L.desks.slots[i]; ctx.drawImage(emptyDesk, s.x, s.y + 3); }
      const slow = vm.speed.slow, ttft = vm.speed.ttftSlow;
      const sorted = people.filter((p) => !p.hidden).sort((a, b) => a.y - b.y);
      for (const p of sorted) {
        const x = Math.round(p.x), y = Math.round(p.y);
        if (p.moving || p.leaving || p.group === 'waiting' || p.group === 'blocked' || p.group === 'rejected' || p.group === 'parked' || p.group === 'evicted') {
          const f = p.moving ? Math.floor(time * 7 + p.phase) % 2 : (p.group === 'waiting' || p.group === 'blocked') && !reduced ? Math.floor(time * 2 + p.phase) % 2 : 0;
          const vr = p.group === 'rejected' && !p.moving ? Object.assign({}, p.vr, { shirt: pal.red }) : p.vr;
          ctx.drawImage(baked('stand', f, vr), x, y);
          if (!p.moving && p.group === 'parked') ctx.drawImage(icon('suitcase', pal.case), x + 5, y + 6);
          if (!p.moving && p.group === 'rejected') ctx.drawImage(icon('cross', pal.red), x + 1, y - 4);
          if (!p.moving && (p.group === 'waiting' || p.group === 'blocked') && vm.crowd.wait > 60 && Math.floor(time * 2 + p.phase) % 2 === 0) ctx.drawImage(icon('bang', pal.red), x + 2, y - 6);
        } else if (p.group === 'sleeping') {
          ctx.drawImage(baked('sleep', 0, p.vr), x, y);
          if (Math.floor(time + p.phase) % 3 !== 0 || reduced) ctx.drawImage(icon('zzz', pal.ink2), x + 2, y - 5 - (reduced ? 0 : Math.floor((time + p.phase) * 2) % 2));
        } else {
          const f = reduced ? 0 : Math.floor(time * 5 + p.phase) % 2;
          ctx.drawImage(baked('seated', f, p.vr), x, y);
          if (ttft && Math.floor(time * 2 + p.phase) % 2 === 0) ctx.drawImage(icon('hourglass', pal.yellow), x + 3, y - 4);
          else if (slow && Math.floor(time * 3 + p.phase) % 3 === 0) ctx.drawImage(icon('sweat', pal.sweat), x + 1, y - 1);
        }
      }
    }
    function drawParticles() {
      for (const q of particles) {
        const t = q.t, mx = (q.x0 + q.x1) / 2, my = Math.min(q.y0, q.y1) - 16;
        const x = (1 - t) * (1 - t) * q.x0 + 2 * (1 - t) * t * mx + t * t * q.x1;
        const y = (1 - t) * (1 - t) * q.y0 + 2 * (1 - t) * t * my + t * t * q.y1;
        rect(x, y, 1, 1, q.c);
      }
    }
    function draw() {
      if (!pal) readPalette();
      ctx.save();
      if (shake > 0) ctx.translate(Math.round((rnd() - 0.5) * 4), Math.round((rnd() - 0.5) * 2));
      drawBackground();
      if (vm) {
        drawSign();
        ctx.save();                                     // racks slide in from the ceiling and spill no bricks onto the sign
        ctx.beginPath(); ctx.rect(L.server.x, L.server.y, L.server.w, L.server.h); ctx.clip();
        drawServerRoom();
        ctx.restore();
        drawZoneLabels();
        drawPeople();
        drawParticles();
      }
      ctx.restore();
      if (flash > 0) { ctx.globalAlpha = Math.min(0.35, flash); rect(0, 0, lw, lh, flashColor); ctx.globalAlpha = 1; }
    }
    function frame(now) {
      raf = requestAnimationFrame(frame);
      const dt = Math.min(0.05, last ? (now - last) / 1000 : 0);
      last = now;
      const t0 = performance.now();
      try { update(dt); draw(); } catch (e) { if (errors++ < 3) console.error(e); if (opts.onError) opts.onError(e); }
      frameMs = frameMs * 0.9 + (performance.now() - t0) * 0.1;
    }

    readPalette();
    resize();
    return {
      setView,
      resize: () => resize(),
      setPalette() { readPalette(); },
      setReducedMotion(v) { reduced = !!v; if (reduced) { particles.length = 0; assign(true); } },
      start() { if (!raf) { last = 0; raf = requestAnimationFrame(frame); } },
      stop() { cancelAnimationFrame(raf); raf = 0; },
      drawOnce() { update(0); draw(); },
      advance(seconds) { for (let t = 0; t < seconds; t += 1 / 60) update(1 / 60); draw(); },   // for tests: run the clock without frames
      stats: () => ({ frameMs, scale, lw, lh, people: people.length, per: sprites.per, particles: particles.length, errors }),
    };
  }
  return { create };
})();
