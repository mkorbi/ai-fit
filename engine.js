/* engine.js - capacity model for AI Fit.
 * Pure functions, no DOM. Works in the browser (globals) and in node (module.exports).
 *
 * The model is a roofline estimate, not a simulator:
 *   memory   weights + KV cache + framework overhead must fit the accelerator memory pool
 *   decode   one step = max(bytes read / bandwidth, FLOPs / peak) + collective latency
 *   prefill  FLOPs / peak for the tokens that are not already cached
 *   load     prefill work steals GPU time from decode (chunked prefill); solved as a fixed point
 */
const Engine = (() => {
  const LINKS_ = typeof LINKS !== 'undefined' ? LINKS : require('./catalog.js').LINKS;
  const ARCHS_ = typeof ARCHS !== 'undefined' ? ARCHS : require('./catalog.js').ARCHS;
  const ENGINES_ = typeof ENGINES !== 'undefined' ? ENGINES : require('./catalog.js').ENGINES;
  const FAMILY_ = typeof ENGINE_FAMILY !== 'undefined' ? ENGINE_FAMILY : require('./catalog.js').ENGINE_FAMILY;
  const FAMILIES_ = typeof ENGINE_FAMILIES !== 'undefined' ? ENGINE_FAMILIES : require('./catalog.js').ENGINE_FAMILIES;
  const BYTES_W  = { bf16: 2, fp8: 1, int8: 1, int4: 0.52, fp4: 0.53 };   // bytes per parameter incl. group scales
  const BYTES_KV = { bf16: 2, fp8: 1, int8: 1, int4: 0.56 };              // bytes per KV element
  const W_PRECS  = ['bf16', 'fp8', 'int8', 'int4', 'fp4'];
  const KV_PRECS = ['bf16', 'fp8', 'int8', 'int4'];
  const PREC_LABEL = { bf16: 'BF16', fp8: 'FP8', int8: 'INT8', int4: 'INT4 (AWQ/GPTQ)', fp4: 'FP4 (NVFP4/MXFP4)' };
  const KV_LABEL = { bf16: 'BF16', fp8: 'FP8', int8: 'INT8', int4: 'INT4' };
  const TP_CANDIDATES = [1, 2, 4, 8, 16, 32, 64];
  const PP_CANDIDATES = [1, 2, 4, 8, 16];
  const DEFAULT_ADV = {
    util: 0.90,          // fraction of accelerator memory the engine may use (vLLM gpu_memory_utilization)
    overheadGB: 1.5,     // fixed per-GPU overhead: CUDA context, graphs, workspace
    overheadFrac: 0.04,  // activation / workspace overhead as a fraction of the per-GPU weight shard
    bwEff: 0.75,         // achieved fraction of peak memory bandwidth during decode
    mfu: 0.50,           // achieved fraction of peak tensor FLOPS (prefill and large-batch decode)
    frag: 0.04,          // KV cache lost to paged-block fragmentation
    ppBubble: 0.10,      // pipeline bubble overhead per extra stage
    collEff: 0.70,       // achieved fraction of link bandwidth in collectives
    hostRestoreGBs: 20,  // GB/s per GPU when restoring a session's KV cache from host memory
    specK: 4,            // draft tokens per speculative step
    specAlpha: 0.70,     // per-token acceptance rate of the draft
    prefillMaxLoad: 0.85 // above this share of GPU time on prefill the replica counts as saturated
  };

  const norm = (m) => Object.assign({ active: m.params, attn: [{ n: m.layers, type: 'full' }] }, m);
  const isMla = (m) => m.attn.some((l) => l.type === 'mla');

  function weightBytes(model, prec, engine) {
    const b = bytesW(prec, engine);
    if (b != null) return model.params * 1e9 * b;                     // whole-file bits per weight (GGUF): embeddings and head included
    return model.params * 1e9 * (0.98 * BYTES_W[prec] + 0.02 * 2);   // ~2% (embeddings, norms, head) stay 16-bit
  }
  function expertSplit(model) {
    if (!model.moe) return { expert: 0, dense: model.params * 1e9 };
    const { experts: E, active: k } = model.moe;
    const per = ((model.params - model.active) * 1e9) / Math.max(1, E - k);
    const expert = Math.min(model.params * 1e9 * 0.99, E * per);
    return { expert, dense: model.params * 1e9 - expert };
  }
  function touched(model, tokens) {           // share of routed experts read in one step for `tokens` tokens
    if (!model.moe) return 1;
    const { experts: E, active: k } = model.moe;
    return 1 - Math.pow(1 - k / E, tokens);
  }
  function layerPerTok(l, model, bytes) {
    if (l.type === 'mla') return (l.dc + l.dr) * bytes;
    if (l.type === 'linear') return 0;
    return 2 * (l.nKv ?? model.nKv) * (l.dHead ?? model.dHead) * bytes;
  }
  /* KV bytes for one session of C tokens, of which the first S tokens are a prefix shared by every session. */
  function kvSplit(model, kvPrec, C, S, engine) {
    const bytes = bytesKv(kvPrec, engine);
    let shared = 0, per = 0, full = 0;
    for (const l of model.attn) {
      const pt = layerPerTok(l, model, bytes);
      if (!pt) continue;
      if (l.window) { const t = l.n * pt * Math.min(C, l.window); per += t; full += t; }
      else { shared += l.n * pt * S; per += l.n * pt * (C - S); full += l.n * pt * C; }
    }
    const st = (model.statePerSeqMB || 0) * 1e6;
    return { shared, perSession: per + st, full: full + st };
  }
  const kvAtCtx = (model, kvPrec, C, engine) => kvSplit(model, kvPrec, C, 0, engine).perSession;
  const kvPerTokenFull = (model, kvPrec, engine) => model.attn.reduce((a, l) => a + l.n * layerPerTok(l, model, bytesKv(kvPrec, engine)), 0);

  /* Attention FLOPs for one decoded token attending over ctx tokens. */
  function attnFlopsPerTok(model, ctx) {
    let f = 0;
    for (const l of model.attn) {
      const nh = l.nHeads ?? model.nHeads, dh = l.dHead ?? model.dHead;
      if (l.type === 'linear') { f += l.n * 4 * nh * dh * dh; continue; }
      const eff = l.window ? Math.min(ctx, l.window) : ctx;
      const dqk = l.type === 'mla' ? l.dc + l.dr : dh, dv = l.type === 'mla' ? l.dc : dh;
      if (model.sparse) f += l.n * (2 * nh * (dqk + dv) * Math.min(eff, model.sparse.topk) + 0.25 * nh * dqk * eff);
      else f += l.n * 2 * nh * (dqk + dv) * eff;
    }
    return f;
  }
  const sumEff = (from, to, w) => {                      // Σ_{i=from+1..to} min(i, w)
    if (to <= w) return (to * to - from * from) / 2;
    const a = Math.min(w, from);
    return (w * w - a * a) / 2 + w * (to - Math.max(w, from));
  };
  /* Attention FLOPs to prefill tokens from+1..to on top of `from` cached tokens (causal, flash-style). */
  function prefillAttnFlops(model, from, to) {
    const n = to - from; if (n <= 0) return 0;
    let f = 0;
    for (const l of model.attn) {
      const nh = l.nHeads ?? model.nHeads, dh = l.dHead ?? model.dHead;
      if (l.type === 'linear') { f += l.n * 4 * nh * dh * dh * n; continue; }
      const dqk = l.type === 'mla' ? l.dc + l.dr : dh, dv = l.type === 'mla' ? l.dc : dh;
      const w = l.window || Infinity;
      if (model.sparse) f += l.n * (2 * nh * (dqk + dv) * sumEff(from, to, Math.min(w, model.sparse.topk)) + 0.25 * nh * dqk * sumEff(from, to, w));
      else f += l.n * 2 * nh * (dqk + dv) * sumEff(from, to, w);
    }
    return f;
  }
  /* 'native' (tensor cores compute in that format), 'weight-only' (weights stored in it, matmuls in BF16), or 'unsupported'. */
  function hwSupport(hw, wPrec) {
    if (wPrec === 'bf16') return 'native';
    const a = ARCHS_[hw.arch];
    if (!a) {                                   // custom hardware without a generation: infer from the TFLOPS keys
      const t = hw.tflops || {};
      if (wPrec === 'fp8') return t.fp8 ? 'native' : 'weight-only';
      if (wPrec === 'int8') return (t.int8 || t.fp8) ? 'native' : 'weight-only';
      if (wPrec === 'fp4') return t.fp4 ? 'native' : 'weight-only';
      return 'weight-only';
    }
    if (a.native.includes(wPrec) && wPrec !== 'int4') return 'native';
    if (a.weightOnly.includes(wPrec) || (wPrec === 'int4' && a.native.includes('int4'))) return 'weight-only';
    return 'unsupported';
  }
  const familyOf = (hw) => FAMILY_[hw.arch] || 'other';
  const engineOf = (id) => (id && id !== 'none' && ENGINES_[id]) || null;
  /* Does the engine run on this hardware at all? Engines list the families they support (catalog.js ENGINES[].families);
   * engine 'none' covers everything, and custom hardware without a known generation is 'unknown' (treated as covered). */
  function engineCovers(hw, id) {
    const e = engineOf(id);
    if (!e || !e.families) return 'yes';
    const f = FAMILY_[hw.arch];
    return !f ? 'unknown' : e.families.includes(f) ? 'yes' : 'no';
  }
  /* "NVIDIA Ampere, Ada Lovelace, Hopper; AMD CDNA 3" for a list of family ids. */
  function familyNames(ids) {
    const groups = [];
    for (const f of FAMILIES_) {
      if (!ids.includes(f.id)) continue;
      const g = groups.find((x) => x.vendor === f.vendor);
      if (g) g.names.push(f.name); else groups.push({ vendor: f.vendor, names: [f.name] });
    }
    return groups.map((g) => (g.names.length === 1 && g.names[0].startsWith(g.vendor) ? g.names[0] : `${g.vendor} ${g.names.join(', ')}`)).join('; ');
  }
  /* How an engine serves requests (catalog.js ENGINES[].serving). The defaults describe a fully capable engine: continuous
   * batching over a paged KV cache, tensor and pipeline parallelism across nodes, every optimization available and no cap
   * on concurrent requests. Engine 'none' and entries without `serving` use them unchanged. */
  const SERVING_FULL = {
    batching: 'continuous',   // 'continuous' (requests join and leave the batch every step) or 'slots' (a fixed number of parallel slots)
    slots: null,              // slots: { default, env } for slot engines: every slot reserves a full context at load time
    maxBatch: null,           // default cap on concurrent requests per replica: { value } | { tiers, default } | { perTokens, min, max }
    paged: true,              // KV cache in pages (fragmentation share applies) rather than one contiguous reservation per slot
    pp: 'pipelined',          // 'pipelined' (micro-batches keep every stage busy) or 'sequential' (layer split: stages take turns)
    maxTp: Infinity, multiNode: true,
    prefixCache: true,        // true = shared prefix stored once per replica; 'per-slot' = reused per slot, stored per session; false
    spec: true, pd: true, dpAttention: true,
    retention: ['host', 'gpu', 'none'],
    hostCacheGB: null,        // host memory the engine may use to park idle sessions, per replica (null = the nodes' RAM)
    queueMax: null,           // requests an engine queues before rejecting (null = unbounded)
  };
  const servingOf = (id) => Object.assign({}, SERVING_FULL, (engineOf(id) || {}).serving);
  const formatOf = (id, kind, prec) => ((((engineOf(id) || {}).formats || {})[kind] || {})[prec]) || {};
  const bytesW = (prec, id) => formatOf(id, 'w', prec).bytes;                           // per-engine override or undefined
  const bytesKv = (prec, id) => formatOf(id, 'kv', prec).bytes ?? BYTES_KV[prec];
  const precLabel = (prec, id) => formatOf(id, 'w', prec).label || PREC_LABEL[prec];
  const kvLabel = (prec, id) => formatOf(id, 'kv', prec).label || KV_LABEL[prec];
  /* Default cap on concurrent requests per replica for a continuous-batching engine. */
  function maxBatchOf(sv, hw, poolTokens, model) {
    const m = sv.maxBatch;
    if (!m) return Infinity;
    if (m.value) return m.value;
    if (m.tiers) {
      const gib = hw.mem * 1e9 / 2 ** 30;
      const t = m.tiers.find((x) => gib >= x.minGiB && !(x.notArch || []).includes(hw.arch));
      return t ? t.value : m.default;
    }
    if (m.perTokens) {
      const est = Math.min(m.max, Math.max(m.min, Math.floor(poolTokens / model.maxCtx * m.perTokens)));
      return Math.max(1, Math.min(est, Math.floor(poolTokens / 2)));
    }
    return Infinity;
  }
  const OPT_NAMES = { prefixCache: 'prefix caching', spec: 'speculative decoding', pd: 'prefill/decode disaggregation', dpAttention: 'data-parallel attention', retention: 'the chosen idle-session mode' };
  /* What the engine actually runs: optimizations it lacks count as off, an idle-session mode it lacks falls back to eviction.
   * `gated` lists what was asked for but is not available. */
  function effective(p) {
    const sv = servingOf(p.engine), o = p.opt || {}, gated = [];
    const opt = { prefixCache: !!o.prefixCache && sv.prefixCache === true, spec: !!o.spec && !!sv.spec, pd: !!o.pd && !!sv.pd };
    for (const k of ['prefixCache', 'spec', 'pd']) if (o[k] && !opt[k]) gated.push(k);
    const dpAttention = !!p.dpAttention && !!sv.dpAttention;
    if (p.dpAttention && !dpAttention) gated.push('dpAttention');
    let retention = p.wl.retention;
    if (!sv.retention.includes(retention)) { gated.push('retention'); retention = sv.retention.includes('none') ? 'none' : sv.retention[0]; }
    return { sv, opt, dpAttention, retention, gated };
  }
  /* Weight-format support as the weaker of silicon capability and the chosen engine's documented kernels. */
  function formatSupport(hw, wPrec, engine) {
    const eng = engineOf(engine);
    if (eng && engineCovers(hw, engine) === 'no') return 'unsupported';
    const cap = hwSupport(hw, wPrec);
    if (!eng || wPrec === 'bf16' || cap === 'unsupported') return cap;
    const e = (eng.weights[wPrec] || {})[familyOf(hw)];
    if (!e) return 'unsupported';
    return e === 'weight-only' || cap === 'weight-only' ? 'weight-only' : 'native';
  }
  /* KV-cache dtype support: the silicon always can (it is just storage); the engine must run there and have kernels for it. */
  function kvSupport(hw, kvPrec, engine) {
    const eng = engineOf(engine);
    if (eng && engineCovers(hw, engine) === 'no') return 'unsupported';
    if (kvPrec === 'bf16' || !eng) return 'supported';
    const fams = (eng.kvCache || {})[kvPrec];
    return fams && fams.includes(familyOf(hw)) ? 'supported' : 'unsupported';
  }
  function peakFlops(hw, wPrec, engine) {
    const t = hw.tflops || {}, fp16 = (t.fp16 || 0) * 1e12;
    const label = { bf16: 'BF16', fp8: 'FP8', int8: 'INT8', int4: 'INT4', fp4: 'FP4' }[wPrec] || wPrec;
    const support = formatSupport(hw, wPrec, engine);
    if (wPrec === 'bf16') return { peak: fp16, used: 'BF16', native: true, support };
    if (support === 'unsupported') return { peak: fp16, used: `${label} not loadable`, native: false, support };
    if (support === 'weight-only') return { peak: fp16, used: `BF16 (${label} weight-only)`, native: false, support };
    const peak = wPrec === 'fp8' ? t.fp8 : wPrec === 'int8' ? (t.int8 || t.fp8) : t.fp4;
    if (!peak) return { peak: fp16, used: `BF16 (no ${label} TFLOPS in catalog)`, native: false, support: 'weight-only' };
    return { peak: peak * 1e12, used: label, native: true, support };
  }
  const nativeKey = (m) => (m.nativePrec || '').split(' ')[0].replace('fp16', 'bf16');

  /* ---------- the core evaluation of one layout ---------- */
  function evaluate(p) {
    const hw = p.hw, model = norm(p.model), adv = Object.assign({}, DEFAULT_ADV, p.adv || {});
    const eff = effective(p), sv = eff.sv, opt = eff.opt;
    const wl = eff.retention === p.wl.retention ? p.wl : Object.assign({}, p.wl, { retention: eff.retention });
    const tp = p.tp, pp = p.pp, R = p.replicas, G = tp * pp, total = G * R;
    const seqPP = sv.pp === 'sequential', computeGpus = seqPP ? tp : G;   // layer split: the stages take turns
    const warnings = [];
    const push = (level, text) => warnings.push({ level, text });

    // memory
    const W = weightBytes(model, p.wPrec, p.engine);
    const split = expertSplit(model);
    const denseBytes = W * split.dense / (model.params * 1e9), expertBytes = W * split.expert / (model.params * 1e9);
    const overhead = adv.overheadGB * 1e9 + adv.overheadFrac * (W / G);
    const capPerGpu = hw.mem * 1e9 * adv.util - overhead;
    const kvAvailRaw = G * capPerGpu - W;
    const mla = isMla(model);
    const kvRepl = eff.dpAttention ? 1 : (mla ? tp : Math.max(1, tp / model.nKv));
    const frag = sv.paged ? adv.frag : 0;
    const kvAvail = kvAvailRaw > 0 ? kvAvailRaw * (1 - frag) / kvRepl : 0;
    const C = Math.max(64, wl.ctx);
    const prefix = wl.prefix > 0 ? Math.min(wl.prefix, C - 1) : 0;
    const S = opt.prefixCache ? prefix : 0;                                     // stored once per replica
    const Scomp = opt.prefixCache || sv.prefixCache === 'per-slot' ? prefix : 0;  // not prefilled again
    const kv = kvSplit(model, p.kvPrec, C, S, p.engine);
    const memSessions = kvAvail > kv.shared ? Math.floor((kvAvail - kv.shared) / kv.perSession) : 0;
    const pk = peakFlops(hw, p.wPrec, p.engine);
    const covers = engineCovers(hw, p.engine);
    const kvOK = kvSupport(hw, p.kvPrec, p.engine) === 'supported';
    const tpOK = tp <= sv.maxTp, nodeOK = sv.multiNode || G <= p.nodeGpus;
    const loadable = covers !== 'no' && pk.support !== 'unsupported' && kvOK && tpOK && nodeOK;
    // slot engines reserve every slot's full context when the model loads: extra memory beyond that buys nothing
    let maxSessions = memSessions, slots = null;
    if (sv.batching === 'slots') {
      const n = p.slots > 0 ? p.slots | 0 : sv.slots.default;
      slots = { n, auto: !(p.slots > 0), env: sv.slots.env, fit: memSessions >= n, reserved: kv.shared + n * kv.perSession };
      slots.unused = slots.fit ? Math.max(0, kvAvail - slots.reserved) : 0;
      maxSessions = slots.fit ? n : 0;
    }
    const fits = loadable && kvAvailRaw > 0 && maxSessions >= 1;
    const poolTokens = kvAvail / Math.max(1, kvPerTokenFull(model, p.kvPrec, p.engine));
    const maxBatch = slots ? slots.n : maxBatchOf(sv, hw, poolTokens, model);   // concurrent requests one replica takes

    // workload shape
    const B = Math.max(1, Math.round(wl.users * wl.activity));
    const residentUsers = wl.retention === 'gpu' ? wl.users : B;
    const sessionsPerRep = Math.ceil(residentUsers / R);
    const bPerRep = Math.ceil(B / R);
    const memOK = fits && sessionsPerRep <= maxSessions;
    const slotLimited = !!slots && fits && !memOK;
    const nodes = Math.ceil(total / p.nodeGpus);
    const idleUsers = Math.max(0, wl.users - B);
    const hostBytes = wl.retention === 'host' ? Math.min(nodes * p.hostRamGB * 1e9 * 0.8, sv.hostCacheGB != null ? R * sv.hostCacheGB * 1e9 : Infinity) : 0;
    const hostSessions = wl.retention === 'host' ? Math.floor(hostBytes / kv.perSession) : 0;
    const hostCoverage = wl.retention === 'host' ? (idleUsers > 0 ? Math.min(1, hostSessions / idleUsers) : 1) : 0;

    // links
    const link = LINKS_[p.link] || LINKS_.nvlink;
    const linkBw = ((link.bw ?? hw.linkBw) || 0) * 1e9 * adv.collEff;
    const netBwPerGpu = (p.net.gbps * 1e9 / 8) / p.nodeGpus * adv.collEff;
    const tpCross = tp > p.nodeGpus, ppCross = G > p.nodeGpus;
    const collective = (cross, bytes, count) => {
      const lat = (cross ? p.net.latencyUs : link.latencyUs) * 1e-6;
      const bw = cross ? netBwPerGpu : linkBw;
      return count * (lat + (bw > 0 ? bytes / bw : 0));
    };

    function decodeStep(b, spec) {
      const Cavg = Math.max(1, C - wl.output / 2);
      const kmul = spec ? adv.specK + 1 : 1;
      const wRead = denseBytes + expertBytes * touched(model, b * kmul);
      const kvRead = b * kvAtCtx(model, p.kvPrec, Cavg, p.engine);
      // pipelined stages overlap micro-batches; a sequential layer split reads every stage's weights and KV one after another
      const tBw = (wRead + (seqPP ? kvRead : kvRead / pp)) / (tp * hw.bw * 1e9 * adv.bwEff);
      const flopsTok = 2 * model.active * 1e9 + attnFlopsPerTok(model, Cavg);
      const tComp = b * kmul * flopsTok / (computeGpus * pk.peak * adv.mfu);
      let tComm = 0;
      if (tp > 1) tComm += collective(tpCross, 2 * (tp - 1) / tp * b * kmul * model.dModel * 2, 2 * model.layers);
      if (pp > 1) tComm += collective(ppCross, b * kmul * model.dModel * 2, pp - 1);
      const bubble = pp > 1 && !seqPP ? 1 + adv.ppBubble * (pp - 1) / pp : 1;
      const step = (Math.max(tBw, tComp) + tComm) * bubble;
      const bound = tComm > Math.max(tBw, tComp) ? 'communication' : (tBw >= tComp ? 'memory bandwidth' : 'compute');
      return { tBw, tComp, tComm, step, bound, wRead, kvRead };
    }
    function prefill(newTok, cachedTok) {
      if (newTok <= 0) return { t: 0, flops: 0 };
      const flops = newTok * 2 * model.active * 1e9 + prefillAttnFlops(model, cachedTok, cachedTok + newTok);
      let t = flops / (tp * pk.peak * adv.mfu);
      if (tp > 1) t += collective(tpCross, 2 * (tp - 1) / tp * newTok * model.dModel * 2, 2 * model.layers);
      if (pp > 1) t += collective(ppCross, newTok * model.dModel * 2, pp - 1);
      return { t, flops };
    }
    const promptLen = Math.max(1, C - wl.output);
    const coldNew = Math.max(1, promptLen - Scomp);
    const cold = prefill(coldNew, Scomp);
    let warm, restoreS = 0, warmNew = coldNew;
    if (wl.retention === 'none') warm = cold;
    else {
      warmNew = Math.min(Math.max(1, wl.newPrompt), promptLen);
      warm = prefill(warmNew, promptLen - warmNew);
      if (wl.retention === 'host') restoreS = kv.perSession / (tp * adv.hostRestoreGBs * 1e9);
    }
    const cov = wl.retention === 'host' ? hostCoverage : 1;
    const reqFlops = cov * warm.flops + (1 - cov) * cold.flops;
    const ttft = cov * (warm.t + restoreS) + (1 - cov) * cold.t;
    const cap = computeGpus * pk.peak * adv.mfu;
    const ttftMax = wl.ttftMax > 0 ? wl.ttftMax : Infinity;
    const ttftOK = ttft <= ttftMax;

    function loadAt(b, spec) {
      spec = spec && !!sv.spec;
      const d = decodeStep(b, spec);
      const acc = spec ? (1 - Math.pow(adv.specAlpha, adv.specK + 1)) / (1 - adv.specAlpha) : 1;
      const o = wl.output / acc;
      // prefill share f of one replica's compute: f = λ · reqFlops / cap with λ = b / (ttft + o · step / (1 − f)).
      // g(f) falls as f rises, so f − g(f) crosses zero exactly once: solve it by bisection (plain iteration oscillates under load).
      const g = (f) => (b / (ttft + o * d.step / (1 - f))) * reqFlops / cap;
      let f = 0;
      if (!opt.pd && reqFlops > 0) {
        const FMAX = 0.999;
        if (g(FMAX) > FMAX) f = FMAX;
        else { let lo = 0, hi = FMAX; for (let i = 0; i < 48; i++) { const mid = (lo + hi) / 2; if (g(mid) > mid) lo = mid; else hi = mid; } f = (lo + hi) / 2; }
      }
      const saturated = f >= adv.prefillMaxLoad;
      const itl = d.step / (1 - f);
      const dur = ttft + o * itl;
      const lambda = b / dur;
      const prefillLoad = lambda * reqFlops / cap;                       // share of one replica's compute (equals f unless disaggregated)
      const pdGpus = opt.pd ? Math.ceil(lambda * reqFlops / (pk.peak * adv.mfu)) : 0;
      return Object.assign(d, { acc, itl, f, saturated, lambda, dur, prefillLoad, pdGpus, perUser: acc / itl, agg: b * acc / itl });
    }
    // with every session resident, the decode batch is the resident sessions' share in flight; one resident session still decodes
    const bMem = wl.retention === 'gpu' ? (maxSessions >= 1 ? Math.max(1, Math.floor(maxSessions * wl.activity)) : 0) : maxSessions;
    const bCap = Math.min(bMem, maxBatch);
    function bSpeedSearch() {
      if (bCap < 1) return 0;
      if (!ttftOK) return 0;
      const ok = (b) => { const r = loadAt(b, opt.spec); return r.perUser >= wl.target && !r.saturated; };
      if (!ok(1)) return 0;
      let lo = 1, hi = bCap;
      if (ok(hi)) return hi;
      while (hi - lo > 1) { const mid = Math.floor((lo + hi) / 2); if (ok(mid)) lo = mid; else hi = mid; }
      return lo;
    }
    const bSpeed = fits ? bSpeedSearch() : 0;
    let maxConc, maxUsers;
    if (wl.retention === 'gpu') {
      const usersMem = maxSessions * R, usersSpeed = Math.floor(bSpeed * R / wl.activity);
      maxUsers = Math.min(usersMem, usersSpeed);
      maxConc = Math.floor(maxUsers * wl.activity);
    } else {
      maxConc = Math.min(maxSessions, bSpeed) * R;
      maxUsers = Math.floor(maxConc / wl.activity);
    }
    const bAt = Math.min(bPerRep, Math.max(1, maxSessions), maxBatch);        // requests one replica actually runs at this load
    const at = fits ? loadAt(bAt, opt.spec) : null;
    const speedOK = !!at && memOK && ttftOK && at.perUser >= wl.target && !at.saturated;
    // what bounds capacity: memory, the engine's slots or batch cap, or the speed and first-token targets
    let capLimit = 'none';
    if (!loadable) capLimit = 'unsupported';
    else if (!fits) capLimit = slots && !slots.fit ? 'slots' : 'memory';
    else if (bSpeed < bCap || bCap < 1) capLimit = !ttftOK ? 'ttft' : (loadAt(Math.min(bCap, bSpeed + 1), opt.spec).saturated ? 'prefill' : 'speed');
    else capLimit = slots ? 'slots' : maxBatch < bMem ? 'max-batch' : 'memory';

    function solveMaxCtx(sessions) {
      if (!fits || sessions < 1) return 0;
      const fitsAt = (c) => { const s = opt.prefixCache && wl.prefix > 0 ? Math.min(wl.prefix, c - 1) : 0; const k = kvSplit(model, p.kvPrec, c, s, p.engine); return k.shared + sessions * k.perSession <= kvAvail; };
      if (!fitsAt(64)) return 0;
      let lo = 64, hi = 1 << 26;
      if (fitsAt(hi)) return hi;
      while (hi - lo > 64) { const mid = Math.floor((lo + hi) / 2); if (fitsAt(mid)) lo = mid; else hi = mid; }
      return lo;
    }
    const maxCtxAtLoad = solveMaxCtx(slots ? slots.n : sessionsPerRep);

    // cost and power
    const pdTotal = at ? at.pdGpus * R : 0;
    const gpusUsed = total + pdTotal;
    const price = hw.price != null ? hw.price * gpusUsed : null;
    const aggTotal = at ? at.agg * R : 0;
    const costPerMTok = price != null && aggTotal > 0 ? price / (aggTotal * 3600 / 1e6) : null;
    const kW = hw.tdp != null ? hw.tdp * gpusUsed / 1000 : null;

    // warnings
    if (wl.ctx > model.maxCtx) push('crit', `Context of ${fmtTok(wl.ctx)} exceeds the model's maximum of ${fmtTok(model.maxCtx)} tokens.`);
    else if (model.nativeCtx && wl.ctx > model.nativeCtx) push('warn', `Beyond the native ${fmtTok(model.nativeCtx)} context; needs RoPE scaling (YaRN), quality may drop.`);
    const eng = engineOf(p.engine), engName = eng ? eng.name : null;
    const archName = (ARCHS_[hw.arch] || {}).name;
    if (covers === 'no') push('crit', `${engName} does not run on ${archName || hw.name}; per its docs (checked ${(eng.familiesSource || eng).checked}) it runs on ${familyNames(eng.families)}.`);
    else if (pk.support === 'unsupported') push('crit', engName && hwSupport(hw, p.wPrec) !== 'unsupported' ? `${engName} has no kernel for ${precLabel(p.wPrec, p.engine)} weights on ${archName || 'this generation'} (per its docs, checked ${eng.checked}).` : `${PREC_LABEL[p.wPrec]} weights cannot be loaded on ${hw.name} (${archName || 'unknown generation'}): no kernel for that format.`);
    else if (!kvOK) push('crit', `${engName} does not support a ${kvLabel(p.kvPrec, p.engine)} KV cache on ${archName || 'this generation'}; use ${Object.keys(eng.kvCache || {}).filter((k) => kvSupport(hw, k, p.engine) === 'supported').concat('bf16').map((k) => kvLabel(k, p.engine)).join(' or ')} KV cache.`);
    else if (!tpOK) push('crit', `${engName} splits layers across GPUs and has no tensor parallelism: use TP 1 and spread the model with PP instead.`);
    else if (!nodeOK) push('crit', `${engName} serves a model within one node (${p.nodeGpus} accelerators); this layout needs ${G}.`);
    else if (!fits) push('crit', kvAvailRaw <= 0 ? `Weights alone (${fmtGB(W)}) do not fit in ${G} × ${hw.mem} GB with headroom.`
      : slots && !slots.fit && memSessions >= 1 ? `${slots.n} parallel slot${slots.n === 1 ? '' : 's'} × ${fmtTok(C)} tokens need ${fmtGB(slots.reserved)} of KV cache; ${fmtGB(kvAvail)} is free after the weights. ${engName} would move layers to the CPU (much slower, not modeled): lower ${slots.env} or the context.`
      : `Only ${fmtGB(kvAvail)} left for KV cache; one session at ${fmtTok(C)} needs ${fmtGB(kv.perSession)}.`);
    else if (slotLimited) push('crit', `${sessionsPerRep} ${wl.retention === 'gpu' ? 'resident sessions' : 'concurrent requests'} per server but ${slots.n} parallel slot${slots.n === 1 ? '' : 's'} (${slots.env}): the rest wait in ${engName}'s queue.`);
    else if (!memOK) push('crit', wl.retention === 'gpu'
      ? `Keeping every user's session in GPU memory needs ${sessionsPerRep} sessions per replica; ${maxSessions} fit.`
      : `${sessionsPerRep} concurrent sessions per replica need ${fmtGB(kv.shared + sessionsPerRep * kv.perSession)} of KV cache; ${fmtGB(kvAvail)} available.`);
    if (fits && !ttftOK) push('crit', `Time to first token of ${fmtTime(ttft)} exceeds the ${fmtTime(ttftMax)} limit: ${fmtTok(warmNew)} tokens must be prefilled on ${tp} accelerator${tp > 1 ? 's' : ''} each turn.`);
    if (at && at.saturated) push('crit', `Prefill saturates the replica: prompts arrive faster than ${fmtTok(reqFlops / (2 * model.active * 1e9))}-token prefills can be computed. Add compute or cache more of the prompt.`);
    if (tpCross) push('warn', p.net.gbps > 0 ? `Tensor parallel spans ${Math.ceil(tp / p.nodeGpus)} nodes; every layer's all-reduce crosses the network.` : 'Tensor parallel spans nodes but no inter-node network is configured.');
    if (ppCross && !tpCross && p.net.gbps === 0) push('crit', 'The layout needs more than one node but no inter-node network is configured.');
    if (kvRepl > 1) push('warn', `KV cache is replicated ${kvRepl}× across tensor-parallel ranks (${mla ? 'MLA has a single latent head' : `${model.nKv} KV heads < TP ${tp}`}). ${sv.dpAttention ? 'Enable data-parallel attention or lower TP.' : 'Lower TP.'}`);
    if (fits && capLimit === 'max-batch') push('info', `${engName} takes at most ${maxBatch.toLocaleString('en-US')} concurrent requests per replica by default (${(sv.maxBatch || {}).name || 'batch limit'}); memory would allow ${memSessions.toLocaleString('en-US')}. Raise it to use the room.`);
    const gatedTxt = eff.gated.filter((k) => k !== 'dpAttention' || tp > 1).map((k) => OPT_NAMES[k]);
    if (gatedTxt.length) push('info', `${engName} has no ${gatedTxt.join(', ')}; ${gatedTxt.length > 1 ? 'they count' : 'it counts'} as off${eff.gated.includes('retention') ? ' (idle sessions are evicted instead)' : ''}.`);
    if (sv.prefixCache === 'per-slot' && prefix > 0) push('info', `${engName} reuses the shared prefix within each slot, so it is not prefilled again, but every session stores its own copy.`);
    if (seqPP && pp > 1) push('info', `${engName} splits the layers across ${pp} accelerators: their memory adds up, their speed does not (a token passes them one after another).`);
    if (slots && fits && R > 1) push('info', `${R} ${engName} servers behind a load balancer, ${slots.n} parallel slot${slots.n === 1 ? '' : 's'} each.`);
    if (slots && fits && slots.unused > 0.25 * kvAvail) push('info', `${fmtGB(slots.unused)} of KV memory per server stays unused: raise ${slots.env} to serve more requests at once.`);
    if (covers === 'unknown' && eng) push('info', `${engName} support is not checked for custom hardware without a chip generation.`);
    if (pk.support === 'weight-only') push('info', `${precLabel(p.wPrec, p.engine)} compute is not native on ${hw.name}: quantized weights save memory but matmuls run in BF16.`);
    const nk = nativeKey(model);
    if (nk && BYTES_W[nk] != null && BYTES_W[p.wPrec] > BYTES_W[nk]) push('info', `The checkpoint ships in ${model.nativePrec}; serving it in ${precLabel(p.wPrec, p.engine)} upcasts the weights to ${(BYTES_W[p.wPrec] / BYTES_W[nk]).toFixed(1)}× the bytes without a quality gain.`);
    else if (nk && BYTES_W[nk] != null && BYTES_W[p.wPrec] < BYTES_W[nk]) push('info', `Needs a ${precLabel(p.wPrec, p.engine)} checkpoint or on-the-fly quantization: the official weights ship in ${model.nativePrec}.`);
    if (p.count != null && p.count - total > 0) push('info', `${p.count - total} of ${p.count} accelerators are idle in this layout.`);
    if (wl.retention === 'host' && hostCoverage < 1) push('warn', `Host memory keeps ${hostSessions} of ${idleUsers} idle sessions warm${sv.hostCacheGB != null ? ` (${engName} caches at most ${fmtGB(sv.hostCacheGB * 1e9)} per server)` : ''}; the rest re-prefill their full context on the next turn.`);
    if (wl.retention === 'none' && wl.users > B) push('info', 'Idle sessions are evicted: every turn re-prefills the whole conversation (cold TTFT applies).');
    if (at && at.pdGpus > 0) push('info', `Prefill/decode disaggregation adds ${at.pdGpus} prefill accelerators per replica.`);

    return {
      hw, model, tp, pp, R, G, total, nodes, gpusUsed, pdTotal, count: p.count, engine: p.engine || 'none', sv, covers,
      effective: { opt, retention: wl.retention, dpAttention: eff.dpAttention }, gated: eff.gated,
      W, denseBytes, expertBytes, overhead, capPerGpu, kvAvailRaw, kvAvail, kvRepl, poolTokens,
      C, S, Scomp, kv, memSessions, maxSessions, slots, slotLimited, maxBatch, fits, memOK, speedOK, B, bPerRep, bAt, sessionsPerRep, residentUsers,
      hostSessions, hostCoverage, idleUsers, bSpeed, bCap, maxConc, maxUsers, maxCtxAtLoad, capLimit, seqPP, computeGpus,
      at, ttft, ttftOK, ttftMax, ttftCold: cold.t, ttftWarm: warm.t + restoreS, restoreS, warmNew, coldNew, reqFlops, cap, pk,
      price, costPerMTok, kW, aggTotal, warnings, tpCross, ppCross, loadable,
      loadAt, decodeStep, solveMaxCtx,
    };
  }

  /* ---------- layout search for a fixed accelerator count ---------- */
  function layouts(model, count, nodeGpus, allowCrossTp, engine) {
    const sv = servingOf(engine);
    const out = [];
    for (const tp of TP_CANDIDATES) {
      if (tp > count || tp > sv.maxTp) break;
      if (model.nHeads % tp !== 0) continue;
      if (tp > nodeGpus && !allowCrossTp) continue;
      for (const pp of PP_CANDIDATES) {
        const G = tp * pp;
        if (G > count || pp > model.layers || (!sv.multiNode && G > nodeGpus)) break;
        out.push({ tp, pp, replicas: Math.floor(count / G) });
      }
    }
    return out;
  }
  function better(a, b) {                       // is layout result a better than b?
    if (!b) return true;
    if (a.fits !== b.fits) return a.fits;
    if (a.maxUsers !== b.maxUsers) return a.maxUsers > b.maxUsers;
    const ua = a.count - a.total, ub = b.count - b.total;
    if (ua !== ub) return ua < ub;
    if (Math.abs(a.ttft - b.ttft) > 1e-3) return a.ttft < b.ttft;
    const sa = a.at ? a.at.perUser : 0, sb = b.at ? b.at.perUser : 0;
    return sa > sb;
  }
  function autoConfig(p) {
    const model = norm(p.model);
    let best = null; const tried = [];
    for (const l of layouts(model, p.count, p.nodeGpus, p.allowCrossTp, p.engine)) {
      const r = evaluate(Object.assign({}, p, l));
      tried.push(r);
      if (better(r, best)) best = r;
    }
    return { best, tried };
  }

  /* smallest layout (tp × pp) on which the model loads with one session of `ctx` tokens (a slot engine: its slots, each of `ctx`) */
  function minGpus(hw, model0, wPrec, kvPrec, ctx, adv0, o = {}) {
    const model = norm(model0), adv = Object.assign({}, DEFAULT_ADV, adv0 || {});
    const engine = o.engine ?? adv.engine, sv = servingOf(engine);
    if (formatSupport(hw, wPrec, engine) === 'unsupported' || kvSupport(hw, kvPrec, engine) !== 'supported') return null;
    const sessions = sv.batching === 'slots' ? (o.slots > 0 ? o.slots : sv.slots.default) : 1;
    const W = weightBytes(model, wPrec, engine), need = sessions * kvAtCtx(model, kvPrec, ctx, engine);
    const nodeGpus = Math.max(hw.nodeGpus, 1), frag = sv.paged ? adv.frag : 0;
    const cands = [];
    for (const tp of TP_CANDIDATES) {
      if (model.nHeads % tp !== 0 || tp > nodeGpus || tp > sv.maxTp) continue;
      for (const pp of PP_CANDIDATES) { if (pp > model.layers || (!sv.multiNode && tp * pp > nodeGpus)) break; cands.push({ G: tp * pp, tp, pp }); }
    }
    cands.sort((a, b) => a.G - b.G || b.tp - a.tp);
    for (const c of cands) {
      const overhead = adv.overheadGB * 1e9 + adv.overheadFrac * (W / c.G);
      const kvRepl = isMla(model) ? (sv.dpAttention ? 1 : c.tp) : Math.max(1, c.tp / model.nKv);
      const avail = (c.G * (hw.mem * 1e9 * adv.util - overhead) - W) * (1 - frag) / kvRepl;
      if (avail >= need) return c;
    }
    return null;
  }

  /* ---------- reverse: smallest hardware for a workload ---------- */
  function betterRev(a, b) {
    if (!b) return true;
    if (a.gpusUsed !== b.gpusUsed) return a.gpusUsed < b.gpusUsed;
    const pa = a.price ?? Infinity, pb = b.price ?? Infinity;
    if (pa !== pb) return pa < pb;
    if (Math.abs(a.ttft - b.ttft) > 1e-3) return a.ttft < b.ttft;
    return a.at.perUser > b.at.perUser;
  }
  const MAX_GPUS = 100000;               // reverse sizing gives up beyond this many accelerators
  /* Why the largest replica count tried for one layout still fails (memory per replica is fine by then). */
  function failReason(r, wl) {
    if (!r.ttftOK) return `Fits, but prefilling ${fmtTok(r.warmNew)} tokens takes ${fmtTime(r.ttft)} on TP ${r.tp}, over the ${fmtTime(r.ttftMax)} limit.`;
    if (r.at && r.at.saturated) return `Fits, but prefill saturates every replica: prompts of ${fmtTok(r.warmNew)} tokens arrive faster than they can be computed.`;
    if (r.at && r.at.perUser < wl.target) return `Fits, but a ${fmtTok(wl.ctx)}-token session cannot reach ${wl.target} tok/s per user (${r.at.perUser.toFixed(1)} tok/s alone).`;
    return `Needs more than ${MAX_GPUS.toLocaleString('en-US')} accelerators.`;
  }
  function reverse(p, hardwareList) {
    const model = norm(p.model), wl = p.wl, eff = effective(p), sv = eff.sv;
    const B = Math.max(1, Math.round(wl.users * wl.activity));
    const demand = eff.retention === 'gpu' ? wl.users : B;   // sessions a replica set must hold
    const results = [];
    for (const hw of hardwareList) {
      const nodeGpus = hw.nodeGpus;
      let best = null, reason = sv.multiNode ? 'Model does not fit on any layout up to 512 accelerators per replica.'
        : `Model does not fit in one ${nodeGpus}-accelerator node, and ${(engineOf(p.engine) || {}).name} serves a model within one node.`;
      const archName = (ARCHS_[hw.arch] || {}).name || 'unknown generation';
      if (engineCovers(hw, p.engine) === 'no') { results.push({ hw, infeasible: true, reason: `${ENGINES_[p.engine].name} does not run on ${archName}; per its docs it runs on ${familyNames(ENGINES_[p.engine].families)}.` }); continue; }
      if (formatSupport(hw, p.wPrec, p.engine) === 'unsupported') { results.push({ hw, infeasible: true, reason: `${PREC_LABEL[p.wPrec]} weights are not loadable on ${archName}${engineOf(p.engine) ? ' with ' + ENGINES_[p.engine].name : ''}.` }); continue; }
      if (kvSupport(hw, p.kvPrec, p.engine) !== 'supported') { results.push({ hw, infeasible: true, reason: `${ENGINES_[p.engine].name} has no ${KV_LABEL[p.kvPrec]} KV cache on this accelerator.` }); continue; }
      for (const tp of TP_CANDIDATES) {
        if (tp > sv.maxTp) break;
        if (model.nHeads % tp !== 0) continue;
        if (tp > nodeGpus && !p.allowCrossTp) continue;
        for (const pp of PP_CANDIDATES) {
          const G = tp * pp;
          if (pp > model.layers || G > 512 || (!sv.multiNode && G > nodeGpus)) break;
          const at = (R) => evaluate(Object.assign({}, p, { hw, nodeGpus, link: hw.link, tp, pp, replicas: R, count: G * R }));
          const ok = (r) => r.memOK && r.speedOK;
          const r1 = at(1);
          if (!r1.fits) continue;                  // memory per replica does not depend on the replica count
          // Feasibility only improves with more replicas (fewer requests each, more host RAM, less prefill per replica),
          // so search for the smallest feasible R upward from the memory bound.
          let lo = Math.max(0, Math.ceil(demand / r1.maxSessions) - 1), hi = lo + 1;
          if (best && G * hi > best.gpusUsed) continue;   // cannot beat the best layout found so far
          let rHi = hi === 1 ? r1 : at(hi), last = rHi;
          while (!ok(rHi)) {
            lo = hi; hi *= 2;
            if (G * hi > MAX_GPUS || (best && G * lo >= best.gpusUsed)) { rHi = null; break; }
            last = rHi = at(hi);
          }
          if (!rHi) { if (!best) reason = failReason(last, wl); continue; }
          while (hi - lo > 1) { const mid = Math.floor((lo + hi) / 2); const rm = at(mid); if (ok(rm)) { hi = mid; rHi = rm; } else lo = mid; }
          if (betterRev(rHi, best)) best = rHi;
        }
      }
      results.push(best ? Object.assign(best, { infeasible: false }) : { hw, infeasible: true, reason });
    }
    results.sort((a, b) => {
      if (a.infeasible !== b.infeasible) return a.infeasible ? 1 : -1;
      if (a.infeasible) return 0;
      if (a.gpusUsed !== b.gpusUsed) return a.gpusUsed - b.gpusUsed;
      return (a.price ?? Infinity) - (b.price ?? Infinity);
    });
    return results;
  }

  /* ---------- formatting helpers shared with the UI ---------- */
  function fmtTok(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(n % 1e6 === 0 ? 0 : 1).replace(/\.0$/, '') + 'M';
    if (n >= 1000) return (n / 1024 >= 1 && n % 1024 === 0) ? (n / 1024) + 'k' : (n / 1000).toFixed(n % 1000 === 0 ? 0 : 1).replace(/\.0$/, '') + 'k';
    return String(Math.round(n));
  }
  function fmtGB(b) {
    if (b >= 1e12) return (b / 1e12).toFixed(2) + ' TB';
    if (b >= 1e9) return (b / 1e9).toFixed(b >= 1e11 ? 0 : 1) + ' GB';
    if (b >= 1e6) return (b / 1e6).toFixed(0) + ' MB';
    if (b >= 1e3) return (b / 1e3).toFixed(0) + ' KB';
    return Math.round(b) + ' B';
  }
  function fmtTime(s) {
    if (!isFinite(s)) return '∞';
    if (s < 1e-3) return (s * 1e6).toFixed(0) + ' µs';
    if (s < 1) return (s * 1e3).toFixed(s < 0.1 ? 1 : 0) + ' ms';
    if (s < 60) return s.toFixed(s < 10 ? 1 : 0) + ' s';
    if (s < 3600) return (s / 60).toFixed(1) + ' min';
    return (s / 3600).toFixed(1) + ' h';
  }

  return { BYTES_W, BYTES_KV, W_PRECS, KV_PRECS, PREC_LABEL, KV_LABEL, DEFAULT_ADV, TP_CANDIDATES, PP_CANDIDATES, MAX_GPUS, SERVING_FULL,
    norm, isMla, weightBytes, expertSplit, touched, kvSplit, kvAtCtx, kvPerTokenFull, attnFlopsPerTok, prefillAttnFlops, peakFlops,
    formatSupport, hwSupport, kvSupport, familyOf, familyNames, engineCovers, servingOf, effective, bytesW, bytesKv, precLabel, kvLabel, maxBatchOf,
    evaluate, autoConfig, layouts, minGpus, reverse, fmtTok, fmtGB, fmtTime };
})();

if (typeof module !== 'undefined') module.exports = Engine;
