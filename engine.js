/* engine.js — capacity model for Context Budget.
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
  const BYTES_W  = { bf16: 2, fp8: 1, int8: 1, int4: 0.52, fp4: 0.53 };   // bytes per parameter incl. group scales
  const BYTES_KV = { bf16: 2, fp8: 1, int8: 1, int4: 0.56 };              // bytes per KV element
  const W_PRECS  = ['bf16', 'fp8', 'int8', 'int4', 'fp4'];
  const KV_PRECS = ['bf16', 'fp8', 'int4'];
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

  function weightBytes(model, prec) {
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
  function kvSplit(model, kvPrec, C, S) {
    const bytes = BYTES_KV[kvPrec];
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
  const kvAtCtx = (model, kvPrec, C) => kvSplit(model, kvPrec, C, 0).perSession;
  const kvPerTokenFull = (model, kvPrec) => model.attn.reduce((a, l) => a + l.n * layerPerTok(l, model, BYTES_KV[kvPrec]), 0);

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
  /* Weight-format support as the weaker of silicon capability and the chosen engine's documented kernels. */
  function formatSupport(hw, wPrec, engine) {
    const cap = hwSupport(hw, wPrec);
    const eng = engine && engine !== 'none' ? ENGINES_[engine] : null;
    if (!eng || wPrec === 'bf16' || cap === 'unsupported') return cap;
    const e = (eng.weights[wPrec] || {})[familyOf(hw)];
    if (!e) return 'unsupported';
    return e === 'weight-only' || cap === 'weight-only' ? 'weight-only' : 'native';
  }
  /* KV-cache dtype support: the silicon always can (it is just storage); the engine must have kernels for it. */
  function kvSupport(hw, kvPrec, engine) {
    if (kvPrec === 'bf16') return 'supported';
    const eng = engine && engine !== 'none' ? ENGINES_[engine] : null;
    if (!eng) return 'supported';
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
    const wl = p.wl, opt = p.opt;
    const tp = p.tp, pp = p.pp, R = p.replicas, G = tp * pp, total = G * R;
    const warnings = [];
    const push = (level, text) => warnings.push({ level, text });

    // memory
    const W = weightBytes(model, p.wPrec);
    const split = expertSplit(model);
    const denseBytes = W * split.dense / (model.params * 1e9), expertBytes = W * split.expert / (model.params * 1e9);
    const overhead = adv.overheadGB * 1e9 + adv.overheadFrac * (W / G);
    const capPerGpu = hw.mem * 1e9 * adv.util - overhead;
    const kvAvailRaw = G * capPerGpu - W;
    const mla = isMla(model);
    const kvRepl = p.dpAttention ? 1 : (mla ? tp : Math.max(1, tp / model.nKv));
    const kvAvail = kvAvailRaw > 0 ? kvAvailRaw * (1 - adv.frag) / kvRepl : 0;
    const C = Math.max(64, wl.ctx);
    const S = opt.prefixCache && wl.prefix > 0 ? Math.min(wl.prefix, C - 1) : 0;
    const kv = kvSplit(model, p.kvPrec, C, S);
    const maxSessions = kvAvail > kv.shared ? Math.floor((kvAvail - kv.shared) / kv.perSession) : 0;
    const pk = peakFlops(hw, p.wPrec, p.engine);
    const kvOK = kvSupport(hw, p.kvPrec, p.engine) === 'supported';
    const loadable = pk.support !== 'unsupported' && kvOK;
    const fits = loadable && kvAvailRaw > 0 && maxSessions >= 1;

    // workload shape
    const B = Math.max(1, Math.round(wl.users * wl.activity));
    const residentUsers = wl.retention === 'gpu' ? wl.users : B;
    const sessionsPerRep = Math.ceil(residentUsers / R);
    const bPerRep = Math.ceil(B / R);
    const memOK = fits && sessionsPerRep <= maxSessions;
    const nodes = Math.ceil(total / p.nodeGpus);
    const idleUsers = Math.max(0, wl.users - B);
    const hostBytes = wl.retention === 'host' ? nodes * p.hostRamGB * 1e9 * 0.8 : 0;
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
      const kvRead = b * kvAtCtx(model, p.kvPrec, Cavg);
      const tBw = (wRead + kvRead / pp) / (tp * hw.bw * 1e9 * adv.bwEff);
      const flopsTok = 2 * model.active * 1e9 + attnFlopsPerTok(model, Cavg);
      const tComp = b * kmul * flopsTok / (G * pk.peak * adv.mfu);
      let tComm = 0;
      if (tp > 1) tComm += collective(tpCross, 2 * (tp - 1) / tp * b * kmul * model.dModel * 2, 2 * model.layers);
      if (pp > 1) tComm += collective(ppCross, b * kmul * model.dModel * 2, pp - 1);
      const bubble = pp > 1 ? 1 + adv.ppBubble * (pp - 1) / pp : 1;
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
    const coldNew = Math.max(1, promptLen - S);
    const cold = prefill(coldNew, S);
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
    const cap = G * pk.peak * adv.mfu;
    const ttftMax = wl.ttftMax > 0 ? wl.ttftMax : Infinity;
    const ttftOK = ttft <= ttftMax;

    function loadAt(b, spec) {
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
    const bCap = wl.retention === 'gpu' ? Math.max(0, Math.floor(maxSessions * wl.activity)) : maxSessions;
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
    const at = fits ? loadAt(Math.min(bPerRep, Math.max(1, maxSessions)), opt.spec) : null;
    const speedOK = !!at && memOK && ttftOK && at.perUser >= wl.target && !at.saturated;

    function solveMaxCtx(sessions) {
      if (!fits || sessions < 1) return 0;
      const fitsAt = (c) => { const s = opt.prefixCache && wl.prefix > 0 ? Math.min(wl.prefix, c - 1) : 0; const k = kvSplit(model, p.kvPrec, c, s); return k.shared + sessions * k.perSession <= kvAvail; };
      if (!fitsAt(64)) return 0;
      let lo = 64, hi = 1 << 26;
      if (fitsAt(hi)) return hi;
      while (hi - lo > 64) { const mid = Math.floor((lo + hi) / 2); if (fitsAt(mid)) lo = mid; else hi = mid; }
      return lo;
    }
    const maxCtxAtLoad = solveMaxCtx(sessionsPerRep);

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
    const engName = p.engine && p.engine !== 'none' && ENGINES_[p.engine] ? ENGINES_[p.engine].name : null;
    if (pk.support === 'unsupported') push('crit', engName && hwSupport(hw, p.wPrec) !== 'unsupported' ? `${engName} has no kernel for ${PREC_LABEL[p.wPrec]} weights on ${(ARCHS_[hw.arch] || {}).name || 'this generation'} (per its docs, checked ${ENGINES_[p.engine].checked}).` : `${PREC_LABEL[p.wPrec]} weights cannot be loaded on ${hw.name} (${(ARCHS_[hw.arch] || {}).name || 'unknown generation'}): no kernel for that format.`);
    if (!kvOK) push('crit', `${engName} does not support a ${KV_LABEL[p.kvPrec]} KV cache on ${(ARCHS_[hw.arch] || {}).name || 'this generation'}; use FP8 or BF16 KV cache.`);
    else if (!fits) push('crit', kvAvailRaw <= 0 ? `Weights alone (${fmtGB(W)}) do not fit in ${G} × ${hw.mem} GB with headroom.` : `Only ${fmtGB(kvAvail)} left for KV cache; one session at ${fmtTok(C)} needs ${fmtGB(kv.perSession)}.`);
    else if (!memOK) push('crit', wl.retention === 'gpu'
      ? `Keeping every user's session in GPU memory needs ${sessionsPerRep} sessions per replica; ${maxSessions} fit.`
      : `${sessionsPerRep} concurrent sessions per replica need ${fmtGB(kv.shared + sessionsPerRep * kv.perSession)} of KV cache; ${fmtGB(kvAvail)} available.`);
    if (fits && !ttftOK) push('crit', `Time to first token of ${fmtTime(ttft)} exceeds the ${fmtTime(ttftMax)} limit: ${fmtTok(warmNew)} tokens must be prefilled on ${tp} accelerator${tp > 1 ? 's' : ''} each turn.`);
    if (at && at.saturated) push('crit', `Prefill saturates the replica: prompts arrive faster than ${fmtTok(reqFlops / (2 * model.active * 1e9))}-token prefills can be computed. Add compute or cache more of the prompt.`);
    if (tpCross) push('warn', p.net.gbps > 0 ? `Tensor parallel spans ${Math.ceil(tp / p.nodeGpus)} nodes; every layer's all-reduce crosses the network.` : 'Tensor parallel spans nodes but no inter-node network is configured.');
    if (ppCross && !tpCross && p.net.gbps === 0) push('crit', 'The layout needs more than one node but no inter-node network is configured.');
    if (kvRepl > 1) push('warn', `KV cache is replicated ${kvRepl}× across tensor-parallel ranks (${mla ? 'MLA has a single latent head' : `${model.nKv} KV heads < TP ${tp}`}). Enable data-parallel attention or lower TP.`);
    if (pk.support === 'weight-only') push('info', `${PREC_LABEL[p.wPrec]} compute is not native on ${hw.name}: quantized weights save memory but matmuls run in BF16.`);
    const nk = nativeKey(model);
    if (nk && BYTES_W[nk] != null && BYTES_W[p.wPrec] > BYTES_W[nk]) push('info', `The checkpoint ships in ${model.nativePrec}; serving it in ${PREC_LABEL[p.wPrec]} upcasts the weights to ${(BYTES_W[p.wPrec] / BYTES_W[nk]).toFixed(1)}× the bytes without a quality gain.`);
    else if (nk && BYTES_W[nk] != null && BYTES_W[p.wPrec] < BYTES_W[nk]) push('info', `Needs a ${PREC_LABEL[p.wPrec]} checkpoint or on-the-fly quantization: the official weights ship in ${model.nativePrec}.`);
    if (p.count != null && p.count - total > 0) push('info', `${p.count - total} of ${p.count} accelerators are idle in this layout.`);
    if (wl.retention === 'host' && hostCoverage < 1) push('warn', `Host memory keeps ${hostSessions} of ${idleUsers} idle sessions warm; the rest re-prefill their full context on the next turn.`);
    if (wl.retention === 'none' && wl.users > B) push('info', 'Idle sessions are evicted: every turn re-prefills the whole conversation (cold TTFT applies).');
    if (at && at.pdGpus > 0) push('info', `Prefill/decode disaggregation adds ${at.pdGpus} prefill accelerators per replica.`);

    return {
      hw, model, tp, pp, R, G, total, nodes, gpusUsed, pdTotal, count: p.count,
      W, denseBytes, expertBytes, overhead, capPerGpu, kvAvailRaw, kvAvail, kvRepl,
      C, S, kv, maxSessions, fits, memOK, speedOK, B, bPerRep, sessionsPerRep, residentUsers,
      hostSessions, hostCoverage, idleUsers, bSpeed, bCap, maxConc, maxUsers, maxCtxAtLoad,
      at, ttft, ttftOK, ttftMax, ttftCold: cold.t, ttftWarm: warm.t + restoreS, restoreS, warmNew, coldNew, reqFlops, cap, pk,
      price, costPerMTok, kW, aggTotal, warnings, tpCross, ppCross, loadable,
      loadAt, decodeStep, solveMaxCtx,
    };
  }

  /* ---------- layout search for a fixed accelerator count ---------- */
  function layouts(model, count, nodeGpus, allowCrossTp) {
    const out = [];
    for (const tp of TP_CANDIDATES) {
      if (tp > count) break;
      if (model.nHeads % tp !== 0) continue;
      if (tp > nodeGpus && !allowCrossTp) continue;
      for (const pp of PP_CANDIDATES) {
        const G = tp * pp;
        if (G > count || pp > model.layers) break;
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
    for (const l of layouts(model, p.count, p.nodeGpus, p.allowCrossTp)) {
      const r = evaluate(Object.assign({}, p, l));
      tried.push(r);
      if (better(r, best)) best = r;
    }
    return { best, tried };
  }

  /* smallest layout (tp × pp) on which the model loads with one session of `ctx` tokens */
  function minGpus(hw, model0, wPrec, kvPrec, ctx, adv0) {
    const model = norm(model0), adv = Object.assign({}, DEFAULT_ADV, adv0 || {});
    if (formatSupport(hw, wPrec, adv.engine) === 'unsupported' || kvSupport(hw, kvPrec, adv.engine) !== 'supported') return null;
    const W = weightBytes(model, wPrec), need = kvAtCtx(model, kvPrec, ctx);
    const cands = [];
    for (const tp of TP_CANDIDATES) {
      if (model.nHeads % tp !== 0 || tp > Math.max(hw.nodeGpus, 1)) continue;
      for (const pp of PP_CANDIDATES) { if (pp > model.layers) break; cands.push({ G: tp * pp, tp, pp }); }
    }
    cands.sort((a, b) => a.G - b.G || b.tp - a.tp);
    for (const c of cands) {
      const overhead = adv.overheadGB * 1e9 + adv.overheadFrac * (W / c.G);
      const kvRepl = isMla(model) ? 1 : Math.max(1, c.tp / model.nKv);
      const avail = (c.G * (hw.mem * 1e9 * adv.util - overhead) - W) * (1 - adv.frag) / kvRepl;
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
  function reverse(p, hardwareList) {
    const model = norm(p.model), wl = p.wl;
    const B = Math.max(1, Math.round(wl.users * wl.activity));
    const results = [];
    for (const hw of hardwareList) {
      const nodeGpus = hw.nodeGpus;
      let best = null, reason = 'Model does not fit on any layout up to 512 accelerators.';
      if (formatSupport(hw, p.wPrec, p.engine) === 'unsupported') { results.push({ hw, infeasible: true, reason: `${PREC_LABEL[p.wPrec]} weights are not loadable on this accelerator (${(ARCHS_[hw.arch] || {}).name || 'unknown generation'})${p.engine && p.engine !== 'none' ? ' with ' + ENGINES_[p.engine].name : ''}.` }); continue; }
      if (kvSupport(hw, p.kvPrec, p.engine) !== 'supported') { results.push({ hw, infeasible: true, reason: `${ENGINES_[p.engine].name} has no ${KV_LABEL[p.kvPrec]} KV cache on this accelerator.` }); continue; }
      for (const tp of TP_CANDIDATES) {
        if (model.nHeads % tp !== 0) continue;
        if (tp > nodeGpus && !p.allowCrossTp) continue;
        for (const pp of PP_CANDIDATES) {
          const G = tp * pp;
          if (pp > model.layers || G > 512) break;
          const r1 = evaluate(Object.assign({}, p, { hw, nodeGpus, link: hw.link, tp, pp, replicas: 1, count: G }));
          if (!r1.fits) continue;
          if (!r1.ttftOK) { reason = `Fits, but prefilling ${fmtTok(r1.warmNew)} tokens takes ${fmtTime(r1.ttft)} on TP ${tp}, over the ${fmtTime(r1.ttftMax)} limit.`; continue; }
          let R;
          if (wl.retention === 'gpu') {
            const perRep = Math.min(r1.maxSessions, Math.floor(r1.bSpeed / wl.activity));
            if (perRep < 1) { reason = `Fits, but ${fmtTok(wl.ctx)} sessions cannot reach ${wl.target} tok/s per user.`; continue; }
            R = Math.ceil(wl.users / perRep);
          } else {
            const perRep = Math.min(r1.maxSessions, r1.bSpeed);
            if (perRep < 1) { reason = `Fits, but a ${fmtTok(wl.ctx)}-token session cannot reach ${wl.target} tok/s per user (${r1.at ? r1.at.perUser.toFixed(1) : '?'} tok/s alone).`; continue; }
            R = Math.ceil(B / perRep);
          }
          const r = evaluate(Object.assign({}, p, { hw, nodeGpus, link: hw.link, tp, pp, replicas: R, count: G * R }));
          if (!(r.memOK && r.speedOK)) continue;
          if (betterRev(r, best)) best = r;
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

  return { BYTES_W, BYTES_KV, W_PRECS, KV_PRECS, PREC_LABEL, KV_LABEL, DEFAULT_ADV, TP_CANDIDATES, PP_CANDIDATES,
    norm, isMla, weightBytes, expertSplit, touched, kvSplit, kvAtCtx, kvPerTokenFull, attnFlopsPerTok, prefillAttnFlops, peakFlops,
    formatSupport, hwSupport, kvSupport, familyOf, evaluate, autoConfig, layouts, minGpus, reverse, fmtTok, fmtGB, fmtTime };
})();

if (typeof module !== 'undefined') module.exports = Engine;
