#!/usr/bin/env node
/* import-model.js — derive or verify catalog model entries from Hugging Face config.json.
 *
 *   node tools/import-model.js Qwen/Qwen3-32B              print a catalog entry for a repo
 *   node tools/import-model.js --check [id ...]            compare catalog entries (those with an `hf` field) with upstream
 *
 * Reads HF_TOKEN from the environment for gated repos. Entries may name an ungated `hfMirror`
 * (a repo that republishes the same config.json); the report says when a mirror was used.
 * Needs node 18+ (global fetch). No dependencies.
 */
const path = require('path');
const { MODELS } = require(path.join(__dirname, '..', 'catalog.js'));
const E = require(path.join(__dirname, '..', 'engine.js'));
const TOKEN = process.env.HF_TOKEN || '';
const HF = 'https://huggingface.co';

async function get(url, json = true) {
  const res = await fetch(url, { headers: TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {}, signal: AbortSignal.timeout(30000) });
  if (!res.ok) { const err = new Error(`${res.status} ${res.statusText} for ${url}`); err.status = res.status; throw err; }
  return json ? res.json() : res.text();
}
async function fetchRepo(repo) {
  const [config, info] = await Promise.all([get(`${HF}/${repo}/resolve/main/config.json`), get(`${HF}/api/models/${repo}`).catch(() => null)]);
  return { repo, config, info };
}
async function fetchWithMirror(entry) {
  try { return Object.assign(await fetchRepo(entry.hf), { mirror: false }); }
  catch (e) {
    if ((e.status === 401 || e.status === 403) && entry.hfMirror) {
      try { return Object.assign(await fetchRepo(entry.hfMirror), { mirror: true }); } catch (e2) { throw new Error(`${entry.hf}: ${e.message}; mirror ${entry.hfMirror}: ${e2.message}`); }
    }
    throw e;
  }
}

/* ---------- config.json → catalog shape ---------- */
function textConfig(cfg) { return cfg.text_config ? Object.assign({}, cfg, cfg.text_config) : cfg; }
function mapConfig(cfg0) {
  const c = textConfig(cfg0);
  const layers = c.num_hidden_layers, dModel = c.hidden_size, nHeads = c.num_attention_heads;
  const nKv = c.num_key_value_heads ?? nHeads;
  const mla = !!c.kv_lora_rank;
  const dHead = c.head_dim ?? (mla ? c.qk_nope_head_dim + c.qk_rope_head_dim : Math.floor(dModel / nHeads));
  const out = { modelType: c.model_type, layers, dModel, nHeads, nKv, dHead, maxCtx: c.max_position_embeddings };
  const rs = c.rope_scaling || c.rope_parameters;
  const rtype = rs && (rs.rope_type || rs.type);
  if (rs && rs.original_max_position_embeddings && ['yarn', 'longrope', 'dynamic'].includes(rtype)) out.nativeCtx = rs.original_max_position_embeddings;
  const E_ = c.n_routed_experts ?? c.num_experts ?? c.num_local_experts;
  const k = c.num_experts_per_tok ?? c.experts_per_token ?? c.num_experts_per_token;
  if (E_ > 1 && k) out.moe = { experts: E_, active: k };
  // attention layer groups
  let types = null;
  if (Array.isArray(c.layer_types)) types = c.layer_types;
  else if (Array.isArray(c.attn_type_list)) types = c.attn_type_list.map((t) => (t === 1 ? 'full_attention' : 'linear_attention'));
  else if (typeof c.hybrid_override_pattern === 'string') types = [...c.hybrid_override_pattern].filter((ch) => ch !== '-').map((ch) => (ch === '*' ? 'full_attention' : 'linear_attention'));
  const attn = [];
  const push = (t, n) => { if (n > 0) attn.push(Object.assign({ n }, t)); };
  if (mla) push({ type: 'mla', dc: c.kv_lora_rank, dr: c.qk_rope_head_dim }, layers);
  else if (types) {
    const counts = {};
    for (const t of types) counts[t] = (counts[t] || 0) + 1;
    for (const [t, n] of Object.entries(counts)) {
      if (t === 'full_attention') push({ type: 'full' }, n);
      else if (t === 'sliding_attention') push({ type: 'swa', window: c.sliding_window }, n);
      else if (t === 'chunked_attention') push({ type: 'chunked', window: c.attention_chunk_size }, n);
      else if (t === 'linear_attention' || t === 'mamba') push({ type: 'linear' }, n);
      else push({ type: 'full', unknownType: t }, n);
    }
  } else if (c.full_attention_interval > 1) {                                        // Qwen3-Next: one full-attention layer every N, the rest linear
    const global = Math.floor(layers / c.full_attention_interval);
    push({ type: 'linear' }, layers - global); push({ type: 'full' }, global);
  } else if (Array.isArray(c.no_rope_layers) && c.attention_chunk_size) {           // Llama 4 without layer_types
    const global = c.no_rope_layers.filter((x) => x === 0).length;
    push({ type: 'chunked', window: c.attention_chunk_size }, layers - global); push({ type: 'full' }, global);
  } else if (c.sliding_window && c.sliding_window_pattern) {                          // Gemma 2/3, Cohere 2 (older configs)
    const global = Math.floor(layers / c.sliding_window_pattern);
    push({ type: 'swa', window: c.sliding_window }, layers - global); push({ type: 'full' }, global);
  } else if (c.sliding_window && c.use_sliding_window !== false && !/^qwen/.test(c.model_type || '')) {
    push({ type: 'swa', window: c.sliding_window }, layers);                          // Mistral 7B v0.1 style: every layer
  } else push({ type: 'full' }, layers);
  if (c.index_topk) out.sparse = { topk: c.index_topk };
  if (c.linear_num_value_heads) out.linear = { nHeads: c.linear_num_value_heads, dHead: c.linear_value_head_dim || c.linear_key_head_dim };
  out.attn = attn.sort((a, b) => (a.type === 'full' ? 1 : 0) - (b.type === 'full' ? 1 : 0));
  out.dtype = c.torch_dtype || c.dtype || null;
  if (c.quantization_config) {
    const q = c.quantization_config;
    out.quant = [q.quant_method, q.fmt, q.bits ? q.bits + 'bit' : null, q.weight_block_size ? 'block ' + q.weight_block_size.join('×') : null, q.activation_scheme].filter(Boolean).join(' ');
  }
  out.moeShape = moeShape(c);
  return out;
}
function moeShape(c) {
  const E_ = c.n_routed_experts ?? c.num_experts ?? c.num_local_experts;
  const k = c.num_experts_per_tok ?? c.experts_per_token ?? c.num_experts_per_token;
  if (!(E_ > 1 && k)) return null;
  const layers = c.num_hidden_layers;
  let moeLayers = layers;
  if (Array.isArray(c.moe_layers)) moeLayers = c.moe_layers.length;
  else {
    if (c.first_k_dense_replace) moeLayers -= c.first_k_dense_replace;
    if (c.interleave_moe_layer_step > 1) moeLayers = Math.floor(layers / c.interleave_moe_layer_step);
    if (c.decoder_sparse_step > 1) moeLayers = Math.floor(layers / c.decoder_sparse_step);
    if (Array.isArray(c.mlp_only_layers)) moeLayers -= c.mlp_only_layers.length;
  }
  const inter = c.moe_intermediate_size ?? c.intermediate_size;
  return { experts: E_, active: k, moeLayers, perExpert: 3 * c.hidden_size * inter };
}
function nativePrecision(info, mapped) {
  const params = info && info.safetensors && info.safetensors.parameters;
  if (mapped.quant) {
    if (/mxfp4/i.test(mapped.quant)) return 'fp4 (MXFP4)';
    if (/nvfp4|fp4/i.test(mapped.quant)) return 'fp4 (NVFP4)';
    if (/fp8/i.test(mapped.quant)) return 'fp8';
    if (/4bit|w4|awq|gptq/i.test(mapped.quant)) return 'int4';
    if (/8bit|w8|int8/i.test(mapped.quant)) return 'int8';
    return mapped.quant;
  }
  if (params) {
    const top = Object.entries(params).sort((a, b) => b[1] - a[1])[0][0];
    return { BF16: 'bf16', F16: 'fp16', F32: 'fp32', F8_E4M3: 'fp8', F8_E5M2: 'fp8' }[top] || top.toLowerCase();
  }
  return mapped.dtype ? mapped.dtype.replace('float16', 'fp16').replace('bfloat16', 'bf16') : null;
}
function mtpParams(c0) {                       // parameters of multi-token-prediction layers stored in the checkpoint but not loaded for plain serving
  const c = textConfig(c0); const n = c.num_nextn_predict_layers || 0; if (!n) return 0;
  const shape = moeShape(c); const inter = c.intermediate_size || 0;
  const dense = 4 * c.hidden_size * c.hidden_size + 3 * c.hidden_size * inter;
  return n * (shape ? shape.experts * shape.perExpert + 2 * c.hidden_size * c.hidden_size : dense);
}
function totalParams(info, cfg) { const t = info && info.safetensors && info.safetensors.total; return t ? t - (cfg ? mtpParams(cfg) : 0) : null; }

/* ---------- catalog entry from upstream ---------- */
function toEntry(repo, up, info, cfg) {
  const total = totalParams(info, cfg);
  const shape = up.moeShape;
  const active = total && shape ? total - (shape.experts - shape.active) * shape.moeLayers * shape.perExpert : total;
  const attn = up.attn.map((l) => { const o = Object.assign({}, l); if (o.type === 'linear' && up.linear) Object.assign(o, up.linear); return o; });
  const entry = {
    id: repo.split('/')[1].toLowerCase(), family: repo.split('/')[0], name: repo.split('/')[1], hf: repo,
    params: total ? +(total / 1e9).toFixed(1) : null, active: active && shape ? +(active / 1e9).toFixed(1) : undefined,
    layers: up.layers, dModel: up.dModel, nHeads: up.nHeads, nKv: up.nKv, dHead: up.dHead,
    attn: attn.length === 1 && attn[0].type === 'full' ? undefined : attn,
    moe: up.moe, sparse: up.sparse, maxCtx: up.maxCtx, nativeCtx: up.nativeCtx, nativePrec: nativePrecision(info, up),
  };
  for (const k of Object.keys(entry)) if (entry[k] === undefined) delete entry[k];
  return entry;
}

/* ---------- comparison ---------- */
const fmtAttn = (attn) => attn.map((l) => `${l.n}×${l.type}${l.window ? '(' + l.window + ')' : ''}${l.dc ? '(' + l.dc + '+' + l.dr + ')' : ''}`).sort().join(' + ');
function compare(entry, up, info, cfg, mirror) {
  const m = E.norm(entry);
  const rows = [];
  const check = (field, cat, upv, opts = {}) => {
    if (upv == null) { rows.push([field, cat, '(absent upstream)', 'skip']); return; }
    let status;
    if (opts.tol) { const rel = Math.abs(cat - upv) / Math.max(Math.abs(upv), 1e-9); status = rel <= opts.tol ? 'ok' : (opts.soft ? 'info' : 'MISMATCH'); }
    else status = String(cat) === String(upv) ? 'ok' : (opts.soft ? 'info' : 'MISMATCH');
    rows.push([field, cat, upv, status]);
  };
  check('layers', m.layers, up.layers); check('dModel', m.dModel, up.dModel); check('nHeads', m.nHeads, up.nHeads);
  check('nKv', m.nKv, up.nKv, { soft: E.isMla(m) }); check('dHead', m.dHead, up.dHead);
  check('moe.experts', m.moe ? m.moe.experts : '-', up.moe ? up.moe.experts : '-');
  check('moe.active', m.moe ? m.moe.active : '-', up.moe ? up.moe.active : '-');
  check('attention layers', fmtAttn(m.attn), fmtAttn(up.attn));
  check('sparse.topk', m.sparse ? m.sparse.topk : '-', up.sparse ? up.sparse.topk : '-');
  // context: catalog may extend beyond config with RoPE scaling when it documents the native size
  const ctxOk = m.maxCtx === up.maxCtx || (m.maxCtx > up.maxCtx && m.nativeCtx === up.maxCtx) || (up.nativeCtx && m.nativeCtx === up.nativeCtx && m.maxCtx === up.maxCtx);
  rows.push(['maxCtx', `${m.maxCtx}${m.nativeCtx ? ' (native ' + m.nativeCtx + ')' : ''}`, `${up.maxCtx}${up.nativeCtx ? ' (native ' + up.nativeCtx + ')' : ''}`, ctxOk ? 'ok' : 'MISMATCH']);
  const total = totalParams(info, cfg);
  if (total) {
    check('params (B)', m.params, +(total / 1e9).toFixed(2), { tol: 0.03 });   // safetensors total minus MTP layers
    if (up.moeShape) check('active (B)', m.active, +((total - (up.moeShape.experts - up.moeShape.active) * up.moeShape.moeLayers * up.moeShape.perExpert) / 1e9).toFixed(2), { tol: 0.2, soft: true });
  } else rows.push(['params (B)', m.params, '(no safetensors metadata)', 'skip']);
  if (mirror) rows.push(['nativePrec', m.nativePrec || '-', '(mirror repo: not comparable)', 'skip']);
  else check('nativePrec', m.nativePrec || '-', nativePrecision(info, up) || '-', { soft: true });
  return rows;
}

/* ---------- quantized checkpoint discovery ---------- */
const SKIP_RE = /gguf|mlx|bnb|bitsandbytes|exl2|exl3|onnx|openvino|coreml|-ov\b|llamafile|torchao|hqq|eetq|qlora|lora|draft|abliterated|uncensored|heretic|nf4|q4_|q8_|-i1-|imatrix|quip|aqlm|qtip|vptq/i;
const FORMAT_RE = /fp8|int4|int8|awq|gptq|w4a16|w8a8|w8a16|nvfp4|fp4|mxfp4|quantized|autofp8|autoround|auto-round|4bit|8bit/i;
const QUANT_ORGS = ['RedHatAI', 'neuralmagic', 'nvidia', 'hugging-quants', 'casperhansen', 'cpatonn', 'QuantTrio', 'ModelCloud', 'Intel', 'amd', 'modularai', 'cognitivecomputations', 'unsloth', 'TheBloke', 'MaziyarPanahi', 'Kwai-Klear', 'RedHatAI'];
function classifyQuant(cfg, id) {
  const q = (cfg.text_config && cfg.text_config.quantization_config) || cfg.quantization_config;
  if (!q) return null;
  const method = String(q.quant_method || '').toLowerCase();
  const blob = JSON.stringify(q).toLowerCase();
  if (method === 'fp8' || method === 'fbgemm_fp8') return 'fp8';
  if (method === 'mxfp4') return 'fp4';
  if (method.startsWith('modelopt')) return /nvfp4|fp4/.test(blob) ? 'fp4' : 'fp8';
  if (/^(awq|gptq|auto-round|autoround)/.test(method)) return q.bits === 8 ? 'int8' : 'int4';   // awq_marlin, gptq_marlin included
  if (method === 'quark') return /mxfp4|fp4/i.test(id) ? 'fp4' : /fp8/i.test(id) ? 'fp8' : 'quark';
  if (method === 'compressed-tensors') {
    const groups = Object.values(q.config_groups || {});
    const w = groups[0] && groups[0].weights;
    if (!w) return /float-quantized|fp8/.test(blob) ? 'fp8' : null;
    const bits = w.num_bits, type = String(w.type || '').toLowerCase();
    if (type === 'float') return bits === 8 ? 'fp8' : bits === 4 ? 'fp4' : null;
    return bits === 8 ? 'int8' : bits === 4 ? 'int4' : null;
  }
  if (method === 'bitsandbytes') return 'bnb';
  return method || null;
}
function classifyModelOpt(hq) {
  const algo = String((hq && hq.quantization && hq.quantization.quant_algo) || '').toUpperCase();
  if (/NVFP4|FP4/.test(algo)) return 'fp4'; if (/FP8/.test(algo)) return 'fp8'; if (/W4A16|INT4/.test(algo)) return 'int4'; if (/INT8/.test(algo)) return 'int8';
  return algo ? algo.toLowerCase() : null;
}
async function findVariants(entry, limit) {
  const org = entry.hf.split('/')[0], base = entry.hf.split('/')[1];
  const terms = [...new Set([base, base.replace(/-?Instruct.*$/i, ''), base.replace(/^(Meta-)?/, '')])].filter((t) => t.length > 4);
  const seen = new Map();
  for (const q of terms) {
    const list = await get(`${HF}/api/models?search=${encodeURIComponent(q)}&limit=100&sort=downloads&direction=-1`).catch(() => []);
    for (const m of list) seen.set(m.id, m);
  }
  const rank = (m) => (m.id.startsWith(org + '/') ? 0 : QUANT_ORGS.includes(m.id.split('/')[0]) ? 1 : 2);
  const cands = [...seen.values()].filter((m) => m.id !== entry.hf && m.id !== entry.hfMirror && FORMAT_RE.test(m.id) && !SKIP_RE.test(m.id))
    .sort((a, b) => rank(a) - rank(b) || (b.downloads || 0) - (a.downloads || 0)).slice(0, limit);
  const out = [];
  for (const m of cands) {
    try {
      const cfg = await get(`${HF}/${m.id}/resolve/main/config.json`);
      const up = mapConfig(cfg);
      const same = up.layers === entry.layers && up.dModel === entry.dModel;
      let fmt = classifyQuant(cfg, m.id);
      if (!fmt) { try { fmt = classifyModelOpt(await get(`${HF}/${m.id}/resolve/main/hf_quant_config.json`)); } catch (e) { /* not a ModelOpt repo */ } }
      out.push({ id: m.id, downloads: m.downloads || 0, gated: !!m.gated, fmt, same });
    } catch (e) { out.push({ id: m.id, downloads: m.downloads || 0, gated: !!m.gated, fmt: null, same: false, err: e.status || e.message }); }
  }
  return out;
}
async function checkVariants(entries) {
  let bad = 0;
  for (const entry of entries) {
    if (!entry.variants) continue;
    for (const [fmt, repo] of Object.entries(entry.variants)) {
      try {
        const cfg = await get(`${HF}/${repo}/resolve/main/config.json`);
        const up = mapConfig(cfg);
        const same = up.layers === entry.layers && up.dModel === entry.dModel;
        let got = classifyQuant(cfg, repo);
        if (!got) { try { got = classifyModelOpt(await get(`${HF}/${repo}/resolve/main/hf_quant_config.json`)); } catch (e) { /* none */ } }
        if (!got && fmt === 'bf16') got = (cfg.text_config || cfg).torch_dtype === 'bfloat16' ? 'bf16' : null;
        const ok = same && got === fmt;
        if (!ok) bad++;
        console.log(`${ok ? 'ok      ' : 'MISMATCH'} ${entry.id.padEnd(18)} ${fmt.padEnd(5)} ${repo.padEnd(64)} format=${got} shape=${same ? 'same' : 'DIFFERENT'}`);
      } catch (e) { bad++; console.log(`MISSING  ${entry.id.padEnd(18)} ${fmt.padEnd(5)} ${repo.padEnd(64)} ${e.status || e.message}`); }
    }
  }
  return bad;
}

async function main() {
  const args0 = process.argv.slice(2);
  if (args0[0] === '--find-variants') {
    const ids = args0.slice(1);
    for (const entry of MODELS.filter((m) => m.hf && (!ids.length || ids.includes(m.id)))) {
      const found = await findVariants(entry, 14);
      console.log(`\n## ${entry.id}  (${entry.hf})`);
      for (const f of found) console.log(`   ${String(f.fmt).padEnd(6)} ${f.same ? 'same ' : 'DIFF '} ${String(f.downloads).padStart(8)} ${f.gated ? 'gated ' : '      '} ${f.id}${f.err ? '  (' + f.err + ')' : ''}`);
    }
    return;
  }
  if (args0[0] === '--check-variants') {
    const ids = args0.slice(1);
    const bad = await checkVariants(MODELS.filter((m) => !ids.length || ids.includes(m.id)));
    console.log(`\n${bad} problems.`); process.exit(bad ? 1 : 0);
  }
  return mainOrig();
}
async function mainOrig() {
  const args = process.argv.slice(2);
  if (!args.length || args[0] === '--help') { console.log('usage: import-model.js <org/repo> | --check [ids] | --find-variants [ids] | --check-variants [ids]'); process.exit(0); }
  if (args[0] === '--check') {
    const ids = args.slice(1);
    const entries = MODELS.filter((m) => m.hf && (!ids.length || ids.includes(m.id)));
    let bad = 0, skipped = 0;
    for (const entry of entries) {
      let fetched;
      try { fetched = await fetchWithMirror(entry); }
      catch (e) { skipped++; console.log(`\n## ${entry.name} [${entry.id}]  —  SKIPPED: ${e.message}`); continue; }
      const up = mapConfig(fetched.config);
      const rows = compare(entry, up, fetched.info, fetched.config, fetched.mirror);
      const mism = rows.filter((r) => r[3] === 'MISMATCH').length; bad += mism > 0 ? 1 : 0;
      console.log(`\n## ${entry.name} [${entry.id}]  ←  ${fetched.repo}${fetched.mirror ? ' (mirror)' : ''}  (${up.modelType})  ${mism ? mism + ' MISMATCH' : 'ok'}`);
      for (const [f, a, b, s] of rows) if (s !== 'ok' || process.env.VERBOSE) console.log(`   ${s.padEnd(8)} ${f.padEnd(18)} catalog: ${String(a).padEnd(34)} upstream: ${b}`);
    }
    console.log(`\n${entries.length} entries checked, ${bad} with mismatches, ${skipped} skipped.`);
    process.exit(bad ? 1 : 0);
  }
  const repo = args[0];
  const fetched = await fetchRepo(repo);
  const up = mapConfig(fetched.config);
  console.log(JSON.stringify(toEntry(repo, up, fetched.info, fetched.config), null, 2));
}
main().catch((e) => { console.error(e.message || e); process.exit(2); });
