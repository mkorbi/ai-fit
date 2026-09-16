#!/usr/bin/env node
/* check-sources.js — verify that every accelerator's source page still resolves and still mentions its key numbers.
 *   node tools/check-sources.js [ids...]
 * The number check is a heuristic text search (pages are marketing HTML), so "not found" means "look", not "wrong".
 */
const path = require('path');
const { HARDWARE } = require(path.join(__dirname, '..', 'catalog.js'));
const UA = { 'User-Agent': 'Mozilla/5.0 (Macintosh) ai-fit-source-check' };
const cache = new Map();
async function pageText(url) {
  if (cache.has(url)) return cache.get(url);
  const p = (async () => {
    const res = await fetch(url, { headers: UA, redirect: 'follow', signal: AbortSignal.timeout(30000) });
    const html = await res.text();
    const txt = html.replace(/<script[\s\S]*?<\/script>/g, ' ').replace(/<style[\s\S]*?<\/style>/g, ' ').replace(/<[^>]+>/g, ' ').replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/\s+/g, ' ');
    return { status: res.status, txt };
  })();
  cache.set(url, p);
  return p;
}
const fmt = (n) => n.toLocaleString('en-US');
function probes(hw) {
  const out = [];
  out.push(['memory', [new RegExp(`\\b${hw.mem}\\s?GB\\b`, 'i')]]);
  const tb = hw.bw / 1000;
  out.push(['bandwidth', [new RegExp(`\\b${fmt(hw.bw)}\\s?GB/s`, 'i'), new RegExp(`\\b${hw.bw}\\s?GB/s`, 'i'), new RegExp(`\\b${tb.toFixed(2).replace(/\\.?0+$/, '')}\\s?TB/s`, 'i'), new RegExp(`\\b${tb.toFixed(1).replace(/\\.0$/, '')}\\s?TB/s`, 'i')]]);
  const t = hw.tflops || {};
  if (t.fp16) out.push(['bf16 tflops (dense or 2× sparse)', [new RegExp(`\\b${fmt(t.fp16)}\\b`), new RegExp(`\\b${fmt(t.fp16 * 2)}\\b`)]]);
  if (t.fp8) out.push(['fp8 tflops (dense or 2× sparse)', [new RegExp(`\\b${fmt(t.fp8)}\\b`), new RegExp(`\\b${fmt(t.fp8 * 2)}\\b`)]]);
  if (t.fp4) out.push(['fp4 tflops (dense or 2× sparse)', [new RegExp(`\\b${fmt(t.fp4)}\\b`), new RegExp(`\\b${fmt(t.fp4 * 2)}\\b`), new RegExp(`\\b${(t.fp4 / 1000).toFixed(1).replace(/\\.0$/, '')}\\s?PFLOPS`, 'i'), new RegExp(`\\b${(t.fp4 * 2 / 1000).toFixed(1).replace(/\\.0$/, '')}\\s?PFLOPS`, 'i')]]);
  if (hw.tdp) out.push(['tdp', [new RegExp(`\\b${fmt(hw.tdp)}\\s?W\\b`)]]);
  // board and rack pages quote aggregates: 8-GPU HGX boards and 72-GPU NVL racks
  for (const n of [8, 72]) {
    const memT = (hw.mem * n / 1000);
    const aggMem = [new RegExp(`\\b${memT.toFixed(1).replace(/\\.0$/, '')}\\s?TB\\b`, 'i'), new RegExp(`\\b${fmt(hw.mem * n)}\\s?GB\\b`, 'i')];
    const res = [];
    if (t.fp4) res.push(new RegExp(`\\b${fmt(Math.round(t.fp4 * n / 1000))}\\s?PFLOPS`, 'i'), new RegExp(`\\b${fmt(Math.round(t.fp4 * 2 * n / 1000))}\\s?PFLOPS`, 'i'), new RegExp(`\\b${(t.fp4 * 2 * n / 1e6).toFixed(1).replace(/\\.0$/, '')}\\s?EFLOPS`, 'i'));
    if (t.fp8) res.push(new RegExp(`\\b${fmt(Math.round(t.fp8 * n / 1000))}\\s?PFLOPS`, 'i'), new RegExp(`\\b${fmt(Math.round(t.fp8 * 2 * n / 1000))}\\s?PFLOPS`, 'i'));
    out.push([`${n}-GPU aggregate memory`, aggMem]);
    if (res.length) out.push([`${n}-GPU aggregate compute`, res]);
  }
  return out;
}
(async () => {
  const ids = process.argv.slice(2);
  let broken = 0;
  for (const hw of HARDWARE.filter((h) => h.source && (!ids.length || ids.includes(h.id)))) {
    let page;
    try { page = await pageText(hw.source.url); } catch (e) { broken++; console.log(`BROKEN  ${hw.id.padEnd(14)} ${hw.source.url}  (${e.message})`); continue; }
    if (page.status >= 400) { broken++; console.log(`BROKEN  ${hw.id.padEnd(14)} ${hw.source.url}  (HTTP ${page.status})`); continue; }
    const found = [], missing = [];
    for (const [name, res] of probes(hw)) (res.some((r) => r.test(page.txt)) ? found : missing).push(name);
    console.log(`${page.status}     ${hw.id.padEnd(14)} ${hw.source.kind.padEnd(12)} found: ${found.join(', ') || '-'}${missing.length ? '   | not found: ' + missing.join(', ') : ''}`);
    if (hw.priceSource) { try { const pp = await pageText(hw.priceSource); if (pp.status >= 400) { broken++; console.log(`BROKEN  ${hw.id.padEnd(14)} price page ${hw.priceSource} (HTTP ${pp.status})`); } } catch (e) { broken++; console.log(`BROKEN  ${hw.id.padEnd(14)} price page ${hw.priceSource} (${e.message})`); } }
  }
  console.log(`\n${broken} broken source links.`);
  process.exit(broken ? 1 : 0);
})();
