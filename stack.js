/* stack.js — AI Fit Engine Stack: the software around the accelerators, layer by layer, for inference, fine-tuning
 * and training, with the default pick of a vLLM reference stack and what each layer needs beside the GPUs.
 * Content lives in STACK (stack-data.js). */
const StackView = (() => {
  'use strict';
  const D = typeof STACK !== 'undefined' ? STACK : require('./stack-data.js');
  const PURPOSES = [['inference', 'Inference serving'], ['finetune', 'Fine-tuning'], ['training', 'Model training']];
  const NEED = { required: 'required', recommended: 'recommended', optional: 'optional', none: 'not needed' };
  const needChip = (n) => 'chip ' + (n === 'required' ? 'good' : n === 'recommended' ? 'info' : '');
  const fmt = (n, d = 0) => (n == null || !isFinite(n)) ? '—' : n.toLocaleString('en-US', { maximumFractionDigits: d });
  const fmtStore = (gb) => (gb >= 1000 ? fmt(gb / 1000, gb >= 10000 ? 0 : 1) + ' TB' : fmt(gb) + ' GB');

  const el = (tag, attrs, ...kids) => {
    const n = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) { if (v == null || v === false) continue; if (k === 'class') n.className = v; else if (k.startsWith('on')) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v === true ? '' : v); }
    for (const kid of kids.flat(Infinity)) { if (kid == null || kid === false) continue; n.append(kid.nodeType ? kid : document.createTextNode(String(kid))); }
    return n;
  };
  const LS = { get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } }, set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* ignore */ } } };

  /* Resources of every layer the purpose needs (required and recommended), split into services that run beside the
   * accelerators and the share that runs on the GPU servers themselves (their host CPUs, RAM and local disks). */
  function totals(purpose) {
    const zero = () => ({ cpu: 0, ramGB: 0, storageGB: 0, layers: [] });
    const t = { side: zero(), gpu: zero() };
    for (const L of D.layers) {
      const need = (L.need || {})[purpose];
      if (need !== 'required' && need !== 'recommended') continue;
      const r = (L.resources || {})[purpose]; if (!r) continue;
      const b = r.gpu || L.id === 'drivers' ? t.gpu : t.side;   // drivers run on the GPU servers too
      b.cpu += +r.cpu || 0; b.ramGB += +r.ramGB || 0; b.storageGB += +r.storageGB || 0; b.layers.push(L.name);
    }
    return t;
  }

  // the default can name several options ("Keycloak (OIDC) + OpenBao + Kyverno"): highlight each one it mentions
  const isPick = (L, o) => !!L.default && (o.name === L.default || L.default.toLowerCase().includes(o.name.replace(/ \(.*\)$/, '').toLowerCase()));
  let root = null, purpose = 'inference';
  const kpis = (b) => el('div', { class: 'kpis stk-kpis' },
    el('div', { class: 'kpi' }, el('div', { class: 'l' }, 'CPU cores'), el('div', { class: 'v' }, fmt(b.cpu))),
    el('div', { class: 'kpi' }, el('div', { class: 'l' }, 'RAM'), el('div', { class: 'v' }, fmtStore(b.ramGB))),
    el('div', { class: 'kpi' }, el('div', { class: 'l' }, 'Disk'), el('div', { class: 'v' }, fmtStore(b.storageGB))));
  function layerRow(L) {
    const need = (L.need || {})[purpose] || 'optional', r = (L.resources || {})[purpose];
    const res = r && need !== 'none' ? [r.cpu != null ? `${fmt(r.cpu, 1)} CPU` : null, r.ramGB != null ? `${fmt(r.ramGB)} GB RAM` : null, r.storageGB ? `${fmtStore(r.storageGB)} disk` : null, r.gpu ? 'uses GPUs' : null].filter(Boolean).join(' · ') : '';
    return el('li', { class: 'stk-layer need-' + need },
      el('div', { class: 'stk-lhead' },
        el('div', null, el('b', { class: 'stk-lname' }, L.name), el('span', { class: needChip(need) }, NEED[need] || need)),
        el('p', { class: 'stk-lpurpose' }, L.purpose)),
      el('ul', { class: 'stk-opts' }, (L.options || []).map((o) => el('li', { class: 'stk-opt' + (isPick(L, o) ? ' pick' : ''), title: o.oneLine },
        el('a', { href: o.url, target: '_blank', rel: 'noopener' }, o.name), o.license ? el('span', { class: 'stk-lic' }, o.license) : null))),
      need === 'none' ? null : el('p', { class: 'stk-lres' }, res ? el('span', null, el('b', null, r.gpu || L.id === 'drivers' ? 'On the GPU servers: ' : 'Needs: '), res) : null, r && r.note ? el('span', { class: 'stk-note' }, ' ', r.note) : null));
  }
  function render(container) {
    if (!root) { root = el('div', { class: 'stk' }); container.replaceChildren(root); purpose = LS.get('cb.stackPurpose', 'inference'); }
    const P = D.purposes[purpose] || {}, T = totals(purpose), PT = P.totals || {};
    const layers = D.layers.slice().reverse();   // drawn top (gateway) to bottom (drivers)
    root.replaceChildren(
      el('div', { class: 'verdict' }, el('strong', null, 'What runs around the GPUs'),
        el('p', null, 'The layers you install and operate to self-host models, with vLLM as the reference engine. Pick a purpose: layers it does not need fade out, highlighted options make up the reference stack, and the totals add up what the software needs: services beside the GPUs and the share it takes on the GPU servers themselves.')),
      el('div', { class: 'stk-switch', role: 'tablist', 'aria-label': 'Purpose' }, PURPOSES.map(([k, label]) =>
        el('button', { class: 'pxbtn' + (k === purpose ? ' on' : ''), type: 'button', role: 'tab', 'aria-selected': String(k === purpose), onclick: () => { purpose = k; LS.set('cb.stackPurpose', k); render(container); } }, label))),
      el('div', { class: 'stk-grid' },
        el('section', { class: 'panel stk-arch' }, el('h3', null, `Architecture: ${P.title || purpose}`), el('p', { class: 'hint' }, `Reference size: ${P.reference || '—'}. Top: what users call; bottom: what the hardware needs.`),
          el('ol', { class: 'stk-layers' }, layers.map(layerRow))),
        el('aside', { class: 'stk-side' },
          el('section', { class: 'panel' }, el('h3', null, 'Resource demand'),
            el('h4', { class: 'stk-sub' }, 'Services beside the GPUs'), kpis(T.side),
            el('p', { class: 'hint' }, `Control plane, storage, monitoring and the rest: ${T.side.layers.join(', ') || 'none'}.`),
            el('h4', { class: 'stk-sub' }, 'On the GPU servers'), kpis(T.gpu),
            el('p', { class: 'hint' }, `Host CPU, RAM and local disk the engine or framework uses on the GPU servers themselves (${T.gpu.layers.join(', ') || 'none'}); check the servers you buy or rent have it.`),
            el('dl', { class: 'stk-dl' },
              [['Control-plane nodes', PT.controlNodes], ['CPU', PT.cpu], ['RAM', PT.ramGB], ['Storage', PT.storage], ['Network', PT.network], ['GPU memory', P.gpuMemoryRule]].filter(([, v]) => v).map(([k, v]) => [el('dt', null, k), el('dd', null, String(v))])),
            PT.note ? el('p', { class: 'hint' }, PT.note) : null),
          el('section', { class: 'panel' }, el('h3', null, 'Common pitfalls'), el('ul', { class: 'warnings' }, (P.pitfalls || []).map((x) => el('li', null, x)))))),
      el('section', { class: 'panel' }, el('h3', null, 'Which purpose needs which layer'),
        el('div', { class: 'tw' }, el('table', { class: 'data stk-matrix' },
          el('thead', null, el('tr', null, el('th', null, 'Layer'), PURPOSES.map(([, l]) => el('th', null, l)), el('th', null, 'Reference pick'))),
          el('tbody', null, D.layers.map((L) => el('tr', null, el('td', null, el('b', null, L.name)),
            PURPOSES.map(([k]) => { const n = (L.need || {})[k] || 'optional'; return el('td', null, el('span', { class: needChip(n) }, NEED[n] || n)); }),
            el('td', null, L.default || '—'))))))),
      el('details', { class: 'panel stk-sources' }, el('summary', null, el('b', null, 'How the numbers were made, and sources')),
        D.resourceBasis ? el('p', { class: 'hint' }, D.resourceBasis) : null,
        el('ul', null, Object.entries(D.sources || {}).map(([k, u]) => el('li', null, el('a', { href: u, target: '_blank', rel: 'noopener' }, k), ' ', el('span', { class: 'stk-lic' }, u.replace(/^https?:\/\//, '').slice(0, 70)))))),
      el('p', { class: 'hint' }, `Options and resource figures checked ${D.checked}. Resource numbers are planning estimates from vendor sizing guides where they exist; measure on your own cluster.`));
  }
  return { render, totals, data: D };
})();
if (typeof module !== 'undefined') module.exports = StackView;
