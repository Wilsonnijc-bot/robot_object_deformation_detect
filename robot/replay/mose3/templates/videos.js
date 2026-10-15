/* Gallery of every video: one view at a time, twelve per page, as on the MoSE3 project page. */
(() => {
  'use strict';
  const INDEX = window.MOSE3_INDEX;
  const VIEWS = INDEX.views.map(v => [v.key, v.label, v.note]);
  const GROUPS = [['all', 'All'], ['COSMOS2.5', 'COSMOS2.5'], ['COSMOS3', 'COSMOS3'], ['LVP_ROBOWM', 'LVP']];
  const METRICS = [['rigid_fit_residual', 'Rigid fit'], ['se3_rotation_spread_deg', 'SE(3) spread °'],
                   ['pair_ratio_mad', 'Pair MAD'], ['embedding_dispersion', 'Emb. spread']];
  const PER = 12;
  const $ = s => document.querySelector(s);
  const cases = INDEX.cases.filter(c => c.status === 'complete');
  const state = {view: INDEX.default_view, group: 'all', sort: 'case', subset: 'interior', page: 0};
  const generator = c => c.case.slice(0, c.case.lastIndexOf('_'));
  const number = c => c.case.slice(c.case.lastIndexOf('_') + 1);
  const label = c => `${generator(c).replace('_ROBOWM', '')} ${number(c)}`;
  const flag = c => (c.masks.link5 && c.masks.link5 !== 'latest_guarded_mask') ? c.masks.link5 : '';
  // Short chip text; the full review label stays in the tooltip.
  const short = f => f.startsWith('User rejected') ? 'mask rejected' : f.startsWith('Excluded') ? 'excluded' : f;
  const score = (c, link, metric) => (state.subset === 'interior' ? c.interior_scores : c.scores)[link][metric];
  const fmt = v => v === null || v === undefined ? '—' : (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3));
  const href = c => `video.html?case=${encodeURIComponent(c.case)}`;

  /* ---- facts ---- */
  const frames = cases.reduce((a, c) => a + c.frames, 0);
  $('#facts').innerHTML = [
    `<b>${cases.length}</b> videos`, `<b>${frames.toLocaleString()}</b> frames`,
    `<b>256</b> Link2 + <b>100</b> Link5 points each`, `<b>${cases.filter(flag).length}</b> flagged Link5 masks`,
  ].map(t => `<li>${t}</li>`).join('');

  /* ---- controls ---- */
  const seg = (el, items, key) => {
    el.innerHTML = items.map(([v, l]) => `<button data-v="${v}" aria-pressed="${state[key] === v}">${l}</button>`).join('');
    el.addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b || b.disabled) return;
      state[key] = b.dataset.v; if (key !== 'view') state.page = 0;
      [...el.children].forEach(x => x.setAttribute('aria-pressed', String(x === b)));
      render();
    });
  };
  seg($('#view'), VIEWS, 'view');
  seg($('#group'), GROUPS, 'group');
  // A view no video has yet (the display-only rigidity pass) stays visible but disabled.
  [...$('#view').children].forEach(b => { b.disabled = !cases.some(c => c.views.includes(b.dataset.v)); });
  $('#sort').innerHTML = `<option value="case">Video</option>` + ['link2', 'link5'].flatMap(link =>
    METRICS.slice(0, 2).map(([m, l]) => `<option value="${link}:${m}">${link} ${l} ↓</option>`)).join('')
    + `<option value="link5:visible">link5 visible, lowest</option>`;
  $('#sort').addEventListener('change', e => { state.sort = e.target.value; state.page = 0; render(); });
  $('#subset').addEventListener('change', e => { state.subset = e.target.value; render(); table(); });

  function ordered() {
    let list = cases.filter(c => state.group === 'all' || generator(c) === state.group);
    if (state.sort === 'link5:visible') return list.sort((a, b) => a.visible_last.link5 - b.visible_last.link5);
    if (state.sort !== 'case') {
      const [link, metric] = state.sort.split(':');
      return list.sort((a, b) => (score(b, link, metric) ?? -1) - (score(a, link, metric) ?? -1));
    }
    return list;
  }

  /* ---- grid: only cells on screen decode video ---- */
  const grid = $('#rgrid');
  const io = new IntersectionObserver(entries => entries.forEach(e => {
    const v = e.target.querySelector('video'); if (!v) return;
    if (e.isIntersecting) v.play().catch(() => {}); else v.pause();
  }), {rootMargin: '120px'});
  function render() {
    const note = VIEWS.find(v => v[0] === state.view);
    $('#vnote').textContent = note ? note[2] : '';
    const list = ordered(), pages = Math.max(1, Math.ceil(list.length / PER));
    state.page = Math.min(state.page, pages - 1);
    const metric = state.sort.includes(':') && state.sort !== 'link5:visible' ? state.sort.split(':')[1] : 'rigid_fit_residual';
    const top = Math.max(...cases.flatMap(c => ['link2', 'link5'].map(l => score(c, l, metric) ?? 0)), 1e-9);
    const metricLabel = METRICS.find(m => m[0] === metric)[1];
    grid.innerHTML = list.slice(state.page * PER, state.page * PER + PER).map(c => {
      const has = c.views.includes(state.view);
      const bars = ['link2', 'link5'].map(l => {
        const v = score(c, l, metric);
        return `<span>${l}</span><span class="bar ${l}"><i style="width:${v === null ? 0 : 100 * v / top}%"></i></span><span>${fmt(v)}</span>`;
      }).join('');
      return `<a class="rcell" href="${href(c)}" title="Open the 3D replay of ${c.case}"><div class="clip">`
        + (has ? `<video src="views/${c.case}/${state.view}.mp4" poster="views/${c.case}/poster.jpg" muted loop playsinline preload="metadata"></video>`
               : `<img src="views/${c.case}/poster.jpg" alt=""><span class="na">view not rendered</span>`)
        + `</div><figcaption><span class="title">${label(c)}${flag(c) ? `<span class="flag" title="Link5 mask: ${flag(c)}">Link5 ${short(flag(c))}</span>` : ''}</span>`
        + `<span class="bars" title="${metricLabel}">${bars}</span></figcaption></a>`;
    }).join('');
    grid.querySelectorAll('.rcell').forEach(cell => io.observe(cell));
    $('#page').textContent = `${state.page + 1} / ${pages}`;
    $('#prev').parentElement.style.display = pages > 1 ? '' : 'none';
  }
  $('#prev').onclick = () => { const p = Math.ceil(ordered().length / PER); state.page = (state.page - 1 + p) % p; render(); };
  $('#next').onclick = () => { const p = Math.ceil(ordered().length / PER); state.page = (state.page + 1) % p; render(); };

  /* ---- sortable table ---- */
  const columns = [
    {key: 'case', label: 'Video', get: c => c.case, cell: c => `<a href="${href(c)}">${c.case}</a>`},
    {key: 'frames', label: 'Frames', get: c => c.frames},
    {key: 'mask', label: 'Link5 mask', get: c => flag(c), cell: c => flag(c) ? `<span class="flag" title="${flag(c)}">${short(flag(c))}</span>` : ''},
    ...['link2', 'link5'].flatMap(link => [
      ...METRICS.map(([m, l]) => ({key: `${link}:${m}`, group: link, label: l, get: c => score(c, link, m),
                                   cell: c => fmt(score(c, link, m))})),
      {key: `${link}:visible`, group: link, label: 'Visible', get: c => c.visible_last[link],
       cell: c => `${(100 * c.visible_last[link]).toFixed(0)}%`},
    ]),
  ];
  const sorting = {key: 'case', dir: 1};
  function table() {
    const rows = [...cases].sort((a, b) => {
      const col = columns.find(c => c.key === sorting.key), x = col.get(a), y = col.get(b);
      return (x === y ? 0 : (x ?? -Infinity) > (y ?? -Infinity) ? 1 : -1) * sorting.dir;
    });
    const groups = `<tr class="group"><th colspan="3"></th>` + ['link2', 'link5'].map(l =>
      `<th colspan="${METRICS.length + 1}" style="text-align:center"><span class="swatch" style="background:var(--${l})"></span>${l}</th>`).join('') + '</tr>';
    $('#scores').innerHTML = `<thead>${groups}<tr>` + columns.map(c =>
      `<th class="sortable" data-k="${c.key}" aria-sort="${sorting.key === c.key ? (sorting.dir > 0 ? 'ascending' : 'descending') : 'none'}">${c.label}</th>`).join('')
      + `</tr></thead><tbody>` + rows.map(r => '<tr>' + columns.map(c => `<td>${c.cell ? c.cell(r) : c.get(r)}</td>`).join('') + '</tr>').join('') + '</tbody>';
  }
  $('#scores').addEventListener('click', e => {
    const th = e.target.closest('th.sortable'); if (!th) return;
    sorting.dir = sorting.key === th.dataset.k ? -sorting.dir : (th.dataset.k === 'case' ? 1 : -1);
    sorting.key = th.dataset.k;
    table();
  });
  render();
  table();
})();
