/* Reg2Inv frame-0 error replay (index.html?case=<id>&view=aligned|camera), adapted from the MoSE3
   error-signal replay: every observed point coloured by its own Reg2Inv anomaly score at the current
   frame, drawn over the frame-0 normal reference, with the per-frame score below and every video indexed. */
(() => {
  'use strict';
  const M = window.MOSE3_3D, INDEX = window.REG2INV_INDEX;
  // White (normal) to red (more error), as the MoSE3 replay; the red end stays visible on black.
  const RAMP = ['#fdf9f9', '#ffdfda', '#ffc7bf', '#ffafa5', '#ff988c', '#fa8074', '#f3665c', '#ea4b43', '#e02628'];
  const REFERENCE = [0.30, 0.50, 0.86], MARKER = [0.31, 0.82, 0.91], UNSCORED = [0.24, 0.255, 0.28];
  const MISSING = 65535, LINKS = INDEX.links;
  const LABEL = {1: 'severe', 0.5: 'moderate', 0: 'none'};
  const VIEWS = {aligned: 'Each frame is moved onto frame 0 with the rigid transform Reg2Inv registered for that link, so '
                          + 'red marks where the link no longer matches its own frame-0 shape.',
                 camera: "Points where the camera saw them at this frame. The blue reference stays at frame 0's position, "
                         + 'so motion of the link and the camera both show.'};
  const $ = s => document.querySelector(s);
  const viewer = $('#viewer'), canvas = $('#scene'), video = $('#video'), status = $('#status');
  const params = new URLSearchParams(location.search);
  const state = {view: params.get('view') === 'camera' ? 'camera' : 'aligned',
                 name: INDEX.cases.some(c => c.case === params.get('case')) ? params.get('case') : INDEX.cases[0].case,
                 selected: null, sort: {key: 'case', dir: 1}};
  const fmt = v => v === null || v === undefined || !Number.isFinite(v) ? '—' : (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3));

  /* ---- colour: one red hue, white (normal) to red (large error) ---- */
  const rampRgb = RAMP.map(h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255));
  function ramp(fraction, out, offset) {
    const x = Math.min(Math.max(fraction, 0), 1) * (rampRgb.length - 1), i = Math.min(Math.floor(x), rampRgb.length - 2), f = x - i;
    for (let k = 0; k < 3; k++) out[offset + k] = rampRgb[i][k] * (1 - f) + rampRgb[i + 1][k] * f;
  }
  const toCss = rgb => `rgb(${rgb.map(v => Math.round(255 * v)).join(',')})`;
  const ICON = {
    dot: `<svg viewBox="0 0 22 22" aria-hidden="true"><circle cx="11" cy="11" r="4.5" fill="${RAMP[6]}"/></svg>`,
    reference: `<svg viewBox="0 0 22 22" aria-hidden="true"><circle cx="11" cy="11" r="3.5" fill="${toCss(REFERENCE)}"/></svg>`,
    gray: `<svg viewBox="0 0 22 22" aria-hidden="true"><circle cx="11" cy="11" r="4.5" fill="#3d4148"/></svg>`,
    selected: `<svg viewBox="0 0 22 22" aria-hidden="true"><circle cx="11" cy="11" r="8" fill="none" stroke="${toCss(MARKER)}" stroke-width="2.4"/></svg>`,
  };
  function colorkey() {
    const rows = [
      ['reference', '<b>Blue dot</b>: the frame-0 normal reference'],
      ['dot', '<b>White to red dot</b>: an observed point at this frame, coloured by its anomaly score'],
      ['gray', '<b>Grey dot</b>: Reg2Inv gave this frame no score'],
      ...(state.selected ? [['selected', '<b>Cyan ring</b>: the point you clicked']] : []),
    ];
    $('#colorkey').innerHTML = `<div class="ckey-title">Reg2Inv point score <span>(feature distance)</span></div>`
      + `<div class="ckey-bar" style="background:linear-gradient(90deg,${RAMP.join(',')})"></div>`
      + `<div class="ckey-ticks"><span>0 · normal</span><span>≥ ${fmt(INDEX.scale)} · error</span></div>`;
    $('#markkey').innerHTML = rows.map(([icon, text]) => `<li><i class="tile">${ICON[icon]}</i><span>${text}</span></li>`).join('');
  }

  /* ---- stage ---- */
  const stage = M.stage(viewer, canvas);
  let current = null;

  function decode(header, base64) {
    const r = M.reader(M.bytes(base64), header.low, header.high), links = {};
    for (const [key, l] of Object.entries(header.links)) {
      const q = r.view(Uint16Array, l.score), score = new Float32Array(q.length);
      for (let i = 0; i < q.length; i++) score[i] = q[i] === MISSING ? NaN : (q[i] / 65534) * l.top;
      links[key] = {reference: r.positions(l.reference), referenceCount: l.reference_count,
        offsets: r.view(Int32Array, l.offsets), position: r.positions(l.position), score,
        transform: r.view(Float32Array, l.transform), series: l.series, status: l.status, fitness: l.fitness,
        max: l.max, maxFrame: l.max_frame, mean: l.mean, label: l.label};
    }
    return {header, links};
  }
  function build(data) {
    stage.clear();
    const h = data.header, size = stage.frame(h);
    stage.controls.target.set(...h.target);
    stage.controls.update();
    let most = 0, references = 0;
    for (const l of Object.values(data.links)) {
      references += l.referenceCount;
      for (let t = 0; t < h.frames; t++) most = Math.max(most, l.offsets[t + 1] - l.offsets[t]);
    }
    current = {data, frames: h.frames, pickRadius: 0.9 * size.world, drawn: [],
      reference: stage.points(references, size.pointSize(0.55)), cloud: stage.points(most * LINKS.length, size.pointSize(0.9)),
      marker: stage.points(1, size.pointSize(2.4), M.ring)};
    const rp = current.reference.geometry.getAttribute('position').array, rc = current.reference.geometry.getAttribute('color').array;
    let offset = 0;
    for (const key of LINKS) {
      const l = data.links[key];
      rp.set(l.reference, offset * 3);
      for (let i = 0; i < l.referenceCount; i++) rc.set(REFERENCE, (offset + i) * 3);
      current.referenceRange = current.referenceRange || {};
      current.referenceRange[key] = [offset, l.referenceCount];
      offset += l.referenceCount;
    }
    current.reference.geometry.getAttribute('position').needsUpdate = current.reference.geometry.getAttribute('color').needsUpdate = true;
    current.marker.renderOrder = 3;
    current.marker.material.depthTest = false;
  }
  const on = key => document.querySelector(`input[data-layer="${key}"]`).checked;
  const activeLinks = () => [...document.querySelectorAll('input[data-link]')].filter(i => i.checked).map(i => i.dataset.link);

  /* ---- draw one frame ---- */
  const color = new Float32Array(3);
  function place(link, frame, i, out, offset) {      // point i of this frame, aligned to frame 0 when requested
    const p = link.position, b = i * 3, T = link.transform, m = frame * 16;
    if (state.view === 'aligned' && Number.isFinite(T[m])) {
      for (let r = 0; r < 3; r++)
        out[offset + r] = T[m + 4 * r] * p[b] + T[m + 4 * r + 1] * p[b + 1] + T[m + 4 * r + 2] * p[b + 2] + T[m + 4 * r + 3];
    } else for (let k = 0; k < 3; k++) out[offset + k] = p[b + k];
  }
  function draw(frame) {
    if (!current) return;
    const {data, cloud, reference} = current, links = activeLinks();
    reference.visible = on('reference');
    // Only the active links' reference points are drawn: they are stored link2 then link5.
    const shownRef = LINKS.filter(k => links.includes(k));
    if (shownRef.length === 1) reference.geometry.setDrawRange(...current.referenceRange[shownRef[0]]);
    else reference.geometry.setDrawRange(0, shownRef.length ? Infinity : 0);
    const cp = cloud.geometry.getAttribute('position').array, cc = cloud.geometry.getAttribute('color').array;
    let shown = 0;
    current.drawn = [];
    for (const key of links) {
      const link = data.links[key], start = link.offsets[frame], end = link.offsets[frame + 1];
      for (let i = start; i < end; i++) {
        const value = link.score[i];
        if (Number.isFinite(value)) ramp(value / INDEX.scale, color, 0); else color.set(UNSCORED);
        place(link, frame, i, cp, shown * 3);
        cc.set(color, shown * 3);
        current.drawn.push([key, i]);
        shown++;
      }
    }
    cloud.geometry.setDrawRange(0, shown);
    cloud.geometry.getAttribute('position').needsUpdate = cloud.geometry.getAttribute('color').needsUpdate = true;
    drawSelection(frame);
    cursor(frame);
  }
  const clock = M.clock({video, play: $('#play'), slider: $('#slider'), label: $('#time'), draw, render: stage.render});

  /* ---- click a point: a card with its score and its frame ---- */
  M.picker(canvas, stage.camera, () => current && current.cloud, () => current.pickRadius, index => {
    const [key, i] = current.drawn[index];
    state.selected = {key, i, frame: clock.drawn};
    colorkey();
    clock.redraw();
  });
  function drawSelection(frame) {
    const marker = current.marker, s = state.selected;
    const valid = s && s.frame === frame && activeLinks().includes(s.key);
    marker.visible = !!valid;
    $('#inspector').hidden = !s;
    if (!s) return;
    const link = current.data.links[s.key];
    if (valid) {
      place(link, frame, s.i, marker.geometry.getAttribute('position').array, 0);
      marker.geometry.getAttribute('color').array.set(MARKER);
      marker.geometry.getAttribute('position').needsUpdate = marker.geometry.getAttribute('color').needsUpdate = true;
    }
    const value = link.score[s.i], start = link.offsets[s.frame], end = link.offsets[s.frame + 1];
    let below = 0, scored = 0;
    for (let i = start; i < end; i++) if (Number.isFinite(link.score[i])) { scored++; if (link.score[i] < value) below++; }
    $('#inspector').innerHTML = `<h4><span class="swatch" style="background:var(--${s.key})"></span>${s.key} · frame ${s.frame}`
      + `<button id="unselect" aria-label="Stop inspecting">×</button></h4><p class="insub">${valid ? 'Reg2Inv point score'
      : 'Shown at the frame where it was clicked; observed points are not tracked between frames'}</p><dl>`
      + `<dt>Point score</dt><dd>${Number.isFinite(value) ? fmt(value) : 'frame not scored'}</dd>`
      + `<dt>Within its frame</dt><dd>${Number.isFinite(value) ? `higher than ${Math.round(100 * below / Math.max(scored, 1))}%` : '—'}</dd>`
      + `<dt>Frame score</dt><dd>${fmt(link.series[s.frame])}</dd>`
      + `<dt>Registration fitness</dt><dd>${fmt(link.fitness[s.frame])}</dd></dl>`;
  }
  const unselect = () => { state.selected = null; colorkey(); clock.redraw(); };
  $('#inspector').addEventListener('click', e => { if (e.target.id === 'unselect') unselect(); });
  addEventListener('keydown', e => { if (e.key === 'Escape' && state.selected) unselect(); });

  /* ---- per-frame chart: Reg2Inv frame score per link ---- */
  const svg = $('#chart'), tip = $('#tip'), NS = 'http://www.w3.org/2000/svg';
  let geometry = null;
  const el = (tag, attrs, parent = svg) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    parent.appendChild(node);
    return node;
  };
  const niceTop = v => { const p = 10 ** Math.floor(Math.log10(v)), m = v / p; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p; };
  function chart() {
    if (!current) return;
    svg.innerHTML = '';
    const width = svg.clientWidth || 800, height = 250, pad = {l: 64, r: 84, t: 24, b: 32}, frames = current.frames;
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const series = activeLinks().map(key => {
      const l = current.data.links[key];
      return {key, values: l.series, max: l.max, maxFrame: l.maxFrame};
    });
    const finite = series.flatMap(s => s.values).filter(v => v !== null && Number.isFinite(v));
    const top = niceTop(Math.max(...finite, 1e-6));
    const x = f => pad.l + (f / Math.max(frames - 1, 1)) * (width - pad.l - pad.r);
    const y = v => height - pad.b - (v / top) * (height - pad.t - pad.b);
    geometry = {x, y, pad, width, height, frames, series};
    for (const v of [0, top / 2, top]) {
      el('line', {x1: pad.l, x2: width - pad.r, y1: y(v), y2: y(v), class: 'grid'});
      el('text', {x: pad.l - 10, y: y(v) + 5, class: 'tick', 'text-anchor': 'end'}).textContent = fmt(v);
    }
    for (const f of [0, Math.round((frames - 1) / 2), frames - 1]) {
      el('text', {x: x(f), y: height - 9, class: 'tick', 'text-anchor': f === 0 ? 'start' : f === frames - 1 ? 'end' : 'middle'}).textContent = `frame ${f}`;
    }
    const ends = [];
    for (const s of series) {
      let d = '', pen = false;
      s.values.forEach((v, i) => {
        if (v === null || !Number.isFinite(v)) { pen = false; return; }
        d += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`; pen = true;
      });
      el('path', {d, class: `series ${s.key}`});
      const last = s.values.map((v, i) => [v, i]).filter(([v]) => v !== null && Number.isFinite(v)).pop();
      if (last) ends.push({x: x(last[1]) + 8, y: y(last[0]) + 5, text: s.key});
    }
    ends.sort((a, b) => a.y - b.y).forEach((label, i) => {
      if (i && label.y - ends[i - 1].y < 16) label.y = ends[i - 1].y + 16;
      el('text', {x: label.x, y: label.y, class: 'direct'}).textContent = label.text;
    });
    for (const s of series) {
      el('circle', {cx: x(s.maxFrame), cy: y(s.max), r: 5, class: `maxdot ${s.key}`});
      const late = s.maxFrame > 0.8 * (frames - 1);
      el('text', {x: x(s.maxFrame) + (late ? -10 : 0), y: y(s.max) - 11, class: 'maxlabel',
                  'text-anchor': late ? 'end' : 'middle'}).textContent = `max ${fmt(s.max)}`;
    }
    geometry.cursor = el('line', {x1: 0, x2: 0, y1: pad.t, y2: height - pad.b, class: 'cursor'});
    geometry.hover = el('line', {x1: 0, x2: 0, y1: pad.t, y2: height - pad.b, class: 'hover', visibility: 'hidden'});
    $('#serieskey').innerHTML = series.map(s => `<span><i class="${s.key}"></i>${s.key}</span>`).join('');
    $('#jumps').innerHTML = series.map(s => `<button data-frame="${s.maxFrame}">${s.key} max at frame ${s.maxFrame}</button>`).join('');
    cursor(Math.max(clock.drawn, 0));
  }
  function cursor(frame) {
    if (!geometry) return;
    geometry.cursor.setAttribute('x1', geometry.x(frame));
    geometry.cursor.setAttribute('x2', geometry.x(frame));
  }
  const frameAt = event => {
    const box = svg.getBoundingClientRect(), g = geometry;
    const px = (event.clientX - box.left) * (g.width / box.width);
    return Math.round(Math.min(Math.max((px - g.pad.l) / (g.width - g.pad.l - g.pad.r), 0), 1) * (g.frames - 1));
  };
  let scrubbing = false;
  svg.addEventListener('pointerdown', e => { if (!geometry) return; scrubbing = true; svg.setPointerCapture(e.pointerId); clock.playing = false; clock.seekFrame(frameAt(e)); });
  svg.addEventListener('pointerup', () => { scrubbing = false; });
  svg.addEventListener('pointermove', e => {
    if (!geometry) return;
    const f = frameAt(e), g = geometry;
    if (scrubbing) clock.seekFrame(f);
    g.hover.setAttribute('visibility', 'visible');
    g.hover.setAttribute('x1', g.x(f)); g.hover.setAttribute('x2', g.x(f));
    tip.hidden = false;
    tip.innerHTML = `<b>frame ${f}</b>` + g.series.map(s => `<span><i class="${s.key}"></i>${s.key} ${fmt(s.values[f])}</span>`).join('');
    const box = svg.getBoundingClientRect(), left = Math.min(e.clientX - box.left + 14, box.width - tip.offsetWidth - 4);
    tip.style.left = `${Math.max(left, 4)}px`;
  });
  svg.addEventListener('pointerleave', () => { if (geometry) geometry.hover.setAttribute('visibility', 'hidden'); tip.hidden = true; });
  $('#jumps').addEventListener('click', e => {
    const b = e.target.closest('button[data-frame]');
    if (b) { clock.playing = false; clock.seekFrame(Number(b.dataset.frame)); }
  });
  new ResizeObserver(() => chart()).observe(svg);

  /* ---- view switcher ---- */
  const seg = $('#view');
  seg.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    state.view = b.dataset.v;
    viewChanged();
  });
  function viewChanged() {
    [...seg.children].forEach(x => x.setAttribute('aria-pressed', String(x.dataset.v === state.view)));
    $('#vnote').textContent = VIEWS[state.view];
    history.replaceState(null, '', `?case=${encodeURIComponent(state.name)}&view=${state.view}`);
    clock.redraw();
  }

  /* ---- index of every video ---- */
  const columns = [
    {key: 'case', label: 'Video', get: c => c.case},
    {key: 'frames', label: 'Frames', get: c => c.frames},
    ...LINKS.flatMap(link => [
      {key: `${link}:label`, group: link, label: 'Label', get: c => c.labels[link]},
      {key: `${link}:mean`, group: link, label: 'Mean', get: c => c.stats[link].mean},
      {key: `${link}:max`, group: link, label: 'Max', get: c => c.stats[link].max},
      {key: `${link}:frame`, group: link, label: 'At frame', get: c => c.stats[link].max_frame},
    ]),
  ];
  const meanTop = Math.max(...INDEX.cases.flatMap(c => LINKS.map(l => c.stats[l].mean)));
  function table() {
    const col = columns.find(c => c.key === state.sort.key);
    const rows = [...INDEX.cases].sort((a, b) => {
      const x = col.get(a), y = col.get(b);
      return (x === y ? 0 : (x ?? -Infinity) > (y ?? -Infinity) ? 1 : -1) * state.sort.dir;
    });
    const cell = (c, column) => {
      const v = column.get(c), link = column.group;
      if (column.key === 'case') return `<td><button class="rowlink" data-case="${c.case}">${c.case}</button>`
        + `${c.warning ? ` <span class="flag" title="${c.warning}">flag</span>` : ''}</td>`;
      if (column.key.endsWith(':label')) return `<td><span class="label l${String(v).replace('.', '')}">${LABEL[v] ?? '—'}</span></td>`;
      if (column.key.endsWith(':mean')) return `<td><span class="cellbar"><span class="bar ${link}"><i style="width:${Math.min(100, 100 * (v ?? 0) / meanTop)}%"></i></span>${fmt(v)}</span></td>`;
      if (column.key.endsWith(':max')) return `<td>${fmt(v)}</td>`;
      return `<td>${v ?? '—'}</td>`;
    };
    $('#cases').innerHTML = `<thead><tr class="group"><th colspan="2"></th>` + LINKS.map(l =>
      `<th colspan="4" style="text-align:center"><span class="swatch" style="background:var(--${l})"></span>${l}</th>`).join('')
      + `</tr><tr>` + columns.map(c => `<th class="sortable" data-k="${c.key}" aria-sort="${state.sort.key === c.key ? (state.sort.dir > 0 ? 'ascending' : 'descending') : 'none'}">${c.label}</th>`).join('')
      + `</tr></thead><tbody>` + rows.map(c => `<tr${c.case === state.name ? ' aria-current="true"' : ''}>`
      + columns.map(column => cell(c, column)).join('') + '</tr>').join('') + '</tbody>';
  }
  $('#cases').addEventListener('click', e => {
    const th = e.target.closest('th.sortable');
    if (th) {
      state.sort = {key: th.dataset.k, dir: state.sort.key === th.dataset.k ? -state.sort.dir : (th.dataset.k === 'case' ? 1 : -1)};
      table();
      return;
    }
    const row = e.target.closest('tr');
    const button = row && row.querySelector('.rowlink');
    if (button && button.dataset.case !== state.name) { load(button.dataset.case); window.scrollTo({top: 0, behavior: 'smooth'}); }
  });
  const a = INDEX.auroc;
  const auroc = l => `<b>${a[l].pooled.toFixed(2)}</b> on ${l} (95% CI ${a[l].ci95[0].toFixed(2)}–${a[l].ci95[1].toFixed(2)}; `
    + `within one generator ${a[l].within.toFixed(2)} over ${a[l].within_pairs} pairs)`;
  $('#aurocline').innerHTML = `Video mean score against the human deformation labels, severe vs none with moderate excluded: AUROC `
    + `${auroc('link2')} and ${auroc('link5')}. Labels are confounded with generator, so the within-generator values matter more.`;
  $('#labelnote').textContent = `Labels: Link2 from ${INDEX.label_sources.link2}; Link5 from ${INDEX.label_sources.link5}. `
    + 'Bars are relative to the largest video mean. Flags mark the five videos with a mask-review warning.';
  const frames = INDEX.cases.reduce((s, c) => s + c.frames, 0);
  $('#herofacts').innerHTML = [`<b>${INDEX.cases.length}</b> videos`, `<b>${frames.toLocaleString()}</b> frames`,
    '<b>2</b> links', `1 in <b>${INDEX.stride * INDEX.stride}</b> points shown`].map(t => `<li>${t}</li>`).join('');

  /* ---- load a video in place ---- */
  async function load(name) {
    state.name = name;
    state.selected = null;
    status.hidden = false;
    status.textContent = 'Loading…';
    const entry = INDEX.cases.find(c => c.case === name);
    $('#title').textContent = name;
    $('#facts').innerHTML = [`<b>${entry.frames}</b> frames`,
      ...LINKS.map(l => `${l}: <b>${LABEL[entry.labels[l]] ?? 'no label'}</b>, mean <b>${fmt(entry.stats[l].mean)}</b>`)]
      .map(t => `<li>${t}</li>`).join('');
    document.title = `${name} · Reg2Inv frame-0 error replay`;
    try {
      const [header, payload] = await M.loadScript(`cases/${name}.js`, 'REG2INV_CASE');
      if (state.name !== name) return;                     // a newer request won
      build(decode(header, payload));
      video.src = `videos/${name}.mp4`;
      clock.set(header.frames, header.fps);
      viewChanged();
      colorkey();
      table();
      chart();
      status.hidden = true;
    } catch (error) {
      status.textContent = String(error.message || error);
    }
  }
  $('#layers').addEventListener('change', e => {
    if (e.target.id === 'pipbox') $('#pip').hidden = !e.target.checked;
    if (e.target.dataset.link) chart();
    clock.redraw();
  });
  $('#howbutton').addEventListener('click', () => {
    const section = $('#about-point');
    section.open = true;
    section.scrollIntoView({behavior: 'smooth', block: 'start'});
  });
  clock.start();
  load(state.name);
})();
