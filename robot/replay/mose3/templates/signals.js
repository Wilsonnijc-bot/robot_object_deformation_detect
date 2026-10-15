/* Error-signal replay (signals.html?case=<id>&signal=<key>): each tracked point coloured by its own
   per-frame value of a link signal, with the per-frame series below and every video indexed. */
(() => {
  'use strict';
  const M = window.MOSE3_3D, INDEX = window.MOSE3_INDEX;
  // White (rigid) to red (more error); lightness falls steadily, and the red end stays visible on black.
  const RAMP = ['#fdf9f9', '#ffdfda', '#ffc7bf', '#ffafa5', '#ff988c', '#fa8074', '#f3665c', '#ea4b43', '#e02628'];
  const RING = [0.557, 0.643, 0.749], MARKER = [0.31, 0.82, 0.91];      // steel-blue predictions, cyan selection
  const UNSCORED = [0.24, 0.255, 0.28], TRAIL = 12, MISSING = 65535;
  const LABEL = {1: 'severe', 0.5: 'moderate', 0: 'none'};
  const $ = s => document.querySelector(s);
  const viewer = $('#viewer'), canvas = $('#scene'), video = $('#video'), status = $('#status');
  const complete = INDEX.cases.filter(c => c.status === 'complete');
  const params = new URLSearchParams(location.search);
  const state = {signal: INDEX.signals.some(s => s.key === params.get('signal')) ? params.get('signal') : INDEX.signals[0].key,
                 name: complete.some(c => c.case === params.get('case')) ? params.get('case') : complete[0].case,
                 selected: null, sort: {key: 'case', dir: 1}};
  const info = () => INDEX.signals.find(s => s.key === state.signal);
  const fmt = v => v === null || v === undefined || !Number.isFinite(v) ? '—' : (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3));
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  /* ---- colour: one red hue, white (near zero) to red (large error) ---- */
  const rampRgb = RAMP.map(h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255));
  function ramp(fraction, out, offset) {
    const x = Math.min(Math.max(fraction, 0), 1) * (rampRgb.length - 1), i = Math.min(Math.floor(x), rampRgb.length - 2), f = x - i;
    for (let k = 0; k < 3; k++) out[offset + k] = rampRgb[i][k] * (1 - f) + rampRgb[i + 1][k] * f;
  }
  // The legend explains every mark drawn on the canvas, using the same colours.
  const toCss = rgb => `rgb(${rgb.map(v => Math.round(255 * v)).join(',')})`;
  const ICON = {
    ring: `<svg viewBox="0 0 22 22" aria-hidden="true"><circle cx="11" cy="11" r="8" fill="none" stroke="${toCss(RING)}" stroke-width="2.4"/></svg>`,
    dot: `<svg viewBox="0 0 22 22" aria-hidden="true"><circle cx="11" cy="11" r="4.5" fill="${RAMP[6]}"/></svg>`,
    line: `<svg viewBox="0 0 22 22" aria-hidden="true"><line x1="2" y1="11" x2="20" y2="11" stroke="${RAMP[6]}" stroke-width="2.4"/></svg>`,
    gray: `<svg viewBox="0 0 22 22" aria-hidden="true"><circle cx="11" cy="11" r="4.5" fill="#3d4148"/></svg>`,
    selected: `<svg viewBox="0 0 22 22" aria-hidden="true"><circle cx="11" cy="11" r="8" fill="none" stroke="${toCss(MARKER)}" stroke-width="2.4"/></svg>`,
  };
  function colorkey() {
    const s = info();
    const rows = [
      ...(s.residual ? [['ring', '<b>Large ring</b>: the rigid prediction, where a rigid motion puts the point']] : []),
      ['dot', `<b>Small dot</b>: the tracked point, where MoSE3 says it went, coloured by its error`],
      ...(s.residual ? [['line', '<b>Line</b>: the residual, from prediction to tracked point']] : []),
      ['gray', '<b>Grey dot</b>: not scored at this frame'],
      ...(state.selected ? [['selected', '<b>Cyan ring</b>: the point you clicked']] : []),
    ];
    // Colour scale on the canvas; the marks are explained in a row under it, so no points are covered.
    $('#colorkey').innerHTML = `<div class="ckey-title">${s.label} <span>(${s.unit})</span></div>`
      + `<div class="ckey-bar" style="background:linear-gradient(90deg,${RAMP.join(',')})"></div>`
      + `<div class="ckey-ticks"><span>0 · rigid</span><span>${fmt(s.scale / 2)}</span><span>≥ ${fmt(s.scale)} · error</span></div>`;
    $('#markkey').innerHTML = rows.map(([icon, text]) => `<li><i class="tile">${ICON[icon]}</i><span>${text}</span></li>`).join('');
  }

  /* ---- stage ---- */
  const stage = M.stage(viewer, canvas);
  const residualMaterial = stage.lineMaterial(2), trailMaterial = stage.lineMaterial(2);
  let current = null;

  function decodeSignals(header, base64, scene) {
    const r = M.reader(M.bytes(base64), scene.header.low, scene.header.high), links = {};
    for (const [key, block] of Object.entries(header.links)) {
      const signals = {};
      for (const [name, s] of Object.entries(block.signals)) {
        const q = r.view(Uint16Array, s.values), values = new Float32Array(q.length);
        for (let i = 0; i < q.length; i++) values[i] = q[i] === MISSING ? NaN : (q[i] / 65534) * s.top;
        signals[name] = {values, series: s.series, max: s.max, maxFrame: s.max_frame,
                         prediction: s.prediction ? r.positions(s.prediction) : null};
      }
      links[key] = {selected: r.view(Uint8Array, block.selected), label: block.label, signals};
    }
    return links;
  }
  function build(scene, signals) {
    stage.clear();
    const h = scene.header, size = stage.frame(h), total = Object.values(scene.links).reduce((a, l) => a + l.count, 0);
    current = {scene, signals, frames: h.frames, fps: h.fps, lift: 0.01 * size.depth, pickRadius: 0.9 * size.world,
      cloud: stage.points(scene.cloud.count, size.pointSize(0.8)), heads: stage.points(total, size.pointSize(0.9)),
      rings: stage.points(total, size.pointSize(1.3), M.ring), marker: stage.points(1, size.pointSize(2.4), M.ring),
      residuals: stage.fatLines(total, residualMaterial), trails: stage.fatLines(total * TRAIL, trailMaterial), drawn: []};
    const gray = current.cloud.geometry.getAttribute('color').array;
    for (let i = 0; i < gray.length; i++) gray[i] = 0.32;
    current.cloud.geometry.getAttribute('color').needsUpdate = true;
    current.marker.renderOrder = 3;
    current.marker.material.depthTest = false;
  }
  const on = key => document.querySelector(`input[data-layer="${key}"]`).checked;
  const activeLinks = () => [...document.querySelectorAll('input[data-link]')].filter(i => i.checked).map(i => i.dataset.link);

  /* ---- draw one frame ---- */
  const lift = new Float32Array(3), head = new Float32Array(3), past = new Float32Array(3), next = new Float32Array(3);
  function liftOf(link, t, n) {                          // the offset ``lifted`` adds along the outward normal
    M.lifted(link, t, n, link.trajectory, head, 0, current.lift);
    const p = (t * link.count + n) * 3;
    for (let k = 0; k < 3; k++) lift[k] = head[k] - link.trajectory[p + k];
    return lift;
  }
  function draw(frame) {
    if (!current) return;
    const {scene, signals, cloud, heads, rings, residuals, trails} = current, s = info();
    cloud.visible = on('scene');
    if (cloud.visible) {
      cloud.geometry.getAttribute('position').array.set(
        scene.cloud.position.subarray(frame * scene.cloud.count * 3, (frame + 1) * scene.cloud.count * 3));
      cloud.geometry.getAttribute('position').needsUpdate = true;
    }
    const showPrediction = s.residual && on('prediction'), showTrails = on('trails');
    const hp = heads.geometry.getAttribute('position').array, hc = heads.geometry.getAttribute('color').array;
    const rp = rings.geometry.getAttribute('position').array, rc = rings.geometry.getAttribute('color').array;
    const lp = residuals.geometry.attributes.instanceStart.data, lc = residuals.geometry.attributes.instanceColorStart.data;
    const tp = trails.geometry.attributes.instanceStart.data, tc = trails.geometry.attributes.instanceColorStart.data;
    const color = new Float32Array(3);
    let shown = 0, ringCount = 0, lineCount = 0, trailCount = 0;
    current.drawn = [];
    for (const key of activeLinks()) {
      const link = scene.links[key], signal = signals[key].signals[state.signal];
      for (let n = 0; n < link.count; n++) {
        const index = frame * link.count + n, value = signal.values[index];
        if (Number.isFinite(value)) ramp(value / s.scale, color, 0); else color.set(UNSCORED);
        M.lifted(link, frame, n, link.trajectory, hp, shown * 3, current.lift);
        hc.set(color, shown * 3);
        current.drawn.push([key, n]);
        shown++;
        if (showPrediction && Number.isFinite(value)) {
          const offset = liftOf(link, frame, n), p = index * 3;
          for (let k = 0; k < 3; k++) {
            rp[ringCount * 3 + k] = signal.prediction[p + k] + offset[k];
            lp.array[lineCount * 6 + k] = rp[ringCount * 3 + k];
            lp.array[lineCount * 6 + 3 + k] = hp[(shown - 1) * 3 + k];
            lc.array[lineCount * 6 + k] = lc.array[lineCount * 6 + 3 + k] = color[k];
          }
          rc.set(RING, ringCount * 3);
          ringCount++; lineCount++;
        }
        if (showTrails) {
          for (let t = Math.max(0, frame - TRAIL); t < frame; t++) {
            M.lifted(link, t, n, link.trajectory, past, 0, current.lift);
            M.lifted(link, t + 1, n, link.trajectory, next, 0, current.lift);
            const o = trailCount * 6, age = (t - frame + TRAIL + 1) / (TRAIL + 1);
            for (let k = 0; k < 3; k++) {
              tp.array[o + k] = past[k]; tp.array[o + 3 + k] = next[k];
              tc.array[o + k] = color[k] * age; tc.array[o + 3 + k] = color[k] * (age + 1 / (TRAIL + 1));
            }
            trailCount++;
          }
        }
      }
    }
    heads.geometry.setDrawRange(0, shown);
    rings.geometry.setDrawRange(0, ringCount);
    for (const g of [heads.geometry, rings.geometry]) {
      g.getAttribute('position').needsUpdate = true;
      g.getAttribute('color').needsUpdate = true;
    }
    residuals.geometry.instanceCount = lineCount;
    trails.geometry.instanceCount = trailCount;
    for (const b of [lp, lc, tp, tc]) b.needsUpdate = true;
    drawSelection(frame);
    cursor(frame);
  }
  const clock = M.clock({video, play: $('#play'), slider: $('#slider'), label: $('#time'), draw, render: stage.render});

  /* ---- click a point: its own curve on the chart and a card ---- */
  M.picker(canvas, stage.camera, () => current && current.heads, () => current.pickRadius, index => {
    const [key, n] = current.drawn[index];
    state.selected = {key, n};
    colorkey();
    chart();
    clock.redraw();
  });
  function drawSelection(frame) {
    const marker = current.marker;
    marker.visible = !!state.selected;
    $('#inspector').hidden = !state.selected;
    if (!state.selected) return;
    const {key, n} = state.selected, link = current.scene.links[key], block = current.signals[key];
    const values = block.signals[state.signal].values;
    M.lifted(link, frame, n, link.trajectory, marker.geometry.getAttribute('position').array, 0, current.lift);
    marker.geometry.getAttribute('color').array.set(MARKER);
    marker.geometry.getAttribute('position').needsUpdate = marker.geometry.getAttribute('color').needsUpdate = true;
    let top = -Infinity, at = null;
    for (let t = 1; t < current.frames; t++) {
      const v = values[t * link.count + n];
      if (Number.isFinite(v) && v > top) { top = v; at = t; }
    }
    const reason = block.selected[n] ? (link.visible[frame * link.count + n] ? 'not scored at this frame' : 'occluded at this frame')
                                     : (link.onEdge[n] ? 'left out: on a depth edge' : 'left out: depth outlier');
    const now = values[frame * link.count + n];
    $('#inspector').innerHTML = `<h4><span class="swatch" style="background:var(--${key})"></span>${key} · point ${link.pointIds[n]}`
      + `<button id="unselect" aria-label="Stop following">×</button></h4><p class="insub">${info().label}</p><dl>`
      + `<dt>This frame</dt><dd>${Number.isFinite(now) ? fmt(now) : reason}</dd>`
      + `<dt>Its maximum</dt><dd>${at === null ? '—' : `${fmt(top)} at frame ${at}`}</dd>`
      + `<dt>Query pixel</dt><dd>${link.queryXy[2 * n].toFixed(1)}, ${link.queryXy[2 * n + 1].toFixed(1)}</dd></dl>`;
  }
  $('#inspector').addEventListener('click', e => {
    if (e.target.id === 'unselect') { state.selected = null; colorkey(); chart(); clock.redraw(); }
  });
  addEventListener('keydown', e => { if (e.key === 'Escape' && state.selected) { state.selected = null; colorkey(); chart(); clock.redraw(); } });

  /* ---- per-frame chart (one axis: the selected signal) ---- */
  const svg = $('#chart'), tip = $('#tip'), NS = 'http://www.w3.org/2000/svg';
  let geometry = null;
  const el = (tag, attrs, parent = svg) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    parent.appendChild(node);
    return node;
  };
  const niceTop = v => { const p = 10 ** Math.floor(Math.log10(v)), m = v / p; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p; };
  function chartSeries() {
    const series = activeLinks().map(key => ({key, values: current.signals[key].signals[state.signal].series,
      max: current.signals[key].signals[state.signal].max, maxFrame: current.signals[key].signals[state.signal].maxFrame}));
    if (state.selected) {
      const {key, n} = state.selected, link = current.scene.links[key], values = current.signals[key].signals[state.signal].values;
      series.push({key, point: current.scene.links[key].pointIds[n],
                   values: Array.from({length: current.frames}, (_, t) => { const v = values[t * link.count + n]; return Number.isFinite(v) ? v : null; })});
    }
    return series;
  }
  function chart() {
    if (!current) return;
    svg.innerHTML = '';
    const width = svg.clientWidth || 800, height = 250, pad = {l: 64, r: 84, t: 24, b: 32}, frames = current.frames;
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const series = chartSeries();
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
      el('path', {d, class: s.point === undefined ? `series ${s.key}` : 'series point'});
      const last = s.values.map((v, i) => [v, i]).filter(([v]) => v !== null && Number.isFinite(v)).pop();
      if (last) ends.push({x: x(last[1]) + 8, y: y(last[0]) + 5, text: s.point === undefined ? s.key : `point ${s.point}`});
    }
    // Line-end labels: keep at least one line of text apart, top to bottom.
    ends.sort((a, b) => a.y - b.y).forEach((label, i) => {
      if (i && label.y - ends[i - 1].y < 16) label.y = ends[i - 1].y + 16;
      el('text', {x: label.x, y: label.y, class: 'direct'}).textContent = label.text;
    });
    for (const s of series.filter(s => s.point === undefined && s.maxFrame !== null && s.maxFrame !== undefined)) {
      el('circle', {cx: x(s.maxFrame), cy: y(s.max), r: 5, class: `maxdot ${s.key}`});
      // Near the right edge the label goes left of the dot, clear of the line-end labels.
      const late = s.maxFrame > 0.8 * (frames - 1);
      el('text', {x: x(s.maxFrame) + (late ? -10 : 0), y: y(s.max) - 11, class: 'maxlabel',
                  'text-anchor': late ? 'end' : 'middle'}).textContent = `max ${fmt(s.max)}`;
    }
    geometry.cursor = el('line', {x1: 0, x2: 0, y1: pad.t, y2: height - pad.b, class: 'cursor'});
    geometry.hover = el('line', {x1: 0, x2: 0, y1: pad.t, y2: height - pad.b, class: 'hover', visibility: 'hidden'});
    $('#serieskey').innerHTML = series.map(s => `<span><i class="${s.point === undefined ? s.key : 'point'}"></i>`
      + `${s.point === undefined ? s.key : `${s.key} point ${s.point}${s.values.some(v => v !== null) ? '' : ' (not scored)'}`}</span>`).join('');
    $('#jumps').innerHTML = series.filter(s => s.point === undefined && s.maxFrame !== null && s.maxFrame !== undefined)
      .map(s => `<button data-frame="${s.maxFrame}">${s.key} max at frame ${s.maxFrame}</button>`).join('');
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
    tip.innerHTML = `<b>frame ${f}</b>` + g.series.map(s => `<span><i class="${s.point === undefined ? s.key : 'point'}"></i>`
      + `${s.point === undefined ? s.key : 'point ' + s.point} ${fmt(s.values[f])}</span>`).join('');
    const box = svg.getBoundingClientRect(), left = Math.min(e.clientX - box.left + 14, box.width - tip.offsetWidth - 4);
    tip.style.left = `${Math.max(left, 4)}px`;
  });
  svg.addEventListener('pointerleave', () => { if (geometry) geometry.hover.setAttribute('visibility', 'hidden'); tip.hidden = true; });
  $('#jumps').addEventListener('click', e => {
    const b = e.target.closest('button[data-frame]');
    if (b) { clock.playing = false; clock.seekFrame(Number(b.dataset.frame)); }
  });
  new ResizeObserver(() => chart()).observe(svg);

  /* ---- signal switcher ---- */
  const seg = $('#signal');
  seg.innerHTML = INDEX.signals.map(s => `<button data-v="${s.key}" aria-pressed="${s.key === state.signal}">${s.label}</button>`).join('');
  seg.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    state.signal = b.dataset.v;
    [...seg.children].forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    signalChanged();
  });
  function signalChanged() {
    const s = info();
    $('#snote').textContent = s.note;
    $('#howbutton').innerHTML = `How is the ${s.label.toLowerCase()} measured? <span aria-hidden="true">↓</span>`;
    $('#predictionlabel').style.opacity = s.residual ? '' : '0.45';
    $('#predictionlabel').querySelector('input').disabled = !s.residual;
    $('#aurocline').innerHTML = `Error signal: ${s.label}, maximum over the clip. AUROC <b>${s.auroc.link2.toFixed(2)}</b> on Link2 and `
      + `<b>${s.auroc.link5.toFixed(2)}</b> on Link5 (severe vs none, moderate excluded). Keeping moderate as the middle `
      + `level, the C-index is ${s.ordinal.link2.toFixed(2)} and ${s.ordinal.link5.toFixed(2)}.`;
    history.replaceState(null, '', `?case=${encodeURIComponent(state.name)}&signal=${state.signal}`);
    colorkey();
    table();
    chart();
    clock.redraw();
  }

  /* ---- index of every video ---- */
  const columns = [
    {key: 'case', label: 'Video', get: c => c.case},
    {key: 'frames', label: 'Frames', get: c => c.frames},
    ...['link2', 'link5'].flatMap(link => [
      {key: `${link}:label`, group: link, label: 'Label', get: c => c.labels[link]},
      {key: `${link}:max`, group: link, label: 'Max', get: c => c.signals[link][state.signal].max},
      {key: `${link}:frame`, group: link, label: 'At frame', get: c => c.signals[link][state.signal].max_frame},
    ]),
  ];
  function table() {
    const s = info(), col = columns.find(c => c.key === state.sort.key);
    const rows = [...complete].sort((a, b) => {
      const x = col.get(a), y = col.get(b);
      return (x === y ? 0 : (x ?? -Infinity) > (y ?? -Infinity) ? 1 : -1) * state.sort.dir;
    });
    const cell = (c, column) => {
      const v = column.get(c), link = column.group;
      if (column.key === 'case') return `<td><button class="rowlink" data-case="${c.case}">${c.case}</button></td>`;
      if (column.key.endsWith(':label')) return `<td><span class="label l${String(v).replace('.', '')}">${LABEL[v] ?? '—'}</span></td>`;
      if (column.key.endsWith(':max')) return `<td><span class="cellbar"><span class="bar ${link}"><i style="width:${Math.min(100, 100 * (v ?? 0) / s.scale)}%"></i></span>${fmt(v)}</span></td>`;
      return `<td>${v ?? '—'}</td>`;
    };
    $('#cases').innerHTML = `<thead><tr class="group"><th colspan="2"></th>` + ['link2', 'link5'].map(l =>
      `<th colspan="3" style="text-align:center"><span class="swatch" style="background:var(--${l})"></span>${l}</th>`).join('')
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
  $('#labelnote').textContent = `Labels: Link2 from ${INDEX.label_sources.link2}; Link5 from ${INDEX.label_sources.link5}. `
    + 'Bars are relative to the colour scale’s upper end.';

  /* ---- load a video in place ---- */
  async function load(name) {
    state.name = name;
    state.selected = null;
    status.hidden = false;
    status.textContent = 'Loading…';
    const entry = complete.find(c => c.case === name);
    $('#title').textContent = name;
    $('#videolink').href = `video.html?case=${encodeURIComponent(name)}`;
    $('#facts').innerHTML = [`<b>${entry.frames}</b> frames`,
      ...['link2', 'link5'].map(l => `${l}: <b>${LABEL[entry.labels[l]] ?? 'no label'}</b>`)].map(t => `<li>${t}</li>`).join('');
    document.title = `${name} · MoSE3 error-signal replay`;
    try {
      const [sceneHeader, sceneData] = await M.loadScript(`cases/${name}.js`, 'MOSE3_SCENE');
      const scene = M.decodeScene(sceneHeader, sceneData);
      const [signalHeader, signalData] = await M.loadScript(`signals/${name}.js`, 'MOSE3_SIGNALS');
      if (state.name !== name) return;                     // a newer request won
      build(scene, decodeSignals(signalHeader, signalData, scene));
      video.src = `videos/${name}.mp4`;
      clock.set(sceneHeader.frames, sceneHeader.fps);
      signalChanged();
      status.hidden = true;
    } catch (error) {
      status.textContent = String(error.message || error);
    }
  }
  $('#layers').addEventListener('change', e => {
    if (e.target.id === 'pipbox') $('#pip').hidden = !e.target.checked;
    if (e.target.dataset.link) {
      if (state.selected && state.selected.key === e.target.dataset.link && !e.target.checked) state.selected = null;
      chart();
    }
    clock.redraw();
  });
  // The button opens the selected signal's explanation, then scrolls to it.
  $('#howbutton').addEventListener('click', () => {
    const section = document.querySelector(`#about-${state.signal}`);
    section.open = true;
    section.scrollIntoView({behavior: 'smooth', block: 'start'});
  });
  document.querySelectorAll('[data-auroc]').forEach(node => {
    const [key, link, kind] = node.dataset.auroc.split(':'), s = INDEX.signals.find(s => s.key === key);
    node.textContent = (kind === 'ordinal' ? s.ordinal : s.auroc)[link].toFixed(2);
  });
  clock.start();
  load(state.name);
})();
