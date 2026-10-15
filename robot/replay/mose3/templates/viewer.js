/* Per-video MoSE3 replay (video.html?case=<id>), built on common3d.js. */
(() => {
  'use strict';
  const M = window.MOSE3_3D, INDEX = window.MOSE3_INDEX;
  const LINK_COLORS = {link2: '#d9822b', link5: '#2f8fd8'};
  const TRAIL = 12, OCCLUDED = 0.35;
  const LABELS = {rigid_fit_residual: 'Rigid fit', se3_rotation_spread_deg: 'SE(3) spread °',
                  pair_ratio_mad: 'Pair MAD', embedding_dispersion: 'Emb. spread'};
  const $ = s => document.querySelector(s);
  const viewer = $('#viewer'), canvas = $('#scene'), video = $('#video'), status = $('#status');
  const complete = INDEX.cases.filter(c => c.status === 'complete');
  const name = new URLSearchParams(location.search).get('case') || (complete[0] || {}).case;
  const entry = INDEX.cases.find(c => c.case === name);
  const flag = c => (c && c.masks.link5 && c.masks.link5 !== 'latest_guarded_mask') ? c.masks.link5 : '';
  const format = v => v === null || v === undefined ? '—' : (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(4));

  const stage = M.stage(viewer, canvas);
  const axisMaterial = stage.lineMaterial(3), trailMaterial = stage.lineMaterial(2.5), pathMaterial = stage.lineMaterial(3.5);
  let current = null;

  function build(data) {
    const h = data.header, total = Object.values(data.links).reduce((a, l) => a + l.count, 0);
    const size = stage.frame(h);
    current = {data, depth: size.depth, lift: 0.01 * size.depth, axisLength: 0.03 * size.depth,
      pickRadius: 0.9 * size.world, cloud: stage.points(data.cloud.count, size.pointSize(1)),
      heads: stage.points(total, size.pointSize(0.7)), marker: stage.points(1, size.pointSize(2.2)),
      axes: stage.fatLines(total * 3, axisMaterial), trails: stage.fatLines(total * TRAIL, trailMaterial),
      path: stage.fatLines(h.frames, pathMaterial), drawnPoints: []};
    current.marker.renderOrder = 2;
    current.marker.material.depthTest = false;
    current.marker.visible = false;
    paintCloud();
  }
  const cloudMode = () => document.querySelector('input[name="cloud"]:checked').value;
  function paintCloud() {
    if (!current) return;
    const mode = cloudMode();
    current.cloud.visible = mode !== 'none';
    if (mode === 'none') return;
    const attribute = current.cloud.geometry.getAttribute('color');
    attribute.array.set(current.data.cloud[mode]);
    attribute.needsUpdate = true;
  }

  /* ---- draw one frame ---- */
  function layers() {
    const on = key => document.querySelector(`input[data-layer="${key}"]`).checked;
    const links = [...document.querySelectorAll('input[data-link]')].filter(i => i.checked).map(i => i.dataset.link);
    return {axes: on('axes'), trails: on('trails'), links};
  }
  function draw(frame) {
    if (!current) return;
    const {data, cloud, heads, axes, trails, lift, axisLength} = current;
    const count = data.cloud.count, position = cloud.geometry.getAttribute('position');
    position.array.set(data.cloud.position.subarray(frame * count * 3, (frame + 1) * count * 3));
    position.needsUpdate = true;
    const state = layers(), palette = cloudMode() === 'pca' ? 'pca' : 'trail';
    const headPosition = heads.geometry.getAttribute('position').array, headColor = heads.geometry.getAttribute('color').array;
    const axisData = axes.geometry.attributes.instanceStart.data, axisColor = axes.geometry.attributes.instanceColorStart.data;
    const trailData = trails.geometry.attributes.instanceStart.data, trailColor = trails.geometry.attributes.instanceColorStart.data;
    let shown = 0, axisCount = 0, trailCount = 0;
    const base = new Float32Array(3), a0 = new Float32Array(3), a1 = new Float32Array(3);
    current.drawnPoints = [];
    for (const key of state.links) {
      const link = data.links[key];
      for (let n = 0; n < link.count; n++) {
        const index = frame * link.count + n, dim = link.visible[index] ? 1 : OCCLUDED;
        M.lifted(link, frame, n, link.trajectory, headPosition, shown * 3, lift);
        for (let k = 0; k < 3; k++) headColor[shown * 3 + k] = link[palette][n * 3 + k] * dim;
        current.drawnPoints.push([key, n]);
        shown++;
        if (state.axes && link.tripod[n] && link.axesOk[index]) {
          M.lifted(link, frame, n, link.origin, base, 0, lift);
          for (let axis = 0; axis < 3; axis++) {
            const d = index * 9 + axis * 3, o = axisCount * 6;
            for (let k = 0; k < 3; k++) {
              axisData.array[o + k] = base[k];
              axisData.array[o + 3 + k] = base[k] + axisLength * link.axes[d + k];
              axisColor.array[o + k] = axisColor.array[o + 3 + k] = M.AXIS_COLORS[axis][k] / 255 * dim;
            }
            axisCount++;
          }
        }
        if (state.trails) {
          for (let t = Math.max(0, frame - TRAIL); t < frame; t++) {
            M.lifted(link, t, n, link.trajectory, a0, 0, lift);
            M.lifted(link, t + 1, n, link.trajectory, a1, 0, lift);
            const o = trailCount * 6, age0 = (t - frame + TRAIL + 1) / (TRAIL + 1), age1 = age0 + 1 / (TRAIL + 1);
            const fade = link.visible[(t + 1) * link.count + n] ? 1 : OCCLUDED;
            for (let k = 0; k < 3; k++) {
              trailData.array[o + k] = a0[k]; trailData.array[o + 3 + k] = a1[k];
              trailColor.array[o + k] = link[palette][n * 3 + k] * age0 * fade;      // fades into the black canvas
              trailColor.array[o + 3 + k] = link[palette][n * 3 + k] * age1 * fade;
            }
            trailCount++;
          }
        }
      }
    }
    heads.geometry.setDrawRange(0, shown);
    heads.geometry.getAttribute('position').needsUpdate = true;
    heads.geometry.getAttribute('color').needsUpdate = true;
    axes.geometry.instanceCount = axisCount;
    trails.geometry.instanceCount = trailCount;
    for (const buffer of [axisData, axisColor, trailData, trailColor]) buffer.needsUpdate = true;
    drawSelection(frame);
    spark(frame);
  }
  const clock = M.clock({video, play: $('#play'), slider: $('#slider'), label: $('#time'), draw, render: stage.render});

  /* ---- click a point to follow it: its whole path, a marker and an inspector card ---- */
  let selected = null;
  function select(key, n) {
    selected = key === null ? null : {key, n};
    const path = current.path, link = selected && current.data.links[key];
    $('#inspector').hidden = !selected;
    $('#hint').hidden = !!selected;
    current.marker.visible = !!selected;
    if (!selected) { path.geometry.instanceCount = 0; clock.redraw(); return; }
    const positions = path.geometry.attributes.instanceStart.data, colors = path.geometry.attributes.instanceColorStart.data;
    const a0 = new Float32Array(3), a1 = new Float32Array(3), frames = current.data.header.frames;
    for (let t = 0; t < frames - 1; t++) {
      M.lifted(link, t, n, link.trajectory, a0, 0, current.lift);
      M.lifted(link, t + 1, n, link.trajectory, a1, 0, current.lift);
      const shade = link.visible[(t + 1) * link.count + n] ? 1 : 0.4;
      for (let k = 0; k < 3; k++) {
        positions.array[t * 6 + k] = a0[k]; positions.array[t * 6 + 3 + k] = a1[k];
        colors.array[t * 6 + k] = colors.array[t * 6 + 3 + k] = shade;
      }
    }
    path.geometry.instanceCount = frames - 1;
    positions.needsUpdate = colors.needsUpdate = true;
    clock.redraw();
  }
  function drawSelection(frame) {
    if (!selected) return;
    const link = current.data.links[selected.key], n = selected.n, index = frame * link.count + n;
    const marker = current.marker.geometry, frames = current.data.header.frames;
    M.lifted(link, frame, n, link.trajectory, marker.getAttribute('position').array, 0, current.lift);
    marker.getAttribute('color').array.set([1, 1, 1]);
    marker.getAttribute('position').needsUpdate = marker.getAttribute('color').needsUpdate = true;
    let visibleFrames = 0;
    for (let t = 0; t < frames; t++) visibleFrames += link.visible[t * link.count + n];
    const vector = link.vector.subarray(n * 16, n * 16 + 16), norm = Math.hypot(...vector) || 1;
    const top = Math.max(...vector.map(v => Math.abs(v / norm)), 1e-6);
    const swatch = `rgb(${[0, 1, 2].map(k => Math.round(255 * link.pca[n * 3 + k])).join(',')})`;
    const residual = link.residual[n];
    $('#inspector').innerHTML = `<h4><span class="swatch" style="background:${LINK_COLORS[selected.key]}"></span>`
      + `${selected.key} · point ${link.pointIds[n]}<button id="unselect" aria-label="Stop following">×</button></h4><dl>`
      + `<dt>Query pixel</dt><dd>${link.queryXy[2 * n].toFixed(1)}, ${link.queryXy[2 * n + 1].toFixed(1)}</dd>`
      + `<dt>This frame</dt><dd>${link.visible[index] ? 'visible' : 'occluded'}</dd>`
      + `<dt>Visible frames</dt><dd>${(100 * visibleFrames / frames).toFixed(0)}%</dd>`
      + `<dt>Rotation since frame 0</dt><dd>${(link.rotation[index] / 100).toFixed(1)}°</dd>`
      + `<dt>Rigid-fit residual</dt><dd>${residual < 0 ? '—' : residual.toFixed(3)}</dd>`
      + `<dt>On a depth edge</dt><dd>${link.onEdge[n] ? 'yes' : 'no'}</dd></dl>`
      + `<div class="vec">${[...vector].map(v => `<i style="height:${Math.max(4, 100 * Math.abs(v / norm) / top)}%;`
        + `background:${v >= 0 ? '#3b63c7' : '#d9822b'}" title="${(v / norm).toFixed(3)}"></i>`).join('')}</div>`
      + `<div class="veclabel"><span>Rigidity vector, 16-D (blue +, orange −)</span>`
      + `<span class="swatch" style="background:${swatch};margin:0" title="Its PCA colour"></span></div>`;
  }
  $('#inspector').addEventListener('click', e => { if (e.target.id === 'unselect') select(null); });
  M.picker(canvas, stage.camera, () => current && current.heads, () => current.pickRadius, index => {
    const [key, n] = current.drawnPoints[index];
    select(key, n);
  });
  addEventListener('keydown', e => { if (e.key === 'Escape' && selected) select(null); });
  $('#layers').addEventListener('change', e => {
    if (e.target.name === 'cloud') paintCloud();
    if (e.target.id === 'pipbox') $('#pip').hidden = !e.target.checked;
    if (e.target.dataset.link && selected && selected.key === e.target.dataset.link && !e.target.checked) select(null);
    clock.redraw();
  });

  /* ---- rigidity strip ---- */
  const metric = $('#metric'), subset = $('#subset'), sparkCanvas = $('#spark');
  const chosen = link => subset.value === 'interior' ? link.interior : link;
  function table(data) {
    const rows = Object.entries(data.header.links).map(([key, link]) => {
      const visible = link.visible_fraction[link.visible_fraction.length - 1];
      const count = subset.value === 'interior' ? `${link.interior.points} of ${link.count}` : link.count;
      return `<tr><td><span class="swatch" style="background:${LINK_COLORS[key]}"></span>${key}</td>`
        + `<td>${count}</td><td>${(100 * visible).toFixed(0)}%</td><td>${link.frame0_reprojection_px.median.toFixed(1)}</td>`
        + INDEX.scores.map(s => `<td>${format(chosen(link).scores[s])}</td>`).join('') + '</tr>';
    }).join('');
    $('#scoretable').innerHTML = `<thead><tr><th>Link</th><th>Points</th><th title="Visible at the last frame">Visible</th>`
      + `<th title="Median distance between each query and MoSE3's frame-0 track, in source pixels">F0 offset px</th>`
      + INDEX.scores.map(s => `<th>${LABELS[s]}</th>`).join('') + `</tr></thead><tbody>${rows}</tbody>`;
  }
  function spark(frame) {
    const ratio = Math.min(window.devicePixelRatio || 1, 2), width = sparkCanvas.clientWidth, height = sparkCanvas.clientHeight;
    sparkCanvas.width = width * ratio; sparkCanvas.height = height * ratio;
    const g = sparkCanvas.getContext('2d');
    g.scale(ratio, ratio);
    const pad = {l: 66, r: 8, t: 10, b: 22}, frames = current.data.header.frames;
    const series = Object.entries(current.data.header.links).map(([key, link]) => [key, chosen(link).series[metric.value]]);
    const values = series.flatMap(([, s]) => s).filter(v => v !== null && Number.isFinite(v));
    const top = Math.max(...values, 1e-9);
    const x = i => pad.l + (i / Math.max(frames - 1, 1)) * (width - pad.l - pad.r);
    const y = v => height - pad.b - (v / top) * (height - pad.t - pad.b);
    const ink = getComputedStyle(document.documentElement).getPropertyValue('--faint').trim() || '#8a9099';
    g.font = '13px ui-monospace, Menlo, monospace'; g.fillStyle = ink; g.strokeStyle = 'rgba(128,132,140,.25)';
    g.beginPath(); g.moveTo(pad.l, y(0)); g.lineTo(width - pad.r, y(0)); g.moveTo(pad.l, y(top)); g.lineTo(width - pad.r, y(top)); g.stroke();
    g.textAlign = 'right'; g.fillText(format(top), pad.l - 6, y(top) + 4); g.fillText('0', pad.l - 6, y(0) + 4);
    g.textAlign = 'left'; g.fillText('frame 0', pad.l, height - 4);
    g.textAlign = 'right'; g.fillText(`frame ${frames - 1}`, width - pad.r, height - 4);
    for (const [key, s] of series) {
      g.strokeStyle = LINK_COLORS[key]; g.lineWidth = 1.6; g.beginPath();
      let pen = false;
      s.forEach((v, i) => {
        if (v === null || !Number.isFinite(v)) { pen = false; return; }
        if (pen) g.lineTo(x(i), y(v)); else { g.moveTo(x(i), y(v)); pen = true; }
      });
      g.stroke();
    }
    g.strokeStyle = ink; g.lineWidth = 1;
    g.beginPath(); g.moveTo(x(frame), pad.t); g.lineTo(x(frame), height - pad.b); g.stroke();
  }
  metric.addEventListener('change', () => clock.redraw());
  subset.addEventListener('change', () => { if (current) table(current.data); clock.redraw(); });

  /* ---- 2D views (the project page's view switcher) ---- */
  function views() {
    const bar = $('#view'), clip = $('#clip');
    const available = INDEX.views.filter(v => entry.views.includes(v.key));
    const show = key => {
      [...bar.children].forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === key)));
      $('#vnote').textContent = INDEX.views.find(v => v.key === key).note;
      clip.src = `views/${name}/${key}.mp4`;
      clip.play().catch(() => {});
    };
    bar.innerHTML = INDEX.views.map(v => `<button data-v="${v.key}" aria-pressed="false"`
      + `${entry.views.includes(v.key) ? '' : ' disabled title="Not rendered yet"'}>${v.label}</button>`).join('');
    bar.addEventListener('click', e => { const b = e.target.closest('button'); if (b && !b.disabled) show(b.dataset.v); });
    clip.poster = `views/${name}/poster.jpg`;
    show(available.some(v => v.key === INDEX.default_view) ? INDEX.default_view : available[0].key);
  }

  /* ---- page ---- */
  async function start() {
    if (!entry || entry.status !== 'complete') { status.textContent = `No replay for ${name}`; return; }
    const position = complete.findIndex(c => c.case === name);
    const step = d => complete[(position + d + complete.length) % complete.length].case;
    $('#prevcase').href = `video.html?case=${encodeURIComponent(step(-1))}`;
    $('#nextcase').href = `video.html?case=${encodeURIComponent(step(1))}`;
    $('#signalslink').href = `signals.html?case=${encodeURIComponent(name)}`;
    document.title = `${name} · MoSE3 replay`;
    $('#title').textContent = name;
    $('#facts').innerHTML = [`<b>${entry.frames}</b> frames at ${entry.fps.toFixed(0)} fps`,
      `<b>${entry.points.link2}</b> Link2 points`, `<b>${entry.points.link5}</b> Link5 points`,
      `${position + 1} of ${complete.length}`].map(s => `<li>${s}</li>`).join('')
      + (flag(entry) ? `<li class="flag">Link5 mask: ${flag(entry)}</li>` : '');
    views();
    try {
      const [header, base64] = await M.loadScript(`cases/${name}.js`, 'MOSE3_SCENE');
      const data = M.decodeScene(header, base64);
      build(data);
      video.src = `videos/${name}.mp4`;
      clock.set(header.frames, header.fps);
      table(data);
      status.hidden = true;
    } catch (error) {
      status.textContent = String(error.message || error);
    }
  }
  clock.start();
  start();
})();
