/* Shared by video.html and signals.html (the error-signal replay): packed-array decoding, on-demand scripts,
   the three.js stage (first-camera framing, round points, fat lines) and the page clock
   that keeps the input video on the same time. Coordinates are the first camera's
   (x right, y down, z forward). */
window.MOSE3_3D = (() => {
  'use strict';
  const AXIS_COLORS = [[255, 32, 32], [0, 255, 64], [0, 180, 255]];   // x, y, z as upstream

  function bytes(base64) {
    const text = atob(base64), out = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i);
    return out.buffer;
  }
  // Positions are 16-bit inside [low, high]; ``reader(buffer)`` returns typed views by [offset, length].
  function reader(buffer, low, high) {
    const view = (Type, [offset, length]) => new Type(buffer, offset, length);
    const span = high ? high.map((v, i) => v - low[i]) : null;
    return {
      view,
      positions(ref) {
        const q = view(Uint16Array, ref), out = new Float32Array(q.length);
        for (let i = 0; i < q.length; i++) out[i] = low[i % 3] + (q[i] / 65535) * span[i % 3];
        return out;
      },
      colors: ref => Float32Array.from(view(Uint8Array, ref), v => v / 255),
    };
  }
  function decodeScene(header, base64) {
    const r = reader(bytes(base64), header.low, header.high);
    const data = {header, cloud: {position: r.positions(header.cloud.position), rgb: r.colors(header.cloud.rgb),
      pca: r.colors(header.cloud.pca), count: header.cloud.count}, links: {}};
    for (const [key, link] of Object.entries(header.links)) {
      data.links[key] = {
        count: link.count, trajectory: r.positions(link.trajectory), origin: r.positions(link.origin),
        axes: Float32Array.from(r.view(Int8Array, link.axes), v => v / 127),
        visible: r.view(Uint8Array, link.visible), axesOk: r.view(Uint8Array, link.axes_ok),
        tripod: r.view(Uint8Array, link.tripod), trail: r.colors(link.trail_rgb), pca: r.colors(link.pca_rgb),
        pointIds: r.view(Int32Array, link.point_ids), queryXy: r.view(Float32Array, link.query_xy),
        vector: r.view(Float32Array, link.rigidity_vector), residual: r.view(Float32Array, link.residual),
        rotation: r.view(Uint16Array, link.rotation_centideg), onEdge: r.view(Uint8Array, link.on_edge),
      };
    }
    return data;
  }
  // Each packed file calls ``window[hook](...)``; resolve with those arguments (file://-safe).
  function loadScript(src, hook) {
    return new Promise((resolve, reject) => {
      window[hook] = (...args) => resolve(args);
      const script = document.createElement('script');
      script.src = src;
      script.onerror = () => reject(new Error(`cannot load ${src}`));
      script.onload = () => script.remove();
      document.body.appendChild(script);
    });
  }
  function texture(draw) {
    const c = document.createElement('canvas');
    c.width = c.height = 64;
    draw(c.getContext('2d'));
    return new THREE.CanvasTexture(c);
  }
  const disc = texture(g => { g.fillStyle = '#fff'; g.beginPath(); g.arc(32, 32, 30, 0, 2 * Math.PI); g.fill(); });
  const ring = texture(g => { g.strokeStyle = '#fff'; g.lineWidth = 9; g.beginPath(); g.arc(32, 32, 26, 0, 2 * Math.PI); g.stroke(); });

  function stage(viewer, canvas) {
    const renderer = new THREE.WebGLRenderer({canvas, antialias: true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    const camera = new THREE.PerspectiveCamera(50, 1, 1e-3, 1e4);
    camera.up.set(0, -1, 0);                                 // set before the controls read it
    const controls = new THREE.OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    const materials = [], objects = [];
    const resize = () => {
      const w = viewer.clientWidth, h = viewer.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
      materials.forEach(m => m.resolution.set(w, h));
    };
    new ResizeObserver(resize).observe(viewer);
    const s = {
      renderer, scene, camera, controls,
      lineMaterial(width) {
        const m = new THREE.LineMaterial({linewidth: width, vertexColors: true});
        materials.push(m);
        resize();
        return m;
      },
      fatLines(maxSegments, material) {
        const geometry = new THREE.LineSegmentsGeometry();
        geometry.setPositions(new Float32Array(Math.max(maxSegments, 1) * 6));
        geometry.setColors(new Float32Array(Math.max(maxSegments, 1) * 6));
        geometry.instanceCount = 0;
        const lines = new THREE.LineSegments2(geometry, material);
        lines.frustumCulled = false;
        scene.add(lines);
        objects.push(lines);
        return lines;
      },
      points(count, size, map = disc) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
        const cloud = new THREE.Points(geometry, new THREE.PointsMaterial({size, vertexColors: true, sizeAttenuation: true,
                                                                           map, alphaTest: 0.5}));
        cloud.frustumCulled = false;
        scene.add(cloud);
        objects.push(cloud);
        return cloud;
      },
      clear() {
        for (const o of objects.splice(0)) { scene.remove(o); o.geometry.dispose(); if (o.isPoints) o.material.dispose(); }
      },
      // Start at the first camera looking at the median depth; returns sizes in world units.
      frame(header) {
        const depth = header.median_depth, [gh] = header.grid_hw;
        const fx = header.intrinsics[0][0], fy = header.intrinsics[1][1], halfTan = 0.5 * gh / fy;
        camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(halfTan));
        camera.position.set(0, 0, 0);
        controls.target.set(0, 0, depth);
        camera.near = depth * 1e-3;
        camera.far = depth * 1e3;
        camera.updateProjectionMatrix();
        controls.update();
        // three.js attenuates point size by half the canvas height, not the focal length; divide
        // by tan(fov/2) so a point covers ~60% of its pixel lattice cell at the median depth.
        const world = 0.6 * header.replay_stride * depth / fx;
        return {depth, world, pointSize: factor => factor * world / halfTan};
      },
      render() { controls.update(); renderer.render(scene, camera); },
    };
    resize();
    return s;
  }

  // Point n of ``link`` at frame t, lifted along its outward normal (z axis of its tripod).
  function lifted(link, t, n, source, out, offset, lift) {
    const p = (t * link.count + n) * 3, a = (t * link.count + n) * 9 + 6;
    const ok = link.axesOk[t * link.count + n];
    for (let k = 0; k < 3; k++) out[offset + k] = source[p + k] + (ok ? lift * link.axes[a + k] : 0);
  }

  // The page owns time; the <video> follows it. ``draw(frame)`` runs when the frame changes.
  function clock({video, play, slider, label, draw, render}) {
    const c = {t: 0, playing: true, dragging: false, frames: 0, fps: 1, drawn: -1};
    let last = performance.now();
    function tick(now) {
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      if (c.frames) {
        const duration = c.frames / c.fps;
        if (c.playing && !c.dragging) { c.t += dt; if (c.t >= duration) c.t -= duration; }
        const frame = Math.min(c.frames - 1, Math.floor(c.t * c.fps));
        if (frame !== c.drawn) { draw(frame); c.drawn = frame; }
        if (video.readyState >= 1) {
          // Hold the last frame at the loop end instead of letting an ended <video> restart early.
          const atEnd = video.ended || (video.duration && c.t >= video.duration - 0.06);
          if (c.playing && !c.dragging && !atEnd) {
            if (Math.abs(video.currentTime - c.t) > 0.12) video.currentTime = c.t;
            if (video.paused) video.play().catch(() => {});
          } else {
            if (!video.paused) video.pause();
            if (!atEnd && Math.abs(video.currentTime - c.t) > 0.02) video.currentTime = c.t;
          }
        }
        if (!c.dragging) slider.value = c.t.toFixed(2);
        label.textContent = `${c.t.toFixed(1)} / ${duration.toFixed(1)} s`;
      }
      play.innerHTML = c.playing ? '&#10074;&#10074;' : '&#9654;';
      render();
      requestAnimationFrame(tick);
    }
    play.addEventListener('click', () => { c.playing = !c.playing; });
    slider.addEventListener('input', () => { c.dragging = true; c.t = parseFloat(slider.value); });
    slider.addEventListener('change', () => { c.dragging = false; });
    addEventListener('keydown', e => {
      if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); c.playing = !c.playing; }
    });
    c.set = (frames, fps) => { c.frames = frames; c.fps = fps; c.t = 0; c.drawn = -1; slider.max = (frames / fps).toFixed(2); };
    c.seekFrame = frame => { c.t = (frame + 0.5) / c.fps; c.drawn = -1; };
    c.redraw = () => { c.drawn = -1; };
    c.start = () => requestAnimationFrame(tick);
    return c;
  }

  // Click (not drag) on the canvas: nearest point of ``target`` within ``radius()`` world units.
  function picker(canvas, camera, target, radius, onPick) {
    const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();
    let down = null;
    canvas.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
    canvas.addEventListener('pointerup', e => {
      const object = target();
      if (!object || !down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
      const box = canvas.getBoundingClientRect();
      pointer.set(((e.clientX - box.left) / box.width) * 2 - 1, -((e.clientY - box.top) / box.height) * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      raycaster.params.Points.threshold = radius();
      const hits = raycaster.intersectObject(object);
      if (hits.length) onPick(hits.reduce((a, b) => (b.distanceToRay < a.distanceToRay ? b : a)).index);
    });
  }

  return {AXIS_COLORS, bytes, reader, decodeScene, loadScript, stage, lifted, clock, picker, disc, ring};
})();
