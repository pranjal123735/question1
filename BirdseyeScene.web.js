/**
 * Web-only Tesla-style bird's-eye scene driven by live detections
 * (bbox + distance_m + track_id). Isolated module — wire from ride fullscreen only.
 */
import { useEffect, useId, useRef } from 'react';
import { Platform, StyleSheet, View } from 'react-native';
import * as THREE from 'three';

function bandColorHex(det) {
  const risk = Number(det?.risk_percent || 0);
  const ttc = Number(det?.ttc_s || 999);
  if (risk >= 75 || ttc < 1.8) return 0xdc2626;
  if (risk >= 40 || ttc < 3.5) return 0xd97706;
  return 0x059669;
}

function isHighRisk(det) {
  const risk = Number(det?.risk_percent || 0);
  const ttc = Number(det?.ttc_s || 999);
  return risk >= 75 || ttc < 1.8;
}

function detToWorldXZ(det, frameW) {
  const [x1, , x2] = det.bbox_xyxy.map(Number);
  const cx = (x1 + x2) / 2;
  const fw = Math.max(320, frameW || 1280);
  const dist = Math.max(0.5, Number(det.distance_m) || 6);
  const nx = cx / fw - 0.5;
  const lateralM = nx * 2 * Math.min(14, dist * 0.48);
  const forwardM = dist;
  return { lateralM, forwardZ: Math.min(55, forwardM) };
}

function trackKey(d) {
  if (d.track_id == null || d.track_id === '') return '';
  return String(d.track_id);
}

function addRangeRings(scene) {
  const group = new THREE.Group();
  group.name = 'rangeRings';
  const radii = [8, 16, 24, 32, 40];
  for (const r of radii) {
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(r - 0.12, r + 0.12, 72),
      new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.12 + (40 - r) * 0.004,
        depthWrite: false,
      })
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.03;
    group.add(ring);
  }
  scene.add(group);
  return group;
}

function makeRadarSweep() {
  const geo = new THREE.RingGeometry(2, 48, 48, 1, 0, Math.PI / 3);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x22d3ee,
    transparent: true,
    opacity: 0.14,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 0.08;
  return mesh;
}

function makeEgoHalo() {
  const torus = new THREE.Mesh(
    new THREE.TorusGeometry(3.2, 0.06, 12, 48),
    new THREE.MeshBasicMaterial({
      color: 0x5eead4,
      transparent: true,
      opacity: 0.35,
      depthWrite: false,
    })
  );
  torus.rotation.x = Math.PI / 2;
  torus.position.y = 0.06;
  return torus;
}

const LABEL_CANVAS_W = 440;
const LABEL_CANVAS_H = 108;

function buildTrackLabelTexture(det) {
  const label = String(det.label ?? 'object').toUpperCase().slice(0, 20);
  const tid = det.track_id != null && det.track_id !== '' ? ` #${det.track_id}` : '';
  const line1 = `${label}${tid}`;
  const dm = Number(det.distance_m);
  const line2 = Number.isFinite(dm) ? `${dm.toFixed(1)} m` : '— m';

  const canvas = document.createElement('canvas');
  canvas.width = LABEL_CANVAS_W;
  canvas.height = LABEL_CANVAS_H;
  const g = canvas.getContext('2d');
  if (!g) return new THREE.CanvasTexture(canvas);

  g.fillStyle = 'rgba(4, 14, 28, 0.92)';
  g.fillRect(0, 0, LABEL_CANVAS_W, LABEL_CANVAS_H);
  g.strokeStyle = 'rgba(34, 211, 238, 0.65)';
  g.lineWidth = 2;
  g.strokeRect(1, 1, LABEL_CANVAS_W - 2, LABEL_CANVAS_H - 2);

  g.fillStyle = '#f0f9ff';
  g.font = 'bold 26px system-ui, Segoe UI, sans-serif';
  g.fillText(line1, 16, 46);

  g.fillStyle = '#5eead4';
  g.font = '600 28px system-ui, Segoe UI, sans-serif';
  g.fillText(line2, 16, 88);

  const tex = new THREE.CanvasTexture(canvas);
  if ('colorSpace' in tex) {
    tex.colorSpace = THREE.SRGBColorSpace;
  }
  tex.needsUpdate = true;
  return tex;
}

function applyLabelSprite(group, det, boxHalfH) {
  const key = `${det.label}|${det.track_id}|${Number(det.distance_m ?? 0).toFixed(2)}`;
  if (group.userData.labelKey === key && group.userData.labelSprite) {
    return;
  }
  group.userData.labelKey = key;

  const tex = buildTrackLabelTexture(det);
  const worldW = 3.85;
  const worldH = worldW * (LABEL_CANVAS_H / LABEL_CANVAS_W);

  if (!group.userData.labelSprite) {
    const mat = new THREE.SpriteMaterial({
      map: tex,
      transparent: true,
      depthTest: true,
      depthWrite: false,
    });
    const sp = new THREE.Sprite(mat);
    sp.position.set(0, boxHalfH + 1.42, 0);
    sp.scale.set(worldW, worldH, 1);
    group.add(sp);
    group.userData.labelSprite = sp;
  } else {
    const sp = group.userData.labelSprite;
    if (sp.material.map) {
      sp.material.map.dispose();
    }
    sp.material.map = tex;
    sp.material.needsUpdate = true;
    sp.position.set(0, boxHalfH + 1.42, 0);
    sp.scale.set(worldW, worldH, 1);
  }
}

function createTrackVoxel(scene, id, d, x, z) {
  const bw = Number(d.bbox_xyxy?.[2]) - Number(d.bbox_xyxy?.[0]);
  const w = 1.1 + Math.min(0.85, Math.abs(bw || 80) / 200);
  const h = 1.35;
  const depth = 1.65;
  const geo = new THREE.BoxGeometry(w, h, depth);
  const col = bandColorHex(d);
  const mat = new THREE.MeshStandardMaterial({
    color: col,
    metalness: 0.22,
    roughness: 0.55,
    emissive: col,
    emissiveIntensity: 0.18,
  });
  const mesh = new THREE.Mesh(geo, mat);
  const edgeGeo = new THREE.EdgesGeometry(geo, 18);
  const edgeMat = new THREE.LineBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.55,
    depthWrite: false,
  });
  const edges = new THREE.LineSegments(edgeGeo, edgeMat);

  const speed = Number(d.speed_kmh || 0);
  const moving = !!d.is_moving && speed > 1;
  let cone = null;
  if (moving) {
    const coneGeo = new THREE.ConeGeometry(0.35, 0.9 + Math.min(speed / 25, 1.2), 8);
    const coneMat = new THREE.MeshBasicMaterial({
      color: col,
      transparent: true,
      opacity: 0.35,
      depthWrite: false,
    });
    cone = new THREE.Mesh(coneGeo, coneMat);
    cone.rotation.x = Math.PI / 2;
    cone.position.set(0, 0.15, -depth * 0.35 - 0.2);
  }

  const hh = h * 0.5;
  const calloutGeom = new THREE.BufferGeometry();
  const callPos = new Float32Array([0, hh, 0, 0, hh + 1.08, 0]);
  calloutGeom.setAttribute('position', new THREE.BufferAttribute(callPos, 3));
  const calloutMat = new THREE.LineBasicMaterial({
    color: 0x7dd3fc,
    transparent: true,
    opacity: 0.55,
    depthWrite: false,
  });
  const labelCallout = new THREE.Line(calloutGeom, calloutMat);

  const group = new THREE.Group();
  group.add(mesh);
  group.add(edges);
  if (cone) group.add(cone);
  group.add(labelCallout);

  const tetherGeom = new THREE.BufferGeometry();
  const tetherPos = new Float32Array(6);
  tetherGeom.setAttribute('position', new THREE.BufferAttribute(tetherPos, 3));
  const tetherMat = new THREE.LineBasicMaterial({
    color: 0x67e8f9,
    transparent: true,
    opacity: 0.22,
    depthWrite: false,
  });
  const tether = new THREE.Line(tetherGeom, tetherMat);
  scene.add(tether);

  group.position.set(x, 0.75, z);
  group.userData = {
    tx: x,
    tz: z,
    mesh,
    edges,
    cone,
    tether,
    bodyGeo: geo,
    edgeGeo,
    tetherGeom,
    calloutGeom,
    labelCallout,
    boxHalfH: hh,
    labelSprite: null,
    labelKey: '',
    lastDet: d,
  };
  scene.add(group);
  applyLabelSprite(group, d, hh);
  return group;
}

function disposeTrackGroup(scene, group) {
  const u = group.userData;
  scene.remove(group);
  if (u.tether) scene.remove(u.tether);
  if (u.labelSprite) {
    if (u.labelSprite.material.map) {
      u.labelSprite.material.map.dispose();
    }
    u.labelSprite.material.dispose();
  }
  u.bodyGeo?.dispose();
  u.edgeGeo?.dispose();
  u.tetherGeom?.dispose();
  u.calloutGeom?.dispose();
  u.mesh?.material?.dispose();
  u.edges?.material?.dispose();
  u.labelCallout?.material?.dispose();
  u.cone?.geometry?.dispose();
  u.cone?.material?.dispose();
}

export default function BirdseyeSceneWeb({ width, height, detections, frameSize, isRunning }) {
  const mountId = `birdseye-${useId().replace(/:/g, '')}`;
  const detectionsRef = useRef(detections);
  const frameSizeRef = useRef(frameSize);
  const isRunningRef = useRef(isRunning);

  detectionsRef.current = detections;
  frameSizeRef.current = frameSize;
  isRunningRef.current = isRunning;

  useEffect(() => {
    if (Platform.OS !== 'web' || typeof document === 'undefined') {
      return undefined;
    }

    let disposed = false;
    let rafWait = 0;
    const ctx = {
      scene: null,
      camera: null,
      renderer: null,
      ego: null,
      egoHalo: null,
      trackGroups: new Map(),
      raf: 0,
      grid: null,
      rangeRings: null,
      sweep: null,
      _cleanupResize: null,
      _resizeObserver: null,
      _el: null,
    };

    const setup = (el) => {
      if (!el || disposed) return;

      const w = Math.max(280, width || 640);
      const h = Math.max(200, height || 480);

      const scene = new THREE.Scene();
      const sky = 0x020617;
      scene.background = new THREE.Color(sky);

      const camera = new THREE.PerspectiveCamera(48, w / Math.max(1, h), 0.5, 220);
      camera.up.set(0, 1, 0);
      camera.position.set(0, 34, -28);
      camera.lookAt(0, 0, 30);

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      renderer.setClearColor(sky, 1);
      renderer.setSize(w, h);
      renderer.setPixelRatio(Math.min(typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1, 2));
      if ('outputColorSpace' in renderer) {
        renderer.outputColorSpace = THREE.SRGBColorSpace;
      }
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.05;

      el.innerHTML = '';
      const canvas = renderer.domElement;
      canvas.style.position = 'absolute';
      canvas.style.left = '0';
      canvas.style.top = '0';
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      canvas.style.display = 'block';
      el.appendChild(canvas);

      const grid = new THREE.GridHelper(120, 48, 0x22d3ee, 0x164e63);
      grid.position.y = 0.01;
      grid.traverse((child) => {
        if (child.material) {
          const m = child.material;
          const mats = Array.isArray(m) ? m : [m];
          mats.forEach((mat) => {
            mat.transparent = true;
            mat.opacity = 0.38;
            mat.depthWrite = false;
          });
        }
      });
      scene.add(grid);

      ctx.rangeRings = addRangeRings(scene);

      const sweep = makeRadarSweep();
      scene.add(sweep);
      ctx.sweep = sweep;

      const egoGeo = new THREE.BoxGeometry(2.2, 0.9, 4.2);
      const egoMat = new THREE.MeshStandardMaterial({
        color: 0x22d3ee,
        metalness: 0.42,
        roughness: 0.38,
        emissive: 0x0e7490,
        emissiveIntensity: 0.62,
      });
      const ego = new THREE.Mesh(egoGeo, egoMat);
      ego.position.set(0, 0.45, 0);
      const egoEdges = new THREE.LineSegments(
        new THREE.EdgesGeometry(egoGeo, 22),
        new THREE.LineBasicMaterial({
          color: 0xe0f2fe,
          transparent: true,
          opacity: 0.65,
          depthWrite: false,
        })
      );
      ego.add(egoEdges);

      const egoHalo = makeEgoHalo();
      scene.add(egoHalo);
      scene.add(ego);

      scene.add(new THREE.AmbientLight(0x7dd3fc, 0.28));
      scene.add(new THREE.HemisphereLight(0x38bdf8, 0x0f172a, 0.45));
      const key = new THREE.DirectionalLight(0xffffff, 0.55);
      key.position.set(-14, 52, 18);
      scene.add(key);
      const rim = new THREE.DirectionalLight(0x67e8f9, 0.28);
      rim.position.set(20, 12, -30);
      scene.add(rim);

      ctx.scene = scene;
      ctx.camera = camera;
      ctx.renderer = renderer;
      ctx.ego = ego;
      ctx.egoHalo = egoHalo;
      ctx.grid = grid;

      const syncTargetsFromDetections = () => {
        const dets = detectionsRef.current || [];
        const fw = frameSizeRef.current?.w || 1280;
        const used = new Set();

        for (const d of dets.slice(0, 24)) {
          const id = trackKey(d);
          if (!id) continue;
          used.add(id);
          const { lateralM, forwardZ } = detToWorldXZ(d, fw);
          const x = lateralM;
          const z = 2.5 + forwardZ * 1.08;

          let group = ctx.trackGroups.get(id);
          if (!group) {
            group = createTrackVoxel(scene, id, d, x, z);
            ctx.trackGroups.set(id, group);
          }
          group.userData.tx = x;
          group.userData.tz = z;
          group.userData.lastDet = d;
          applyLabelSprite(group, d, group.userData.boxHalfH ?? 0.68);
          const { mesh } = group.userData;
          if (mesh?.material) {
            const col = bandColorHex(d);
            mesh.material.color.setHex(col);
            mesh.material.emissive.setHex(col);
            mesh.material.emissiveIntensity = isHighRisk(d) ? 0.35 : 0.18;
          }
        }

        for (const [id, group] of [...ctx.trackGroups.entries()]) {
          if (!used.has(id)) {
            disposeTrackGroup(scene, group);
            ctx.trackGroups.delete(id);
          }
        }
      };

      const animate = () => {
        if (disposed) return;
        ctx.raf = requestAnimationFrame(animate);
        const t = Date.now() * 0.001;
        syncTargetsFromDetections();

        const alpha = isRunningRef.current ? 0.2 : 0.07;
        const egoY = 0.45;
        const egoZ = 0;

        for (const group of ctx.trackGroups.values()) {
          const u = group.userData;
          if (u.tx == null) continue;
          group.position.x += (u.tx - group.position.x) * alpha;
          group.position.z += (u.tz - group.position.z) * alpha;
          group.position.y = 0.75;

          const det = u.lastDet;
          if (det && u.mesh?.material && isHighRisk(det)) {
            const pulse = 0.28 + Math.sin(t * 6) * 0.22;
            u.mesh.material.emissiveIntensity = pulse;
          } else if (u.mesh?.material) {
            u.mesh.material.emissiveIntensity = 0.18;
          }

          if (u.tether && u.tether.geometry?.attributes?.position) {
            const pos = u.tether.geometry.attributes.position.array;
            pos[0] = 0;
            pos[1] = egoY;
            pos[2] = egoZ + 1.2;
            pos[3] = group.position.x;
            pos[4] = group.position.y * 0.4 + 0.2;
            pos[5] = group.position.z;
            u.tether.geometry.attributes.position.needsUpdate = true;
          }
        }

        if (ctx.sweep) {
          ctx.sweep.rotation.y = t * 0.95;
        }
        if (ctx.egoHalo) {
          ctx.egoHalo.rotation.y = t * 0.35;
          const s = 1 + Math.sin(t * 2.2) * 0.04;
          ctx.egoHalo.scale.setScalar(s);
        }
        if (ctx.ego) {
          ctx.ego.rotation.y = Math.sin(Date.now() * 0.0007) * 0.02;
        }
        if (ctx.rangeRings) {
          ctx.rangeRings.rotation.y = t * 0.04;
        }

        renderer.render(scene, camera);
      };
      animate();

      const onResize = () => {
        if (!ctx.renderer || !ctx.camera || !ctx._el) return;
        const rw = Math.max(280, ctx._el.clientWidth || w);
        const rh = Math.max(200, ctx._el.clientHeight || h);
        ctx.camera.aspect = rw / Math.max(1, rh);
        ctx.camera.updateProjectionMatrix();
        ctx.renderer.setSize(rw, rh);
      };
      if (typeof window !== 'undefined') {
        window.addEventListener('resize', onResize);
      }

      ctx._cleanupResize = onResize;
      ctx._el = el;
      onResize();

      if (typeof ResizeObserver !== 'undefined') {
        ctx._resizeObserver = new ResizeObserver(() => onResize());
        ctx._resizeObserver.observe(el);
      }
    };

    const tryStart = () => {
      if (disposed) return;
      const el =
        document.querySelector(`[data-testid="${mountId}"]`) || document.getElementById(mountId);
      if (!el) {
        rafWait = requestAnimationFrame(tryStart);
        return;
      }
      setup(el);
    };
    tryStart();

    return () => {
      disposed = true;
      cancelAnimationFrame(rafWait);
      if (ctx._resizeObserver) {
        ctx._resizeObserver.disconnect();
        ctx._resizeObserver = null;
      }
      if (typeof window !== 'undefined' && ctx._cleanupResize) {
        window.removeEventListener('resize', ctx._cleanupResize);
      }
      cancelAnimationFrame(ctx.raf);
      if (ctx.trackGroups && ctx.scene) {
        for (const g of [...ctx.trackGroups.values()]) {
          disposeTrackGroup(ctx.scene, g);
        }
        ctx.trackGroups.clear();
      }
      if (ctx.renderer) {
        ctx.renderer.dispose();
        if (ctx._el && ctx.renderer.domElement.parentNode === ctx._el) {
          ctx._el.removeChild(ctx.renderer.domElement);
        }
      }
      if (ctx.scene) {
        ctx.scene.traverse((obj) => {
          if (obj.geometry) obj.geometry.dispose();
          if (obj.material) {
            if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
            else obj.material.dispose();
          }
        });
      }
    };
  }, [mountId, width, height]);

  if (Platform.OS !== 'web') {
    return null;
  }

  return (
    <View
      testID={mountId}
      nativeID={mountId}
      collapsable={false}
      style={[StyleSheet.absoluteFillObject, { zIndex: 1 }]}
      pointerEvents="none"
    />
  );
}
