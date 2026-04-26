import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Animated,
  Easing,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  useWindowDimensions,
  View,
} from 'react-native';

const CAPTURE_INTERVAL_MS = 450;
const ALERT_COOLDOWN_MS = 5000;
const RADAR_GHOST_MS = 480;
const DEFAULT_BACKEND_URL =
  process.env.EXPO_PUBLIC_BACKEND_URL ||
  (typeof window !== 'undefined' ? window.location.origin : 'http://127.0.0.1:8001');

/** Web-only glass panel (react-native-web). */
const HUD_SHEET_GLASS =
  Platform.OS === 'web'
    ? {
        backgroundColor: 'rgba(6, 11, 28, 0.88)',
        borderColor: 'rgba(45, 212, 191, 0.42)',
        borderWidth: 1,
        backdropFilter: 'blur(22px)',
        WebkitBackdropFilter: 'blur(22px)',
      }
    : {
        backgroundColor: 'rgba(6, 11, 28, 0.96)',
        borderColor: 'rgba(45, 212, 191, 0.35)',
        borderWidth: 1,
      };

function getRiskBand(det) {
  const risk = Number(det?.risk_percent || 0);
  const ttc = Number(det?.ttc_s || 999);
  if (risk >= 75 || ttc < 1.8) return 'DANGER';
  if (risk >= 40 || ttc < 3.5) return 'CAUTION';
  return 'SAFE';
}

function riskColor(band) {
  if (band === 'DANGER') return '#DC2626';
  if (band === 'CAUTION') return '#D97706';
  return '#059669';
}

function sparkBarColor(score) {
  if (score >= 72) return '#F87171';
  if (score >= 45) return '#FBBF24';
  if (score >= 22) return '#22D3EE';
  return 'rgba(45, 212, 191, 0.4)';
}

function formatSessionClock(totalSec) {
  const s = Math.max(0, Math.floor(totalSec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`;
}

function detectionBand(det) {
  return getRiskBand(det);
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function threatScore(det) {
  const distanceM = Number(det?.distance_m || 120);
  const speedKmh = Number(det?.speed_kmh || 0);
  const risk = Number(det?.risk_percent || 0);
  const moving = !!det?.is_moving;

  const distanceFactor = clamp01((45 - distanceM) / 45);
  const speedFactor = moving ? clamp01(speedKmh / 70) : 0;
  const riskFactor = clamp01(risk / 100);

  const raw = distanceFactor * 0.45 + speedFactor * 0.25 + riskFactor * 0.3;
  return Math.round(raw * 100);
}

function radarPoint(det, frameW, frameH) {
  const [x1, y1, x2] = det.bbox_xyxy;
  const cx = (x1 + x2) / 2;
  const nx = clamp01(cx / Math.max(frameW, 1)); // 0..1 left->right
  const distanceM = Number(det?.distance_m || 120);
  const ny = clamp01(distanceM / 80); // farther => lower on radar
  return { xNorm: nx, yNorm: ny };
}

function mapBoxToOverlay(bbox, frameW, frameH, viewW, viewH) {
  const [x1, y1, x2, y2] = bbox;
  return {
    left: (x1 / frameW) * viewW,
    top: (y1 / frameH) * viewH,
    width: ((x2 - x1) / frameW) * viewW,
    height: ((y2 - y1) / frameH) * viewH,
  };
}

export default function App() {
  const { width: winW, height: winH } = useWindowDimensions();
  const compact = winW < 520;
  const hostRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const timerRef = useRef(0);
  const inFlightRef = useRef(false);
  const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND_URL);
  const [status, setStatus] = useState('Starting camera...');
  const [detections, setDetections] = useState([]);
  const [layout, setLayout] = useState({ w: 0, h: 0 });
  const [error, setError] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [frameSize, setFrameSize] = useState({ w: 1280, h: 720 });
  const [globalBand, setGlobalBand] = useState('SAFE');
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [calibration, setCalibration] = useState({
    focal_like: '900',
    meters_per_px: '0.05',
    default_object_height_m: '1.5',
  });
  const lastAlertAtRef = useRef(0);
  const lastAlertKeyRef = useRef('');
  const scanPhase = useRef(new Animated.Value(0)).current;
  const sessionStartMs = useRef(Date.now());
  const [sessionSec, setSessionSec] = useState(0);
  const lastFrameAtRef = useRef(0);
  const [fps, setFps] = useState(0);
  const [pipelineMs, setPipelineMs] = useState(0);
  /** @type {null | 'main' | 'radar' | 'cal' | 'trip' | 'threat'} */
  const [expandedPanel, setExpandedPanel] = useState(null);
  const [lastTripSnapshot, setLastTripSnapshot] = useState(null);
  const [tripDetail, setTripDetail] = useState(null);
  /** Rolling threat score 0–100 for bottom sparkline (client-side). */
  const [threatHistory, setThreatHistory] = useState([]);
  /** Backend frame quality hints (glare / low light / low contrast). */
  const [frameDiagnostics, setFrameDiagnostics] = useState(null);
  /** Fading radar dots for tracks that just disappeared (reduces empty flicker). */
  const [radarGhosts, setRadarGhosts] = useState([]);
  const prevDetectionsRef = useRef([]);

  const normalizedUrl = useMemo(() => backendUrl.replace(/\/+$/, ''), [backendUrl]);
  const ngrokHeaders = useMemo(
    () =>
      normalizedUrl.includes('ngrok-free.dev') ? { 'ngrok-skip-browser-warning': 'true' } : null,
    [normalizedUrl]
  );
  const withTunnelHeaders = (init = {}) => {
    if (!ngrokHeaders) return init;
    return { ...init, headers: { ...(init.headers || {}), ...ngrokHeaders } };
  };

  useEffect(() => {
    if (typeof document === 'undefined') {
      return undefined;
    }
    let meta = document.querySelector('meta[name="viewport"]');
    if (!meta) {
      meta = document.createElement('meta');
      meta.setAttribute('name', 'viewport');
      document.head.appendChild(meta);
    }
    meta.setAttribute(
      'content',
      'width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover'
    );
    const fontId = 'hud-fonts-ride';
    if (!document.getElementById(fontId)) {
      const link = document.createElement('link');
      link.id = fontId;
      link.rel = 'stylesheet';
      link.href =
        'https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;800&family=Rajdhani:wght@500;600;700&display=swap';
      document.head.appendChild(link);
    }
    return undefined;
  }, []);

  useEffect(() => {
    if (typeof document === 'undefined' || !hostRef.current) {
      return undefined;
    }

    const host = hostRef.current;
    const video = document.createElement('video');
    const canvas = document.createElement('canvas');

    video.setAttribute('playsinline', 'true');
    video.setAttribute('webkit-playsinline', 'true');
    video.muted = true;
    video.autoplay = true;
    video.style.position = 'absolute';
    video.style.left = '0';
    video.style.top = '0';
    video.style.width = '100%';
    video.style.height = '100%';
    video.style.objectFit = 'cover';
    video.style.zIndex = '1';
    host.appendChild(video);
    videoRef.current = video;
    canvasRef.current = canvas;

    let stream;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        video.srcObject = stream;
        await video.play();
        setFrameSize({
          w: Math.max(1, video.videoWidth || 1280),
          h: Math.max(1, video.videoHeight || 720),
        });
        setStatus('Camera ready. Press Start Detection.');
      } catch (e) {
        setStatus('Camera error');
        setError(e.message);
      }
    })();

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
      if (video.parentNode) {
        video.parentNode.removeChild(video);
      }
      videoRef.current = null;
      canvasRef.current = null;
    };
  }, []);

  useEffect(() => {
    const loadCalibration = async () => {
      try {
        const res = await fetch(`${normalizedUrl}/calibration`, withTunnelHeaders());
        if (!res.ok) return;
        const cfg = await res.json();
        setCalibration({
          focal_like: String(cfg.focal_like ?? 900),
          meters_per_px: String(cfg.meters_per_px ?? 0.05),
          default_object_height_m: String(cfg.default_object_height_m ?? 1.5),
        });
      } catch {
        // Ignore calibration prefetch errors; user can still run detection.
      }
    };
    loadCalibration();
  }, [normalizedUrl]);

  useEffect(() => {
    if (expandedPanel !== 'trip') {
      setTripDetail(null);
      return undefined;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${normalizedUrl}/trip/stats`, withTunnelHeaders());
        if (!res.ok || cancelled) return;
        const j = await res.json();
        if (!cancelled) setTripDetail(j);
      } catch {
        if (!cancelled) setTripDetail(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [expandedPanel, normalizedUrl]);

  const analyzeFrame = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2 || inFlightRef.current) {
      return;
    }

    const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
    inFlightRef.current = true;
    try {
      const fw = Math.max(1, video.videoWidth || frameSize.w);
      const fh = Math.max(1, video.videoHeight || frameSize.h);
      setFrameSize({ w: fw, h: fh });
      canvas.width = fw;
      canvas.height = fh;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, fw, fh);

      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.76));
      if (!blob) {
        throw new Error('Unable to capture frame blob');
      }

      const form = new FormData();
      form.append('file', blob, 'frame.jpg');
      const res = await fetch(
        `${normalizedUrl}/analyze-image`,
        withTunnelHeaders({
        method: 'POST',
        body: form,
        })
      );
      if (!res.ok) {
        throw new Error(`Backend HTTP ${res.status}`);
      }
      const body = await res.json();
      const all = body.detections || [];
      setFrameDiagnostics(body.frame_diagnostics ?? null);

      const gNow = Date.now();
      const liveIds = new Set(all.map((d) => d.track_id));
      setRadarGhosts((ghosts) => {
        let next = ghosts.filter((g) => g.expiresAt > gNow && !liveIds.has(g.track_id));
        const prev = prevDetectionsRef.current;
        for (const pd of prev) {
          if (!liveIds.has(pd.track_id)) {
            const p = radarPoint(pd, fw, fh);
            next = next.filter((g) => g.track_id !== pd.track_id);
            next.push({
              track_id: pd.track_id,
              xNorm: p.xNorm,
              yNorm: p.yNorm,
              band: detectionBand(pd),
              expiresAt: gNow + RADAR_GHOST_MS,
            });
          }
        }
        prevDetectionsRef.current = all;
        return next.slice(-24);
      });

      const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
      setPipelineMs((prev) => (prev ? prev * 0.72 + (t1 - t0) * 0.28 : t1 - t0));
      const nowMs = Date.now();
      if (lastFrameAtRef.current > 0) {
        const inst = 1000 / Math.max(1, nowMs - lastFrameAtRef.current);
        const capped = Math.max(0, Math.min(inst, 60));
        setFps((prev) => (prev ? prev * 0.75 + capped * 0.25 : capped));
      }
      lastFrameAtRef.current = nowMs;
      if (body.trip) {
        setLastTripSnapshot(body.trip);
      }
      setDetections(all);
      setStatus(`Detecting... ${all.length || 0} objects`);
      setError(null);

      let band = 'SAFE';
      let topThreat = null;
      let histScore = 0;
      if (all.length) {
        const ranked = [...all].sort((a, b) => threatScore(b) - threatScore(a));
        topThreat = ranked[0];
        band = getRiskBand(topThreat);
        histScore = threatScore(topThreat);
      }
      setGlobalBand(band);
      setThreatHistory((prev) => [...prev, histScore].slice(-56));

      if (!all.length || !topThreat) {
        return;
      }

      if (
        voiceEnabled &&
        typeof window !== 'undefined' &&
        window.speechSynthesis &&
        (band === 'DANGER' || band === 'CAUTION')
      ) {
        const now = Date.now();
        const key = `${topThreat.label}-${topThreat.track_id}-${band}`;
        const canSpeak =
          now - lastAlertAtRef.current > ALERT_COOLDOWN_MS && key !== lastAlertKeyRef.current;
        if (canSpeak) {
          const msg =
            band === 'DANGER'
              ? `Danger. ${topThreat.label} ahead.`
              : `Caution. ${topThreat.label} ahead.`;
          window.speechSynthesis.cancel();
          const utter = new window.SpeechSynthesisUtterance(msg);
          utter.rate = 1.08;
          utter.pitch = 1.0;
          window.speechSynthesis.speak(utter);
          lastAlertAtRef.current = now;
          lastAlertKeyRef.current = key;
        }
      }
    } catch (e) {
      setStatus('Detection paused');
      setError(e.message);
      setIsRunning(false);
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = 0;
      }
    } finally {
      inFlightRef.current = false;
    }
  };

  const toggleRun = () => {
    if (isRunning) {
      setIsRunning(false);
      setStatus('Detection stopped');
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = 0;
      }
      return;
    }

    setStatus('Detection running...');
    setIsRunning(true);
    analyzeFrame();
    timerRef.current = window.setInterval(analyzeFrame, CAPTURE_INTERVAL_MS);
  };

  const resetTripStats = async () => {
    try {
      await fetch(`${normalizedUrl}/trip/reset`, withTunnelHeaders({ method: 'POST' }));
      const res = await fetch(`${normalizedUrl}/trip/stats`, withTunnelHeaders());
      if (res.ok) {
        const j = await res.json();
        setTripDetail(j);
        setLastTripSnapshot({
          frames: j.frames,
          danger_frames: j.danger_frames,
          caution_frames: j.caution_frames,
          safe_frames: j.safe_frames,
          near_miss_count: j.near_miss_count,
          trip_elapsed_s: j.trip_elapsed_s,
        });
      }
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  };

  const saveCalibration = async () => {
    try {
      const payload = {
        focal_like: Number(calibration.focal_like),
        meters_per_px: Number(calibration.meters_per_px),
        default_object_height_m: Number(calibration.default_object_height_m),
      };
      const res = await fetch(
        `${normalizedUrl}/calibration`,
        withTunnelHeaders({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        })
      );
      if (!res.ok) throw new Error(`Calibration HTTP ${res.status}`);
      setStatus('Calibration saved');
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  };

  const top = detections?.[0] || null;
  const radarSheetH = Math.min(compact ? 140 : 160, Math.max(96, winH * 0.22));
  const BOTTOM_HUD_HEIGHT = 118;
  const FAB_COLUMN_W = 78;
  const FAB_SIZE = 54;
  const FAB_GAP = 12;
  const fabBaseBottom = BOTTOM_HUD_HEIGHT + 10;
  const sheetBottom = expandedPanel ? 14 : fabBaseBottom;
  const sheetMaxH = Math.min(560, winH * 0.88);
  const labelMaxW = Math.min(layout.w > 0 ? layout.w * 0.92 : 320, 360);

  const sparkSamples = useMemo(() => {
    const s = threatHistory.slice(-48);
    if (s.length >= 12) return s;
    return [...Array(Math.max(0, 12 - s.length)).fill(0), ...s];
  }, [threatHistory]);

  const dockDots = useMemo(() => {
    return [...detections].sort((a, b) => threatScore(b) - threatScore(a)).slice(0, 12);
  }, [detections]);

  const fabStack = useMemo(() => {
    const row = [
      { id: 'main', a11y: 'Open ride controls', glyph: '☰', tint: 'main' },
      { id: 'radar', a11y: 'Open mini radar', glyph: '◎', tint: 'radar' },
      { id: 'cal', a11y: 'Open calibration', glyph: '⚙', tint: 'cal' },
      { id: 'trip', a11y: 'Open trip stats and near-miss log', glyph: '▣', tint: 'trip' },
    ];
    if (top) {
      row.push({ id: 'threat', a11y: 'Open top threat details', glyph: '◆', tint: 'threat' });
    }
    return row;
  }, [top]);

  useEffect(() => {
    if (expandedPanel === 'threat' && !top) {
      setExpandedPanel(null);
    }
  }, [expandedPanel, top]);

  useEffect(() => {
    if (expandedPanel) {
      return undefined;
    }
    const id = setInterval(() => {
      setSessionSec((Date.now() - sessionStartMs.current) / 1000);
    }, 1000);
    return () => clearInterval(id);
  }, [expandedPanel]);

  useEffect(() => {
    if (expandedPanel) {
      scanPhase.stopAnimation();
      return undefined;
    }
    const loop = Animated.loop(
      Animated.timing(scanPhase, {
        toValue: 1,
        duration: 2800,
        easing: Easing.linear,
        useNativeDriver: true,
      })
    );
    loop.start();
    return () => {
      loop.stop();
      scanPhase.setValue(0);
    };
  }, [expandedPanel, scanPhase]);

  const sweepTranslateY = scanPhase.interpolate({
    inputRange: [0, 1],
    outputRange: [2, 40],
  });
  const idleRingOpacity = scanPhase.interpolate({
    inputRange: [0, 0.5, 1],
    outputRange: [0.12, 0.28, 0.12],
  });

  const nearestDistanceM = useMemo(() => {
    if (!detections.length) return null;
    return Math.min(...detections.map((d) => Number(d.distance_m || 999)));
  }, [detections]);

  const movingCount = useMemo(() => detections.filter((d) => d.is_moving).length, [detections]);

  const avgRisk = useMemo(() => {
    if (!detections.length) return 0;
    return detections.reduce((acc, d) => acc + Number(d.risk_percent || 0), 0) / detections.length;
  }, [detections]);

  const threatLadder = useMemo(() => {
    return [...detections]
      .sort((a, b) => threatScore(b) - threatScore(a))
      .slice(0, 3)
      .map((d) => ({
        id: d.track_id,
        label: d.label,
        dist: Number(d.distance_m || 0),
        ttc: Number(d.ttc_s || 999),
        risk: Number(d.risk_percent || 0),
        band: detectionBand(d),
      }));
  }, [detections]);

  const laneOffsetPct = useMemo(() => {
    if (!top || !frameSize.w) return 50;
    const [x1, , x2] = top.bbox_xyxy || [frameSize.w * 0.45, 0, frameSize.w * 0.55];
    const cx = (Number(x1) + Number(x2)) / 2;
    return Math.max(0, Math.min(100, (cx / Math.max(1, frameSize.w)) * 100));
  }, [top, frameSize.w]);

  const showIdleAtmosphere = !expandedPanel && detections.length === 0;

  return (
    <View
      ref={hostRef}
      collapsable={false}
      style={styles.container}
      onLayout={(e) => {
        const { width, height } = e.nativeEvent.layout;
        setLayout({ w: width, h: height });
      }}
    >
      {!expandedPanel ? (
        <View
          pointerEvents="none"
          style={[styles.hudStateTint, isRunning ? styles.hudStateTintLive : styles.hudStateTintIdle]}
        />
      ) : null}

      {showIdleAtmosphere ? (
        <View style={styles.idleAtmosphere} pointerEvents="none">
          <View style={styles.idleVignette} />
          {[12, 24, 36, 48, 60, 72, 84].map((pct) => (
            <View
              key={`hgrid-${pct}`}
              style={[styles.idleGridH, { top: `${pct}%` }]}
            />
          ))}
          {[14, 28, 42, 56, 70].map((pct) => (
            <View
              key={`vgrid-${pct}`}
              style={[styles.idleGridV, { left: `${pct}%` }]}
            />
          ))}
          <View style={styles.idleCornerTL}>
            <View style={styles.idleCornerH} />
            <View style={styles.idleCornerV} />
          </View>
          <View style={styles.idleCornerTR}>
            <View style={[styles.idleCornerH, { alignSelf: 'flex-end' }]} />
            <View style={[styles.idleCornerV, { alignSelf: 'flex-end' }]} />
          </View>
          <View style={styles.idleCornerBL}>
            <View style={{ flexDirection: 'row', alignItems: 'flex-end' }}>
              <View style={styles.idleCornerV} />
              <View style={[styles.idleCornerH, { marginLeft: -2 }]} />
            </View>
          </View>
          <View style={styles.idleCornerBR}>
            <View style={{ flexDirection: 'row', alignItems: 'flex-end', justifyContent: 'flex-end' }}>
              <View style={styles.idleCornerV} />
              <View style={[styles.idleCornerH, { marginRight: -2 }]} />
            </View>
          </View>
          <View style={styles.idleMiddle}>
            <Animated.View
              style={[styles.idleCenterRing, { opacity: idleRingOpacity }]}
              pointerEvents="none"
            />
            <View style={styles.idleCenterBlock} pointerEvents="none">
              <Text style={styles.idleTagline}>
                {isRunning ? 'ACTIVE SCAN' : 'OPTICAL ARRAY'}
              </Text>
              <Text style={styles.idleTaglineDim}>
                {isRunning ? 'AWAITING TARGETS' : 'SYSTEM STANDBY'}
              </Text>
              <Text style={styles.idleStatusEcho} numberOfLines={2}>
                {status}
              </Text>
            </View>
          </View>
        </View>
      ) : null}

      {!expandedPanel ? (
        <View style={[styles.hudTelemetryBar, compact && styles.hudTelemetryBarCompact]} pointerEvents="none">
          <Text style={styles.hudTelLeft}>{isRunning ? `LIVE ${fps.toFixed(1)} FPS` : 'STANDBY'}</Text>
          <Text style={[styles.hudTelMid, compact && styles.hudTelMidCompact]}>
            {compact
              ? `${globalBand} · ${movingCount}/${detections.length} MOV`
              : `${movingCount}/${detections.length} MOV · ${globalBand} · RISK ${avgRisk.toFixed(0)}%`}
          </Text>
          <Text style={[styles.hudTelRight, compact && styles.hudTelRightCompact]}>
            {nearestDistanceM == null
              ? compact
                ? `${pipelineMs.toFixed(0)}ms`
                : `LAT ${pipelineMs.toFixed(0)}ms`
              : compact
                ? `${nearestDistanceM.toFixed(1)}m`
                : `${nearestDistanceM.toFixed(1)}m · ${pipelineMs.toFixed(0)}ms`}
          </Text>
        </View>
      ) : null}

      {!expandedPanel ? (
        <View style={[styles.hudThreatLadder, compact && styles.hudThreatLadderCompact]} pointerEvents="none">
          <Text style={styles.hudPanelTitle}>THREAT LADDER</Text>
          {threatLadder.length === 0 ? (
            <Text style={styles.hudPanelEmpty}>No active tracks</Text>
          ) : (
            threatLadder.slice(0, compact ? 2 : 3).map((t, idx) => (
              <View key={`lad-${t.id}-${idx}`} style={styles.hudLadderRow}>
                <View style={[styles.hudLadderPill, { backgroundColor: riskColor(t.band) }]} />
                <Text style={[styles.hudLadderText, compact && styles.hudLadderTextCompact]} numberOfLines={1}>
                  {idx + 1}. {t.label} · {t.dist.toFixed(1)}m
                  {!compact ? ` · ${t.ttc >= 900 ? 'TTC N/A' : `TTC ${t.ttc.toFixed(1)}s`}` : ''}
                </Text>
              </View>
            ))
          )}
        </View>
      ) : null}

      {!expandedPanel ? (
        <View style={[styles.hudGuidanceBar, compact && styles.hudGuidanceBarCompact]} pointerEvents="none">
          <Text style={styles.hudPanelTitle}>LANE GUIDANCE</Text>
          <View style={styles.hudGuidanceTrack}>
            <View style={styles.hudGuidanceCenter} />
            <View style={[styles.hudGuidanceMarker, { left: `${laneOffsetPct}%` }]} />
          </View>
          <Text style={[styles.hudGuidanceMeta, compact && styles.hudGuidanceMetaCompact]} numberOfLines={1}>
            {top ? `${top.label} offset ${(laneOffsetPct - 50).toFixed(0)}%` : 'No target lock'}
          </Text>
        </View>
      ) : null}

      {!expandedPanel ? (
        <View style={[styles.hudSystemCard, compact && styles.hudSystemCardCompact]} pointerEvents="none">
          <Text style={styles.hudPanelTitle}>SYSTEM</Text>
          <Text style={[styles.hudSystemLine, compact && styles.hudSystemLineCompact]}>
            Frame {Math.round(frameSize.w)}x{Math.round(frameSize.h)}
          </Text>
          <Text style={[styles.hudSystemLine, compact && styles.hudSystemLineCompact]}>
            Latency {pipelineMs.toFixed(0)}ms
          </Text>
          <Text style={[styles.hudSystemLine, compact && styles.hudSystemLineCompact]}>
            Voice {voiceEnabled ? 'ON' : 'OFF'}
          </Text>
          {frameDiagnostics ? (
            <Text style={[styles.hudSystemLine, compact && styles.hudSystemLineCompact]} numberOfLines={1}>
              Scene {frameDiagnostics.quality_hint}
              {!compact
                ? ` · br ${Math.round((frameDiagnostics.brightness_01 || 0) * 100)}%`
                : ''}
            </Text>
          ) : null}
          {!compact ? (
            <Text style={styles.hudSystemLine}>Backend {backendUrl.replace(/^https?:\/\//, '')}</Text>
          ) : null}
        </View>
      ) : null}

      <View style={styles.boxLayer} pointerEvents="none">
        {detections.map((d, idx) => {
          const box = mapBoxToOverlay(d.bbox_xyxy, frameSize.w, frameSize.h, layout.w, layout.h);
          const band = detectionBand(d);
          const color = riskColor(band);
          const score = threatScore(d);
          return (
            <View
              key={`${d.track_id}-${idx}`}
              style={[
                styles.detectionBox,
                {
                  left: box.left,
                  top: box.top,
                  width: box.width,
                  height: box.height,
                  borderColor: color,
                  shadowColor: color,
                },
              ]}
            >
              <View style={[styles.detectionLabel, { backgroundColor: color, maxWidth: labelMaxW }]}>
                <Text
                  style={[styles.detectionLabelText, compact && styles.detectionLabelTextCompact]}
                  numberOfLines={compact ? 3 : 2}
                >
                  {d.label} #{d.track_id} | {Number(d.distance_m).toFixed(1)}m |{' '}
                  {d.is_moving ? `${Number(d.speed_kmh).toFixed(0)}km/h` : 'static'} |{' '}
                  {Math.round(Number(d.risk_percent))}% | {band} | T{score} | A{' '}
                  {Number(d.track_age_s ?? 0).toFixed(1)}s
                </Text>
              </View>
            </View>
          );
        })}
      </View>

      {!expandedPanel ? (
        <View
          pointerEvents="none"
          style={[
            styles.hudBottomDock,
            compact && styles.hudBottomDockCompact,
            { height: BOTTOM_HUD_HEIGHT, paddingRight: FAB_COLUMN_W },
          ]}
        >
          <View style={styles.hudDockTickBar}>
            {Array.from({ length: 36 }).map((_, i) => (
              <View key={`tick-${i}`} style={styles.hudDockTickCell}>
                <View style={[styles.hudDockTick, i % 6 === 0 ? styles.hudDockTickMajor : null]} />
              </View>
            ))}
          </View>
          <View style={styles.hudDockMetricRow}>
            <Text style={styles.hudDockMetricLeft}>
              MODE {isRunning ? 'TRACK' : 'STBY'} · SESSION {formatSessionClock(sessionSec)}
            </Text>
            <Text style={styles.hudDockMetricRight}>
              OBJ {detections.length} · MOV {movingCount} · FPS {fps.toFixed(1)}
            </Text>
          </View>
          <View style={styles.hudDockSparkRow}>
            {sparkSamples.map((v, i) => (
              <View key={`spark-${i}`} style={styles.hudSparkCell}>
                <View
                  style={[
                    styles.hudSparkBar,
                    { height: Math.max(2, (v / 100) * 14), backgroundColor: sparkBarColor(v) },
                  ]}
                />
              </View>
            ))}
          </View>
          <View style={styles.hudDockRadar}>
            <View style={styles.hudDockRadarLine} />
            {dockDots.length === 0 ? (
              <Animated.View
                style={[
                  styles.hudDockSweep,
                  { transform: [{ translateY: sweepTranslateY }] },
                ]}
              />
            ) : null}
            {dockDots.map((d, idx) => {
              const p = radarPoint(d, frameSize.w, frameSize.h);
              const left = p.xNorm * 100;
              const topPct = p.yNorm * 100;
              const band = detectionBand(d);
              return (
                <View
                  key={`dock-${d.track_id}-${idx}`}
                  style={[
                    styles.hudDockDot,
                    {
                      left: `${left}%`,
                      top: `${topPct}%`,
                      backgroundColor: riskColor(band),
                    },
                  ]}
                />
              );
            })}
            {radarGhosts.map((g) => {
              const left = g.xNorm * 100;
              const topPct = g.yNorm * 100;
              const fade = Math.max(0.08, Math.min(0.45, (g.expiresAt - Date.now()) / RADAR_GHOST_MS * 0.45));
              return (
                <View
                  key={`dock-ghost-${g.track_id}`}
                  style={[
                    styles.hudDockDot,
                    {
                      left: `${left}%`,
                      top: `${topPct}%`,
                      backgroundColor: riskColor(g.band),
                      opacity: fade,
                    },
                  ]}
                />
              );
            })}
          </View>
          <View style={styles.hudDockFooter}>
            <Text style={styles.hudDockTitle}>LANE · RANGE</Text>
            <Text style={styles.hudDockMeta}>
              {globalBand} · {detections.length} track{detections.length === 1 ? '' : 's'}
            </Text>
          </View>
        </View>
      ) : null}

      {!expandedPanel ? (
        <View
          pointerEvents="none"
          style={[
            styles.hudFabRail,
            {
              height: fabStack.length * (FAB_SIZE + FAB_GAP) - FAB_GAP + 36,
              bottom: fabBaseBottom - 10,
            },
          ]}
        />
      ) : null}

      {!expandedPanel
        ? fabStack.map((fab, i) => (
            <Pressable
              key={fab.id}
              accessibilityRole="button"
              accessibilityLabel={fab.a11y}
              onPress={() => setExpandedPanel(fab.id)}
              style={[
                styles.panelFab,
                fab.tint === 'main' && styles.panelFabMain,
                fab.tint === 'radar' && styles.panelFabRadar,
                fab.tint === 'cal' && styles.panelFabCal,
                fab.tint === 'trip' && styles.panelFabTrip,
                fab.tint === 'threat' && styles.panelFabThreat,
                { bottom: fabBaseBottom + i * (FAB_SIZE + FAB_GAP) },
              ]}
            >
              <Text style={styles.panelFabGlyph}>{fab.glyph}</Text>
              {fab.id === 'radar' && detections.length > 0 ? (
                <View style={styles.panelFabBadge}>
                  <Text style={styles.panelFabBadgeText}>{Math.min(99, detections.length)}</Text>
                </View>
              ) : null}
              {fab.id === 'main' && error ? (
                <View style={styles.panelFabBadge}>
                  <Text style={styles.panelFabBadgeText}>!</Text>
                </View>
              ) : null}
              {fab.id === 'main' && globalBand === 'DANGER' && !error ? <View style={styles.panelFabDot} /> : null}
              {fab.id === 'trip' && (lastTripSnapshot?.near_miss_count ?? 0) > 0 ? (
                <View style={styles.panelFabBadge}>
                  <Text style={styles.panelFabBadgeText}>
                    {Math.min(99, lastTripSnapshot.near_miss_count)}
                  </Text>
                </View>
              ) : null}
              {fab.id === 'threat' && globalBand === 'DANGER' ? <View style={styles.panelFabDot} /> : null}
            </Pressable>
          ))
        : null}

      {expandedPanel ? (
        <>
          <Pressable
            style={styles.panelFabBackdrop}
            onPress={() => setExpandedPanel(null)}
            accessibilityRole="button"
            accessibilityLabel="Close panel"
          />
          <View style={[styles.panelFabSheet, HUD_SHEET_GLASS, { bottom: sheetBottom, maxHeight: sheetMaxH }]}>
            <View style={styles.hudSheetAccentTop} pointerEvents="none" />
            <View style={styles.panelFabSheetHeader}>
              <Text
                style={[
                  styles.hudSheetTitle,
                  styles.panelFabSheetTitle,
                  compact && styles.sectionTitleSmall,
                ]}
              >
                {expandedPanel === 'main' && 'Ride controls'}
                {expandedPanel === 'radar' && 'Mini radar'}
                {expandedPanel === 'cal' && 'Calibration'}
                {expandedPanel === 'trip' && 'Trip · Phase 4'}
                {expandedPanel === 'threat' && top && 'Top threat'}
              </Text>
              <Pressable onPress={() => setExpandedPanel(null)} hitSlop={12} style={styles.panelFabCloseHit}>
                <Text style={styles.panelFabClose}>CLOSE</Text>
              </Pressable>
            </View>

            {expandedPanel === 'main' ? (
              <ScrollView
                keyboardShouldPersistTaps="handled"
                showsVerticalScrollIndicator
                style={[styles.mainSheetScroll, { maxHeight: sheetMaxH - 52 }]}
                contentContainerStyle={styles.mainSheetScrollContent}
              >
                <TextInput
                  style={[styles.input, compact && styles.inputCompact]}
                  value={backendUrl}
                  onChangeText={setBackendUrl}
                  autoCapitalize="none"
                  autoCorrect={false}
                  placeholder="http://127.0.0.1:8001"
                  placeholderTextColor="#A5B4FC"
                />
                {compact ? (
                  <View style={styles.quickRow}>
                    <Pressable style={[styles.button, styles.buttonHalf, styles.buttonHalfLeft]} onPress={toggleRun}>
                      <Text style={[styles.buttonText, styles.buttonTextSmall]}>
                        {isRunning ? 'Stop' : 'Start'}
                      </Text>
                    </Pressable>
                    <Pressable
                      style={[styles.buttonAlt, styles.buttonHalf]}
                      onPress={() => setVoiceEnabled((v) => !v)}
                    >
                      <Text style={[styles.buttonText, styles.buttonTextSmall]}>
                        Voice: {voiceEnabled ? 'ON' : 'OFF'}
                      </Text>
                    </Pressable>
                  </View>
                ) : (
                  <>
                    <Pressable style={styles.button} onPress={toggleRun}>
                      <Text style={styles.buttonText}>
                        {isRunning ? 'Stop Detection' : 'Start Detection'}
                      </Text>
                    </Pressable>
                    <Pressable style={styles.buttonAlt} onPress={() => setVoiceEnabled((v) => !v)}>
                      <Text style={styles.buttonText}>Voice Alerts: {voiceEnabled ? 'ON' : 'OFF'}</Text>
                    </Pressable>
                  </>
                )}
                <View style={[styles.riskBadge, { backgroundColor: riskColor(globalBand) }]}>
                  <Text style={[styles.riskBadgeText, compact && styles.riskBadgeTextSmall]}>
                    Ride Risk: {globalBand}
                  </Text>
                </View>
                <Text style={[styles.status, compact && styles.statusSmall]}>{status}</Text>
                {error ? (
                  <Text style={[styles.error, compact && styles.statusSmall]}>Error: {error}</Text>
                ) : null}
              </ScrollView>
            ) : null}

            {expandedPanel === 'radar' ? (
              <>
                <View style={[styles.radarBody, { height: radarSheetH }]}>
                  <View style={styles.radarSweepLine} />
                  {detections.slice(0, 12).map((d, idx) => {
                    const p = radarPoint(d, frameSize.w, frameSize.h);
                    const left = p.xNorm * 100;
                    const topPct = p.yNorm * 100;
                    const band = detectionBand(d);
                    return (
                      <View
                        key={`radar-${d.track_id}-${idx}`}
                        style={[
                          styles.radarDot,
                          compact && styles.radarDotSmall,
                          {
                            left: `${left}%`,
                            top: `${topPct}%`,
                            backgroundColor: riskColor(band),
                          },
                        ]}
                      />
                    );
                  })}
                  {radarGhosts.map((g) => {
                    const left = g.xNorm * 100;
                    const topPct = g.yNorm * 100;
                    const fade = Math.max(0.08, Math.min(0.45, (g.expiresAt - Date.now()) / RADAR_GHOST_MS * 0.45));
                    return (
                      <View
                        key={`radar-ghost-${g.track_id}`}
                        style={[
                          styles.radarDot,
                          compact && styles.radarDotSmall,
                          {
                            left: `${left}%`,
                            top: `${topPct}%`,
                            backgroundColor: riskColor(g.band),
                            opacity: fade,
                          },
                        ]}
                      />
                    );
                  })}
                </View>
                <Text style={[styles.radarHint, compact && styles.radarHintSmall]}>
                  Bottom near · Top far · Left/right lane
                </Text>
              </>
            ) : null}

            {expandedPanel === 'cal' ? (
              <ScrollView
                keyboardShouldPersistTaps="handled"
                showsVerticalScrollIndicator={false}
                style={[styles.panelFabSheetScroll, { maxHeight: Math.min(360, winH * 0.42) }]}
              >
                <View style={[styles.calibrationCard, styles.calibrationCardInSheet]}>
                  <View style={[styles.calibrationRow, compact && styles.calibrationRowStack]}>
                    <Text style={[styles.calibrationLabel, compact && styles.calibrationLabelStack]}>
                      Focal-like
                    </Text>
                    <TextInput
                      style={[styles.calibrationInput, compact && styles.calibrationInputFull]}
                      value={calibration.focal_like}
                      onChangeText={(v) => setCalibration((c) => ({ ...c, focal_like: v }))}
                      keyboardType="numeric"
                    />
                  </View>
                  <View style={[styles.calibrationRow, compact && styles.calibrationRowStack]}>
                    <Text style={[styles.calibrationLabel, compact && styles.calibrationLabelStack]}>
                      Meters / pixel
                    </Text>
                    <TextInput
                      style={[styles.calibrationInput, compact && styles.calibrationInputFull]}
                      value={calibration.meters_per_px}
                      onChangeText={(v) => setCalibration((c) => ({ ...c, meters_per_px: v }))}
                      keyboardType="numeric"
                    />
                  </View>
                  <View style={[styles.calibrationRow, compact && styles.calibrationRowStack]}>
                    <Text style={[styles.calibrationLabel, compact && styles.calibrationLabelStack]}>
                      Default height (m)
                    </Text>
                    <TextInput
                      style={[styles.calibrationInput, compact && styles.calibrationInputFull]}
                      value={calibration.default_object_height_m}
                      onChangeText={(v) => setCalibration((c) => ({ ...c, default_object_height_m: v }))}
                      keyboardType="numeric"
                    />
                  </View>
                  <Pressable style={styles.saveCalButton} onPress={saveCalibration}>
                    <Text style={styles.saveCalButtonText}>Save calibration</Text>
                  </Pressable>
                </View>
              </ScrollView>
            ) : null}

            {expandedPanel === 'threat' && top ? (
              <View style={styles.topCard}>
                <Text style={[styles.topTitle, compact && styles.sectionTitleSmall]}>
                  {top.label} #{top.track_id}
                </Text>
                <Text style={[styles.topLine, compact && styles.topLineSmall]}>
                  Distance: {Number(top.distance_m).toFixed(2)} m
                </Text>
                <Text style={[styles.topLine, compact && styles.topLineSmall]}>
                  Speed: {top.is_moving ? `${Number(top.speed_kmh).toFixed(1)} km/h` : 'static'}
                </Text>
                <Text style={[styles.topLine, compact && styles.topLineSmall]}>
                  Threat: {threatScore(top)}/100
                </Text>
                <Text style={[styles.topLine, compact && styles.topLineSmall]}>
                  TTC: {Number(top.ttc_s) >= 900 ? 'N/A' : `${Number(top.ttc_s).toFixed(2)} s`}
                </Text>
                <Text style={[styles.topLine, compact && styles.topLineSmall]}>
                  Risk: {Math.round(Number(top.risk_percent))}%
                </Text>
              </View>
            ) : null}

            {expandedPanel === 'trip' ? (
              <ScrollView
                keyboardShouldPersistTaps="handled"
                showsVerticalScrollIndicator
                style={[styles.tripSheetScroll, { maxHeight: sheetMaxH - 52 }]}
                contentContainerStyle={styles.tripSheetScrollContent}
              >
                <Text style={styles.tripIntro}>
                  Session counters and a debounced near-miss log (moving vehicles, caution/danger
                  only). Stored in memory on the server until you reset.
                </Text>
                {!tripDetail && !lastTripSnapshot ? (
                  <Text style={[styles.statusSmall, styles.tripMuted]}>Start detection to load trip data.</Text>
                ) : null}
                {tripDetail || lastTripSnapshot ? (
                  <>
                    <View style={styles.tripStatsBlock}>
                      <Text style={styles.tripStatLine}>
                        Trip time:{' '}
                        {(tripDetail || lastTripSnapshot).trip_elapsed_s != null
                          ? `${Number((tripDetail || lastTripSnapshot).trip_elapsed_s).toFixed(0)} s`
                          : '—'}
                      </Text>
                      <Text style={styles.tripStatLine}>
                        Frames: {(tripDetail || lastTripSnapshot).frames ?? 0}
                      </Text>
                      <Text style={styles.tripStatLine}>
                        Danger / caution / safe frames:{' '}
                        {(tripDetail || lastTripSnapshot).danger_frames ?? 0} /{' '}
                        {(tripDetail || lastTripSnapshot).caution_frames ?? 0} /{' '}
                        {(tripDetail || lastTripSnapshot).safe_frames ?? 0}
                      </Text>
                      <Text style={styles.tripStatLine}>
                        Logged events: {(tripDetail || lastTripSnapshot).near_miss_count ?? 0}
                      </Text>
                    </View>
                    <Pressable style={styles.tripResetBtn} onPress={resetTripStats}>
                      <Text style={styles.tripResetBtnText}>Reset trip</Text>
                    </Pressable>
                    <Text style={styles.tripSectionTitle}>Recent events</Text>
                    {!tripDetail ? (
                      <Text style={[styles.statusSmall, styles.tripMuted]}>Loading…</Text>
                    ) : (tripDetail.events || []).length === 0 ? (
                      <Text style={[styles.statusSmall, styles.tripMuted]}>
                        No caution or danger vehicle events yet.
                      </Text>
                    ) : (
                      (tripDetail.events || []).map((ev, idx) => (
                        <View key={`${ev.ts_s}-${ev.track_id}-${idx}`} style={styles.tripEventCard}>
                          <Text style={styles.tripEventTitle}>
                            {ev.severity} · {ev.label} #{ev.track_id}
                          </Text>
                          <Text style={styles.tripEventMeta}>
                            {new Date(ev.ts_s * 1000).toLocaleTimeString()} ·{' '}
                            {Number(ev.distance_m).toFixed(1)} m · risk {Number(ev.risk_percent).toFixed(0)}%
                            {ev.ttc_s != null ? ` · TTC ${Number(ev.ttc_s).toFixed(1)} s` : ''}
                          </Text>
                        </View>
                      ))
                    )}
                  </>
                ) : null}
              </ScrollView>
            ) : null}
          </View>
        </>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    width: '100%',
    minHeight: '100%',
    backgroundColor: '#030712',
    position: 'relative',
  },
  idleAtmosphere: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 9,
  },
  idleVignette: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.28)',
  },
  idleGridH: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: 'rgba(34, 211, 238, 0.07)',
  },
  idleGridV: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 1,
    backgroundColor: 'rgba(167, 139, 250, 0.06)',
  },
  idleCornerTL: {
    position: 'absolute',
    top: '7%',
    left: '4%',
  },
  idleCornerTR: {
    position: 'absolute',
    top: '7%',
    right: '4%',
    alignItems: 'flex-end',
  },
  idleCornerBL: {
    position: 'absolute',
    bottom: '22%',
    left: '4%',
  },
  idleCornerBR: {
    position: 'absolute',
    bottom: '22%',
    right: '4%',
    alignItems: 'flex-end',
  },
  idleCornerH: {
    width: 36,
    height: 2,
    backgroundColor: 'rgba(34, 211, 238, 0.65)',
    borderRadius: 1,
  },
  idleCornerV: {
    width: 2,
    height: 36,
    marginTop: -2,
    backgroundColor: 'rgba(34, 211, 238, 0.65)',
    borderRadius: 1,
  },
  idleMiddle: {
    ...StyleSheet.absoluteFillObject,
    bottom: 128,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  idleCenterRing: {
    width: 132,
    height: 132,
    borderRadius: 66,
    borderWidth: 1,
    borderColor: 'rgba(45, 212, 191, 0.35)',
    marginBottom: 14,
    shadowColor: '#22D3EE',
    shadowOpacity: 0.2,
    shadowRadius: 20,
    shadowOffset: { width: 0, height: 0 },
  },
  idleCenterBlock: {
    alignItems: 'center',
    maxWidth: 320,
    backgroundColor: 'rgba(3, 7, 18, 0.55)',
    borderWidth: 1,
    borderColor: 'rgba(34, 211, 238, 0.22)',
    borderRadius: 4,
    paddingVertical: 14,
    paddingHorizontal: 18,
  },
  idleTagline: {
    color: '#E0F2FE',
    fontSize: 15,
    fontWeight: '800',
    letterSpacing: 4,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
  },
  idleTaglineDim: {
    color: '#67E8F9',
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 3,
    marginTop: 6,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
  },
  idleStatusEcho: {
    color: '#A5F3FC',
    fontSize: 13,
    marginTop: 12,
    textAlign: 'center',
    lineHeight: 18,
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
    fontWeight: '600',
    opacity: 0.9,
  },
  boxLayer: { ...StyleSheet.absoluteFillObject, zIndex: 16 },
  detectionBox: {
    position: 'absolute',
    borderWidth: 1,
    borderRadius: 2,
    borderColor: '#22D3EE',
    backgroundColor: 'rgba(2, 6, 23, 0.12)',
    shadowColor: '#22D3EE',
    shadowOpacity: 0.35,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 0 },
  },
  detectionLabel: {
    position: 'absolute',
    left: -2,
    top: -26,
    backgroundColor: 'rgba(8, 15, 35, 0.92)',
    borderRadius: 2,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderLeftWidth: 3,
    borderLeftColor: '#2DD4BF',
    borderTopWidth: 1,
    borderRightWidth: 1,
    borderBottomWidth: 1,
    borderTopColor: 'rgba(45, 212, 191, 0.35)',
    borderRightColor: 'rgba(45, 212, 191, 0.2)',
    borderBottomColor: 'rgba(45, 212, 191, 0.2)',
  },
  detectionLabelText: {
    color: '#F0FDFA',
    fontSize: 11,
    fontWeight: '700',
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
    letterSpacing: 0.4,
  },
  detectionLabelTextCompact: { fontSize: 10, lineHeight: 13 },
  mainSheetScroll: {
    width: '100%',
  },
  mainSheetScrollContent: {
    paddingBottom: 12,
  },
  quickRow: {
    flexDirection: 'row',
    alignItems: 'stretch',
    marginBottom: 6,
  },
  buttonHalf: {
    flex: 1,
    marginBottom: 0,
    marginHorizontal: 0,
    minHeight: 44,
    justifyContent: 'center',
  },
  buttonHalfLeft: {
    marginRight: 8,
  },
  buttonTextSmall: { fontSize: 13 },
  inputCompact: {
    paddingVertical: 8,
    marginBottom: 6,
    fontSize: 13,
  },
  riskBadgeTextSmall: { fontSize: 14 },
  sectionTitleSmall: { fontSize: 13, marginBottom: 4 },
  statusSmall: { fontSize: 12 },
  topLineSmall: { fontSize: 11 },
  radarDotSmall: {
    width: 8,
    height: 8,
    marginLeft: -4,
    marginTop: -4,
  },
  radarHintSmall: { fontSize: 10, marginTop: 4 },
  calibrationRowStack: {
    flexDirection: 'column',
    alignItems: 'stretch',
    marginBottom: 8,
  },
  calibrationLabelStack: {
    marginRight: 0,
    marginBottom: 4,
    flex: 0,
  },
  calibrationInputFull: {
    width: '100%',
    alignSelf: 'stretch',
    textAlign: 'left',
  },
  saveCalButton: {
    alignItems: 'center',
    paddingVertical: 12,
    borderRadius: 4,
    backgroundColor: 'rgba(30, 41, 59, 0.95)',
    marginTop: 4,
    borderWidth: 1,
    borderColor: 'rgba(45, 212, 191, 0.45)',
  },
  saveCalButtonText: {
    color: '#CCFBF1',
    fontWeight: '800',
    fontSize: 14,
    letterSpacing: 1,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
  },
  input: {
    borderWidth: 1,
    borderColor: 'rgba(34, 211, 238, 0.35)',
    borderRadius: 4,
    color: '#ECFEFF',
    paddingHorizontal: 12,
    paddingVertical: 10,
    marginBottom: 8,
    backgroundColor: 'rgba(15, 23, 42, 0.85)',
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
    fontSize: 15,
    fontWeight: '600',
  },
  button: {
    alignItems: 'center',
    paddingVertical: 12,
    borderRadius: 4,
    backgroundColor: 'rgba(8, 145, 178, 0.55)',
    marginBottom: 8,
    minHeight: 46,
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(34, 211, 238, 0.65)',
    shadowColor: '#22D3EE',
    shadowOpacity: 0.25,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 0 },
  },
  buttonAlt: {
    alignItems: 'center',
    paddingVertical: 12,
    borderRadius: 4,
    backgroundColor: 'rgba(30, 41, 59, 0.9)',
    marginBottom: 8,
    minHeight: 46,
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.45)',
  },
  buttonText: {
    color: '#F8FAFC',
    fontWeight: '700',
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
    letterSpacing: 0.8,
    fontSize: 15,
  },
  riskBadge: {
    alignItems: 'center',
    borderRadius: 4,
    paddingVertical: 10,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.12)',
  },
  riskBadgeText: {
    color: '#fff',
    fontWeight: '800',
    letterSpacing: 2,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
    fontSize: 13,
  },
  status: {
    color: '#A5F3FC',
    marginBottom: 4,
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
    fontSize: 14,
  },
  error: {
    color: '#FCA5A5',
    marginBottom: 4,
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
  },
  topCard: {
    borderRadius: 4,
    backgroundColor: 'rgba(15, 23, 42, 0.75)',
    borderWidth: 1,
    borderColor: 'rgba(56, 189, 248, 0.25)',
    padding: 10,
    borderLeftWidth: 3,
    borderLeftColor: 'rgba(34, 211, 238, 0.7)',
  },
  topTitle: {
    color: '#F0F9FF',
    fontWeight: '800',
    marginBottom: 4,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
    letterSpacing: 0.6,
  },
  topLine: {
    color: '#BAE6FD',
    fontSize: 13,
    marginBottom: 2,
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
    fontWeight: '600',
  },
  hudStateTint: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 5,
  },
  hudStateTintLive: {
    backgroundColor: 'rgba(6, 182, 212, 0.07)',
  },
  hudStateTintIdle: {
    backgroundColor: 'rgba(99, 102, 241, 0.055)',
  },
  hudTelemetryBar: {
    position: 'absolute',
    top: 4,
    left: 8,
    right: 8,
    zIndex: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: 'rgba(34, 211, 238, 0.28)',
    backgroundColor: 'rgba(3, 7, 18, 0.72)',
  },
  hudTelemetryBarCompact: {
    top: 6,
    left: 6,
    right: 6,
    paddingVertical: 5,
    paddingHorizontal: 8,
  },
  hudTelLeft: {
    color: '#A5F3FC',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1.2,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
  },
  hudTelMid: {
    color: '#E0F2FE',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1.6,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
  },
  hudTelMidCompact: {
    fontSize: 9,
    letterSpacing: 0.8,
    textAlign: 'center',
  },
  hudTelRight: {
    color: '#7DD3FC',
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 1,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
  },
  hudTelRightCompact: {
    fontSize: 9,
    letterSpacing: 0.8,
  },
  hudPanelTitle: {
    color: '#67E8F9',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1.4,
    marginBottom: 4,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
  },
  hudThreatLadder: {
    position: 'absolute',
    left: 8,
    top: 34,
    width: 210,
    zIndex: 12,
    paddingHorizontal: 8,
    paddingVertical: 7,
    borderWidth: 1,
    borderColor: 'rgba(34, 211, 238, 0.26)',
    backgroundColor: 'rgba(3, 7, 18, 0.62)',
    borderRadius: 4,
  },
  hudThreatLadderCompact: {
    top: 35,
    left: 6,
    width: '55%',
    minWidth: 160,
    maxWidth: 220,
    paddingHorizontal: 7,
    paddingVertical: 5,
  },
  hudPanelEmpty: {
    color: '#94A3B8',
    fontSize: 11,
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
  },
  hudLadderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 3,
  },
  hudLadderPill: {
    width: 6,
    height: 6,
    borderRadius: 2,
    marginRight: 6,
  },
  hudLadderText: {
    color: '#BAE6FD',
    fontSize: 11,
    fontWeight: '600',
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
  },
  hudLadderTextCompact: {
    fontSize: 10,
  },
  hudGuidanceBar: {
    position: 'absolute',
    left: '24%',
    right: '24%',
    top: 34,
    zIndex: 12,
    paddingHorizontal: 8,
    paddingVertical: 6,
    borderWidth: 1,
    borderColor: 'rgba(34, 211, 238, 0.26)',
    backgroundColor: 'rgba(3, 7, 18, 0.55)',
    borderRadius: 4,
  },
  hudGuidanceBarCompact: {
    left: 6,
    right: 6,
    top: 112,
    paddingHorizontal: 7,
    paddingVertical: 5,
  },
  hudGuidanceTrack: {
    height: 16,
    borderWidth: 1,
    borderColor: 'rgba(34, 211, 238, 0.3)',
    backgroundColor: 'rgba(2, 6, 23, 0.8)',
    borderRadius: 3,
    position: 'relative',
    overflow: 'hidden',
  },
  hudGuidanceCenter: {
    position: 'absolute',
    left: '50%',
    top: 0,
    bottom: 0,
    width: 1,
    backgroundColor: 'rgba(167, 139, 250, 0.85)',
  },
  hudGuidanceMarker: {
    position: 'absolute',
    top: 1,
    width: 10,
    height: 12,
    marginLeft: -5,
    borderRadius: 2,
    backgroundColor: '#22D3EE',
  },
  hudGuidanceMeta: {
    marginTop: 4,
    color: '#A5F3FC',
    fontSize: 11,
    textAlign: 'center',
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
  },
  hudGuidanceMetaCompact: {
    fontSize: 10,
    marginTop: 3,
  },
  hudSystemCard: {
    position: 'absolute',
    right: 86,
    top: 34,
    width: 175,
    zIndex: 12,
    paddingHorizontal: 8,
    paddingVertical: 7,
    borderWidth: 1,
    borderColor: 'rgba(34, 211, 238, 0.26)',
    backgroundColor: 'rgba(3, 7, 18, 0.62)',
    borderRadius: 4,
  },
  hudSystemCardCompact: {
    right: 6,
    top: 35,
    width: 132,
    paddingHorizontal: 7,
    paddingVertical: 5,
  },
  hudSystemLine: {
    color: '#A5F3FC',
    fontSize: 11,
    marginBottom: 2,
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
  },
  hudSystemLineCompact: {
    fontSize: 10,
    marginBottom: 1,
  },
  hudBottomDock: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 13,
    flexDirection: 'column',
    backgroundColor: 'rgba(3, 7, 18, 0.92)',
    borderTopWidth: 1,
    borderTopColor: 'rgba(34, 211, 238, 0.4)',
    paddingTop: 6,
    paddingLeft: 8,
    justifyContent: 'flex-end',
  },
  hudBottomDockCompact: {
    paddingLeft: 6,
    paddingTop: 5,
  },
  hudDockTickBar: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    height: 6,
    marginBottom: 3,
    paddingHorizontal: 2,
  },
  hudDockTickCell: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'flex-end',
    height: 6,
  },
  hudDockTick: {
    width: 1,
    height: 3,
    backgroundColor: 'rgba(34, 211, 238, 0.35)',
    borderRadius: 1,
  },
  hudDockTickMajor: {
    height: 5,
    backgroundColor: 'rgba(34, 211, 238, 0.65)',
  },
  hudDockMetricRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 4,
    paddingHorizontal: 2,
  },
  hudDockMetricLeft: {
    color: '#7DD3FC',
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 1,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
  },
  hudDockMetricRight: {
    color: '#A5F3FC',
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 1,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
  },
  hudDockSparkRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    height: 16,
    marginBottom: 4,
    paddingHorizontal: 2,
    gap: 1,
  },
  hudSparkCell: {
    flex: 1,
    height: 16,
    justifyContent: 'flex-end',
    alignItems: 'center',
  },
  hudSparkBar: {
    width: '72%',
    maxWidth: 5,
    borderRadius: 1,
  },
  hudDockRadar: {
    flex: 1,
    minHeight: 38,
    maxHeight: 46,
    marginBottom: 4,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: 'rgba(34, 211, 238, 0.3)',
    backgroundColor: 'rgba(2, 6, 23, 0.95)',
    position: 'relative',
    overflow: 'hidden',
  },
  hudDockRadarLine: {
    position: 'absolute',
    left: '50%',
    top: 0,
    bottom: 0,
    width: 1,
    backgroundColor: 'rgba(34, 211, 238, 0.35)',
  },
  hudDockSweep: {
    position: 'absolute',
    left: 6,
    right: 6,
    top: 0,
    height: 2,
    backgroundColor: 'rgba(34, 211, 238, 0.85)',
    borderRadius: 1,
    shadowColor: '#22D3EE',
    shadowOpacity: 0.9,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 0 },
  },
  hudDockDot: {
    position: 'absolute',
    width: 8,
    height: 8,
    borderRadius: 2,
    marginLeft: -4,
    marginTop: -4,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.35)',
  },
  hudDockFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingRight: 4,
    paddingBottom: 2,
  },
  hudDockTitle: {
    color: '#67E8F9',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 2,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
  },
  hudDockMeta: {
    color: '#A5F3FC',
    fontSize: 12,
    fontWeight: '600',
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
  },
  hudFabRail: {
    position: 'absolute',
    right: 36,
    width: 3,
    zIndex: 21,
    backgroundColor: 'rgba(34, 211, 238, 0.18)',
    borderRadius: 2,
    borderWidth: 1,
    borderColor: 'rgba(6, 182, 212, 0.35)',
    shadowColor: '#22D3EE',
    shadowOpacity: 0.5,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 0 },
  },
  panelFab: {
    position: 'absolute',
    right: 16,
    width: 54,
    height: 54,
    borderRadius: 14,
    zIndex: 27,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    backgroundColor: 'rgba(6, 10, 22, 0.92)',
    overflow: 'visible',
  },
  panelFabMain: {
    borderColor: 'rgba(167, 139, 250, 0.95)',
    shadowColor: '#A78BFA',
    shadowOpacity: 0.55,
    shadowRadius: 14,
    shadowOffset: { width: 0, height: 0 },
    backgroundColor: 'rgba(46, 16, 80, 0.75)',
  },
  panelFabRadar: {
    borderColor: 'rgba(34, 211, 238, 0.9)',
    shadowColor: '#22D3EE',
    shadowOpacity: 0.5,
    shadowRadius: 14,
    shadowOffset: { width: 0, height: 0 },
    backgroundColor: 'rgba(8, 47, 73, 0.85)',
  },
  panelFabCal: {
    borderColor: 'rgba(148, 163, 184, 0.85)',
    shadowColor: '#94A3B8',
    shadowOpacity: 0.35,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 0 },
    backgroundColor: 'rgba(15, 23, 42, 0.9)',
  },
  panelFabTrip: {
    borderColor: 'rgba(45, 212, 191, 0.9)',
    shadowColor: '#2DD4BF',
    shadowOpacity: 0.45,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 0 },
    backgroundColor: 'rgba(6, 78, 59, 0.75)',
  },
  panelFabThreat: {
    borderColor: 'rgba(251, 146, 60, 0.95)',
    shadowColor: '#FB923C',
    shadowOpacity: 0.5,
    shadowRadius: 14,
    shadowOffset: { width: 0, height: 0 },
    backgroundColor: 'rgba(67, 20, 7, 0.8)',
  },
  panelFabGlyph: {
    color: '#F8FAFC',
    fontSize: 21,
    fontWeight: '800',
    textShadowColor: 'rgba(0, 0, 0, 0.75)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 3,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
  },
  panelFabBadge: {
    position: 'absolute',
    top: -2,
    right: -2,
    minWidth: 20,
    height: 20,
    paddingHorizontal: 4,
    borderRadius: 4,
    backgroundColor: 'rgba(127, 29, 29, 0.95)',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(254, 202, 202, 0.9)',
  },
  panelFabBadgeText: {
    color: '#FFFBEB',
    fontSize: 10,
    fontWeight: '800',
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
  },
  panelFabDot: {
    position: 'absolute',
    top: 6,
    right: 6,
    width: 9,
    height: 9,
    borderRadius: 2,
    backgroundColor: '#FACC15',
    borderWidth: 1,
    borderColor: '#FDE047',
    shadowColor: '#FACC15',
    shadowOpacity: 0.9,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 0 },
  },
  panelFabBackdrop: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 25,
    backgroundColor: 'rgba(2, 6, 18, 0.78)',
  },
  panelFabSheet: {
    position: 'absolute',
    left: 12,
    right: 12,
    zIndex: 30,
    borderRadius: 6,
    padding: 14,
    maxWidth: 520,
    alignSelf: 'center',
    overflow: 'hidden',
  },
  hudSheetAccentTop: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 3,
    backgroundColor: '#22D3EE',
    opacity: 0.85,
  },
  hudSheetTitle: {
    color: '#ECFEFF',
    fontWeight: '800',
    marginBottom: 0,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
    letterSpacing: 1.4,
    fontSize: 14,
    textTransform: 'uppercase',
  },
  panelFabSheetHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
    marginTop: 4,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(34, 211, 238, 0.2)',
  },
  panelFabCloseHit: { paddingVertical: 6, paddingHorizontal: 6 },
  panelFabClose: {
    color: '#67E8F9',
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1.2,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
  },
  panelFabSheetTitle: { marginBottom: 0 },
  panelFabSheetScroll: {
    width: '100%',
  },
  calibrationCardInSheet: {
    marginBottom: 0,
    backgroundColor: 'transparent',
    borderWidth: 0,
    paddingHorizontal: 0,
    paddingVertical: 0,
  },
  tripSheetScroll: { width: '100%' },
  tripSheetScrollContent: { paddingBottom: 16 },
  tripIntro: {
    color: '#7DD3FC',
    fontSize: 13,
    lineHeight: 18,
    marginBottom: 10,
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
    fontWeight: '500',
  },
  tripMuted: { color: '#94A3B8', fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined },
  tripStatsBlock: {
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderRadius: 4,
    borderWidth: 1,
    borderColor: 'rgba(34, 211, 238, 0.25)',
    padding: 12,
    marginBottom: 10,
    borderLeftWidth: 3,
    borderLeftColor: 'rgba(167, 139, 250, 0.8)',
  },
  tripStatLine: {
    color: '#E0F2FE',
    fontSize: 14,
    marginBottom: 4,
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
    fontWeight: '600',
  },
  tripResetBtn: {
    alignItems: 'center',
    paddingVertical: 12,
    borderRadius: 4,
    backgroundColor: 'rgba(69, 10, 10, 0.85)',
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(248, 113, 113, 0.55)',
  },
  tripResetBtnText: {
    color: '#FECACA',
    fontWeight: '800',
    fontSize: 13,
    letterSpacing: 1.2,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
  },
  tripSectionTitle: {
    color: '#F0F9FF',
    fontWeight: '800',
    fontSize: 12,
    marginBottom: 6,
    letterSpacing: 1.5,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
    textTransform: 'uppercase',
  },
  tripEventCard: {
    backgroundColor: 'rgba(8, 15, 35, 0.88)',
    borderRadius: 4,
    borderWidth: 1,
    borderColor: 'rgba(56, 189, 248, 0.2)',
    padding: 10,
    marginBottom: 6,
    borderLeftWidth: 2,
    borderLeftColor: 'rgba(251, 191, 36, 0.75)',
  },
  tripEventTitle: {
    color: '#F8FAFC',
    fontWeight: '800',
    fontSize: 13,
    marginBottom: 2,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
    letterSpacing: 0.4,
  },
  tripEventMeta: {
    color: '#7DD3FC',
    fontSize: 12,
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
    fontWeight: '500',
  },
  radarBody: {
    height: 120,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: 'rgba(34, 211, 238, 0.35)',
    backgroundColor: 'rgba(2, 6, 23, 0.95)',
    position: 'relative',
    overflow: 'hidden',
  },
  radarSweepLine: {
    position: 'absolute',
    left: '50%',
    top: 0,
    bottom: 0,
    width: 1,
    backgroundColor: 'rgba(34, 211, 238, 0.45)',
  },
  radarDot: {
    position: 'absolute',
    width: 10,
    height: 10,
    borderRadius: 999,
    marginLeft: -5,
    marginTop: -5,
  },
  radarHint: {
    color: '#7DD3FC',
    fontSize: 11,
    marginTop: 6,
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
    letterSpacing: 0.5,
  },
  calibrationCard: {
    marginBottom: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(15, 23, 42, 0.65)',
    borderWidth: 1,
    borderColor: 'rgba(56, 189, 248, 0.2)',
    padding: 8,
  },
  calibrationTitle: {
    color: '#E0F2FE',
    fontWeight: '800',
    marginBottom: 6,
    fontFamily: Platform.OS === 'web' ? 'Orbitron, sans-serif' : undefined,
    letterSpacing: 1,
    fontSize: 12,
    textTransform: 'uppercase',
  },
  calibrationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  calibrationLabel: {
    color: '#BAE6FD',
    fontSize: 13,
    flex: 1,
    marginRight: 8,
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
    fontWeight: '600',
  },
  calibrationInput: {
    width: 110,
    borderWidth: 1,
    borderColor: 'rgba(34, 211, 238, 0.35)',
    borderRadius: 4,
    color: '#F0FDFA',
    paddingHorizontal: 8,
    paddingVertical: 6,
    backgroundColor: 'rgba(15, 23, 42, 0.9)',
    textAlign: 'right',
    fontSize: 14,
    fontFamily: Platform.OS === 'web' ? 'Rajdhani, system-ui, sans-serif' : undefined,
    fontWeight: '600',
  },
});
