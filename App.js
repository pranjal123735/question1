import { useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';

const DEFAULT_BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL || 'http://127.0.0.1:8001';

function MetricCard({ item }) {
  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>
        {item.label} #{item.track_id}
      </Text>
      <Text style={styles.cardLine}>Distance: {Number(item.distance_m).toFixed(2)} m</Text>
      <Text style={styles.cardLine}>
        Speed: {item.is_moving ? `${Number(item.speed_kmh).toFixed(1)} km/h` : 'static'}
      </Text>
      <Text style={styles.cardLine}>Collision Risk: {Math.round(Number(item.risk_percent))}%</Text>
      <Text style={styles.cardLine}>TTC: {Number(item.ttc_s).toFixed(2)} s</Text>
      <Text style={styles.cardLine}>Confidence: {Math.round(Number(item.confidence) * 100)}%</Text>
    </View>
  );
}

export default function App() {
  const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND_URL);
  const [status, setStatus] = useState('Idle');
  const [detections, setDetections] = useState([]);
  const [error, setError] = useState(null);

  const normalizedUrl = useMemo(() => backendUrl.replace(/\/+$/, ''), [backendUrl]);
  const requestInit = useMemo(
    () =>
      normalizedUrl.includes('ngrok-free.dev')
        ? { headers: { 'ngrok-skip-browser-warning': 'true' } }
        : undefined,
    [normalizedUrl]
  );

  const checkBackend = async () => {
    try {
      setError(null);
      setStatus('Checking backend...');
      const res = await fetch(`${normalizedUrl}/health`, requestInit);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const body = await res.json();
      setStatus(`Backend ready (${body.mode})`);
    } catch (e) {
      setStatus('Backend check failed');
      setError(e.message);
    }
  };

  const loadDemo = async () => {
    try {
      setError(null);
      setStatus('Fetching demo detections...');
      const res = await fetch(`${normalizedUrl}/demo-detections`, requestInit);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const body = await res.json();
      setDetections(body.detections || []);
      setStatus(`Loaded ${body.detections?.length ?? 0} detections`);
    } catch (e) {
      setStatus('Demo fetch failed');
      setError(e.message);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Car Vision MVP</Text>
      <Text style={styles.text}>
        This app now connects to a Python backend for vehicle metrics (distance, speed, collision risk).
      </Text>

      <TextInput
        style={styles.input}
        value={backendUrl}
        onChangeText={setBackendUrl}
        autoCapitalize="none"
        autoCorrect={false}
        placeholder="Backend URL"
        placeholderTextColor="#93A3B8"
      />

      <View style={styles.row}>
        <Pressable style={styles.button} onPress={checkBackend}>
          <Text style={styles.buttonText}>Check Backend</Text>
        </Pressable>
        <Pressable style={[styles.button, styles.buttonAlt]} onPress={loadDemo}>
          <Text style={styles.buttonText}>Load Demo</Text>
        </Pressable>
      </View>

      <Text style={styles.status}>{status}</Text>
      {error ? <Text style={styles.error}>Error: {error}</Text> : null}

      <ScrollView contentContainerStyle={styles.list}>
        {detections.map((item) => (
          <MetricCard key={`${item.track_id}-${item.label}`} item={item} />
        ))}
        {!detections.length ? (
          <Text style={styles.empty}>
            No detections yet. Start backend and tap "Load Demo", then connect real camera frames next.
          </Text>
        ) : null}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F172A', padding: 16, paddingTop: 52 },
  title: { color: '#fff', fontSize: 28, fontWeight: '700', marginBottom: 8 },
  text: { color: '#CBD5E1', fontSize: 14, lineHeight: 20, marginBottom: 16 },
  input: {
    borderColor: '#334155',
    borderWidth: 1,
    borderRadius: 8,
    color: '#fff',
    paddingHorizontal: 12,
    paddingVertical: 10,
    marginBottom: 12,
    backgroundColor: '#111827',
  },
  row: { flexDirection: 'row', gap: 10, marginBottom: 10 },
  button: {
    flex: 1,
    paddingVertical: 11,
    borderRadius: 8,
    alignItems: 'center',
    backgroundColor: '#2563EB',
  },
  buttonAlt: { backgroundColor: '#0E7490' },
  buttonText: { color: '#fff', fontWeight: '700' },
  status: { color: '#E2E8F0', marginBottom: 6 },
  error: { color: '#FCA5A5', marginBottom: 8 },
  list: { paddingBottom: 20 },
  card: {
    backgroundColor: '#111827',
    borderRadius: 10,
    borderColor: '#334155',
    borderWidth: 1,
    padding: 12,
    marginBottom: 10,
  },
  cardTitle: { color: '#F8FAFC', fontSize: 16, fontWeight: '700', marginBottom: 6 },
  cardLine: { color: '#CBD5E1', fontSize: 13, marginBottom: 2 },
  empty: { color: '#94A3B8', marginTop: 8 },
});
