import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "./api";
import ErrorBoundary from "./components/ErrorBoundary";
import LeakPanel from "./components/LeakPanel";
import Legend from "./components/Legend";
import ModeToggle from "./components/ModeToggle";
import ReportModal from "./components/ReportModal";
import Sensitivity from "./components/Sensitivity";
import SplatViewer, { type ViewerStatus } from "./components/SplatViewer";
import StatsBar from "./components/StatsBar";
import type { Leak, LeaksPayload, Report, SceneMeta, SceneSummary, SplatMode } from "./types";

const ANALYZE_DEBOUNCE_MS = 250;
const DEFAULT_CENTER: [number, number, number] = [0, 1.5, 0];
const DEFAULT_MIN_DELTA_C = 1.5;
const SELECTION_MATCH_M = 0.75;
const NO_LEAKS: Leak[] = [];

/** Leak ids are rank-based and ranks move as the threshold changes, so a selection is carried
 *  across a re-analysis by position: the new leak nearest the old one's centroid, if close enough. */
export function carrySelection(previous: Leak | undefined, next: Leak[]): string | null {
  if (!previous) return null;
  let best: Leak | null = null;
  let bestDistance = SELECTION_MATCH_M;
  for (const leak of next) {
    const d = Math.hypot(...leak.centroid.map((v, i) => v - previous.centroid[i]));
    if (d < bestDistance) {
      best = leak;
      bestDistance = d;
    }
  }
  return best?.id ?? null;
}

export default function App() {
  const [scenes, setScenes] = useState<SceneSummary[]>([]);
  const [sceneId, setSceneId] = useState<string | null>(null);
  const [meta, setMeta] = useState<SceneMeta | null>(null);
  const [mode, setMode] = useState<SplatMode>("thermal");
  const [payload, setPayload] = useState<LeaksPayload | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [minDelta, setMinDelta] = useState(DEFAULT_MIN_DELTA_C);
  const [analyzing, setAnalyzing] = useState(false);
  const [version, setVersion] = useState(() => Date.now()); // unique per page load so a stale cached splat is never reused
  const [status, setStatus] = useState<ViewerStatus>({ state: "loading" });
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [reportBusy, setReportBusy] = useState(false);
  const debounceRef = useRef<number | null>(null);
  const analyzeSeq = useRef(0);
  const payloadRef = useRef<LeaksPayload | null>(null);
  payloadRef.current = payload;

  const fail = useCallback((e: unknown) => {
    const message = e instanceof ApiError ? e.message : (e as Error)?.message ?? String(e);
    setError(message);
    console.error(e);
  }, []);

  useEffect(() => {
    api
      .listScenes()
      .then((list) => {
        setScenes(list);
        if (list.length === 0) setError("No scenes found. Run: uv run python -m heatsplat.cli synth --scene demo-house");
        else setSceneId(list[0].scene_id);
      })
      .catch(fail);
  }, [fail]);

  useEffect(() => {
    if (!sceneId) return;
    let stale = false;
    setSelectedId(null);
    // Re-analysis rewrites the scene's leak files on disk, so a scene always opens at the default
    // threshold rather than wherever the last session left the slider.
    const leaksAtDefault = api.getLeaks(sceneId).then((stored) =>
      stored.params.min_delta_c === DEFAULT_MIN_DELTA_C ? stored : api.analyze(sceneId, { min_delta_c: DEFAULT_MIN_DELTA_C }),
    );
    Promise.all([leaksAtDefault, api.getScene(sceneId)])
      .then(([leaks, sceneMeta]) => {
        if (stale) return;
        setPayload(leaks);
        setMinDelta(leaks.params.min_delta_c);
        setMeta(sceneMeta);
        setVersion((v) => v + 1);
      })
      .catch((e) => {
        if (!stale) fail(e);
      });
    return () => {
      stale = true;
    };
  }, [sceneId, fail]);

  const onThreshold = (value: number) => {
    setMinDelta(value);
    if (!sceneId) return;
    if (debounceRef.current) window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(() => {
      const seq = ++analyzeSeq.current;
      setAnalyzing(true);
      api
        .analyze(sceneId, { min_delta_c: value })
        .then((result) => {
          if (seq !== analyzeSeq.current) return;
          setSelectedId((id) => carrySelection(payloadRef.current?.leaks.find((l) => l.id === id), result.leaks));
          setPayload(result);
          setVersion((v) => v + 1);
        })
        .catch(fail)
        .finally(() => {
          if (seq === analyzeSeq.current) setAnalyzing(false);
        });
    }, ANALYZE_DEBOUNCE_MS);
  };

  const onReport = () => {
    if (!sceneId) return;
    setReportBusy(true);
    api.report(sceneId).then(setReport).catch(fail).finally(() => setReportBusy(false));
  };

  const sceneCenter = useMemo<[number, number, number]>(() => {
    if (!meta?.bounds) return DEFAULT_CENTER;
    const { min, max } = meta.bounds;
    return [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  }, [meta]);

  // Only the leaks splat changes when the threshold changes; other modes keep their cached file.
  const splatVersion = mode === "leaks" ? version : 1;
  const colorRange = meta?.color_range_c ?? meta?.temp_range_c ?? null;
  const leaks = payload?.leaks ?? NO_LEAKS;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden="true" />
          <span>Heatsplat</span>
        </div>
        <select className="scene-select" value={sceneId ?? ""} onChange={(e) => setSceneId(e.target.value)} aria-label="Scene" disabled={scenes.length === 0}>
          {scenes.map((s) => (
            <option key={s.scene_id} value={s.scene_id}>
              {s.name} · {s.n_gaussians.toLocaleString()} Gaussians
            </option>
          ))}
        </select>
        <ModeToggle mode={mode} onChange={setMode} disabled={!sceneId} />
        <span className={`status ${status.state}`} role="status">
          {status.state === "loading" ? status.message ?? "Loading" : status.state === "error" ? "Load failed" : "Live"}
        </span>
      </header>

      <main className="stage">
        {sceneId && (
          <ErrorBoundary label="3D viewer">
            <SplatViewer
              sceneId={sceneId}
              mode={mode}
              version={splatVersion}
              leaks={leaks}
              selectedId={selectedId}
              sceneCenter={sceneCenter}
              onSelect={setSelectedId}
              onStatus={setStatus}
            />
          </ErrorBoundary>
        )}
        <Legend mode={mode} colorRange={colorRange} />
        {status.state === "error" && <div className="stage-error">{status.message}</div>}
      </main>

      <aside className="panel">
        <div className="panel-body">
          <StatsBar payload={payload} />
          <Sensitivity value={minDelta} onChange={onThreshold} busy={analyzing} />
          <LeakPanel leaks={leaks} selectedId={selectedId} onSelect={setSelectedId} />
        </div>
        <div className="panel-footer">
          <button className="btn primary report-btn" onClick={onReport} disabled={!payload || reportBusy}>
            {reportBusy ? "Writing report…" : "Generate audit report"}
          </button>
        </div>
      </aside>

      {report && <ReportModal report={report} onClose={() => setReport(null)} />}
      {error && (
        <div className="toast" role="alert">
          <span>{error}</span>
          <button onClick={() => setError(null)} aria-label="Dismiss">
            ✕
          </button>
        </div>
      )}
    </div>
  );
}
