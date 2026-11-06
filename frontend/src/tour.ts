import { useCallback, useEffect, useRef, useState } from "react";
import type { Leak, LeaksPayload, SplatMode } from "./types";

export interface TourStep {
  caption: string;
  mode: SplatMode;
  selectIndex: number | null; // index into the ranked leaks, or null for no selection
  home: boolean; // fly the camera back to the home view
  ms: number;
}

const money = (v: number, currency: string) => new Intl.NumberFormat("en-US", { style: "currency", currency, maximumFractionDigits: 0 }).format(v);

/** The scripted walkthrough a presenter narrates: scan, heat, anomalies, the worst two leaks, the bill. */
export function buildTour(payload: LeaksPayload | null): TourStep[] {
  const leaks: Leak[] = payload?.leaks ?? [];
  const steps: TourStep[] = [
    { caption: "A phone walk-around becomes a Gaussian splat: every surface, in 3D", mode: "rgb", selectIndex: null, home: true, ms: 4500 },
    { caption: "Thermal frames baked onto every Gaussian: surface temperature you can orbit", mode: "thermal", selectIndex: null, home: false, ms: 5000 },
    { caption: "Only the anomalies: patches warmer than the two metres around them", mode: "leaks", selectIndex: null, home: false, ms: 3500 },
  ];
  leaks.slice(0, 2).forEach((leak, i) => {
    steps.push({
      caption: `#${leak.rank} ${leak.likely_cause}: +${leak.mean_delta_c.toFixed(1)} °C over ${leak.area_m2.toFixed(1)} m², ${money(leak.cost.cost_per_year, leak.cost.currency)} a year`,
      mode: "leaks",
      selectIndex: i,
      home: false,
      ms: 4500,
    });
  });
  if (payload) {
    const s = payload.summary;
    steps.push({
      caption: `${s.n_leaks} leaks, ${Math.round(s.total_watts)} W walking out of the wall: ${money(s.total_cost_per_year, s.currency)} a year`,
      mode: "thermal",
      selectIndex: null,
      home: true,
      ms: 4500,
    });
  }
  return steps;
}

export interface TourControls {
  running: boolean;
  stepIndex: number;
  steps: TourStep[];
  start: () => void;
  stop: () => void;
}

interface TourActions {
  setMode: (mode: SplatMode) => void;
  select: (leakId: string | null) => void;
  goHome: () => void;
}

/** Runs the tour timeline against the app's state setters. Any user interaction should call stop(). */
export function useTour(payload: LeaksPayload | null, actions: TourActions): TourControls {
  const [running, setRunning] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [steps, setSteps] = useState<TourStep[]>([]);
  const actionsRef = useRef(actions);
  actionsRef.current = actions;
  const payloadRef = useRef(payload);
  payloadRef.current = payload;

  const stop = useCallback(() => setRunning(false), []);
  const start = useCallback(() => {
    setSteps(buildTour(payloadRef.current));
    setStepIndex(0);
    setRunning(true);
  }, []);

  useEffect(() => {
    if (!running) return;
    const step = steps[stepIndex];
    if (!step) {
      setRunning(false);
      return;
    }
    const leaks = payloadRef.current?.leaks ?? [];
    const a = actionsRef.current;
    a.setMode(step.mode);
    a.select(step.selectIndex === null ? null : (leaks[step.selectIndex]?.id ?? null));
    if (step.home) a.goHome();
    const timer = window.setTimeout(() => setStepIndex((i) => i + 1), step.ms);
    return () => window.clearTimeout(timer);
  }, [running, stepIndex, steps]);

  return { running, stepIndex, steps, start, stop };
}
