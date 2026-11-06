import type { TourControls } from "../tour";

export default function TourOverlay({ tour }: { tour: TourControls }) {
  if (!tour.running) return null;
  const step = tour.steps[tour.stepIndex];
  if (!step) return null;
  return (
    <div className="tour" role="status" aria-live="polite">
      <div className="tour-progress" key={tour.stepIndex} style={{ animationDuration: `${step.ms}ms` }} />
      <div className="tour-row">
        <span className="tour-step mono">
          {tour.stepIndex + 1}/{tour.steps.length}
        </span>
        <span className="tour-caption">{step.caption}</span>
        <button className="tour-stop" onClick={tour.stop} aria-label="Stop tour">
          Esc
        </button>
      </div>
    </div>
  );
}
