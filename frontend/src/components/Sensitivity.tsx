const MIN = 1.5; // the pipeline default; below it the soft edges of adjacent leaks merge
const MAX = 6;
const STEP = 0.25;

export default function Sensitivity({ value, onChange, busy }: { value: number; onChange: (v: number) => void; busy: boolean }) {
  return (
    <div className="sensitivity">
      <div className="row">
        <label htmlFor="min-delta">Detection threshold</label>
        <span className="mono">
          ≥ {value.toFixed(2)} °C {busy && <span className="spinner" aria-label="re-analysing" />}
        </span>
      </div>
      <input id="min-delta" type="range" min={MIN} max={MAX} step={STEP} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      <div className="row hint">
        <span>Sensitive (default)</span>
        <span>Only the worst</span>
      </div>
    </div>
  );
}
