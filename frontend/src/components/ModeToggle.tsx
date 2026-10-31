import type { SplatMode } from "../types";

const MODES: { id: SplatMode; label: string; hint: string }[] = [
  { id: "rgb", label: "RGB", hint: "Photographic scan" },
  { id: "thermal", label: "Thermal", hint: "Baked surface temperature" },
  { id: "leaks", label: "Leaks", hint: "Only anomalies lit" },
];

export default function ModeToggle({ mode, onChange, disabled }: { mode: SplatMode; onChange: (m: SplatMode) => void; disabled?: boolean }) {
  return (
    <div className="segmented" role="tablist" aria-label="View mode">
      {MODES.map((m) => (
        <button
          key={m.id}
          role="tab"
          aria-selected={mode === m.id}
          className={mode === m.id ? "active" : ""}
          title={m.hint}
          disabled={disabled}
          onClick={() => onChange(m.id)}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
