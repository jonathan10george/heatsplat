import type { SplatMode } from "../types";

export default function Legend({ mode, colorRange }: { mode: SplatMode; colorRange: [number, number] | null }) {
  if (mode === "rgb") return null;
  const [lo, hi] = colorRange ?? [0, 1];
  return (
    <div className="legend">
      {mode === "thermal" ? (
        <>
          <span className="mono">{lo.toFixed(1)} °C</span>
          <span className="ironbow" aria-hidden="true" />
          <span className="mono">{hi.toFixed(1)} °C</span>
        </>
      ) : (
        <>
          <span className="swatch dim" aria-hidden="true" /> <span>normal</span>
          <span className="ironbow short" aria-hidden="true" /> <span>warmer than surroundings</span>
        </>
      )}
    </div>
  );
}
