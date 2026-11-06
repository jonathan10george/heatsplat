/** A one-shot scan line that sweeps the stage when the thermal layer switches on. Keyed by the parent so each switch replays it. */
export default function ScanSweep({ play }: { play: number }) {
  if (play === 0) return null;
  return <div className="sweep" key={play} aria-hidden="true" />;
}
