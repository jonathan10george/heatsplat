import type { Leak } from "../types";

const money = (v: number, currency: string) => new Intl.NumberFormat("en-US", { style: "currency", currency }).format(v);

function LeakRow({ leak, selected, onSelect }: { leak: Leak; selected: boolean; onSelect: () => void }) {
  return (
    <button className={`leak-row ${selected ? "selected" : ""}`} onClick={onSelect} aria-pressed={selected}>
      <span className="rank">{leak.rank}</span>
      <span className="leak-main">
        <span className="cause">{leak.likely_cause}</span>
        <span className="meta mono">
          +{leak.mean_delta_c.toFixed(1)} °C · {leak.area_m2.toFixed(2)} m²
        </span>
      </span>
      <span className="cost mono">{money(leak.cost.cost_per_year, leak.cost.currency)}/yr</span>
    </button>
  );
}

function LeakDetail({ leak }: { leak: Leak }) {
  const [x, y, z] = leak.centroid;
  return (
    <div className="leak-detail">
      <h3>
        Leak {leak.rank} <span className="pill">{leak.id}</span>
      </h3>
      <p className="cause-big">{leak.likely_cause}</p>
      <dl>
        <dt>Warmer than surroundings</dt>
        <dd className="mono">
          +{leak.mean_delta_c.toFixed(1)} °C <small>(peak +{leak.max_delta_c.toFixed(1)})</small>
        </dd>
        <dt>Affected area</dt>
        <dd className="mono">{leak.area_m2.toFixed(2)} m²</dd>
        <dt>Extra heat loss</dt>
        <dd className="mono">
          {Math.round(leak.cost.watts)} W · {Math.round(leak.cost.kwh_per_year)} kWh/yr
        </dd>
        <dt>Estimated cost</dt>
        <dd className="mono accent-text">{money(leak.cost.cost_per_year, leak.cost.currency)} / yr</dd>
        <dt>Location</dt>
        <dd className="mono">
          x {x.toFixed(1)} · h {y.toFixed(1)} · z {z.toFixed(1)} m
        </dd>
      </dl>
    </div>
  );
}

export default function LeakPanel({ leaks, selectedId, onSelect }: { leaks: Leak[]; selectedId: string | null; onSelect: (id: string) => void }) {
  const selected = leaks.find((l) => l.id === selectedId) ?? null;
  return (
    <div className="leak-panel">
      <h2>Ranked leaks</h2>
      {leaks.length === 0 ? (
        <p className="empty">No leaks at this threshold. Lower it to see weaker anomalies.</p>
      ) : (
        <div className="leak-list">
          {leaks.map((leak) => (
            <LeakRow key={leak.id} leak={leak} selected={leak.id === selectedId} onSelect={() => onSelect(leak.id)} />
          ))}
        </div>
      )}
      {selected && <LeakDetail leak={selected} />}
    </div>
  );
}
