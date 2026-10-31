import type { LeaksPayload } from "../types";

const money = (v: number, currency: string) => new Intl.NumberFormat("en-US", { style: "currency", currency }).format(v);

export default function StatsBar({ payload }: { payload: LeaksPayload | null }) {
  const s = payload?.summary;
  return (
    <div className="stats">
      <div className="stat">
        <span className="stat-label">Leaks</span>
        <span className="stat-value">{s ? s.n_leaks : "–"}</span>
      </div>
      <div className="stat">
        <span className="stat-label">Heat loss</span>
        <span className="stat-value">
          {s ? Math.round(s.total_watts) : "–"}
          <small> W</small>
        </span>
      </div>
      <div className="stat accent">
        <span className="stat-label">Est. cost / yr</span>
        <span className="stat-value">{s ? money(s.total_cost_per_year, s.currency) : "–"}</span>
      </div>
    </div>
  );
}
