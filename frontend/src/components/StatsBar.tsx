import type { LeaksPayload } from "../types";
import { useCountUp } from "../hooks/useCountUp";
import { CO2_KG_PER_KWH, TREE_KG_CO2_PER_YEAR, carbonFor } from "../impact";

const money = (v: number, currency: string) => new Intl.NumberFormat("en-US", { style: "currency", currency }).format(v);

function Tile({ label, value, unit, sub, accent, title }: { label: string; value: string; unit?: string; sub?: string; accent?: boolean; title?: string }) {
  return (
    <div className={`stat ${accent ? "accent" : ""}`} title={title}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">
        {value}
        {unit && <small> {unit}</small>}
      </span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  );
}

export default function StatsBar({ payload }: { payload: LeaksPayload | null }) {
  const s = payload?.summary;
  const kwh = payload?.leaks.reduce((sum, leak) => sum + leak.cost.kwh_per_year, 0) ?? 0;
  const carbon = carbonFor(kwh);
  const leaks = useCountUp(s?.n_leaks ?? 0);
  const watts = useCountUp(s?.total_watts ?? 0);
  const cost = useCountUp(s?.total_cost_per_year ?? 0);
  const co2 = useCountUp(carbon.kgPerYear);
  const currency = s?.currency ?? "USD";
  return (
    <div className="stats">
      <Tile label="Leaks" value={s ? String(Math.round(leaks)) : "–"} />
      <Tile label="Heat loss" value={s ? String(Math.round(watts)) : "–"} unit="W" sub={s ? `${Math.round(kwh).toLocaleString()} kWh / yr` : undefined} />
      <Tile label="Est. cost / yr" value={s ? money(cost, currency) : "–"} accent sub={payload ? `at ${money(payload.cost_model.price_per_kwh, currency)} / kWh` : undefined} />
      <Tile
        label="CO₂ / yr"
        value={s ? Math.round(co2).toLocaleString() : "–"}
        unit="kg"
        sub={s ? `≈ ${Math.max(1, Math.round(carbon.trees))} trees' worth` : undefined}
        title={`Electric heat on the US average grid, ${CO2_KG_PER_KWH} kg CO₂ per kWh; one mature tree absorbs about ${TREE_KG_CO2_PER_YEAR} kg a year`}
      />
    </div>
  );
}
