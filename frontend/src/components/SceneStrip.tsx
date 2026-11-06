import type { SceneSummary } from "../types";

const money = (v: number, currency: string) => new Intl.NumberFormat("en-US", { style: "currency", currency, maximumFractionDigits: 0 }).format(v);

/** Short label: the part before the first comma or " via ", so chips stay one line. */
function shortName(name: string): string {
  return name.split(/,| via /)[0].trim();
}

export default function SceneStrip({ scenes, current, onSelect }: { scenes: SceneSummary[]; current: string | null; onSelect: (id: string) => void }) {
  if (scenes.length === 0) return null;
  return (
    <div className="scene-strip" role="tablist" aria-label="Scenes">
      {scenes.map((s) => (
        <button key={s.scene_id} role="tab" aria-selected={s.scene_id === current} className={`scene-chip ${s.scene_id === current ? "active" : ""}`} onClick={() => onSelect(s.scene_id)} title={s.name}>
          <span className="chip-name">{shortName(s.name)}</span>
          <span className="chip-meta mono">
            {s.n_leaks} leak{s.n_leaks === 1 ? "" : "s"} · {money(s.total_cost_per_year, s.currency)}/yr
          </span>
        </button>
      ))}
    </div>
  );
}
