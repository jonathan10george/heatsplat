export type SplatMode = "rgb" | "thermal" | "leaks";

export interface SceneSummary {
  scene_id: string;
  name: string;
  n_gaussians: number;
  temp_min_c: number;
  temp_max_c: number;
  n_leaks: number;
  total_cost_per_year: number;
  total_watts: number;
  currency: string;
}

export interface SceneMeta {
  scene_id: string;
  name?: string;
  n_gaussians: number;
  bounds?: { min: [number, number, number]; max: [number, number, number] };
  temp_range_c?: [number, number];
  color_range_c?: [number, number];
}

export interface LeakCost {
  watts: number;
  kwh_per_year: number;
  cost_per_year: number;
  currency: string;
}

export interface Leak {
  id: string;
  rank: number;
  centroid: [number, number, number];
  bbox_min: [number, number, number];
  bbox_max: [number, number, number];
  n_points: number;
  mean_delta_c: number;
  max_delta_c: number;
  area_m2: number;
  severity: number;
  likely_cause: string;
  cost: LeakCost;
}

export interface LeaksPayload {
  scene_id: string;
  params: { min_delta_c: number; min_zscore: number; cluster_eps_m: number; cluster_min_points: number; sign: number };
  cost_model: { price_per_kwh: number; heating_hours_per_year: number; currency: string };
  summary: { n_leaks: number; total_cost_per_year: number; total_watts: number; currency: string; n_candidates: number };
  leaks: Leak[];
}

export interface AnalyzeRequest {
  min_delta_c?: number;
  min_zscore?: number;
  cluster_eps_m?: number;
  cluster_min_points?: number;
  sign?: 1 | -1;
  price_per_kwh?: number;
  heating_hours_per_year?: number;
}

export interface Report {
  scene_id: string;
  generator: string;
  markdown: string;
}

export interface CameraPose {
  position: [number, number, number];
  target: [number, number, number];
}
