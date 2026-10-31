import type { AnalyzeRequest, LeaksPayload, Report, SceneMeta, SceneSummary, SplatMode } from "./types";

export class ApiError extends Error {
  constructor(message: string, public readonly status: number) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch (error) {
    throw new ApiError(`Cannot reach the Heatsplat API (${(error as Error).message})`, 0);
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  listScenes: () => request<SceneSummary[]>("/api/scenes"),
  getScene: (sceneId: string) => request<SceneMeta>(`/api/scenes/${encodeURIComponent(sceneId)}`),
  getLeaks: (sceneId: string) => request<LeaksPayload>(`/api/scenes/${encodeURIComponent(sceneId)}/leaks`),
  analyze: (sceneId: string, body: AnalyzeRequest) =>
    request<LeaksPayload>(`/api/scenes/${encodeURIComponent(sceneId)}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  report: (sceneId: string) => request<Report>(`/api/scenes/${encodeURIComponent(sceneId)}/report`, { method: "POST" }),
  splatUrl: (sceneId: string, mode: SplatMode, version: number) =>
    `/api/scenes/${encodeURIComponent(sceneId)}/splat?mode=${mode}&v=${version}`,
};
