import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import { api } from "../api";
import type { Leak, SplatMode, CameraPose } from "../types";

export type ViewerStatus = { state: "loading" | "ready" | "error"; message?: string };

interface Props {
  sceneId: string;
  mode: SplatMode;
  version: number;
  leaks: Leak[];
  selectedId: string | null;
  sceneCenter: [number, number, number];
  homePose: CameraPose;
  onSelect: (id: string) => void;
  onStatus: (status: ViewerStatus) => void;
}

const ACCENT = 0xff7a1a;
const SELECTED = 0xffffff;
const MARKER_LIFT_M = 0.35;
const FLY_DISTANCE_M = 6.5;
const FLY_MS = 700;
const CLICK_SLOP_PX = 6;

/** Outward unit normal of a leak: the thinnest bbox axis, pointing away from the scene centre. */
export function leakNormal(leak: Leak, sceneCenter: [number, number, number]): THREE.Vector3 {
  const extents = leak.bbox_max.map((hi, i) => hi - leak.bbox_min[i]);
  const axis = extents.indexOf(Math.min(...extents));
  const sign = Math.sign(leak.centroid[axis] - sceneCenter[axis]) || 1;
  const n = new THREE.Vector3();
  n.setComponent(axis, sign);
  return n;
}

function labelSprite(text: string, selected: boolean): THREE.Sprite {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 8, 0, Math.PI * 2);
  ctx.fillStyle = selected ? "#ffffff" : "#ff7a1a";
  ctx.fill();
  ctx.lineWidth = 8;
  ctx.strokeStyle = selected ? "#ff7a1a" : "#1a0d04";
  ctx.stroke();
  ctx.fillStyle = selected ? "#ff7a1a" : "#1a0d04";
  ctx.font = "bold 60px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, size / 2, size / 2 + 4);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, depthTest: false, depthWrite: false, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.setScalar(selected ? 0.75 : 0.6);
  sprite.renderOrder = 1000;
  return sprite;
}

function buildMarkers(leaks: Leak[], selectedId: string | null, sceneCenter: [number, number, number]): THREE.Group {
  const group = new THREE.Group();
  group.name = "leak-markers";
  for (const leak of leaks) {
    const selected = leak.id === selectedId;
    const normal = leakNormal(leak, sceneCenter);
    const sprite = labelSprite(String(leak.rank), selected);
    sprite.position.set(...leak.centroid).addScaledVector(normal, MARKER_LIFT_M);
    sprite.userData.leakId = leak.id;
    group.add(sprite);

    const box = new THREE.Box3(new THREE.Vector3(...leak.bbox_min), new THREE.Vector3(...leak.bbox_max)).expandByScalar(0.08);
    const helper = new THREE.Box3Helper(box, selected ? SELECTED : ACCENT);
    (helper.material as THREE.LineBasicMaterial).depthTest = false;
    (helper.material as THREE.LineBasicMaterial).transparent = true;
    (helper.material as THREE.LineBasicMaterial).opacity = selected ? 1 : 0.8;
    helper.renderOrder = 999;
    group.add(helper);
  }
  return group;
}

function disposeGroup(group: THREE.Group) {
  group.traverse((obj) => {
    const mat = (obj as THREE.Sprite).material as THREE.Material | undefined;
    if (mat) {
      const map = (mat as THREE.SpriteMaterial).map;
      if (map) map.dispose();
      mat.dispose();
    }
    const geometry = (obj as THREE.Mesh).geometry as THREE.BufferGeometry | undefined;
    if (geometry) geometry.dispose();
  });
}

const SPLAT_LOAD_OPTIONS = {
  format: GaussianSplats3D.SceneFormat.Splat,
  splatAlphaRemovalThreshold: 5,
  showLoadingUI: false,
  progressiveLoad: false,
};

export default function SplatViewer({ sceneId, mode, version, leaks, selectedId, sceneCenter, homePose, onSelect, onStatus }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const queueRef = useRef<Promise<unknown>>(Promise.resolve());
  const markersRef = useRef<THREE.Group | null>(null);
  const markerSceneRef = useRef<THREE.Scene>(new THREE.Scene());
  const poseRef = useRef<CameraPose>(homePose);
  const lastSceneRef = useRef<string | null>(null);
  const userInteractedRef = useRef(false);
  const homePoseRef = useRef(homePose);
  homePoseRef.current = homePose;
  const flyRef = useRef<number | null>(null);
  const [viewerReady, setViewerReady] = useState(0);

  // Scene bounds usually arrive after the viewer exists: frame the scene then, unless the user
  // has already taken the camera somewhere.
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || userInteractedRef.current) return;
    viewer.camera.position.set(...homePose.position);
    viewer.controls.target.set(...homePose.target);
    viewer.controls.update?.();
    poseRef.current = homePose;
  }, [homePose, viewerReady]);
  const leaksRef = useRef(leaks);
  const selectRef = useRef(onSelect);
  const statusRef = useRef(onStatus);
  const versionRef = useRef(version);
  leaksRef.current = leaks;
  selectRef.current = onSelect;
  statusRef.current = onStatus;

  // One viewer per (scene, mode). We own the WebGL renderer so the library's dispose() does not
  // try to remove our container from <body>, and so the GL context is released promptly. Each
  // viewer runs its own frame loop: the library's update() + render(), then the markers drawn
  // last so they sit on top of the near-opaque splats. On cleanup the viewer is disposed only
  // once its load has settled, because disposing mid-load makes the library's internal promises
  // reject noisily.
  useEffect(() => {
    if (lastSceneRef.current !== sceneId) {
      // A new scene gets its own framing; a mode switch keeps the camera where the user left it.
      poseRef.current = homePoseRef.current;
      userInteractedRef.current = false;
      lastSceneRef.current = sceneId;
    }
    const container = containerRef.current;
    if (!container) return;
    let cancelled = false;
    let alive = true;

    statusRef.current({ state: "loading", message: `Loading ${mode} splat` });
    const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    const resize = new ResizeObserver(() => renderer.setSize(container.clientWidth, container.clientHeight));
    resize.observe(container);

    const viewer = new GaussianSplats3D.Viewer({
      renderer,
      rootElement: container,
      cameraUp: [0, 1, 0],
      initialCameraPosition: poseRef.current.position,
      initialCameraLookAt: poseRef.current.target,
      sharedMemoryForWorkers: false,
      sphericalHarmonicsDegree: 0,
      selfDrivenMode: false,
      useBuiltInControls: true,
      dynamicScene: false,
      sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant, // default is a slow radial fade; markers would appear seconds before the house
      logLevel: GaussianSplats3D.LogLevel.None,
    });
    viewerRef.current = viewer;
    if (import.meta.env.DEV) Object.assign(window as any, { __heatsplatViewer: viewer, __THREE: THREE });
    // The splat mesh only shows up if at least one ordinary draw call precedes it in the frame
    // (observed with gaussian-splats-3d 0.4.7 + three 0.170: with an empty scene pass the splat
    // pass draws all its triangles but nothing reaches the canvas). A one-triangle mesh that
    // writes neither colour nor depth provides that draw call at zero visible cost.
    const warmup = new THREE.Mesh(
      new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0, 0, 0.001, 0, 0.001, 0, 0], 3)),
      new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false }),
    );
    warmup.frustumCulled = false;
    warmup.name = "scene-pass-warmup";
    viewer.threeScene.add(warmup);
    viewer.controls.addEventListener("start", () => {
      userInteractedRef.current = true;
      viewer.controls.autoRotate = false;
    });

    const frame = () => {
      if (!alive) return;
      viewer.update();
      viewer.render();
      if (viewerRef.current === viewer) {
        const savedAutoClear = renderer.autoClear;
        renderer.autoClear = false;
        renderer.render(markerSceneRef.current, viewer.camera);
        renderer.autoClear = savedAutoClear;
      }
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);

    // The library returns its own thenable (no .finally), so adopt it into a native Promise.
    const load = Promise.resolve(viewer.addSplatScene(api.splatUrl(sceneId, mode, versionRef.current), SPLAT_LOAD_OPTIONS))
      .then(() => {
        if (cancelled) return;
        viewer.controls.autoRotate = !userInteractedRef.current;
        viewer.controls.autoRotateSpeed = 0.5;
        setViewerReady((n) => n + 1); // markers effect re-runs against the new viewer
        statusRef.current({ state: "ready" });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        console.error("splat load failed", error);
        statusRef.current({ state: "error", message: `Could not load the ${mode} splat: ${(error as Error)?.message ?? error}` });
      });
    queueRef.current = load;

    return () => {
      cancelled = true;
      poseRef.current = {
        position: viewer.camera.position.toArray() as [number, number, number],
        target: viewer.controls.target.toArray() as [number, number, number],
      };
      if (viewerRef.current === viewer) viewerRef.current = null;
      void queueRef.current.finally(async () => {
        alive = false;
        resize.disconnect();
        try {
          await viewer.dispose();
        } catch (error) {
          console.warn("viewer dispose failed", error);
        }
        warmup.geometry.dispose();
        warmup.material.dispose();
        renderer.dispose();
        renderer.forceContextLoss();
        renderer.domElement.remove();
      });
    };
  }, [sceneId, mode]);

  // A new data version (after re-analysis) swaps the splat file inside the existing viewer:
  // no renderer teardown, and the previous frame stays on screen until the new file is ready.
  useEffect(() => {
    if (version === versionRef.current) return;
    versionRef.current = version;
    const viewer = viewerRef.current;
    if (!viewer) return;
    queueRef.current = queueRef.current
      .catch(() => undefined)
      .then(async () => {
        if (viewerRef.current !== viewer) return;
        if (viewer.getSceneCount() > 0) await viewer.removeSplatScene(0, false);
        if (viewerRef.current !== viewer) return;
        await viewer.addSplatScene(api.splatUrl(sceneId, mode, version), SPLAT_LOAD_OPTIONS);
      })
      .catch((error: unknown) => {
        if (viewerRef.current !== viewer) return;
        console.error("splat swap failed", error);
        statusRef.current({ state: "error", message: `Could not reload the ${mode} splat: ${(error as Error)?.message ?? error}` });
      });
  }, [version, sceneId, mode]);

  // Rebuild markers when leaks or selection change (cheap, no splat reload).
  useEffect(() => {
    if (!viewerRef.current || viewerReady === 0) return;
    const markerScene = markerSceneRef.current;
    if (markersRef.current) {
      markerScene.remove(markersRef.current);
      disposeGroup(markersRef.current);
    }
    const group = buildMarkers(leaks, selectedId, sceneCenter);
    markerScene.add(group);
    markersRef.current = group;
  }, [leaks, selectedId, sceneCenter, viewerReady]);

  // Release marker GPU resources on unmount.
  useEffect(
    () => () => {
      if (markersRef.current) {
        markerSceneRef.current.remove(markersRef.current);
        disposeGroup(markersRef.current);
        markersRef.current = null;
      }
    },
    [],
  );

  // Fly the camera to the selected leak. Depends on the selection only: a re-analysis that keeps
  // the same selection must not yank the camera back.
  useEffect(() => {
    const viewer = viewerRef.current;
    const leak = leaksRef.current.find((l) => l.id === selectedId);
    if (!viewer || !leak) return;
    viewer.controls.autoRotate = false;
    userInteractedRef.current = true;
    const normal = leakNormal(leak, sceneCenter);
    const target = new THREE.Vector3(...leak.centroid);
    const destination = target.clone().addScaledVector(normal, FLY_DISTANCE_M).add(new THREE.Vector3(0, 1.2, 0));
    const startPos = viewer.camera.position.clone();
    const startTarget = viewer.controls.target.clone();
    const t0 = performance.now();
    if (flyRef.current) cancelAnimationFrame(flyRef.current);
    const step = (now: number) => {
      if (viewerRef.current !== viewer) return;
      const t = Math.min(1, (now - t0) / FLY_MS);
      const ease = t * t * (3 - 2 * t);
      viewer.camera.position.lerpVectors(startPos, destination, ease);
      viewer.controls.target.lerpVectors(startTarget, target, ease);
      viewer.controls.update();
      if (t < 1) flyRef.current = requestAnimationFrame(step);
    };
    flyRef.current = requestAnimationFrame(step);
    return () => {
      if (flyRef.current) cancelAnimationFrame(flyRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId]);

  // Click-to-select on marker sprites.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let down: { x: number; y: number } | null = null;
    const raycaster = new THREE.Raycaster();
    const onDown = (e: PointerEvent) => {
      down = { x: e.clientX, y: e.clientY };
    };
    const onUp = (e: PointerEvent) => {
      const viewer = viewerRef.current;
      const markers = markersRef.current;
      if (!down || !viewer || !markers) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      down = null;
      if (moved > CLICK_SLOP_PX) return;
      const rect = container.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(ndc, viewer.camera);
      const sprites = markers.children.filter((c) => (c as THREE.Sprite).isSprite);
      const hit = raycaster.intersectObjects(sprites, false)[0];
      if (hit?.object.userData.leakId) selectRef.current(hit.object.userData.leakId as string);
    };
    container.addEventListener("pointerdown", onDown);
    container.addEventListener("pointerup", onUp);
    return () => {
      container.removeEventListener("pointerdown", onDown);
      container.removeEventListener("pointerup", onUp);
    };
  }, []);

  return <div ref={containerRef} className="viewer" />;
}
