/*
// src/components/FaceCanvas.tsx
import React, { useEffect, useRef, useState } from "react";
import { Stage, Layer, Image as KonvaImage, Circle, Line, Text, Group } from "react-konva";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import { nanoid } from "nanoid";

type Landmark = [number, number, number?];

const DEFAULT_STAGE_WIDTH = 800;
const DEFAULT_STAGE_HEIGHT = 800;

function fitImage(imageWidth: number, imageHeight: number, stageW: number, stageH: number) {
  const imgRatio = imageWidth / imageHeight;
  const stageRatio = stageW / stageH;
  let width = stageW;
  let height = stageH;
  if (imgRatio > stageRatio) {
    width = stageW;
    height = Math.round(stageW / imgRatio);
  } else {
    height = stageH;
    width = Math.round(stageH * imgRatio);
  }
  return { width, height };
}

export default function FaceCanvas(): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [imgEl, setImgEl] = useState<HTMLImageElement | null>(null);
  const stageRef = useRef<any>(null);

  const [landmarks, setLandmarks] = useState<Landmark[]>([]);
  const [detector, setDetector] = useState<any | null>(null);
  const [points, setPoints] = useState<{ id: string; x: number; y: number }[]>([]);
  const [stageSize, setStageSize] = useState({ width: DEFAULT_STAGE_WIDTH, height: DEFAULT_STAGE_HEIGHT });

  // file -> data URL
  useEffect(() => {
    if (!file) {
      setImageSrc(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setImageSrc(String(reader.result));
    reader.onerror = (e) => {
      console.error("FileReader error", e);
      setImageSrc(null);
    };
    reader.readAsDataURL(file);
  }, [file]);

  // data URL -> native Image element (crossOrigin anonymous)
  useEffect(() => {
    if (!imageSrc) {
      setImgEl(null);
      return;
    }
    let mounted = true;
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = imageSrc;
    image.onload = () => {
      if (!mounted) return;
      setImgEl(image);
    };
    image.onerror = (err) => {
      console.error("Image load error", err);
      if (mounted) setImgEl(null);
    };
    return () => {
      mounted = false;
    };
  }, [imageSrc]);

  // create detector (try modern createDetector; fallback to legacy load)
  useEffect(() => {
    let mounted = true;
    let created: any = null;

    async function setup() {
      try {
        await tf.setBackend("webgl");
        await tf.ready();

        const modelConst = (faceLandmarksDetection as any).SupportedModels
          ? (faceLandmarksDetection as any).SupportedModels.MediaPipeFaceMesh
          : (faceLandmarksDetection as any).MediaPipeFaceMesh; // defensive

        // modern config (tfjs runtime)
        const detectorConfigTfjs = {
          runtime: "tfjs",
          maxFaces: 1,
          refineLandmarks: true,
        };

        // Try modern API first (createDetector). Use any-cast to avoid TS signature mismatch
        try {
          created = await (faceLandmarksDetection as any).createDetector(modelConst, detectorConfigTfjs);
          if (mounted) {
            setDetector(created);
            console.log("Created detector via createDetector (tfjs).");
          } else {
            if (created?.dispose) created.dispose();
          }
        } catch (errCreate) {
          console.warn("createDetector failed; attempting legacy load fallback:", errCreate);
          // fallback to older API: faceLandmarksDetection.load(...)
          try {
            const legacy = await (faceLandmarksDetection as any).load(
              (faceLandmarksDetection as any).SupportedPackages?.mediapipeFacemesh ??
                (faceLandmarksDetection as any).SupportedPackages,
              { maxFaces: 1 }
            );
            created = legacy;
            if (mounted) {
              setDetector(created);
              console.log("Detector created via legacy load().");
            } else {
              // no-op
            }
          } catch (errLegacy) {
            console.error("Both createDetector and legacy load() failed:", errLegacy);
            throw errLegacy;
          }
        }
      } catch (err) {
        console.error("Detector setup overall failed:", err);
      }
    }

    setup();

    return () => {
      mounted = false;
      if (created && typeof created.dispose === "function") {
        try {
          created.dispose();
        } catch (e) {
          // ignore
        }
      }
    };
  }, []);

  // set stage size to fit image
  useEffect(() => {
    if (!imgEl) {
      setStageSize({ width: 800, height: 600 });
      return;
    }
    const w = imgEl.naturalWidth || DEFAULT_STAGE_WIDTH;
    const h = imgEl.naturalHeight || DEFAULT_STAGE_HEIGHT;
    const fit = fitImage(w, h, DEFAULT_STAGE_WIDTH, DEFAULT_STAGE_HEIGHT);
    setStageSize({ width: fit.width, height: fit.height });
  }, [imgEl]);

  // detection: robust attempt of multiple estimateFaces signatures
  useEffect(() => {
    if (!detector || !imgEl) return;
    let cancelled = false;

    async function detectOnce() {
      try {
        const naturalW = imgEl.naturalWidth || DEFAULT_STAGE_WIDTH;
        const naturalH = imgEl.naturalHeight || DEFAULT_STAGE_HEIGHT;
        const fit = fitImage(naturalW, naturalH, stageSize.width, stageSize.height);

        const off = document.createElement("canvas");
        off.width = fit.width;
        off.height = fit.height;
        const ctx = off.getContext("2d")!;
        ctx.clearRect(0, 0, off.width, off.height);
        ctx.drawImage(imgEl, 0, 0, fit.width, fit.height);

        console.log("Detector:", detector);
        console.log("estimateFaces type:", typeof (detector as any).estimateFaces);

        let predictions: any = null;
        let lastErr: any = null;

        // Try multiple call shapes
        try {
          predictions = await (detector as any).estimateFaces(off);
          console.log("estimateFaces(off) succeeded");
        } catch (e1) {
          lastErr = e1;
          console.warn("estimateFaces(off) failed ->", e1);
          try {
            predictions = await (detector as any).estimateFaces({ input: off });
            console.log("estimateFaces({input: off}) succeeded");
          } catch (e2) {
            lastErr = e2;
            console.warn("estimateFaces({input: off}) failed ->", e2);
            try {
              predictions = await (detector as any).estimateFaces(off, { flipHorizontal: false });
              console.log("estimateFaces(off, config) succeeded");
            } catch (e3) {
              lastErr = e3;
              console.warn("estimateFaces(off, config) failed ->", e3);
            }
          }
        }

        if (!predictions) {
          console.error("All estimateFaces forms failed; last error:", lastErr);
          throw lastErr;
        }

        if (cancelled) return;

        if (!predictions || predictions.length === 0) {
          setLandmarks([]);
          setPoints([]);
          return;
        }

        const face = predictions[0] as any;
        let mesh: Landmark[] = [];

        if (face?.scaledMesh && Array.isArray(face.scaledMesh)) {
          mesh = face.scaledMesh as Landmark[];
        } else if (face?.mesh && Array.isArray(face.mesh)) {
          mesh = face.mesh as Landmark[];
        } else if (face?.keypoints && Array.isArray(face.keypoints)) {
          mesh = (face.keypoints as any[]).map((kp) => [kp.x, kp.y, kp.z ?? 0]);
        } else if (face?.keypoints3D && Array.isArray(face.keypoints3D)) {
          mesh = face.keypoints3D as Landmark[];
        } else {
          console.warn("Unknown prediction shape:", face);
        }

        setLandmarks(mesh || []);

        const lipsIndices = [13, 14, 78, 308];
        const defaultPoints = lipsIndices
          .map((i) => {
            const p = mesh && mesh[i];
            if (!p) return null;
            return { id: nanoid(), x: p[0], y: p[1] };
          })
          .filter(Boolean) as { id: string; x: number; y: number }[];

        setPoints(defaultPoints);
      } catch (err) {
        console.error("detectOnce final error:", err);
      }
    }

    detectOnce();

    return () => {
      cancelled = true;
    };
  }, [detector, imgEl, stageSize]);

  function onAddPoint() {
    setPoints((p) => [...p, { id: nanoid(), x: stageSize.width / 2, y: stageSize.height / 2 }]);
  }

  function onExport() {
    if (!stageRef.current) return;
    try {
      const uri = stageRef.current.toDataURL({ pixelRatio: 2 });
      const link = document.createElement("a");
      link.download = `face_map_${Date.now()}.png`;
      link.href = uri;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error("Export error:", err);
      alert("Export failed (canvas may be tainted). Make sure image is loaded from same origin or crossOrigin='anonymous'.");
    }
  }

  return (
    <div style={{ border: "1px solid #ddd", padding: 12, borderRadius: 8, maxWidth: 1000 }}>
      <div style={{ marginBottom: 12 }}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            if (!e.target.files?.[0]) return;
            setFile(e.target.files[0]);
          }}
        />
        <button onClick={onAddPoint} style={{ marginLeft: 8 }}>
          Add point
        </button>
        <button onClick={onExport} style={{ marginLeft: 8 }}>
          Export PNG
        </button>
      </div>

      <div style={{ display: "flex", gap: 12 }}>
        <div style={{ border: "1px solid #eee", padding: 6 }}>
          <Stage width={stageSize.width} height={stageSize.height} ref={stageRef}>
            <Layer>
              {imgEl ? (
                <KonvaImage image={imgEl as any} x={0} y={0} width={stageSize.width} height={stageSize.height} />
              ) : null}
            </Layer>

            <Layer>
              {landmarks.length > 0
                ? landmarks.map((lm, idx) => {
                    const x = lm[0];
                    const y = lm[1];
                    return <Circle key={`lm-${idx}`} x={x} y={y} radius={2} fill="rgba(0,150,255,0.9)" />;
                  })
                : null}

              {landmarks.length > 155 ? (
                <>
                  <Line
                    points={[landmarks[10][0], landmarks[10][1], landmarks[152][0], landmarks[152][1]]}
                    stroke="rgba(255,0,0,0.7)"
                    strokeWidth={1}
                  />
                  <Line
                    points={[landmarks[33][0], landmarks[33][1], landmarks[263][0], landmarks[263][1]]}
                    stroke="rgba(0,200,0,0.5)"
                    strokeWidth={1}
                  />
                </>
              ) : null}

              {points.length > 0
                ? points.map((p) => (
                    <Group
                      key={p.id}
                      x={p.x}
                      y={p.y}
                      draggable
                      onDragEnd={(e) => {
                        const nx = e.target.x();
                        const ny = e.target.y();
                        setPoints((prev) => prev.map((q) => (q.id === p.id ? { ...q, x: nx, y: ny } : q)));
                      }}
                    >
                      <Circle radius={8} fill="rgba(255,165,0,0.9)" stroke="black" strokeWidth={1} />
                      <Text text="●" fontSize={12} offsetX={6} offsetY={6} />
                    </Group>
                  ))
                : null}
            </Layer>
          </Stage>
        </div>

        <div style={{ width: 300 }}>
          <h3>Session</h3>
          <div>
            <strong>Detected points:</strong> {landmarks.length}
          </div>

          <div style={{ marginTop: 12 }}>
            <h4>Injection points</h4>
            {points.length > 0
              ? points.map((p) => (
                  <div key={p.id} style={{ marginBottom: 6 }}>
                    <small>ID: {p.id}</small>
                    <div>
                      x: {Math.round(p.x)}, y: {Math.round(p.y)}{" "}
                      <button
                        onClick={() => setPoints((ps) => ps.filter((q) => q.id !== p.id))}
                        style={{ marginLeft: 6 }}
                      >
                        remove
                      </button>
                    </div>
                  </div>
                ))
              : null}
          </div>

          <div style={{ marginTop: 12 }}>
            <small style={{ color: "#666" }}>
              Note: All image processing is performed locally in your browser. No images are uploaded.
            </small>
          </div>
        </div>
      </div>
    </div>
  );
}
  */

/* WORKING VERSION!
// src/components/FaceCanvas.tsx
import React, { useEffect, useRef, useState } from "react";
import {
  Stage,
  Layer,
  Image as KonvaImageElement,
  Circle,
  Line,
  Text,
  Group,
} from "react-konva";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import { nanoid } from "nanoid";

type Landmark = [number, number, number?];

const DEFAULT_STAGE_WIDTH = 800;
const DEFAULT_STAGE_HEIGHT = 800;

function fitImage(imageWidth: number, imageHeight: number, stageW: number, stageH: number) {
  const imgRatio = imageWidth / imageHeight;
  const stageRatio = stageW / stageH;
  let width = stageW;
  let height = stageH;
  if (imgRatio > stageRatio) {
    width = stageW;
    height = Math.round(stageW / imgRatio);
  } else {
    height = stageH;
    width = Math.round(stageH * imgRatio);
  }
  return { width, height };
}

// @ts-ignore
export default function FaceCanvas(): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  // typed as HTMLImageElement | null for correctness — we'll cast to any for Konva rendering
  const [imgEl, setImgEl] = useState<HTMLImageElement | null>(null);

  // use a generic ref type to avoid strict Konva typing complaints
  const stageRef = useRef<any>(null);

  const [landmarks, setLandmarks] = useState<Landmark[]>([]);
  const [detector, setDetector] = useState<any | null>(null);
  const [points, setPoints] = useState<{ id: string; x: number; y: number }[]>([]);
  const [stageSize, setStageSize] = useState({ width: DEFAULT_STAGE_WIDTH, height: DEFAULT_STAGE_HEIGHT });

  // file -> data URL
  useEffect(() => {
    if (!file) {
      setImageSrc(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setImageSrc(String(reader.result));
    reader.onerror = (e) => {
      console.error("FileReader error", e);
      setImageSrc(null);
    };
    reader.readAsDataURL(file);
  }, [file]);

  // data URL -> HTMLImageElement (crossOrigin to avoid tainting)
  useEffect(() => {
    if (!imageSrc) {
      setImgEl(null);
      return;
    }
    let mounted = true;
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = imageSrc;
    image.onload = () => {
      if (!mounted) return;
      setImgEl(image);
    };
    image.onerror = (err) => {
      console.error("Image load error", err);
      if (mounted) setImgEl(null);
    };
    return () => {
      mounted = false;
    };
  }, [imageSrc]);

  // create detector (modern API with fallback)
  useEffect(() => {
    let mounted = true;
    let created: any = null;

    async function setup() {
      try {
        await tf.setBackend("webgl");
        await tf.ready();

        const modelConst = (faceLandmarksDetection as any).SupportedModels?.MediaPipeFaceMesh ??
                           (faceLandmarksDetection as any).MediaPipeFaceMesh;

        const detectorConfigTfjs = { runtime: "tfjs", maxFaces: 1, refineLandmarks: true };

        try {
          created = await (faceLandmarksDetection as any).createDetector(modelConst, detectorConfigTfjs);
          if (mounted) setDetector(created);
          else if (created?.dispose) created.dispose();
        } catch (errCreate) {
          console.warn("createDetector failed; trying legacy load():", errCreate);
          try {
            // legacy load fallback
            created = await (faceLandmarksDetection as any).load(
              (faceLandmarksDetection as any).SupportedPackages?.mediapipeFacemesh ??
                (faceLandmarksDetection as any).SupportedPackages,
              { maxFaces: 1 }
            );
            if (mounted) setDetector(created);
          } catch (errLegacy) {
            console.error("Both createDetector and legacy load() failed:", errLegacy);
            throw errLegacy;
          }
        }
      } catch (err) {
        console.error("Detector setup error:", err);
      }
    }

    setup();

    return () => {
      mounted = false;
      if (created && typeof created.dispose === "function") {
        try {
          created.dispose();
        } catch {}
      }
    };
  }, []);

  // stage size adjust
  useEffect(() => {
    if (!imgEl) {
      setStageSize({ width: 800, height: 600 });
      return;
    }
    const w = imgEl.naturalWidth || DEFAULT_STAGE_WIDTH;
    const h = imgEl.naturalHeight || DEFAULT_STAGE_HEIGHT;
    const fit = fitImage(w, h, DEFAULT_STAGE_WIDTH, DEFAULT_STAGE_HEIGHT);
    setStageSize({ width: fit.width, height: fit.height });
  }, [imgEl]);

  // detection with multiple estimateFaces call forms
  useEffect(() => {
    if (!detector || !imgEl) return;
    let cancelled = false;

    async function detectOnce() {
      try {
        const naturalW = imgEl.naturalWidth || DEFAULT_STAGE_WIDTH;
        const naturalH = imgEl.naturalHeight || DEFAULT_STAGE_HEIGHT;
        const fit = fitImage(naturalW, naturalH, stageSize.width, stageSize.height);

        const off = document.createElement("canvas");
        off.width = fit.width;
        off.height = fit.height;
        const ctx = off.getContext("2d")!;
        ctx.clearRect(0, 0, off.width, off.height);
        ctx.drawImage(imgEl, 0, 0, fit.width, fit.height);

        console.log("Detector ready:", detector);
        console.log("estimateFaces exists:", typeof (detector as any).estimateFaces);

        let preds: any = null;
        let lastErr: any = null;

        try {
          preds = await (detector as any).estimateFaces(off);
          console.log("estimateFaces(off) OK");
        } catch (e1) {
          lastErr = e1;
          try {
            preds = await (detector as any).estimateFaces({ input: off });
            console.log("estimateFaces({input:off}) OK");
          } catch (e2) {
            lastErr = e2;
            try {
              preds = await (detector as any).estimateFaces(off, { flipHorizontal: false });
              console.log("estimateFaces(off, config) OK");
            } catch (e3) {
              lastErr = e3;
            }
          }
        }

        if (!preds) {
          console.error("All estimateFaces attempts failed:", lastErr);
          throw lastErr;
        }

        if (cancelled) return;

        if (!preds || preds.length === 0) {
          setLandmarks([]);
          setPoints([]);
          return;
        }

        const face = preds[0] as any;
        let mesh: Landmark[] = [];

        if (Array.isArray(face?.scaledMesh)) mesh = face.scaledMesh as Landmark[];
        else if (Array.isArray(face?.mesh)) mesh = face.mesh as Landmark[];
        else if (Array.isArray(face?.keypoints)) mesh = (face.keypoints as any[]).map(kp => [kp.x, kp.y, kp.z ?? 0]);
        else if (Array.isArray(face?.keypoints3D)) mesh = face.keypoints3D as Landmark[];
        else console.warn("Unknown prediction shape", face);

        setLandmarks(mesh || []);

        const lipsIndices = [13, 14, 78, 308];
        const defaultPoints = lipsIndices
          .map(i => {
            const p = mesh?.[i];
            if (!p) return null;
            return { id: nanoid(), x: p[0], y: p[1] };
          })
          .filter(Boolean) as { id: string; x: number; y: number }[];

        setPoints(defaultPoints);
      } catch (err) {
        console.error("detectOnce error:", err);
      }
    }

    detectOnce();

    return () => {
      cancelled = true;
    };
  }, [detector, imgEl, stageSize]);

  function onAddPoint() {
    setPoints(p => [...p, { id: nanoid(), x: stageSize.width / 2, y: stageSize.height / 2 }]);
  }

  function onExport() {
    if (!stageRef.current) return;
    try {
      // @ts-ignore: Konva stage instance typing mismatch sometimes throws; runtime is fine
      const uri = stageRef.current.toDataURL({ pixelRatio: 2 });
      const link = document.createElement("a");
      link.download = `face_map_${Date.now()}.png`;
      link.href = uri;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error("Export error:", err);
      alert("Export failed (canvas may be tainted). Ensure image had crossOrigin='anonymous'.");
    }
  }

  return (
    <div style={{ border: "1px solid #ddd", padding: 12, borderRadius: 8, maxWidth: 1000 }}>
      <div style={{ marginBottom: 12 }}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            if (!e.target.files?.[0]) return;
            setFile(e.target.files[0]);
          }}
        />
        <button onClick={onAddPoint} style={{ marginLeft: 8 }}>
          Add point
        </button>
        <button onClick={onExport} style={{ marginLeft: 8 }}>
          Export PNG
        </button>
      </div>

      <div style={{ display: "flex", gap: 12 }}>
        <div style={{ border: "1px solid #eee", padding: 6 }}>
          <Stage width={stageSize.width} height={stageSize.height} ref={stageRef}>
            <Layer>
              {/* cast to any for Konva image prop to avoid strict signature mismatch }
              {imgEl ? (
                // @ts-ignore: Konva's Image prop expects many possible types; runtime OK
                <KonvaImageElement image={imgEl as any} x={0} y={0} width={stageSize.width} height={stageSize.height} />
              ) : null}
            </Layer>

            <Layer>
              {landmarks.length > 0
                ? landmarks.map((lm, idx) => {
                    const x = lm[0];
                    const y = lm[1];
                    return <Circle key={`lm-${idx}`} x={x} y={y} radius={2} fill="rgba(0,150,255,0.9)" />;
                  })
                : null}

              {landmarks.length > 155 ? (
                <>
                  <Line
                    points={[landmarks[10][0], landmarks[10][1], landmarks[152][0], landmarks[152][1]]}
                    stroke="rgba(255,0,0,0.7)"
                    strokeWidth={1}
                  />
                  <Line
                    points={[landmarks[33][0], landmarks[33][1], landmarks[263][0], landmarks[263][1]]}
                    stroke="rgba(0,200,0,0.5)"
                    strokeWidth={1}
                  />
                </>
              ) : null}

              {points.length > 0
                ? points.map((p) => (
                    <Group
                      key={p.id}
                      x={p.x}
                      y={p.y}
                      draggable
                      onDragEnd={(e) => {
                        const nx = (e.target as any).x();
                        const ny = (e.target as any).y();
                        setPoints(prev => prev.map(q => (q.id === p.id ? { ...q, x: nx, y: ny } : q)));
                      }}
                    >
                      <Circle radius={8} fill="rgba(255,165,0,0.9)" stroke="black" strokeWidth={1} />
                      <Text text="●" fontSize={12} offsetX={6} offsetY={6} />
                    </Group>
                  ))
                : null}
            </Layer>
          </Stage>
        </div>

        <div style={{ width: 300 }}>
          <h3>Session</h3>
          <div>
            <strong>Detected points:</strong> {landmarks.length}
          </div>

          <div style={{ marginTop: 12 }}>
            <h4>Injection points</h4>
            {points.length > 0
              ? points.map(p => (
                  <div key={p.id} style={{ marginBottom: 6 }}>
                    <small>ID: {p.id}</small>
                    <div>
                      x: {Math.round(p.x)}, y: {Math.round(p.y)}{" "}
                      <button onClick={() => setPoints(ps => ps.filter(q => q.id !== p.id))} style={{ marginLeft: 6 }}>
                        remove
                      </button>
                    </div>
                  </div>
                ))
              : null}
          </div>

          <div style={{ marginTop: 12 }}>
            <small style={{ color: "#666" }}>Note: All image processing runs locally in your browser.</small>
          </div>
        </div>
      </div>
    </div>
  );
}

*/

/* works but lines and points are not calibrated
// src/components/FaceCanvas.tsx
import React, { useEffect, useRef, useState } from "react";
import {
  Stage,
  Layer,
  Image as KonvaImageElement,
  Circle,
  Line,
  Text,
  Group,
} from "react-konva";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import { nanoid } from "nanoid";

type Landmark = [number, number, number?];

const DEFAULT_STAGE_WIDTH = 800;
const DEFAULT_STAGE_HEIGHT = 800;

function fitImage(imageWidth: number, imageHeight: number, stageW: number, stageH: number) {
  const imgRatio = imageWidth / imageHeight;
  const stageRatio = stageW / stageH;
  let width = stageW;
  let height = stageH;
  if (imgRatio > stageRatio) {
    width = stageW;
    height = Math.round(stageW / imgRatio);
  } else {
    height = stageH;
    width = Math.round(stageH * imgRatio);
  }
  return { width, height };
}

// vector helpers
function vecSub(a: { x: number; y: number }, b: { x: number; y: number }) {
  return { x: a.x - b.x, y: a.y - b.y };
}
function vecDot(a: { x: number; y: number }, b: { x: number; y: number }) {
  return a.x * b.x + a.y * b.y;
}
function vecLen(a: { x: number; y: number }) {
  return Math.hypot(a.x, a.y);
}
function vecScale(a: { x: number; y: number }, s: number) {
  return { x: a.x * s, y: a.y * s };
}
function vecAdd(a: { x: number; y: number }, b: { x: number; y: number }) {
  return { x: a.x + b.x, y: a.y + b.y };
}
function perpDistanceToLine(point: { x: number; y: number }, a: { x: number; y: number }, b: { x: number; y: number }) {
  // returns {dist, side} where side >0 or <0 indicates which side (by cross)
  const v = vecSub(b, a);
  const w = vecSub(point, a);
  const vlen2 = v.x * v.x + v.y * v.y;
  if (vlen2 === 0) return { dist: vecLen(w), side: 0 };
  // projection scalar
  const t = vecDot(w, v) / vlen2;
  // projection point
  const proj = vecAdd(a, vecScale(v, t));
  const perp = vecSub(point, proj);
  const dist = vecLen(perp);
  // cross product z (v x w)
  const cross = v.x * w.y - v.y * w.x;
  const side = cross; // sign indicates side
  return { dist, side };
}
// @ts-ignore
export default function FaceCanvas(): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [imgEl, setImgEl] = useState<HTMLImageElement | null>(null);
  const stageRef = useRef<any>(null);

  const [landmarks, setLandmarks] = useState<Landmark[]>([]);
  const [detector, setDetector] = useState<any | null>(null);
  const [points, setPoints] = useState<{ id: string; x: number; y: number }[]>([]);
  const [stageSize, setStageSize] = useState({ width: DEFAULT_STAGE_WIDTH, height: DEFAULT_STAGE_HEIGHT });

  // toggles for reference lines
  const [showMidline, setShowMidline] = useState(true);
  const [showEyeLine, setShowEyeLine] = useState(true);
  const [showThirds, setShowThirds] = useState(true);

  // convert selected file to data URL
  useEffect(() => {
    if (!file) {
      setImageSrc(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setImageSrc(String(reader.result));
    reader.onerror = (e) => {
      console.error("FileReader error", e);
      setImageSrc(null);
    };
    reader.readAsDataURL(file);
  }, [file]);

  // data URL -> native Image element (crossOrigin)
  useEffect(() => {
    if (!imageSrc) {
      setImgEl(null);
      return;
    }
    let mounted = true;
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = imageSrc;
    image.onload = () => {
      if (!mounted) return;
      setImgEl(image);
    };
    image.onerror = (err) => {
      console.error("Image load error", err);
      if (mounted) setImgEl(null);
    };
    return () => {
      mounted = false;
    };
  }, [imageSrc]);

  // create detector (modern API with fallback to legacy)
  useEffect(() => {
    let mounted = true;
    let created: any = null;

    async function setup() {
      try {
        await tf.setBackend("webgl");
        await tf.ready();

        const modelConst = (faceLandmarksDetection as any).SupportedModels?.MediaPipeFaceMesh ??
                           (faceLandmarksDetection as any).MediaPipeFaceMesh;

        const detectorConfigTfjs = { runtime: "tfjs", maxFaces: 1, refineLandmarks: true };

        try {
          created = await (faceLandmarksDetection as any).createDetector(modelConst, detectorConfigTfjs);
          if (mounted) setDetector(created);
          else if (created?.dispose) created.dispose();
        } catch (errCreate) {
          console.warn("createDetector failed; trying legacy load():", errCreate);
          try {
            created = await (faceLandmarksDetection as any).load(
              (faceLandmarksDetection as any).SupportedPackages?.mediapipeFacemesh ??
                (faceLandmarksDetection as any).SupportedPackages,
              { maxFaces: 1 }
            );
            if (mounted) setDetector(created);
          } catch (errLegacy) {
            console.error("Both createDetector and legacy load() failed:", errLegacy);
            throw errLegacy;
          }
        }
      } catch (err) {
        console.error("Detector setup error:", err);
      }
    }

    setup();

    return () => {
      mounted = false;
      if (created && typeof created.dispose === "function") {
        try {
          created.dispose();
        } catch {}
      }
    };
  }, []);

  // update stage size to fit image
  useEffect(() => {
    if (!imgEl) {
      setStageSize({ width: 800, height: 600 });
      return;
    }
    const naturalW = imgEl.naturalWidth || DEFAULT_STAGE_WIDTH;
    const naturalH = imgEl.naturalHeight || DEFAULT_STAGE_HEIGHT;
    const fit = fitImage(naturalW, naturalH, DEFAULT_STAGE_WIDTH, DEFAULT_STAGE_HEIGHT);
    setStageSize({ width: fit.width, height: fit.height });
  }, [imgEl]);

  // detection (re-uses robust attempt)
  useEffect(() => {
    if (!detector || !imgEl) return;
    let cancelled = false;

    async function detectOnce() {
      try {
        const naturalW = imgEl.naturalWidth || DEFAULT_STAGE_WIDTH;
        const naturalH = imgEl.naturalHeight || DEFAULT_STAGE_HEIGHT;
        const fit = fitImage(naturalW, naturalH, stageSize.width, stageSize.height);

        const off = document.createElement("canvas");
        off.width = fit.width;
        off.height = fit.height;
        const ctx = off.getContext("2d")!;
        ctx.clearRect(0, 0, off.width, off.height);
        ctx.drawImage(imgEl, 0, 0, fit.width, fit.height);

        let predictions: any = null;
        let lastErr: any = null;

        try {
          predictions = await (detector as any).estimateFaces(off);
        } catch (e1) {
          lastErr = e1;
          try {
            predictions = await (detector as any).estimateFaces({ input: off });
          } catch (e2) {
            lastErr = e2;
            try {
              predictions = await (detector as any).estimateFaces(off, { flipHorizontal: false });
            } catch (e3) {
              lastErr = e3;
            }
          }
        }

        if (!predictions) {
          console.error("All estimateFaces attempts failed:", lastErr);
          throw lastErr;
        }

        if (cancelled) return;

        if (!predictions || predictions.length === 0) {
          setLandmarks([]);
          setPoints([]);
          return;
        }

        const face = predictions[0] as any;
        let mesh: Landmark[] = [];

        if (Array.isArray(face?.scaledMesh)) mesh = face.scaledMesh as Landmark[];
        else if (Array.isArray(face?.mesh)) mesh = face.mesh as Landmark[];
        else if (Array.isArray(face?.keypoints)) mesh = (face.keypoints as any[]).map(kp => [kp.x, kp.y, kp.z ?? 0]);
        else if (Array.isArray(face?.keypoints3D)) mesh = face.keypoints3D as Landmark[];
        else console.warn("Unknown prediction shape", face);

        setLandmarks(mesh || []);

        // populate default points if none exist (don't overwrite existing points)
        if ((!points || points.length === 0) && mesh && mesh.length > 0) {
          const lipsIndices = [13, 14, 78, 308];
          const defaultPoints = lipsIndices
            .map((i) => {
              const p = mesh && mesh[i];
              if (!p) return null;
              return { id: nanoid(), x: p[0], y: p[1] };
            })
            .filter(Boolean) as { id: string; x: number; y: number }[];
          setPoints(defaultPoints);
        }
      } catch (err) {
        console.error("detectOnce error:", err);
      }
    }

    detectOnce();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detector, imgEl]);

  function onAddPoint() {
    setPoints((p) => [...p, { id: nanoid(), x: stageSize.width / 2, y: stageSize.height / 2 }]);
  }

  function onExport() {
    if (!stageRef.current) return;
    try {
      // @ts-ignore (kept intentionally)
      const uri = stageRef.current.toDataURL({ pixelRatio: 2 });
      const link = document.createElement("a");
      link.download = `face_map_${Date.now()}.png`;
      link.href = uri;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error("Export error:", err);
      alert("Export failed (canvas may be tainted). Make sure image is loaded from same origin or crossOrigin='anonymous'.");
    }
  }

  // compute reference line points (midline endpoints, eye points, thirds)
  const midlineEndpoints =
    landmarks.length > 152 && landmarks[10] && landmarks[152]
      ? { a: { x: landmarks[10][0], y: landmarks[10][1] }, b: { x: landmarks[152][0], y: landmarks[152][1] } }
      : // fallback: use minY and maxY of landmarks with center x
        (() => {
          if (!landmarks || landmarks.length === 0) return null;
          let minY = Infinity,
            maxY = -Infinity;
          let minX = Infinity,
            maxX = -Infinity;
          for (const lm of landmarks) {
            if (!lm) continue;
            const x = lm[0],
              y = lm[1];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
          const cx = (minX + maxX) / 2;
          return { a: { x: cx, y: minY }, b: { x: cx, y: maxY } };
        })();

  const eyeLine =
    landmarks.length > 263 && landmarks[33] && landmarks[263]
      ? { l: { x: landmarks[33][0], y: landmarks[33][1] }, r: { x: landmarks[263][0], y: landmarks[263][1] } }
      : null;

  const thirdsYs = (() => {
    if (!landmarks || landmarks.length === 0) return null;
    let minY = Infinity,
      maxY = -Infinity;
    let minX = Infinity,
      maxX = -Infinity;
    for (const lm of landmarks) {
      if (!lm) continue;
      const x = lm[0],
        y = lm[1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const h = maxY - minY;
    if (!isFinite(h) || h <= 0) return null;
    const y1 = minY + h / 3;
    const y2 = minY + (2 * h) / 3;
    return { y1, y2, xmin: minX, xmax: maxX };
  })();

  // compute offsets for each annotation point relative to midline
  const pointOffsets = points.map((pt) => {
    if (!midlineEndpoints) return { id: pt.id, dist: 0, side: 0 };
    const { a, b } = midlineEndpoints;
    const { dist, side } = perpDistanceToLine({ x: pt.x, y: pt.y }, a, b);
    // side sign normalization: positive => left or right depending on orientation; we'll just show L/R
    return { id: pt.id, dist: Math.round(dist), side: side };
  });

  // helper to format side as L/R
  function sideLabel(side: number) {
    if (side === 0) return "C";
    // sign positive/negative depends on orientation; decide convention: positive => left
    return side > 0 ? "L" : "R";
  }

  return (
    <div style={{ border: "1px solid #ddd", padding: 12, borderRadius: 8, maxWidth: 1100 }}>
      <div style={{ marginBottom: 12, display: "flex", gap: 12, alignItems: "center" }}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            if (!e.target.files?.[0]) return;
            setFile(e.target.files[0]);
            // reset points/landmarks on new image
            setPoints([]);
            setLandmarks([]);
          }}
        />
        <button onClick={onAddPoint}>Add point</button>
        <button onClick={onExport}>Export PNG</button>

        <div style={{ marginLeft: 16, display: "flex", gap: 8, alignItems: "center" }}>
          <label>
            <input type="checkbox" checked={showMidline} onChange={(e) => setShowMidline(e.target.checked)} /> Midline
          </label>
          <label>
            <input type="checkbox" checked={showEyeLine} onChange={(e) => setShowEyeLine(e.target.checked)} /> Eye-line
          </label>
          <label>
            <input type="checkbox" checked={showThirds} onChange={(e) => setShowThirds(e.target.checked)} /> Thirds
          </label>
        </div>
      </div>

      <div style={{ display: "flex", gap: 12 }}>
        <div style={{ border: "1px solid #eee", padding: 6 }}>
          <Stage width={stageSize.width} height={stageSize.height} ref={stageRef}>
            <Layer>
              {imgEl ? (
                // @ts-ignore
                <KonvaImageElement image={imgEl as any} x={0} y={0} width={stageSize.width} height={stageSize.height} />
              ) : null}
            </Layer>

            <Layer>
              {/* draw landmarks as tiny dots }
              {landmarks.length > 0
                ? landmarks.map((lm, idx) => {
                    const x = lm[0];
                    const y = lm[1];
                    return <Circle key={`lm-${idx}`} x={x} y={y} radius={1.2} fill="rgba(0,150,255,0.9)" />;
                  })
                : null}

              {/* reference: midline }
              {showMidline && midlineEndpoints ? (
                <>
                  <Line
                    points={[midlineEndpoints.a.x, midlineEndpoints.a.y, midlineEndpoints.b.x, midlineEndpoints.b.y]}
                    stroke="rgba(255,0,0,0.85)"
                    strokeWidth={1.5}
                    dash={[6, 4]}
                  />
                  <Text
                    text="Midline"
                    x={(midlineEndpoints.a.x + midlineEndpoints.b.x) / 2 + 6}
                    y={(midlineEndpoints.a.y + midlineEndpoints.b.y) / 2 + 6}
                    fontSize={12}
                    fill="rgba(255,0,0,0.9)"
                  />
                </>
              ) : null}

              {/* reference: eye-line }
              {showEyeLine && eyeLine ? (
                <>
                  <Line
                    points={[eyeLine.l.x, eyeLine.l.y, eyeLine.r.x, eyeLine.r.y]}
                    stroke="rgba(0,200,0,0.9)"
                    strokeWidth={1.5}
                  />
                  <Text
                    text="Eye-line"
                    x={(eyeLine.l.x + eyeLine.r.x) / 2 + 6}
                    y={(eyeLine.l.y + eyeLine.r.y) / 2 - 16}
                    fontSize={12}
                    fill="rgba(0,160,0,0.9)"
                  />
                </>
              ) : null}

              {/* thirds }
              {showThirds && thirdsYs ? (
                <>
                  <Line points={[thirdsYs.xmin, thirdsYs.y1, thirdsYs.xmax, thirdsYs.y1]} stroke="rgba(0,0,0,0.5)" strokeWidth={1} dash={[4, 4]} />
                  <Text text="1/3" x={thirdsYs.xmax + 6} y={thirdsYs.y1 - 6} fontSize={11} fill="#333" />
                  <Line points={[thirdsYs.xmin, thirdsYs.y2, thirdsYs.xmax, thirdsYs.y2]} stroke="rgba(0,0,0,0.5)" strokeWidth={1} dash={[4, 4]} />
                  <Text text="2/3" x={thirdsYs.xmax + 6} y={thirdsYs.y2 - 6} fontSize={11} fill="#333" />
                </>
              ) : null}

              {/* draggable annotation points + offset labels }
              {points.map((p) => {
                const offset = pointOffsets.find((o) => o.id === p.id);
                const label = offset ? `${sideLabel(offset.side)} ${offset.dist}px` : "";
                return (
                  <Group
                    key={p.id}
                    x={p.x}
                    y={p.y}
                    draggable
                    onDragEnd={(e) => {
                      const nx = (e.target as any).x();
                      const ny = (e.target as any).y();
                      setPoints((prev) => prev.map((q) => (q.id === p.id ? { ...q, x: nx, y: ny } : q)));
                    }}
                  >
                    <Circle radius={8} fill="rgba(255,165,0,0.95)" stroke="black" strokeWidth={1} />
                    <Text text="●" fontSize={12} offsetX={6} offsetY={6} />
                    <Text text={label} x={12} y={-10} fontSize={11} fill="#222" />
                  </Group>
                );
              })}
            </Layer>
          </Stage>
        </div>

        <div style={{ width: 340 }}>
          <h3>Session</h3>
          <div>
            <strong>Detected landmarks:</strong> {landmarks.length}
          </div>

          <div style={{ marginTop: 12 }}>
            <h4>Reference Lines</h4>
            <div>Midline: {midlineEndpoints ? `from (${Math.round(midlineEndpoints.a.x)},${Math.round(midlineEndpoints.a.y)}) to (${Math.round(midlineEndpoints.b.x)},${Math.round(midlineEndpoints.b.y)})` : "n/a"}</div>
            <div>Eye-line: {eyeLine ? `L(${Math.round(eyeLine.l.x)},${Math.round(eyeLine.l.y)}) R(${Math.round(eyeLine.r.x)},${Math.round(eyeLine.r.y)})` : "n/a"}</div>
            <div>Thirds: {thirdsYs ? `y1=${Math.round(thirdsYs.y1)} y2=${Math.round(thirdsYs.y2)}` : "n/a"}</div>
          </div>

          <div style={{ marginTop: 12 }}>
            <h4>Injection points</h4>
            {points.length === 0 ? <div>No points yet</div> : null}
            {points.map((p) => {
              const off = pointOffsets.find((o) => o.id === p.id);
              return (
                <div key={p.id} style={{ marginBottom: 8 }}>
                  <small>ID: {p.id}</small>
                  <div>
                    x: {Math.round(p.x)}, y: {Math.round(p.y)}{" "}
                    <button onClick={() => setPoints((ps) => ps.filter((q) => q.id !== p.id))} style={{ marginLeft: 6 }}>
                      remove
                    </button>
                  </div>
                  <div style={{ color: "#444", marginTop: 4 }}>
                    Offset to midline: {off ? `${sideLabel(off.side)} ${off.dist}px` : "—"}
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ marginTop: 12 }}>
            <small style={{ color: "#666" }}>
              Tip: Toggle Midline / Eye-line / Thirds to help align injection points. Distances are in pixels.
            </small>
          </div>
        </div>
      </div>
    </div>
  );
}
*/

/*
// src/components/FaceCanvas.tsx
import React, { useEffect, useRef, useState } from "react";
import {
  Stage,
  Layer,
  Image as KonvaImageElement,
  Circle,
  Line,
  Text,
  Group,
} from "react-konva";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import { nanoid } from "nanoid";

type Landmark = [number, number, number?];

const DEFAULT_STAGE_WIDTH = 800;
const DEFAULT_STAGE_HEIGHT = 800;

function fitImage(imageWidth: number, imageHeight: number, stageW: number, stageH: number) {
  const imgRatio = imageWidth / imageHeight;
  const stageRatio = stageW / stageH;
  let width = stageW;
  let height = stageH;
  if (imgRatio > stageRatio) {
    width = stageW;
    height = Math.round(stageW / imgRatio);
  } else {
    height = stageH;
    width = Math.round(stageH * imgRatio);
  }
  return { width, height };
}

// vector helpers (unchanged)
function vecSub(a: { x: number; y: number }, b: { x: number; y: number }) {
  return { x: a.x - b.x, y: a.y - b.y };
}
function vecDot(a: { x: number; y: number }, b: { x: number; y: number }) {
  return a.x * b.x + a.y * b.y;
}
function vecLen(a: { x: number; y: number }) {
  return Math.hypot(a.x, a.y);
}
function vecScale(a: { x: number; y: number }, s: number) {
  return { x: a.x * s, y: a.y * s };
}
function vecAdd(a: { x: number; y: number }, b: { x: number; y: number }) {
  return { x: a.x + b.x, y: a.y + b.y };
}
function perpDistanceToLine(point: { x: number; y: number }, a: { x: number; y: number }, b: { x: number; y: number }) {
  const v = vecSub(b, a);
  const w = vecSub(point, a);
  const vlen2 = v.x * v.x + v.y * v.y;
  if (vlen2 === 0) return { dist: vecLen(w), side: 0 };
  const t = vecDot(w, v) / vlen2;
  const proj = vecAdd(a, vecScale(v, t));
  const perp = vecSub(point, proj);
  const dist = vecLen(perp);
  const cross = v.x * w.y - v.y * w.x;
  const side = cross;
  return { dist, side };
}
// @ts-ignore
export default function FaceCanvas(): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [imgEl, setImgEl] = useState<HTMLImageElement | null>(null);
  const stageRef = useRef<any>(null);

  const [imageFit, setImageFit] = useState<{ width: number; height: number } | null>(null); // NEW: store fit used for both stage & detection
  const [landmarks, setLandmarks] = useState<Landmark[]>([]);
  const [detector, setDetector] = useState<any | null>(null);
  const [points, setPoints] = useState<{ id: string; x: number; y: number }[]>([]);
  const [stageSize, setStageSize] = useState({ width: DEFAULT_STAGE_WIDTH, height: DEFAULT_STAGE_HEIGHT });

  // toggles
  const [showMidline, setShowMidline] = useState(true);
  const [showEyeLine, setShowEyeLine] = useState(true);
  const [showThirds, setShowThirds] = useState(true);

  // file -> data URL
  useEffect(() => {
    if (!file) {
      setImageSrc(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setImageSrc(String(reader.result));
    reader.onerror = (e) => {
      console.error("FileReader error", e);
      setImageSrc(null);
    };
    reader.readAsDataURL(file);
  }, [file]);

  // data URL -> HTMLImageElement (crossOrigin)
  useEffect(() => {
    if (!imageSrc) {
      setImgEl(null);
      setImageFit(null);
      setStageSize({ width: DEFAULT_STAGE_WIDTH, height: DEFAULT_STAGE_HEIGHT });
      return;
    }
    let mounted = true;
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = imageSrc;
    image.onload = () => {
      if (!mounted) return;
      setImgEl(image);

      // compute fit based on natural size and DEFAULT stage bounds (this single fit is used everywhere)
      const naturalW = image.naturalWidth || DEFAULT_STAGE_WIDTH;
      const naturalH = image.naturalHeight || DEFAULT_STAGE_HEIGHT;
      const fit = fitImage(naturalW, naturalH, DEFAULT_STAGE_WIDTH, DEFAULT_STAGE_HEIGHT);

      // set both imageFit and stageSize from same fit
      setImageFit(fit);
      setStageSize({ width: fit.width, height: fit.height });
    };
    image.onerror = (err) => {
      console.error("Image load error", err);
      if (mounted) {
        setImgEl(null);
        setImageFit(null);
      }
    };
    return () => {
      mounted = false;
    };
  }, [imageSrc]);

  // detector creation (unchanged logic)
  useEffect(() => {
    let mounted = true;
    let created: any = null;

    async function setup() {
      try {
        await tf.setBackend("webgl");
        await tf.ready();

        const modelConst = (faceLandmarksDetection as any).SupportedModels?.MediaPipeFaceMesh ??
                           (faceLandmarksDetection as any).MediaPipeFaceMesh;

        const detectorConfigTfjs = { runtime: "tfjs", maxFaces: 1, refineLandmarks: true };

        try {
          created = await (faceLandmarksDetection as any).createDetector(modelConst, detectorConfigTfjs);
          if (mounted) setDetector(created);
          else if (created?.dispose) created.dispose();
        } catch (errCreate) {
          console.warn("createDetector failed; trying legacy load():", errCreate);
          try {
            created = await (faceLandmarksDetection as any).load(
              (faceLandmarksDetection as any).SupportedPackages?.mediapipeFacemesh ??
                (faceLandmarksDetection as any).SupportedPackages,
              { maxFaces: 1 }
            );
            if (mounted) setDetector(created);
          } catch (errLegacy) {
            console.error("Both createDetector and legacy load() failed:", errLegacy);
            throw errLegacy;
          }
        }
      } catch (err) {
        console.error("Detector setup error:", err);
      }
    }

    setup();

    return () => {
      mounted = false;
      if (created && typeof created.dispose === "function") {
        try {
          created.dispose();
        } catch {}
      }
    };
  }, []);

  // DETECTION: run only when detector AND imageFit are ready
  useEffect(() => {
    if (!detector || !imgEl || !imageFit) return;
    let cancelled = false;

    async function detectOnce() {
      try {
        // Use imageFit for canvas size (guarantees same coordinates as stage)
        const off = document.createElement("canvas");
        off.width = imageFit.width;
        off.height = imageFit.height;
        const ctx = off.getContext("2d")!;
        ctx.clearRect(0, 0, off.width, off.height);
        // draw image scaled to the same fit used by stage
        ctx.drawImage(imgEl, 0, 0, imageFit.width, imageFit.height);

        let predictions: any = null;
        let lastErr: any = null;

        try {
          predictions = await (detector as any).estimateFaces(off);
        } catch (e1) {
          lastErr = e1;
          try {
            predictions = await (detector as any).estimateFaces({ input: off });
          } catch (e2) {
            lastErr = e2;
            try {
              predictions = await (detector as any).estimateFaces(off, { flipHorizontal: false });
            } catch (e3) {
              lastErr = e3;
            }
          }
        }

        if (!predictions) {
          console.error("All estimateFaces attempts failed:", lastErr);
          throw lastErr;
        }

        if (cancelled) return;

        if (!predictions || predictions.length === 0) {
          setLandmarks([]);
          // do not override user-added points
          return;
        }

        const face = predictions[0] as any;
        let mesh: Landmark[] = [];

        if (Array.isArray(face?.scaledMesh)) mesh = face.scaledMesh as Landmark[];
        else if (Array.isArray(face?.mesh)) mesh = face.mesh as Landmark[];
        else if (Array.isArray(face?.keypoints)) mesh = (face.keypoints as any[]).map(kp => [kp.x, kp.y, kp.z ?? 0]);
        else if (Array.isArray(face?.keypoints3D)) mesh = face.keypoints3D as Landmark[];
        else console.warn("Unknown prediction shape", face);

        // IMPORTANT: mesh coordinates correspond to the off canvas size (imageFit.width/height)
        // We store them directly and the Stage uses the same size (imageFit), so drawing lines/points will align.
        setLandmarks(mesh || []);

        // populate default points only when none exist (do not overwrite user points)
        if ((!points || points.length === 0) && mesh && mesh.length > 0) {
          const lipsIndices = [13, 14, 78, 308];
          const defaultPoints = lipsIndices
            .map((i) => {
              const p = mesh && mesh[i];
              if (!p) return null;
              return { id: nanoid(), x: p[0], y: p[1] };
            })
            .filter(Boolean) as { id: string; x: number; y: number }[];
          setPoints(defaultPoints);
        }
      } catch (err) {
        console.error("detectOnce error:", err);
      }
    }

    detectOnce();

    return () => {
      cancelled = true;
    };
  }, [detector, imgEl, imageFit]); // run after imageFit is set

  // keep stage size in sync with imageFit whenever imageFit changes
  useEffect(() => {
    if (imageFit) {
      setStageSize({ width: imageFit.width, height: imageFit.height });
    }
  }, [imageFit]);

  function onAddPoint() {
    setPoints((p) => [...p, { id: nanoid(), x: (imageFit?.width ?? stageSize.width) / 2, y: (imageFit?.height ?? stageSize.height) / 2 }]);
  }

  function onExport() {
    if (!stageRef.current) return;
    try {
      // @ts-ignore (kept intentionally)
      const uri = stageRef.current.toDataURL({ pixelRatio: 2 });
      const link = document.createElement("a");
      link.download = `face_map_${Date.now()}.png`;
      link.href = uri;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error("Export error:", err);
      alert("Export failed (canvas may be tainted). Make sure image is loaded from same origin or crossOrigin='anonymous'.");
    }
  }

  // compute midline / eye-line / thirds (same logic as before)
  const midlineEndpoints =
    landmarks.length > 152 && landmarks[10] && landmarks[152]
      ? { a: { x: landmarks[10][0], y: landmarks[10][1] }, b: { x: landmarks[152][0], y: landmarks[152][1] } }
      : (() => {
          if (!landmarks || landmarks.length === 0) return null;
          let minY = Infinity,
            maxY = -Infinity;
          let minX = Infinity,
            maxX = -Infinity;
          for (const lm of landmarks) {
            if (!lm) continue;
            const x = lm[0],
              y = lm[1];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
          const cx = (minX + maxX) / 2;
          return { a: { x: cx, y: minY }, b: { x: cx, y: maxY } };
        })();

  const eyeLine =
    landmarks.length > 263 && landmarks[33] && landmarks[263]
      ? { l: { x: landmarks[33][0], y: landmarks[33][1] }, r: { x: landmarks[263][0], y: landmarks[263][1] } }
      : null;

  const thirdsYs = (() => {
    if (!landmarks || landmarks.length === 0) return null;
    let minY = Infinity,
      maxY = -Infinity;
    let minX = Infinity,
      maxX = -Infinity;
    for (const lm of landmarks) {
      if (!lm) continue;
      const x = lm[0],
        y = lm[1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const h = maxY - minY;
    if (!isFinite(h) || h <= 0) return null;
    const y1 = minY + h / 3;
    const y2 = minY + (2 * h) / 3;
    return { y1, y2, xmin: minX, xmax: maxX };
  })();

  const pointOffsets = points.map((pt) => {
    if (!midlineEndpoints) return { id: pt.id, dist: 0, side: 0 };
    const { a, b } = midlineEndpoints;
    const { dist, side } = perpDistanceToLine({ x: pt.x, y: pt.y }, a, b);
    return { id: pt.id, dist: Math.round(dist), side: side };
  });

  function sideLabel(side: number) {
    if (side === 0) return "C";
    return side > 0 ? "L" : "R";
  }

  return (
    <div style={{ border: "1px solid #ddd", padding: 12, borderRadius: 8, maxWidth: 1100 }}>
      <div style={{ marginBottom: 12, display: "flex", gap: 12, alignItems: "center" }}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            if (!e.target.files?.[0]) return;
            setFile(e.target.files[0]);
            setPoints([]);
            setLandmarks([]);
            setImageFit(null);
          }}
        />
        <button onClick={onAddPoint}>Add point</button>
        <button onClick={onExport}>Export PNG</button>

        <div style={{ marginLeft: 16, display: "flex", gap: 8, alignItems: "center" }}>
          <label>
            <input type="checkbox" checked={showMidline} onChange={(e) => setShowMidline(e.target.checked)} /> Midline
          </label>
          <label>
            <input type="checkbox" checked={showEyeLine} onChange={(e) => setShowEyeLine(e.target.checked)} /> Eye-line
          </label>
          <label>
            <input type="checkbox" checked={showThirds} onChange={(e) => setShowThirds(e.target.checked)} /> Thirds
          </label>
        </div>
      </div>

      <div style={{ display: "flex", gap: 12 }}>
        <div style={{ border: "1px solid #eee", padding: 6 }}>
          <Stage width={stageSize.width} height={stageSize.height} ref={stageRef}>
            <Layer>
              {imgEl ? (
                // @ts-ignore
                <KonvaImageElement image={imgEl as any} x={0} y={0} width={stageSize.width} height={stageSize.height} />
              ) : null}
            </Layer>

            <Layer>
              {landmarks.length > 0
                ? landmarks.map((lm, idx) => {
                    const x = lm[0];
                    const y = lm[1];
                    return <Circle key={`lm-${idx}`} x={x} y={y} radius={1.2} fill="rgba(0,150,255,0.9)" />;
                  })
                : null}

              {showMidline && midlineEndpoints ? (
                <>
                  <Line
                    points={[midlineEndpoints.a.x, midlineEndpoints.a.y, midlineEndpoints.b.x, midlineEndpoints.b.y]}
                    stroke="rgba(255,0,0,0.85)"
                    strokeWidth={1.5}
                    dash={[6, 4]}
                  />
                  <Text
                    text="Midline"
                    x={(midlineEndpoints.a.x + midlineEndpoints.b.x) / 2 + 6}
                    y={(midlineEndpoints.a.y + midlineEndpoints.b.y) / 2 + 6}
                    fontSize={12}
                    fill="rgba(255,0,0,0.9)"
                  />
                </>
              ) : null}

              {showEyeLine && eyeLine ? (
                <>
                  <Line
                    points={[eyeLine.l.x, eyeLine.l.y, eyeLine.r.x, eyeLine.r.y]}
                    stroke="rgba(0,200,0,0.9)"
                    strokeWidth={1.5}
                  />
                  <Text
                    text="Eye-line"
                    x={(eyeLine.l.x + eyeLine.r.x) / 2 + 6}
                    y={(eyeLine.l.y + eyeLine.r.y) / 2 - 16}
                    fontSize={12}
                    fill="rgba(0,160,0,0.9)"
                  />
                </>
              ) : null}

              {showThirds && thirdsYs ? (
                <>
                  <Line points={[thirdsYs.xmin, thirdsYs.y1, thirdsYs.xmax, thirdsYs.y1]} stroke="rgba(0,0,0,0.5)" strokeWidth={1} dash={[4, 4]} />
                  <Text text="1/3" x={thirdsYs.xmax + 6} y={thirdsYs.y1 - 6} fontSize={11} fill="#333" />
                  <Line points={[thirdsYs.xmin, thirdsYs.y2, thirdsYs.xmax, thirdsYs.y2]} stroke="rgba(0,0,0,0.5)" strokeWidth={1} dash={[4, 4]} />
                  <Text text="2/3" x={thirdsYs.xmax + 6} y={thirdsYs.y2 - 6} fontSize={11} fill="#333" />
                </>
              ) : null}

              {points.map((p) => {
                const offset = pointOffsets.find((o) => o.id === p.id);
                const label = offset ? `${sideLabel(offset.side)} ${offset.dist}px` : "";
                return (
                  <Group
                    key={p.id}
                    x={p.x}
                    y={p.y}
                    draggable
                    onDragEnd={(e) => {
                      const nx = (e.target as any).x();
                      const ny = (e.target as any).y();
                      setPoints((prev) => prev.map((q) => (q.id === p.id ? { ...q, x: nx, y: ny } : q)));
                    }}
                  >
                    <Circle radius={8} fill="rgba(255,165,0,0.95)" stroke="black" strokeWidth={1} />
                    <Text text="●" fontSize={12} offsetX={6} offsetY={6} />
                    <Text text={label} x={12} y={-10} fontSize={11} fill="#222" />
                  </Group>
                );
              })}
            </Layer>
          </Stage>
        </div>

        <div style={{ width: 340 }}>
          <h3>Session</h3>
          <div>
            <strong>Detected landmarks:</strong> {landmarks.length}
          </div>

          <div style={{ marginTop: 12 }}>
            <h4>Reference Lines</h4>
            <div>Midline: {midlineEndpoints ? `from (${Math.round(midlineEndpoints.a.x)},${Math.round(midlineEndpoints.a.y)}) to (${Math.round(midlineEndpoints.b.x)},${Math.round(midlineEndpoints.b.y)})` : "n/a"}</div>
            <div>Eye-line: {eyeLine ? `L(${Math.round(eyeLine.l.x)},${Math.round(eyeLine.l.y)}) R(${Math.round(eyeLine.r.x)},${Math.round(eyeLine.r.y)})` : "n/a"}</div>
            <div>Thirds: {thirdsYs ? `y1=${Math.round(thirdsYs.y1)} y2=${Math.round(thirdsYs.y2)}` : "n/a"}</div>
          </div>

          <div style={{ marginTop: 12 }}>
            <h4>Injection points</h4>
            {points.length === 0 ? <div>No points yet</div> : null}
            {points.map((p) => {
              const off = pointOffsets.find((o) => o.id === p.id);
              return (
                <div key={p.id} style={{ marginBottom: 8 }}>
                  <small>ID: {p.id}</small>
                  <div>
                    x: {Math.round(p.x)}, y: {Math.round(p.y)}{" "}
                    <button onClick={() => setPoints((ps) => ps.filter((q) => q.id !== p.id))} style={{ marginLeft: 6 }}>
                      remove
                    </button>
                  </div>
                  <div style={{ color: "#444", marginTop: 4 }}>
                    Offset to midline: {off ? `${sideLabel(off.side)} ${off.dist}px` : "—"}
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ marginTop: 12 }}>
            <small style={{ color: "#666" }}>
              Tip: Toggle Midline / Eye-line / Thirds to help align injection points. Distances are in pixels.
            </small>
          </div>
        </div>
      </div>
    </div>
  );
}
*/

/* works but adding toggle for landmarks
// src/components/FaceCanvas.tsx
import React, { useEffect, useRef, useState } from "react";
import {
  Stage,
  Layer,
  Image as KonvaImageElement,
  Circle,
  Line,
  Text,
  Group,
} from "react-konva";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import { nanoid } from "nanoid";

type Landmark = [number, number, number?];

type AnnotationPoint = {
  id: string;
  x: number;
  y: number;
  product?: string;
  dose?: string;
};

const DEFAULT_STAGE_WIDTH = 800;
const DEFAULT_STAGE_HEIGHT = 800;

function fitImage(imageWidth: number, imageHeight: number, stageW: number, stageH: number) {
  const imgRatio = imageWidth / imageHeight;
  const stageRatio = stageW / stageH;
  let width = stageW;
  let height = stageH;
  if (imgRatio > stageRatio) {
    width = stageW;
    height = Math.round(stageW / imgRatio);
  } else {
    height = stageH;
    width = Math.round(stageH * imgRatio);
  }
  return { width, height };
}

// vector helpers
function vecSub(a: { x: number; y: number }, b: { x: number; y: number }) {
  return { x: a.x - b.x, y: a.y - b.y };
}
function vecDot(a: { x: number; y: number }, b: { x: number; y: number }) {
  return a.x * b.x + a.y * b.y;
}
function vecLen(a: { x: number; y: number }) {
  return Math.hypot(a.x, a.y);
}
function vecScale(a: { x: number; y: number }, s: number) {
  return { x: a.x * s, y: a.y * s };
}
function vecAdd(a: { x: number; y: number }, b: { x: number; y: number }) {
  return { x: a.x + b.x, y: a.y + b.y };
}
function perpDistanceToLine(point: { x: number; y: number }, a: { x: number; y: number }, b: { x: number; y: number }) {
  const v = vecSub(b, a);
  const w = vecSub(point, a);
  const vlen2 = v.x * v.x + v.y * v.y;
  if (vlen2 === 0) return { dist: vecLen(w), side: 0 };
  const t = vecDot(w, v) / vlen2;
  const proj = vecAdd(a, vecScale(v, t));
  const perp = vecSub(point, proj);
  const dist = vecLen(perp);
  const cross = v.x * w.y - v.y * w.x;
  const side = cross;
  return { dist, side };
}

// @ts-ignore
export default function FaceCanvas(): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [imgEl, setImgEl] = useState<HTMLImageElement | null>(null);
  const stageRef = useRef<any>(null);

  const [imageFit, setImageFit] = useState<{ width: number; height: number } | null>(null);
  const [landmarks, setLandmarks] = useState<Landmark[]>([]);
  const [detector, setDetector] = useState<any | null>(null);
  const [points, setPoints] = useState<AnnotationPoint[]>([]);
  const [stageSize, setStageSize] = useState({ width: DEFAULT_STAGE_WIDTH, height: DEFAULT_STAGE_HEIGHT });

  // toggles
  const [showMidline, setShowMidline] = useState(true);
  const [showEyeLine, setShowEyeLine] = useState(true);
  const [showThirds, setShowThirds] = useState(true);

  // file -> data URL
  useEffect(() => {
    if (!file) {
      setImageSrc(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setImageSrc(String(reader.result));
    reader.onerror = (e) => {
      console.error("FileReader error", e);
      setImageSrc(null);
    };
    reader.readAsDataURL(file);
  }, [file]);

  // data URL -> image element + compute fit
  useEffect(() => {
    if (!imageSrc) {
      setImgEl(null);
      setImageFit(null);
      setStageSize({ width: DEFAULT_STAGE_WIDTH, height: DEFAULT_STAGE_HEIGHT });
      return;
    }
    let mounted = true;
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = imageSrc;
    image.onload = () => {
      if (!mounted) return;
      setImgEl(image);
      const naturalW = image.naturalWidth || DEFAULT_STAGE_WIDTH;
      const naturalH = image.naturalHeight || DEFAULT_STAGE_HEIGHT;
      const fit = fitImage(naturalW, naturalH, DEFAULT_STAGE_WIDTH, DEFAULT_STAGE_HEIGHT);
      setImageFit(fit);
      setStageSize({ width: fit.width, height: fit.height });
    };
    image.onerror = (err) => {
      console.error("Image load error", err);
      if (mounted) {
        setImgEl(null);
        setImageFit(null);
      }
    };
    return () => {
      mounted = false;
    };
  }, [imageSrc]);

  // create detector
  useEffect(() => {
    let mounted = true;
    let created: any = null;

    async function setup() {
      try {
        await tf.setBackend("webgl");
        await tf.ready();

        const modelConst = (faceLandmarksDetection as any).SupportedModels?.MediaPipeFaceMesh ??
                           (faceLandmarksDetection as any).MediaPipeFaceMesh;

        const detectorConfigTfjs = { runtime: "tfjs", maxFaces: 1, refineLandmarks: true };

        try {
          created = await (faceLandmarksDetection as any).createDetector(modelConst, detectorConfigTfjs);
          if (mounted) setDetector(created);
          else if (created?.dispose) created.dispose();
        } catch (errCreate) {
          console.warn("createDetector failed; trying legacy load():", errCreate);
          try {
            created = await (faceLandmarksDetection as any).load(
              (faceLandmarksDetection as any).SupportedPackages?.mediapipeFacemesh ??
                (faceLandmarksDetection as any).SupportedPackages,
              { maxFaces: 1 }
            );
            if (mounted) setDetector(created);
          } catch (errLegacy) {
            console.error("Both createDetector and legacy load() failed:", errLegacy);
            throw errLegacy;
          }
        }
      } catch (err) {
        console.error("Detector setup error:", err);
      }
    }

    setup();

    return () => {
      mounted = false;
      if (created && typeof created.dispose === "function") {
        try {
          created.dispose();
        } catch {}
      }
    };
  }, []);

  // detection (run after imageFit)
  useEffect(() => {
    if (!detector || !imgEl || !imageFit) return;
    let cancelled = false;

    async function detectOnce() {
      try {
        const off = document.createElement("canvas");
        off.width = imageFit.width;
        off.height = imageFit.height;
        const ctx = off.getContext("2d")!;
        ctx.clearRect(0, 0, off.width, off.height);
        ctx.drawImage(imgEl, 0, 0, imageFit.width, imageFit.height);

        let predictions: any = null;
        let lastErr: any = null;

        try {
          predictions = await (detector as any).estimateFaces(off);
        } catch (e1) {
          lastErr = e1;
          try {
            predictions = await (detector as any).estimateFaces({ input: off });
          } catch (e2) {
            lastErr = e2;
            try {
              predictions = await (detector as any).estimateFaces(off, { flipHorizontal: false });
            } catch (e3) {
              lastErr = e3;
            }
          }
        }

        if (!predictions) {
          console.error("All estimateFaces attempts failed:", lastErr);
          throw lastErr;
        }

        if (cancelled) return;

        if (!predictions || predictions.length === 0) {
          setLandmarks([]);
          return;
        }

        const face = predictions[0] as any;
        let mesh: Landmark[] = [];

        if (Array.isArray(face?.scaledMesh)) mesh = face.scaledMesh as Landmark[];
        else if (Array.isArray(face?.mesh)) mesh = face.mesh as Landmark[];
        else if (Array.isArray(face?.keypoints)) mesh = (face.keypoints as any[]).map(kp => [kp.x, kp.y, kp.z ?? 0]);
        else if (Array.isArray(face?.keypoints3D)) mesh = face.keypoints3D as Landmark[];
        else console.warn("Unknown prediction shape", face);

        setLandmarks(mesh || []);

        // populate some sensible default points (lips) if there are no points yet
        if ((!points || points.length === 0) && mesh && mesh.length > 0) {
          const lipsIndices = [13, 14, 78, 308];
          const defaultPoints = lipsIndices
            .map((i) => {
              const p = mesh && mesh[i];
              if (!p) return null;
              return { id: nanoid(), x: p[0], y: p[1], product: "", dose: "" };
            })
            .filter(Boolean) as AnnotationPoint[];
          setPoints(defaultPoints);
        }
      } catch (err) {
        console.error("detectOnce error:", err);
      }
    }

    detectOnce();

    return () => {
      cancelled = true;
    };
  }, [detector, imgEl, imageFit]); // re-run when imageFit set

  // keep stage size in sync with imageFit whenever imageFit changes
  useEffect(() => {
    if (imageFit) {
      setStageSize({ width: imageFit.width, height: imageFit.height });
    }
  }, [imageFit]);

  function onAddPoint() {
    setPoints((p) => [...p, { id: nanoid(), x: (imageFit?.width ?? stageSize.width) / 2, y: (imageFit?.height ?? stageSize.height) / 2, product: "", dose: "" }]);
  }

  function onExport() {
    if (!stageRef.current) return;
    try {
      // @ts-ignore (kept intentionally)
      const uri = stageRef.current.toDataURL({ pixelRatio: 2 });
      const link = document.createElement("a");
      link.download = `face_map_${Date.now()}.png`;
      link.href = uri;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error("Export error:", err);
      alert("Export failed (canvas may be tainted). Make sure image is loaded from same origin or crossOrigin='anonymous'.");
    }
  }

  // reference calculations
  const midlineEndpoints =
    landmarks.length > 152 && landmarks[10] && landmarks[152]
      ? { a: { x: landmarks[10][0], y: landmarks[10][1] }, b: { x: landmarks[152][0], y: landmarks[152][1] } }
      : (() => {
          if (!landmarks || landmarks.length === 0) return null;
          let minY = Infinity,
            maxY = -Infinity;
          let minX = Infinity,
            maxX = -Infinity;
          for (const lm of landmarks) {
            if (!lm) continue;
            const x = lm[0],
              y = lm[1];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
          const cx = (minX + maxX) / 2;
          return { a: { x: cx, y: minY }, b: { x: cx, y: maxY } };
        })();

  const eyeLine =
    landmarks.length > 263 && landmarks[33] && landmarks[263]
      ? { l: { x: landmarks[33][0], y: landmarks[33][1] }, r: { x: landmarks[263][0], y: landmarks[263][1] } }
      : null;

  const thirdsYs = (() => {
    if (!landmarks || landmarks.length === 0) return null;
    let minY = Infinity,
      maxY = -Infinity;
    let minX = Infinity,
      maxX = -Infinity;
    for (const lm of landmarks) {
      if (!lm) continue;
      const x = lm[0],
        y = lm[1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const h = maxY - minY;
    if (!isFinite(h) || h <= 0) return null;
    const y1 = minY + h / 3;
    const y2 = minY + (2 * h) / 3;
    return { y1, y2, xmin: minX, xmax: maxX };
  })();

  // offsets
  const pointOffsets = points.map((pt) => {
    if (!midlineEndpoints) return { id: pt.id, dist: 0, side: 0 };
    const { a, b } = midlineEndpoints;
    const { dist, side } = perpDistanceToLine({ x: pt.x, y: pt.y }, a, b);
    return { id: pt.id, dist: Math.round(dist), side: side };
  });
  function sideLabel(side: number) {
    if (side === 0) return "C";
    return side > 0 ? "L" : "R";
  }

  // ---------- PRESETS ----------
  // Indices chosen to approximate the anatomical regions in MediaPipe FaceMesh.
  // If indices aren't available, we attempt fallbacks.
  const presetTemplates = {
    lips: {
      // a small cluster: upper lip center, lower lip center, left lip, right lip
      indices: [13, 14, 78, 308],
      label: "Lips",
    },
    nasolabial: {
      // approximate nasolabial fold points (left inner cheek, right inner cheek, two slightly below)
      // these indices are approximate; will fallback if not present
      indices: [50, 280, 146, 376],
      label: "Nasolabial",
    },
    marionette: {
      // marionette line region near mouth corners / lower face
      indices: [57, 287, 164, 393],
      label: "Marionette",
    },
  } as const;

  function addPreset(name: keyof typeof presetTemplates) {
    const tpl = presetTemplates[name];
    if (!landmarks || landmarks.length === 0) {
      // no landmarks: add generic points across midline / thirds
      // distribute points vertically around midline center
      const cx = (stageSize.width / 2);
      const cy = (stageSize.height / 2);
      const fallbackPoints: AnnotationPoint[] = [ -1, 0, 1, 2 ].map((i) => ({
        id: nanoid(),
        x: cx + (i - 1.5) * 12,
        y: cy + (i - 1.5) * 10,
        product: "",
        dose: "",
      }));
      setPoints((p) => [...p, ...fallbackPoints]);
      return;
    }

    const newPts: AnnotationPoint[] = [];
    for (const idx of tpl.indices) {
      const lm = landmarks[idx];
      if (lm && typeof lm[0] === "number") {
        newPts.push({ id: nanoid(), x: lm[0], y: lm[1], product: "", dose: "" });
      } else {
        // fallback: place relative to midline / thirds
        if (midlineEndpoints && thirdsYs) {
          // place around midline at thirds positions
          const cx = (midlineEndpoints.a.x + midlineEndpoints.b.x) / 2;
          // distribute horizontally slightly
          const dx = (newPts.length % 2 === 0) ? -12 : 12;
          const y = newPts.length < 2 ? thirdsYs.y1 : thirdsYs.y2;
          newPts.push({ id: nanoid(), x: cx + dx, y, product: "", dose: "" });
        } else {
          // final fallback: center-ish
          newPts.push({ id: nanoid(), x: stageSize.width / 2 + (newPts.length - 1) * 10, y: stageSize.height / 2, product: "", dose: "" });
        }
      }
    }
    setPoints((p) => [...p, ...newPts]);
  }

  // update point metadata helpers
  function updatePointMeta(id: string, data: Partial<Pick<AnnotationPoint, "product" | "dose" | "x" | "y">>) {
    setPoints((prev) => prev.map((pt) => (pt.id === id ? { ...pt, ...data } : pt)));
  }

  // delete
  function removePoint(id: string) {
    setPoints((ps) => ps.filter((p) => p.id !== id));
  }

  // snapping helper (optional; not auto-enabled)
  function snapPointToNearestLandmark(id: string, radius = 12) {
    const pt = points.find((p) => p.id === id);
    if (!pt || !landmarks || landmarks.length === 0) return;
    let best: { idx: number; dist: number } | null = null;
    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i];
      if (!lm) continue;
      const dx = lm[0] - pt.x;
      const dy = lm[1] - pt.y;
      const d = Math.hypot(dx, dy);
      if (d <= radius && (!best || d < best.dist)) best = { idx: i, dist: d };
    }
    if (best) {
      const lm = landmarks[best.idx];
      updatePointMeta(id, { x: lm[0], y: lm[1] });
    }
  }

  return (
    <div style={{ border: "1px solid #ddd", padding: 12, borderRadius: 8, maxWidth: 1200 }}>
      <div style={{ marginBottom: 12, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            if (!e.target.files?.[0]) return;
            setFile(e.target.files[0]);
            setPoints([]);
            setLandmarks([]);
            setImageFit(null);
          }}
        />
        <button onClick={onAddPoint}>Add point</button>
        <button onClick={onExport}>Export PNG</button>

        <div style={{ marginLeft: 8, display: "flex", gap: 6, alignItems: "center" }}>
          <strong>Presets:</strong>
          <button onClick={() => addPreset("lips")}>Lips</button>
          <button onClick={() => addPreset("nasolabial")}>Nasolabial</button>
          <button onClick={() => addPreset("marionette")}>Marionette</button>
        </div>

        <div style={{ marginLeft: "auto", display: "flex", gap: 12 }}>
          <label>
            <input type="checkbox" checked={showMidline} onChange={(e) => setShowMidline(e.target.checked)} /> Midline
          </label>
          <label>
            <input type="checkbox" checked={showEyeLine} onChange={(e) => setShowEyeLine(e.target.checked)} /> Eye-line
          </label>
          <label>
            <input type="checkbox" checked={showThirds} onChange={(e) => setShowThirds(e.target.checked)} /> Thirds
          </label>
        </div>
      </div>

      <div style={{ display: "flex", gap: 12 }}>
        <div style={{ border: "1px solid #eee", padding: 6 }}>
          <Stage width={stageSize.width} height={stageSize.height} ref={stageRef}>
            <Layer>
              {imgEl ? (
                // @ts-ignore
                <KonvaImageElement image={imgEl as any} x={0} y={0} width={stageSize.width} height={stageSize.height} />
              ) : null}
            </Layer>

            <Layer>
              {landmarks.length > 0
                ? landmarks.map((lm, idx) => {
                    const x = lm[0];
                    const y = lm[1];
                    return <Circle key={`lm-${idx}`} x={x} y={y} radius={1.2} fill="rgba(0,150,255,0.9)" />;
                  })
                : null}

              {showMidline && midlineEndpoints ? (
                <>
                  <Line
                    points={[midlineEndpoints.a.x, midlineEndpoints.a.y, midlineEndpoints.b.x, midlineEndpoints.b.y]}
                    stroke="rgba(255,0,0,0.85)"
                    strokeWidth={1.5}
                    dash={[6, 4]}
                  />
                  <Text
                    text="Midline"
                    x={(midlineEndpoints.a.x + midlineEndpoints.b.x) / 2 + 6}
                    y={(midlineEndpoints.a.y + midlineEndpoints.b.y) / 2 + 6}
                    fontSize={12}
                    fill="rgba(255,0,0,0.9)"
                  />
                </>
              ) : null}

              {showEyeLine && eyeLine ? (
                <>
                  <Line
                    points={[eyeLine.l.x, eyeLine.l.y, eyeLine.r.x, eyeLine.r.y]}
                    stroke="rgba(0,200,0,0.9)"
                    strokeWidth={1.5}
                  />
                  <Text
                    text="Eye-line"
                    x={(eyeLine.l.x + eyeLine.r.x) / 2 + 6}
                    y={(eyeLine.l.y + eyeLine.r.y) / 2 - 16}
                    fontSize={12}
                    fill="rgba(0,160,0,0.9)"
                  />
                </>
              ) : null}

              {showThirds && thirdsYs ? (
                <>
                  <Line points={[thirdsYs.xmin, thirdsYs.y1, thirdsYs.xmax, thirdsYs.y1]} stroke="rgba(0,0,0,0.5)" strokeWidth={1} dash={[4, 4]} />
                  <Text text="1/3" x={thirdsYs.xmax + 6} y={thirdsYs.y1 - 6} fontSize={11} fill="#333" />
                  <Line points={[thirdsYs.xmin, thirdsYs.y2, thirdsYs.xmax, thirdsYs.y2]} stroke="rgba(0,0,0,0.5)" strokeWidth={1} dash={[4, 4]} />
                  <Text text="2/3" x={thirdsYs.xmax + 6} y={thirdsYs.y2 - 6} fontSize={11} fill="#333" />
                </>
              ) : null}

              {points.map((p) => {
                const offset = pointOffsets.find((o) => o.id === p.id);
                const label = p.product ? `${p.product} ${p.dose ?? ""}`.trim() : `${offset ? sideLabel(offset.side) + " " + offset.dist + "px" : ""}`;
                return (
                  <Group
                    key={p.id}
                    x={p.x}
                    y={p.y}
                    draggable
                    onDragEnd={(e) => {
                      const nx = (e.target as any).x();
                      const ny = (e.target as any).y();
                      updatePointMeta(p.id, { x: nx, y: ny });
                    }}
                  >
                    <Circle radius={8} fill="rgba(255,165,0,0.95)" stroke="black" strokeWidth={1} />
                    <Text text="●" fontSize={12} offsetX={6} offsetY={6} />
                    <Text text={label} x={12} y={-10} fontSize={11} fill="#222" />
                  </Group>
                );
              })}
            </Layer>
          </Stage>
        </div>

        <div style={{ width: 380 }}>
          <h3>Session</h3>
          <div>
            <strong>Detected landmarks:</strong> {landmarks.length}
          </div>

          <div style={{ marginTop: 12 }}>
            <h4>Reference Lines</h4>
            <div>Midline: {midlineEndpoints ? `from (${Math.round(midlineEndpoints.a.x)},${Math.round(midlineEndpoints.a.y)}) to (${Math.round(midlineEndpoints.b.x)},${Math.round(midlineEndpoints.b.y)})` : "n/a"}</div>
            <div>Eye-line: {eyeLine ? `L(${Math.round(eyeLine.l.x)},${Math.round(eyeLine.l.y)}) R(${Math.round(eyeLine.r.x)},${Math.round(eyeLine.r.y)})` : "n/a"}</div>
            <div>Thirds: {thirdsYs ? `y1=${Math.round(thirdsYs.y1)} y2=${Math.round(thirdsYs.y2)}` : "n/a"}</div>
          </div>

          <div style={{ marginTop: 12 }}>
            <h4>Injection points</h4>
            {points.length === 0 ? <div>No points yet</div> : null}
            {points.map((p) => {
              const off = pointOffsets.find((o) => o.id === p.id);
              return (
                <div key={p.id} style={{ marginBottom: 10, padding: 8, borderRadius: 6, background: "#0f0f0f", color: "#fff" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <small style={{ opacity: 0.9 }}>ID: {p.id}</small>
                    <div style={{ display: "flex", gap: 6 }}>
                      <button onClick={() => removePoint(p.id)}>remove</button>
                      <button onClick={() => snapPointToNearestLandmark(p.id)}>snap</button>
                    </div>
                  </div>

                  <div style={{ marginTop: 6 }}>
                    <div>
                      x: {Math.round(p.x)}, y: {Math.round(p.y)}
                      <button onClick={() => updatePointMeta(p.id, { x: Math.round(p.x), y: Math.round(p.y) })} style={{ marginLeft: 6 }}>
                        refresh
                      </button>
                    </div>

                    <div style={{ marginTop: 6 }}>
                      <label style={{ display: "block", fontSize: 12, marginBottom: 4 }}>Product</label>
                      <input
                        value={p.product ?? ""}
                        placeholder="Product name"
                        onChange={(e) => updatePointMeta(p.id, { product: e.target.value })}
                        style={{ width: "100%", padding: "6px 8px", borderRadius: 4 }}
                      />
                    </div>

                    <div style={{ marginTop: 6 }}>
                      <label style={{ display: "block", fontSize: 12, marginBottom: 4 }}>Dose</label>
                      <input
                        value={p.dose ?? ""}
                        placeholder="e.g. 0.1 mL (0.5 unit)"
                        onChange={(e) => updatePointMeta(p.id, { dose: e.target.value })}
                        style={{ width: "100%", padding: "6px 8px", borderRadius: 4 }}
                      />
                    </div>

                    <div style={{ marginTop: 6, color: "#ccc", fontSize: 13 }}>
                      Offset to midline: {off ? `${sideLabel(off.side)} ${off.dist}px` : "—"}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ marginTop: 8 }}>
            <small style={{ color: "#999" }}>
              Presets add template points anchored to detected landmarks when possible. Edit each point’s Product and Dose inline.
            </small>
          </div>
        </div>
      </div>
    </div>
  );
}
*/

/* adding muscles and vascularity
// src/components/FaceCanvas.tsx
import React, { useEffect, useRef, useState } from "react";
import {
  Stage,
  Layer,
  Image as KonvaImageElement,
  Circle,
  Line,
  Text,
  Group,
} from "react-konva";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import { nanoid } from "nanoid";

type Landmark = [number, number, number?];

type AnnotationPoint = {
  id: string;
  x: number;
  y: number;
  product?: string;
  dose?: string;
};

const DEFAULT_STAGE_WIDTH = 800;
const DEFAULT_STAGE_HEIGHT = 800;

function fitImage(imageWidth: number, imageHeight: number, stageW: number, stageH: number) {
  const imgRatio = imageWidth / imageHeight;
  const stageRatio = stageW / stageH;
  let width = stageW;
  let height = stageH;
  if (imgRatio > stageRatio) {
    width = stageW;
    height = Math.round(stageW / imgRatio);
  } else {
    height = stageH;
    width = Math.round(stageH * imgRatio);
  }
  return { width, height };
}

// vector helpers
function vecSub(a: { x: number; y: number }, b: { x: number; y: number }) {
  return { x: a.x - b.x, y: a.y - b.y };
}
function vecDot(a: { x: number; y: number }, b: { x: number; y: number }) {
  return a.x * b.x + a.y * b.y;
}
function vecLen(a: { x: number; y: number }) {
  return Math.hypot(a.x, a.y);
}
function vecScale(a: { x: number; y: number }, s: number) {
  return { x: a.x * s, y: a.y * s };
}
function vecAdd(a: { x: number; y: number }, b: { x: number; y: number }) {
  return { x: a.x + b.x, y: a.y + b.y };
}
function perpDistanceToLine(point: { x: number; y: number }, a: { x: number; y: number }, b: { x: number; y: number }) {
  const v = vecSub(b, a);
  const w = vecSub(point, a);
  const vlen2 = v.x * v.x + v.y * v.y;
  if (vlen2 === 0) return { dist: vecLen(w), side: 0 };
  const t = vecDot(w, v) / vlen2;
  const proj = vecAdd(a, vecScale(v, t));
  const perp = vecSub(point, proj);
  const dist = vecLen(perp);
  const cross = v.x * w.y - v.y * w.x;
  const side = cross;
  return { dist, side };
}

// @ts-ignore
export default function FaceCanvas(): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [imgEl, setImgEl] = useState<HTMLImageElement | null>(null);
  const stageRef = useRef<any>(null);

  const [imageFit, setImageFit] = useState<{ width: number; height: number } | null>(null);
  const [landmarks, setLandmarks] = useState<Landmark[]>([]);
  const [detector, setDetector] = useState<any | null>(null);
  const [points, setPoints] = useState<AnnotationPoint[]>([]);
  const [stageSize, setStageSize] = useState({ width: DEFAULT_STAGE_WIDTH, height: DEFAULT_STAGE_HEIGHT });

  // toggles
  const [showMidline, setShowMidline] = useState(true);
  const [showEyeLine, setShowEyeLine] = useState(true);
  const [showThirds, setShowThirds] = useState(true);

  // NEW: toggle for showing/hiding the small landmark dots
  const [showLandmarks, setShowLandmarks] = useState<boolean>(true);

  // file -> data URL
  useEffect(() => {
    if (!file) {
      setImageSrc(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setImageSrc(String(reader.result));
    reader.onerror = (e) => {
      console.error("FileReader error", e);
      setImageSrc(null);
    };
    reader.readAsDataURL(file);
  }, [file]);

  // data URL -> image element + compute fit
  useEffect(() => {
    if (!imageSrc) {
      setImgEl(null);
      setImageFit(null);
      setStageSize({ width: DEFAULT_STAGE_WIDTH, height: DEFAULT_STAGE_HEIGHT });
      return;
    }
    let mounted = true;
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = imageSrc;
    image.onload = () => {
      if (!mounted) return;
      setImgEl(image);
      const naturalW = image.naturalWidth || DEFAULT_STAGE_WIDTH;
      const naturalH = image.naturalHeight || DEFAULT_STAGE_HEIGHT;
      const fit = fitImage(naturalW, naturalH, DEFAULT_STAGE_WIDTH, DEFAULT_STAGE_HEIGHT);
      setImageFit(fit);
      setStageSize({ width: fit.width, height: fit.height });
    };
    image.onerror = (err) => {
      console.error("Image load error", err);
      if (mounted) {
        setImgEl(null);
        setImageFit(null);
      }
    };
    return () => {
      mounted = false;
    };
  }, [imageSrc]);

  // create detector
  useEffect(() => {
    let mounted = true;
    let created: any = null;

    async function setup() {
      try {
        await tf.setBackend("webgl");
        await tf.ready();

        const modelConst = (faceLandmarksDetection as any).SupportedModels?.MediaPipeFaceMesh ??
                           (faceLandmarksDetection as any).MediaPipeFaceMesh;

        const detectorConfigTfjs = { runtime: "tfjs", maxFaces: 1, refineLandmarks: true };

        try {
          created = await (faceLandmarksDetection as any).createDetector(modelConst, detectorConfigTfjs);
          if (mounted) setDetector(created);
          else if (created?.dispose) created.dispose();
        } catch (errCreate) {
          console.warn("createDetector failed; trying legacy load():", errCreate);
          try {
            created = await (faceLandmarksDetection as any).load(
              (faceLandmarksDetection as any).SupportedPackages?.mediapipeFacemesh ??
                (faceLandmarksDetection as any).SupportedPackages,
              { maxFaces: 1 }
            );
            if (mounted) setDetector(created);
          } catch (errLegacy) {
            console.error("Both createDetector and legacy load() failed:", errLegacy);
            throw errLegacy;
          }
        }
      } catch (err) {
        console.error("Detector setup error:", err);
      }
    }

    setup();

    return () => {
      mounted = false;
      if (created && typeof created.dispose === "function") {
        try {
          created.dispose();
        } catch {}
      }
    };
  }, []);

  // detection (run after imageFit)
  useEffect(() => {
    if (!detector || !imgEl || !imageFit) return;
    let cancelled = false;

    async function detectOnce() {
      try {
        const off = document.createElement("canvas");
        off.width = imageFit.width;
        off.height = imageFit.height;
        const ctx = off.getContext("2d")!;
        ctx.clearRect(0, 0, off.width, off.height);
        ctx.drawImage(imgEl, 0, 0, imageFit.width, imageFit.height);

        let predictions: any = null;
        let lastErr: any = null;

        try {
          predictions = await (detector as any).estimateFaces(off);
        } catch (e1) {
          lastErr = e1;
          try {
            predictions = await (detector as any).estimateFaces({ input: off });
          } catch (e2) {
            lastErr = e2;
            try {
              predictions = await (detector as any).estimateFaces(off, { flipHorizontal: false });
            } catch (e3) {
              lastErr = e3;
            }
          }
        }

        if (!predictions) {
          console.error("All estimateFaces attempts failed:", lastErr);
          throw lastErr;
        }

        if (cancelled) return;

        if (!predictions || predictions.length === 0) {
          setLandmarks([]);
          return;
        }

        const face = predictions[0] as any;
        let mesh: Landmark[] = [];

        if (Array.isArray(face?.scaledMesh)) mesh = face.scaledMesh as Landmark[];
        else if (Array.isArray(face?.mesh)) mesh = face.mesh as Landmark[];
        else if (Array.isArray(face?.keypoints)) mesh = (face.keypoints as any[]).map(kp => [kp.x, kp.y, kp.z ?? 0]);
        else if (Array.isArray(face?.keypoints3D)) mesh = face.keypoints3D as Landmark[];
        else console.warn("Unknown prediction shape", face);

        setLandmarks(mesh || []);

        // populate some sensible default points (lips) if there are no points yet
        if ((!points || points.length === 0) && mesh && mesh.length > 0) {
          const lipsIndices = [13, 14, 78, 308];
          const defaultPoints = lipsIndices
            .map((i) => {
              const p = mesh && mesh[i];
              if (!p) return null;
              return { id: nanoid(), x: p[0], y: p[1], product: "", dose: "" };
            })
            .filter(Boolean) as AnnotationPoint[];
          setPoints(defaultPoints);
        }
      } catch (err) {
        console.error("detectOnce error:", err);
      }
    }

    detectOnce();

    return () => {
      cancelled = true;
    };
  }, [detector, imgEl, imageFit]); // re-run when imageFit is set

  // keep stage size in sync with imageFit whenever imageFit changes
  useEffect(() => {
    if (imageFit) {
      setStageSize({ width: imageFit.width, height: imageFit.height });
    }
  }, [imageFit]);

  function onAddPoint() {
    setPoints((p) => [...p, { id: nanoid(), x: (imageFit?.width ?? stageSize.width) / 2, y: (imageFit?.height ?? stageSize.height) / 2, product: "", dose: "" }]);
  }

  function onExport() {
    if (!stageRef.current) return;
    try {
      // @ts-ignore (kept intentionally)
      const uri = stageRef.current.toDataURL({ pixelRatio: 2 });
      const link = document.createElement("a");
      link.download = `face_map_${Date.now()}.png`;
      link.href = uri;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error("Export error:", err);
      alert("Export failed (canvas may be tainted). Make sure image is loaded from same origin or crossOrigin='anonymous'.");
    }
  }

  // reference calculations
  const midlineEndpoints =
    landmarks.length > 152 && landmarks[10] && landmarks[152]
      ? { a: { x: landmarks[10][0], y: landmarks[10][1] }, b: { x: landmarks[152][0], y: landmarks[152][1] } }
      : (() => {
          if (!landmarks || landmarks.length === 0) return null;
          let minY = Infinity,
            maxY = -Infinity;
          let minX = Infinity,
            maxX = -Infinity;
          for (const lm of landmarks) {
            if (!lm) continue;
            const x = lm[0],
              y = lm[1];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
          const cx = (minX + maxX) / 2;
          return { a: { x: cx, y: minY }, b: { x: cx, y: maxY } };
        })();

  const eyeLine =
    landmarks.length > 263 && landmarks[33] && landmarks[263]
      ? { l: { x: landmarks[33][0], y: landmarks[33][1] }, r: { x: landmarks[263][0], y: landmarks[263][1] } }
      : null;

  const thirdsYs = (() => {
    if (!landmarks || landmarks.length === 0) return null;
    let minY = Infinity,
      maxY = -Infinity;
    let minX = Infinity,
      maxX = -Infinity;
    for (const lm of landmarks) {
      if (!lm) continue;
      const x = lm[0],
        y = lm[1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const h = maxY - minY;
    if (!isFinite(h) || h <= 0) return null;
    const y1 = minY + h / 3;
    const y2 = minY + (2 * h) / 3;
    return { y1, y2, xmin: minX, xmax: maxX };
  })();

  // offsets
  const pointOffsets = points.map((pt) => {
    if (!midlineEndpoints) return { id: pt.id, dist: 0, side: 0 };
    const { a, b } = midlineEndpoints;
    const { dist, side } = perpDistanceToLine({ x: pt.x, y: pt.y }, a, b);
    return { id: pt.id, dist: Math.round(dist), side: side };
  });
  function sideLabel(side: number) {
    if (side === 0) return "C";
    return side > 0 ? "L" : "R";
  }

  // ---------- PRESETS ----------
  const presetTemplates = {
    lips: {
      indices: [13, 14, 78, 308],
      label: "Lips",
    },
    nasolabial: {
      indices: [50, 280, 146, 376],
      label: "Nasolabial",
    },
    marionette: {
      indices: [57, 287, 164, 393],
      label: "Marionette",
    },
  } as const;

  function addPreset(name: keyof typeof presetTemplates) {
    const tpl = presetTemplates[name];
    if (!landmarks || landmarks.length === 0) {
      const cx = (stageSize.width / 2);
      const cy = (stageSize.height / 2);
      const fallbackPoints: AnnotationPoint[] = [ -1, 0, 1, 2 ].map((i) => ({
        id: nanoid(),
        x: cx + (i - 1.5) * 12,
        y: cy + (i - 1.5) * 10,
        product: "",
        dose: "",
      }));
      setPoints((p) => [...p, ...fallbackPoints]);
      return;
    }

    const newPts: AnnotationPoint[] = [];
    for (const idx of tpl.indices) {
      const lm = landmarks[idx];
      if (lm && typeof lm[0] === "number") {
        newPts.push({ id: nanoid(), x: lm[0], y: lm[1], product: "", dose: "" });
      } else {
        if (midlineEndpoints && thirdsYs) {
          const cx = (midlineEndpoints.a.x + midlineEndpoints.b.x) / 2;
          const dx = (newPts.length % 2 === 0) ? -12 : 12;
          const y = newPts.length < 2 ? thirdsYs.y1 : thirdsYs.y2;
          newPts.push({ id: nanoid(), x: cx + dx, y, product: "", dose: "" });
        } else {
          newPts.push({ id: nanoid(), x: stageSize.width / 2 + (newPts.length - 1) * 10, y: stageSize.height / 2, product: "", dose: "" });
        }
      }
    }
    setPoints((p) => [...p, ...newPts]);
  }

  // update point metadata helpers
  function updatePointMeta(id: string, data: Partial<Pick<AnnotationPoint, "product" | "dose" | "x" | "y">>) {
    setPoints((prev) => prev.map((pt) => (pt.id === id ? { ...pt, ...data } : pt)));
  }

  // delete
  function removePoint(id: string) {
    setPoints((ps) => ps.filter((p) => p.id !== id));
  }

  // snapping helper
  function snapPointToNearestLandmark(id: string, radius = 12) {
    const pt = points.find((p) => p.id === id);
    if (!pt || !landmarks || landmarks.length === 0) return;
    let best: { idx: number; dist: number } | null = null;
    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i];
      if (!lm) continue;
      const dx = lm[0] - pt.x;
      const dy = lm[1] - pt.y;
      const d = Math.hypot(dx, dy);
      if (d <= radius && (!best || d < best.dist)) best = { idx: i, dist: d };
    }
    if (best) {
      const lm = landmarks[best.idx];
      updatePointMeta(id, { x: lm[0], y: lm[1] });
    }
  }

  return (
    <div style={{ border: "1px solid #ddd", padding: 12, borderRadius: 8, maxWidth: 1200 }}>
      <div style={{ marginBottom: 12, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            if (!e.target.files?.[0]) return;
            setFile(e.target.files[0]);
            setPoints([]);
            setLandmarks([]);
            setImageFit(null);
          }}
        />
        <button onClick={onAddPoint}>Add point</button>
        <button onClick={onExport}>Export PNG</button>

        <div style={{ marginLeft: 8, display: "flex", gap: 6, alignItems: "center" }}>
          <strong>Presets:</strong>
          <button onClick={() => addPreset("lips")}>Lips</button>
          <button onClick={() => addPreset("nasolabial")}>Nasolabial</button>
          <button onClick={() => addPreset("marionette")}>Marionette</button>
        </div>

        {/* NEW: Landmark toggle (doesn't affect annotation points, midline, etc.) }
        <div style={{ display: "flex", gap: 10, alignItems: "center", marginLeft: 8 }}>
          <label>
            <input
              type="checkbox"
              checked={showLandmarks}
              onChange={(e) => setShowLandmarks(e.target.checked)}
            />{" "}
            Landmarks
          </label>
        </div>

        <div style={{ marginLeft: "auto", display: "flex", gap: 12 }}>
          <label>
            <input type="checkbox" checked={showMidline} onChange={(e) => setShowMidline(e.target.checked)} /> Midline
          </label>
          <label>
            <input type="checkbox" checked={showEyeLine} onChange={(e) => setShowEyeLine(e.target.checked)} /> Eye-line
          </label>
          <label>
            <input type="checkbox" checked={showThirds} onChange={(e) => setShowThirds(e.target.checked)} /> Thirds
          </label>
        </div>
      </div>

      <div style={{ display: "flex", gap: 12 }}>
        <div style={{ border: "1px solid #eee", padding: 6 }}>
          <Stage width={stageSize.width} height={stageSize.height} ref={stageRef}>
            <Layer>
              {imgEl ? (
                // @ts-ignore
                <KonvaImageElement image={imgEl as any} x={0} y={0} width={stageSize.width} height={stageSize.height} />
              ) : null}
            </Layer>

            <Layer>
              {/* landmarks: only render when showLandmarks=true }
              {showLandmarks && landmarks.length > 0
                ? landmarks.map((lm, idx) => {
                    const x = lm[0];
                    const y = lm[1];
                    return <Circle key={`lm-${idx}`} x={x} y={y} radius={1.2} fill="rgba(0,150,255,0.9)" />;
                  })
                : null}

              {showMidline && midlineEndpoints ? (
                <>
                  <Line
                    points={[midlineEndpoints.a.x, midlineEndpoints.a.y, midlineEndpoints.b.x, midlineEndpoints.b.y]}
                    stroke="rgba(255,0,0,0.85)"
                    strokeWidth={1.5}
                    dash={[6, 4]}
                  />
                  <Text
                    text="Midline"
                    x={(midlineEndpoints.a.x + midlineEndpoints.b.x) / 2 + 6}
                    y={(midlineEndpoints.a.y + midlineEndpoints.b.y) / 2 + 6}
                    fontSize={12}
                    fill="rgba(255,0,0,0.9)"
                  />
                </>
              ) : null}

              {showEyeLine && eyeLine ? (
                <>
                  <Line
                    points={[eyeLine.l.x, eyeLine.l.y, eyeLine.r.x, eyeLine.r.y]}
                    stroke="rgba(0,200,0,0.9)"
                    strokeWidth={1.5}
                  />
                  <Text
                    text="Eye-line"
                    x={(eyeLine.l.x + eyeLine.r.x) / 2 + 6}
                    y={(eyeLine.l.y + eyeLine.r.y) / 2 - 16}
                    fontSize={12}
                    fill="rgba(0,160,0,0.9)"
                  />
                </>
              ) : null}

              {showThirds && thirdsYs ? (
                <>
                  <Line points={[thirdsYs.xmin, thirdsYs.y1, thirdsYs.xmax, thirdsYs.y1]} stroke="rgba(0,0,0,0.5)" strokeWidth={1} dash={[4, 4]} />
                  <Text text="1/3" x={thirdsYs.xmax + 6} y={thirdsYs.y1 - 6} fontSize={11} fill="#333" />
                  <Line points={[thirdsYs.xmin, thirdsYs.y2, thirdsYs.xmax, thirdsYs.y2]} stroke="rgba(0,0,0,0.5)" strokeWidth={1} dash={[4, 4]} />
                  <Text text="2/3" x={thirdsYs.xmax + 6} y={thirdsYs.y2 - 6} fontSize={11} fill="#333" />
                </>
              ) : null}

              {points.map((p) => {
                const offset = pointOffsets.find((o) => o.id === p.id);
                const label = p.product ? `${p.product} ${p.dose ?? ""}`.trim() : `${offset ? sideLabel(offset.side) + " " + offset.dist + "px" : ""}`;
                return (
                  <Group
                    key={p.id}
                    x={p.x}
                    y={p.y}
                    draggable
                    onDragEnd={(e) => {
                      const nx = (e.target as any).x();
                      const ny = (e.target as any).y();
                      updatePointMeta(p.id, { x: nx, y: ny });
                    }}
                  >
                    <Circle radius={8} fill="rgba(255,165,0,0.95)" stroke="black" strokeWidth={1} />
                    <Text text="●" fontSize={12} offsetX={6} offsetY={6} />
                    <Text text={label} x={12} y={-10} fontSize={11} fill="#222" />
                  </Group>
                );
              })}
            </Layer>
          </Stage>
        </div>

        <div style={{ width: 380 }}>
          <h3>Session</h3>
          <div>
            <strong>Detected landmarks:</strong> {landmarks.length}
          </div>

          <div style={{ marginTop: 12 }}>
            <h4>Reference Lines</h4>
            <div>Midline: {midlineEndpoints ? `from (${Math.round(midlineEndpoints.a.x)},${Math.round(midlineEndpoints.a.y)}) to (${Math.round(midlineEndpoints.b.x)},${Math.round(midlineEndpoints.b.y)})` : "n/a"}</div>
            <div>Eye-line: {eyeLine ? `L(${Math.round(eyeLine.l.x)},${Math.round(eyeLine.l.y)}) R(${Math.round(eyeLine.r.x)},${Math.round(eyeLine.r.y)})` : "n/a"}</div>
            <div>Thirds: {thirdsYs ? `y1=${Math.round(thirdsYs.y1)} y2=${Math.round(thirdsYs.y2)}` : "n/a"}</div>
          </div>

          <div style={{ marginTop: 12 }}>
            <h4>Injection points</h4>
            {points.length === 0 ? <div>No points yet</div> : null}
            {points.map((p) => {
              const off = pointOffsets.find((o) => o.id === p.id);
              return (
                <div key={p.id} style={{ marginBottom: 10, padding: 8, borderRadius: 6, background: "#0f0f0f", color: "#fff" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <small style={{ opacity: 0.9 }}>ID: {p.id}</small>
                    <div style={{ display: "flex", gap: 6 }}>
                      <button onClick={() => removePoint(p.id)}>remove</button>
                      <button onClick={() => snapPointToNearestLandmark(p.id)}>snap</button>
                    </div>
                  </div>

                  <div style={{ marginTop: 6 }}>
                    <div>
                      x: {Math.round(p.x)}, y: {Math.round(p.y)}
                      <button onClick={() => updatePointMeta(p.id, { x: Math.round(p.x), y: Math.round(p.y) })} style={{ marginLeft: 6 }}>
                        refresh
                      </button>
                    </div>

                    <div style={{ marginTop: 6 }}>
                      <label style={{ display: "block", fontSize: 12, marginBottom: 4 }}>Product</label>
                      <input
                        value={p.product ?? ""}
                        placeholder="Product name"
                        onChange={(e) => updatePointMeta(p.id, { product: e.target.value })}
                        style={{ width: "100%", padding: "6px 8px", borderRadius: 4 }}
                      />
                    </div>

                    <div style={{ marginTop: 6 }}>
                      <label style={{ display: "block", fontSize: 12, marginBottom: 4 }}>Dose</label>
                      <input
                        value={p.dose ?? ""}
                        placeholder="e.g. 0.1 mL (0.5 unit)"
                        onChange={(e) => updatePointMeta(p.id, { dose: e.target.value })}
                        style={{ width: "100%", padding: "6px 8px", borderRadius: 4 }}
                      />
                    </div>

                    <div style={{ marginTop: 6, color: "#ccc", fontSize: 13 }}>
                      Offset to midline: {off ? `${sideLabel(off.side)} ${off.dist}px` : "—"}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ marginTop: 8 }}>
            <small style={{ color: "#999" }}>
              Presets add template points anchored to detected landmarks when possible. Edit each point’s Product and Dose inline.
            </small>
          </div>
        </div>
      </div>
    </div>
  );
}
*/

/*
// src/components/FaceCanvas.tsx
import React, { useEffect, useRef, useState } from "react";
import {
  Stage,
  Layer,
  Image as KonvaImageElement,
  Circle,
  Line,
  Text,
  Group,
  Rect,
} from "react-konva";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import { nanoid } from "nanoid";

type Landmark = [number, number, number?];

type AnnotationPoint = {
  id: string;
  x: number;
  y: number;
  product?: string;
  dose?: string;
};

const DEFAULT_STAGE_WIDTH = 800;
const DEFAULT_STAGE_HEIGHT = 800;

function fitImage(imageWidth: number, imageHeight: number, stageW: number, stageH: number) {
  const imgRatio = imageWidth / imageHeight;
  const stageRatio = stageW / stageH;
  let width = stageW;
  let height = stageH;
  if (imgRatio > stageRatio) {
    width = stageW;
    height = Math.round(stageW / imgRatio);
  } else {
    height = stageH;
    width = Math.round(stageH * imgRatio);
  }
  return { width, height };
}

// vector helpers
function vecSub(a: { x: number; y: number }, b: { x: number; y: number }) {
  return { x: a.x - b.x, y: a.y - b.y };
}
function vecDot(a: { x: number; y: number }, b: { x: number; y: number }) {
  return a.x * b.x + a.y * b.y;
}
function vecLen(a: { x: number; y: number }) {
  return Math.hypot(a.x, a.y);
}
function vecScale(a: { x: number; y: number }, s: number) {
  return { x: a.x * s, y: a.y * s };
}
function vecAdd(a: { x: number; y: number }, b: { x: number; y: number }) {
  return { x: a.x + b.x, y: a.y + b.y };
}
function perpDistanceToLine(point: { x: number; y: number }, a: { x: number; y: number }, b: { x: number; y: number }) {
  const v = vecSub(b, a);
  const w = vecSub(point, a);
  const vlen2 = v.x * v.x + v.y * v.y;
  if (vlen2 === 0) return { dist: vecLen(w), side: 0 };
  const t = vecDot(w, v) / vlen2;
  const proj = vecAdd(a, vecScale(v, t));
  const perp = vecSub(point, proj);
  const dist = vecLen(perp);
  const cross = v.x * w.y - v.y * w.x;
  const side = cross;
  return { dist, side };
}

// @ts-ignore
export default function FaceCanvas(): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [imgEl, setImgEl] = useState<HTMLImageElement | null>(null);
  const stageRef = useRef<any>(null);

  const [imageFit, setImageFit] = useState<{ width: number; height: number } | null>(null);
  const [landmarks, setLandmarks] = useState<Landmark[]>([]);
  const [detector, setDetector] = useState<any | null>(null);
  const [points, setPoints] = useState<AnnotationPoint[]>([]);
  const [stageSize, setStageSize] = useState({ width: DEFAULT_STAGE_WIDTH, height: DEFAULT_STAGE_HEIGHT });

  // toggles
  const [showMidline, setShowMidline] = useState(true);
  const [showEyeLine, setShowEyeLine] = useState(true);
  const [showThirds, setShowThirds] = useState(true);
  const [showLandmarks, setShowLandmarks] = useState<boolean>(true);

  // NEW toggles for anatomy overlays
  const [showMuscles, setShowMuscles] = useState<boolean>(false);
  const [showVessels, setShowVessels] = useState<boolean>(false);

  // file -> data URL
  useEffect(() => {
    if (!file) {
      setImageSrc(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setImageSrc(String(reader.result));
    reader.onerror = (e) => {
      console.error("FileReader error", e);
      setImageSrc(null);
    };
    reader.readAsDataURL(file);
  }, [file]);

  // data URL -> image element + compute fit
  useEffect(() => {
    if (!imageSrc) {
      setImgEl(null);
      setImageFit(null);
      setStageSize({ width: DEFAULT_STAGE_WIDTH, height: DEFAULT_STAGE_HEIGHT });
      return;
    }
    let mounted = true;
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = imageSrc;
    image.onload = () => {
      if (!mounted) return;
      setImgEl(image);
      const naturalW = image.naturalWidth || DEFAULT_STAGE_WIDTH;
      const naturalH = image.naturalHeight || DEFAULT_STAGE_HEIGHT;
      const fit = fitImage(naturalW, naturalH, DEFAULT_STAGE_WIDTH, DEFAULT_STAGE_HEIGHT);
      setImageFit(fit);
      setStageSize({ width: fit.width, height: fit.height });
    };
    image.onerror = (err) => {
      console.error("Image load error", err);
      if (mounted) {
        setImgEl(null);
        setImageFit(null);
      }
    };
    return () => {
      mounted = false;
    };
  }, [imageSrc]);

  // create detector
  useEffect(() => {
    let mounted = true;
    let created: any = null;

    async function setup() {
      try {
        await tf.setBackend("webgl");
        await tf.ready();

        const modelConst = (faceLandmarksDetection as any).SupportedModels?.MediaPipeFaceMesh ??
                           (faceLandmarksDetection as any).MediaPipeFaceMesh;

        const detectorConfigTfjs = { runtime: "tfjs", maxFaces: 1, refineLandmarks: true };

        try {
          created = await (faceLandmarksDetection as any).createDetector(modelConst, detectorConfigTfjs);
          if (mounted) setDetector(created);
          else if (created?.dispose) created.dispose();
        } catch (errCreate) {
          console.warn("createDetector failed; trying legacy load():", errCreate);
          try {
            created = await (faceLandmarksDetection as any).load(
              (faceLandmarksDetection as any).SupportedPackages?.mediapipeFacemesh ??
                (faceLandmarksDetection as any).SupportedPackages,
              { maxFaces: 1 }
            );
            if (mounted) setDetector(created);
          } catch (errLegacy) {
            console.error("Both createDetector and legacy load() failed:", errLegacy);
            throw errLegacy;
          }
        }
      } catch (err) {
        console.error("Detector setup error:", err);
      }
    }

    setup();

    return () => {
      mounted = false;
      if (created && typeof created.dispose === "function") {
        try {
          created.dispose();
        } catch {}
      }
    };
  }, []);

  // detection (run after imageFit)
  useEffect(() => {
    if (!detector || !imgEl || !imageFit) return;
    let cancelled = false;

    async function detectOnce() {
      try {
        const off = document.createElement("canvas");
        off.width = imageFit.width;
        off.height = imageFit.height;
        const ctx = off.getContext("2d")!;
        ctx.clearRect(0, 0, off.width, off.height);
        ctx.drawImage(imgEl, 0, 0, imageFit.width, imageFit.height);

        let predictions: any = null;
        let lastErr: any = null;

        try {
          predictions = await (detector as any).estimateFaces(off);
        } catch (e1) {
          lastErr = e1;
          try {
            predictions = await (detector as any).estimateFaces({ input: off });
          } catch (e2) {
            lastErr = e2;
            try {
              predictions = await (detector as any).estimateFaces(off, { flipHorizontal: false });
            } catch (e3) {
              lastErr = e3;
            }
          }
        }

        if (!predictions) {
          console.error("All estimateFaces attempts failed:", lastErr);
          throw lastErr;
        }

        if (cancelled) return;

        if (!predictions || predictions.length === 0) {
          setLandmarks([]);
          return;
        }

        const face = predictions[0] as any;
        let mesh: Landmark[] = [];

        if (Array.isArray(face?.scaledMesh)) mesh = face.scaledMesh as Landmark[];
        else if (Array.isArray(face?.mesh)) mesh = face.mesh as Landmark[];
        else if (Array.isArray(face?.keypoints)) mesh = (face.keypoints as any[]).map(kp => [kp.x, kp.y, kp.z ?? 0]);
        else if (Array.isArray(face?.keypoints3D)) mesh = face.keypoints3D as Landmark[];
        else console.warn("Unknown prediction shape", face);

        setLandmarks(mesh || []);

        // populate some sensible default points (lips) if there are no points yet
        if ((!points || points.length === 0) && mesh && mesh.length > 0) {
          const lipsIndices = [13, 14, 78, 308];
          const defaultPoints = lipsIndices
            .map((i) => {
              const p = mesh && mesh[i];
              if (!p) return null;
              return { id: nanoid(), x: p[0], y: p[1], product: "", dose: "" };
            })
            .filter(Boolean) as AnnotationPoint[];
          setPoints(defaultPoints);
        }
      } catch (err) {
        console.error("detectOnce error:", err);
      }
    }

    detectOnce();

    return () => {
      cancelled = true;
    };
  }, [detector, imgEl, imageFit]); // re-run when imageFit is set

  // keep stage size in sync with imageFit whenever imageFit changes
  useEffect(() => {
    if (imageFit) {
      setStageSize({ width: imageFit.width, height: imageFit.height });
    }
  }, [imageFit]);

  function onAddPoint() {
    setPoints((p) => [...p, { id: nanoid(), x: (imageFit?.width ?? stageSize.width) / 2, y: (imageFit?.height ?? stageSize.height) / 2, product: "", dose: "" }]);
  }

  function onExport() {
    if (!stageRef.current) return;
    try {
      // @ts-ignore (kept intentionally)
      const uri = stageRef.current.toDataURL({ pixelRatio: 2 });
      const link = document.createElement("a");
      link.download = `face_map_${Date.now()}.png`;
      link.href = uri;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error("Export error:", err);
      alert("Export failed (canvas may be tainted). Make sure image is loaded from same origin or crossOrigin='anonymous'.");
    }
  }

  // reference calculations (unchanged)
  const midlineEndpoints =
    landmarks.length > 152 && landmarks[10] && landmarks[152]
      ? { a: { x: landmarks[10][0], y: landmarks[10][1] }, b: { x: landmarks[152][0], y: landmarks[152][1] } }
      : (() => {
          if (!landmarks || landmarks.length === 0) return null;
          let minY = Infinity,
            maxY = -Infinity;
          let minX = Infinity,
            maxX = -Infinity;
          for (const lm of landmarks) {
            if (!lm) continue;
            const x = lm[0],
              y = lm[1];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
          const cx = (minX + maxX) / 2;
          return { a: { x: cx, y: minY }, b: { x: cx, y: maxY } };
        })();

  const eyeLine =
    landmarks.length > 263 && landmarks[33] && landmarks[263]
      ? { l: { x: landmarks[33][0], y: landmarks[33][1] }, r: { x: landmarks[263][0], y: landmarks[263][1] } }
      : null;

  const thirdsYs = (() => {
    if (!landmarks || landmarks.length === 0) return null;
    let minY = Infinity,
      maxY = -Infinity;
    let minX = Infinity,
      maxX = -Infinity;
    for (const lm of landmarks) {
      if (!lm) continue;
      const x = lm[0],
        y = lm[1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const h = maxY - minY;
    if (!isFinite(h) || h <= 0) return null;
    const y1 = minY + h / 3;
    const y2 = minY + (2 * h) / 3;
    return { y1, y2, xmin: minX, xmax: maxX };
  })();

  // offsets
  const pointOffsets = points.map((pt) => {
    if (!midlineEndpoints) return { id: pt.id, dist: 0, side: 0 };
    const { a, b } = midlineEndpoints;
    const { dist, side } = perpDistanceToLine({ x: pt.x, y: pt.y }, a, b);
    return { id: pt.id, dist: Math.round(dist), side: side };
  });
  function sideLabel(side: number) {
    if (side === 0) return "C";
    return side > 0 ? "L" : "R";
  }

  // ---------- PRESETS (unchanged) ----------
  const presetTemplates = {
    lips: { indices: [13, 14, 78, 308], label: "Lips" },
    nasolabial: { indices: [50, 280, 146, 376], label: "Nasolabial" },
    marionette: { indices: [57, 287, 164, 393], label: "Marionette" },
  } as const;

  function addPreset(name: keyof typeof presetTemplates) {
    const tpl = presetTemplates[name];
    if (!landmarks || landmarks.length === 0) {
      const cx = stageSize.width / 2;
      const cy = stageSize.height / 2;
      const fallbackPoints: AnnotationPoint[] = [-1, 0, 1, 2].map((i) => ({
        id: nanoid(),
        x: cx + (i - 1.5) * 12,
        y: cy + (i - 1.5) * 10,
        product: "",
        dose: "",
      }));
      setPoints((p) => [...p, ...fallbackPoints]);
      return;
    }

    const newPts: AnnotationPoint[] = [];
    for (const idx of tpl.indices) {
      const lm = landmarks[idx];
      if (lm && typeof lm[0] === "number") {
        newPts.push({ id: nanoid(), x: lm[0], y: lm[1], product: "", dose: "" });
      } else {
        if (midlineEndpoints && thirdsYs) {
          const cx = (midlineEndpoints.a.x + midlineEndpoints.b.x) / 2;
          const dx = (newPts.length % 2 === 0) ? -12 : 12;
          const y = newPts.length < 2 ? thirdsYs.y1 : thirdsYs.y2;
          newPts.push({ id: nanoid(), x: cx + dx, y, product: "", dose: "" });
        } else {
          newPts.push({ id: nanoid(), x: stageSize.width / 2 + (newPts.length - 1) * 10, y: stageSize.height / 2, product: "", dose: "" });
        }
      }
    }
    setPoints((p) => [...p, ...newPts]);
  }

  // update point metadata helpers
  function updatePointMeta(id: string, data: Partial<Pick<AnnotationPoint, "product" | "dose" | "x" | "y">>) {
    setPoints((prev) => prev.map((pt) => (pt.id === id ? { ...pt, ...data } : pt)));
  }

  // delete
  function removePoint(id: string) {
    setPoints((ps) => ps.filter((p) => p.id !== id));
  }

  // snapping helper
  function snapPointToNearestLandmark(id: string, radius = 12) {
    const pt = points.find((p) => p.id === id);
    if (!pt || !landmarks || landmarks.length === 0) return;
    let best: { idx: number; dist: number } | null = null;
    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i];
      if (!lm) continue;
      const dx = lm[0] - pt.x;
      const dy = lm[1] - pt.y;
      const d = Math.hypot(dx, dy);
      if (d <= radius && (!best || d < best.dist)) best = { idx: i, dist: d };
    }
    if (best) {
      const lm = landmarks[best.idx];
      updatePointMeta(id, { x: lm[0], y: lm[1] });
    }
  }

  // ---------- MUSCLES & VASCULAR OVERLAY HELPERS ----------
  // NOTE: The indices below are approximate MediaPipe FaceMesh indices commonly used to map regions.
  // These overlays are visual aids (not clinical maps). You can adjust indices to better match your images.

  // Helper: convert landmark index array -> flattened points array for Konva Line/Polygon
  function indicesToPoints(indices: number[]) {
    const pts: number[] = [];
    for (const i of indices) {
      const lm = landmarks[i];
      if (!lm || typeof lm[0] !== "number") return null; // if any index missing, abort (so overlay doesn't distort)
      pts.push(lm[0], lm[1]);
    }
    return pts;
  }

  // muscle polygons (each is an array of landmark indices forming a polygon)
  const musclePolygonsIndices: { id: string; label: string; indices: number[] }[] = [
    // Orbicularis oris outer ring (approx)
    { id: "orb_oris_outer", label: "Orbicularis oris (outer)", indices: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291] },
    // Orbicularis oris inner ring (approx)
    { id: "orb_oris_inner", label: "Orbicularis oris (inner)", indices: [78, 95, 88, 178, 87, 14, 308, 324, 318, 402, 317] },
    // Zygomaticus region (left + right as two polygons using mouth corner -> cheek)
    { id: "zygomaticus_left", label: "Zygomaticus (L)", indices: [61, 226, 345, 454, 334] },
    { id: "zygomaticus_right", label: "Zygomaticus (R)", indices: [291, 446, 323, 454 - 0 /* placeholder , 123].map((x) => x) }, // right polygon uses approximate indices
    // Mentalis / lower chin region
    { id: "mentalis", label: "Mentalis", indices: [152, 148, 176, 149, 150] },
  ];

  // vascular paths: arrays of indices forming a polyline (not closed)
  const vesselPathsIndices: { id: string; label: string; indices: number[] }[] = [
    // facial artery path (approx) - from near mandible up toward nasolabial fold
    { id: "facial_artery_l", label: "Facial artery (L)", indices: [234, 93, 132, 58, 0, 1, 2] },
    { id: "facial_artery_r", label: "Facial artery (R)", indices: [454, 323, 361, 288, 267, 266, 265] },
    // angular artery near nose / medial canthus
    { id: "angular_l", label: "Angular (L)", indices: [2, 98, 4, 5] },
    { id: "angular_r", label: "Angular (R)", indices: [265, 362, 257, 256] },
    // superior/inferior labial approximate paths across lips
    { id: "sup_labial", label: "Superior labial", indices: [61, 62, 63, 64, 65, 66] },
    { id: "inf_labial", label: "Inferior labial", indices: [291, 292, 293, 294, 295, 296] },
  ];

  // Convert indices arrays to Konva-friendly point arrays, skipping those that fail
  const muscleShapes = musclePolygonsIndices
    .map((m) => {
      const pts = indicesToPoints(m.indices);
      return pts ? { ...m, points: pts } : null;
    })
    .filter(Boolean) as Array<{ id: string; label: string; points: number[] }>;

  const vesselShapes = vesselPathsIndices
    .map((v) => {
      const pts = indicesToPoints(v.indices);
      return pts ? { ...v, points: pts } : null;
    })
    .filter(Boolean) as Array<{ id: string; label: string; points: number[] }>;

  // ---------- RENDER ----------
  return (
    <div style={{ border: "1px solid #ddd", padding: 12, borderRadius: 8, maxWidth: 1300 }}>
      <div style={{ marginBottom: 12, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            if (!e.target.files?.[0]) return;
            setFile(e.target.files[0]);
            setPoints([]);
            setLandmarks([]);
            setImageFit(null);
          }}
        />
        <button onClick={onAddPoint}>Add point</button>
        <button onClick={onExport}>Export PNG</button>

        <div style={{ marginLeft: 8, display: "flex", gap: 6, alignItems: "center" }}>
          <strong>Presets:</strong>
          <button onClick={() => addPreset("lips")}>Lips</button>
          <button onClick={() => addPreset("nasolabial")}>Nasolabial</button>
          <button onClick={() => addPreset("marionette")}>Marionette</button>
        </div>

        {/* Landmark toggle }
        <div style={{ display: "flex", gap: 10, alignItems: "center", marginLeft: 8 }}>
          <label>
            <input type="checkbox" checked={showLandmarks} onChange={(e) => setShowLandmarks(e.target.checked)} /> Landmarks
          </label>
        </div>

        {/* Anatomy toggles }
        <div style={{ display: "flex", gap: 10, alignItems: "center", marginLeft: 8 }}>
          <label>
            <input type="checkbox" checked={showMuscles} onChange={(e) => setShowMuscles(e.target.checked)} /> Muscles
          </label>
          <label>
            <input type="checkbox" checked={showVessels} onChange={(e) => setShowVessels(e.target.checked)} /> Vessels
          </label>
        </div>

        <div style={{ marginLeft: "auto", display: "flex", gap: 12 }}>
          <label>
            <input type="checkbox" checked={showMidline} onChange={(e) => setShowMidline(e.target.checked)} /> Midline
          </label>
          <label>
            <input type="checkbox" checked={showEyeLine} onChange={(e) => setShowEyeLine(e.target.checked)} /> Eye-line
          </label>
          <label>
            <input type="checkbox" checked={showThirds} onChange={(e) => setShowThirds(e.target.checked)} /> Thirds
          </label>
        </div>
      </div>

      <div style={{ display: "flex", gap: 12 }}>
        <div style={{ border: "1px solid #eee", padding: 6 }}>
          <Stage width={stageSize.width} height={stageSize.height} ref={stageRef}>
            <Layer>
              {imgEl ? (
                // @ts-ignore
                <KonvaImageElement image={imgEl as any} x={0} y={0} width={stageSize.width} height={stageSize.height} />
              ) : null}
            </Layer>

            <Layer>
              {/* landmarks: conditional }
              {showLandmarks && landmarks.length > 0
                ? landmarks.map((lm, idx) => {
                    const x = lm[0];
                    const y = lm[1];
                    return <Circle key={`lm-${idx}`} x={x} y={y} radius={1.2} fill="rgba(0,150,255,0.9)" />;
                  })
                : null}

              {/* MUSCLE OVERLAYS (filled translucent polygons) }
              {showMuscles &&
                muscleShapes.map((m) => (
                  <React.Fragment key={m.id}>
                    <Line points={m.points} closed stroke="rgba(180,0,120,0.9)" strokeWidth={1} fill="rgba(180,0,120,0.18)" />
                    {/* label near centroid }
                    {m.points.length >= 4 ? (
                      <Text
                        text={m.label}
                        fontSize={11}
                        fill="#9b0055"
                        x={m.points[0] + 8}
                        y={m.points[1] + 8}
                      />
                    ) : null}
                  </React.Fragment>
                ))}

              {/* VESSEL OVERLAYS (thin stroked polylines) }
              {showVessels &&
                vesselShapes.map((v) => (
                  <React.Fragment key={v.id}>
                    <Line points={v.points} stroke="rgba(220,20,60,0.95)" strokeWidth={2} dash={[6, 2]} lineCap="round" />
                    {/* label at first point }
                    {v.points.length >= 2 ? (
                      <Text text={v.label} fontSize={11} fill="#c71b2b" x={v.points[0] + 6} y={v.points[1] - 12} />
                    ) : null}
                  </React.Fragment>
                ))}

              {/* midline / eyeline / thirds }
              {showMidline && midlineEndpoints ? (
                <>
                  <Line
                    points={[midlineEndpoints.a.x, midlineEndpoints.a.y, midlineEndpoints.b.x, midlineEndpoints.b.y]}
                    stroke="rgba(255,0,0,0.85)"
                    strokeWidth={1.5}
                    dash={[6, 4]}
                  />
                  <Text
                    text="Midline"
                    x={(midlineEndpoints.a.x + midlineEndpoints.b.x) / 2 + 6}
                    y={(midlineEndpoints.a.y + midlineEndpoints.b.y) / 2 + 6}
                    fontSize={12}
                    fill="rgba(255,0,0,0.9)"
                  />
                </>
              ) : null}

              {showEyeLine && eyeLine ? (
                <>
                  <Line
                    points={[eyeLine.l.x, eyeLine.l.y, eyeLine.r.x, eyeLine.r.y]}
                    stroke="rgba(0,200,0,0.9)"
                    strokeWidth={1.5}
                  />
                  <Text
                    text="Eye-line"
                    x={(eyeLine.l.x + eyeLine.r.x) / 2 + 6}
                    y={(eyeLine.l.y + eyeLine.r.y) / 2 - 16}
                    fontSize={12}
                    fill="rgba(0,160,0,0.9)"
                  />
                </>
              ) : null}

              {showThirds && thirdsYs ? (
                <>
                  <Line points={[thirdsYs.xmin, thirdsYs.y1, thirdsYs.xmax, thirdsYs.y1]} stroke="rgba(0,0,0,0.5)" strokeWidth={1} dash={[4, 4]} />
                  <Text text="1/3" x={thirdsYs.xmax + 6} y={thirdsYs.y1 - 6} fontSize={11} fill="#333" />
                  <Line points={[thirdsYs.xmin, thirdsYs.y2, thirdsYs.xmax, thirdsYs.y2]} stroke="rgba(0,0,0,0.5)" strokeWidth={1} dash={[4, 4]} />
                  <Text text="2/3" x={thirdsYs.xmax + 6} y={thirdsYs.y2 - 6} fontSize={11} fill="#333" />
                </>
              ) : null}

              {/* annotation points }
              {points.map((p) => {
                const offset = pointOffsets.find((o) => o.id === p.id);
                const label = p.product ? `${p.product} ${p.dose ?? ""}`.trim() : `${offset ? sideLabel(offset.side) + " " + offset.dist + "px" : ""}`;
                return (
                  <Group
                    key={p.id}
                    x={p.x}
                    y={p.y}
                    draggable
                    onDragEnd={(e) => {
                      const nx = (e.target as any).x();
                      const ny = (e.target as any).y();
                      updatePointMeta(p.id, { x: nx, y: ny });
                    }}
                  >
                    <Circle radius={8} fill="rgba(255,165,0,0.95)" stroke="black" strokeWidth={1} />
                    <Text text="●" fontSize={12} offsetX={6} offsetY={6} />
                    <Text text={label} x={12} y={-10} fontSize={11} fill="#222" />
                  </Group>
                );
              })}
            </Layer>
          </Stage>
        </div>

        <div style={{ width: 420 }}>
          <h3>Session</h3>
          <div>
            <strong>Detected landmarks:</strong> {landmarks.length}
          </div>

          <div style={{ marginTop: 12 }}>
            <h4>Reference Lines</h4>
            <div>Midline: {midlineEndpoints ? `from (${Math.round(midlineEndpoints.a.x)},${Math.round(midlineEndpoints.a.y)}) to (${Math.round(midlineEndpoints.b.x)},${Math.round(midlineEndpoints.b.y)})` : "n/a"}</div>
            <div>Eye-line: {eyeLine ? `L(${Math.round(eyeLine.l.x)},${Math.round(eyeLine.l.y)}) R(${Math.round(eyeLine.r.x)},${Math.round(eyeLine.r.y)})` : "n/a"}</div>
            <div>Thirds: {thirdsYs ? `y1=${Math.round(thirdsYs.y1)} y2=${Math.round(thirdsYs.y2)}` : "n/a"}</div>
          </div>

          <div style={{ marginTop: 12 }}>
            <h4>Overlays</h4>
            <div>Muscles: {showMuscles ? "ON" : "OFF"}</div>
            <div>Vessels: {showVessels ? "ON" : "OFF"}</div>

            <div style={{ marginTop: 8 }}>
              <h5 style={{ marginBottom: 6 }}>Legend</h5>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <div style={{ width: 18, height: 12, background: "rgba(180,0,120,0.18)", border: "1px solid rgba(180,0,120,0.9)" }} />
                <div>Muscle region (approx)</div>
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 6 }}>
                <div style={{ width: 40, height: 4, background: "transparent", borderTop: "2px dashed rgba(220,20,60,0.95)" }} />
                <div>Vessel path (approx)</div>
              </div>
            </div>
          </div>

          <div style={{ marginTop: 12 }}>
            <h4>Injection points</h4>
            {points.length === 0 ? <div>No points yet</div> : null}
            {points.map((p) => {
              const off = pointOffsets.find((o) => o.id === p.id);
              return (
                <div key={p.id} style={{ marginBottom: 10, padding: 8, borderRadius: 6, background: "#0f0f0f", color: "#fff" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <small style={{ opacity: 0.9 }}>ID: {p.id}</small>
                    <div style={{ display: "flex", gap: 6 }}>
                      <button onClick={() => removePoint(p.id)}>remove</button>
                      <button onClick={() => snapPointToNearestLandmark(p.id)}>snap</button>
                    </div>
                  </div>

                  <div style={{ marginTop: 6 }}>
                    <div>
                      x: {Math.round(p.x)}, y: {Math.round(p.y)}
                      <button onClick={() => updatePointMeta(p.id, { x: Math.round(p.x), y: Math.round(p.y) })} style={{ marginLeft: 6 }}>
                        refresh
                      </button>
                    </div>

                    <div style={{ marginTop: 6 }}>
                      <label style={{ display: "block", fontSize: 12, marginBottom: 4 }}>Product</label>
                      <input
                        value={p.product ?? ""}
                        placeholder="Product name"
                        onChange={(e) => updatePointMeta(p.id, { product: e.target.value })}
                        style={{ width: "100%", padding: "6px 8px", borderRadius: 4 }}
                      />
                    </div>

                    <div style={{ marginTop: 6 }}>
                      <label style={{ display: "block", fontSize: 12, marginBottom: 4 }}>Dose</label>
                      <input
                        value={p.dose ?? ""}
                        placeholder="e.g. 0.1 mL (0.5 unit)"
                        onChange={(e) => updatePointMeta(p.id, { dose: e.target.value })}
                        style={{ width: "100%", padding: "6px 8px", borderRadius: 4 }}
                      />
                    </div>

                    <div style={{ marginTop: 6, color: "#ccc", fontSize: 13 }}>
                      Offset to midline: {off ? `${sideLabel(off.side)} ${off.dist}px` : "—"}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ marginTop: 8 }}>
            <small style={{ color: "#999" }}>
              Muscle and vascular overlays are approximations based on FaceMesh indices. If a shape looks off for your images I can tweak the landmark indices for each region.
            </small>
          </div>
        </div>
      </div>
    </div>
  );
} */

// src/components/FaceCanvas.tsx
import React, { useEffect, useRef, useState } from "react";
import {
  Stage,
  Layer,
  Image as KonvaImageElement,
  Circle,
  Line,
  Text,
  Group,
} from "react-konva";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import { nanoid } from "nanoid";

type Landmark = [number, number, number?];

type AnnotationPoint = {
  id: string;
  x: number;
  y: number;
  product?: string;
  dose?: string;
};

const DEFAULT_STAGE_WIDTH = 800;
const DEFAULT_STAGE_HEIGHT = 800;

function fitImage(imageWidth: number, imageHeight: number, stageW: number, stageH: number) {
  const imgRatio = imageWidth / imageHeight;
  const stageRatio = stageW / stageH;
  let width = stageW;
  let height = stageH;
  if (imgRatio > stageRatio) {
    width = stageW;
    height = Math.round(stageW / imgRatio);
  } else {
    height = stageH;
    width = Math.round(stageH * imgRatio);
  }
  return { width, height };
}

// vector helpers
function vecSub(a: { x: number; y: number }, b: { x: number; y: number }) {
  return { x: a.x - b.x, y: a.y - b.y };
}
function vecDot(a: { x: number; y: number }, b: { x: number; y: number }) {
  return a.x * b.x + a.y * b.y;
}
function vecLen(a: { x: number; y: number }) {
  return Math.hypot(a.x, a.y);
}
function vecScale(a: { x: number; y: number }, s: number) {
  return { x: a.x * s, y: a.y * s };
}
function vecAdd(a: { x: number; y: number }, b: { x: number; y: number }) {
  return { x: a.x + b.x, y: a.y + b.y };
}
function perpDistanceToLine(point: { x: number; y: number }, a: { x: number; y: number }, b: { x: number; y: number }) {
  const v = vecSub(b, a);
  const w = vecSub(point, a);
  const vlen2 = v.x * v.x + v.y * v.y;
  if (vlen2 === 0) return { dist: vecLen(w), side: 0 };
  const t = vecDot(w, v) / vlen2;
  const proj = vecAdd(a, vecScale(v, t));
  const perp = vecSub(point, proj);
  const dist = vecLen(perp);
  const cross = v.x * w.y - v.y * w.x;
  const side = cross;
  return { dist, side };
}

// helper: convert indices -> [x,y] points. returns null if any landmark missing
function indicesToXY(indices: number[], landmarks: Landmark[] | null) {
  if (!landmarks || landmarks.length === 0) return null;
  const pts: { x: number; y: number }[] = [];
  for (const i of indices) {
    const lm = landmarks[i];
    if (!lm || typeof lm[0] !== "number") return null;
    pts.push({ x: lm[0], y: lm[1] });
  }
  return pts;
}

// helper: flatten xy to number[] for Konva Line
function flattenPts(pts: { x: number; y: number }[]) {
  const out: number[] = [];
  for (const p of pts) out.push(p.x, p.y);
  return out;
}

// helper: sample along polyline (not smooth curve) to compute fiber positions
function samplesAlong(pts: { x: number; y: number }[], n = 10) {
  if (pts.length < 2) return [];
  // compute cumulative lengths
  const segLens: number[] = [];
  let total = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    const dx = pts[i + 1].x - pts[i].x;
    const dy = pts[i + 1].y - pts[i].y;
    const l = Math.hypot(dx, dy);
    segLens.push(l);
    total += l;
  }
  if (total === 0) return [];
  const samples: { x: number; y: number; nx: number; ny: number }[] = [];
  for (let s = 0; s < n; s++) {
    const t = (s + 0.5) / n * total;
    // find segment
    let accum = 0;
    let segIdx = 0;
    while (segIdx < segLens.length && accum + segLens[segIdx] < t) {
      accum += segLens[segIdx];
      segIdx++;
    }
    if (segIdx >= segLens.length) segIdx = segLens.length - 1;
    const localT = (t - accum) / (segLens[segIdx] || 1);
    const a = pts[segIdx];
    const b = pts[segIdx + 1];
    const x = a.x + (b.x - a.x) * localT;
    const y = a.y + (b.y - a.y) * localT;
    // compute normal (perp) from segment vector
    const vx = b.x - a.x;
    const vy = b.y - a.y;
    const len = Math.hypot(vx, vy) || 1;
    const nx = -vy / len;
    const ny = vx / len;
    samples.push({ x, y, nx, ny });
  }
  return samples;
}

// @ts-ignore
export default function FaceCanvas(): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [imgEl, setImgEl] = useState<HTMLImageElement | null>(null);
  const stageRef = useRef<any>(null);

  const [imageFit, setImageFit] = useState<{ width: number; height: number } | null>(null);
  const [landmarks, setLandmarks] = useState<Landmark[]>([]);
  const [detector, setDetector] = useState<any | null>(null);
  const [points, setPoints] = useState<AnnotationPoint[]>([]);
  const [stageSize, setStageSize] = useState({ width: DEFAULT_STAGE_WIDTH, height: DEFAULT_STAGE_HEIGHT });

  // toggles
  const [showMidline, setShowMidline] = useState(true);
  const [showEyeLine, setShowEyeLine] = useState(true);
  const [showThirds, setShowThirds] = useState(true);
  const [showLandmarks, setShowLandmarks] = useState<boolean>(true);

  // anatomy
  const [showMuscles, setShowMuscles] = useState<boolean>(true);
  const [showVessels, setShowVessels] = useState<boolean>(false);

  // file -> data URL
  useEffect(() => {
    if (!file) {
      setImageSrc(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setImageSrc(String(reader.result));
    reader.onerror = (e) => {
      console.error("FileReader error", e);
      setImageSrc(null);
    };
    reader.readAsDataURL(file);
  }, [file]);

  // data URL -> image element + compute fit
  useEffect(() => {
    if (!imageSrc) {
      setImgEl(null);
      setImageFit(null);
      setStageSize({ width: DEFAULT_STAGE_WIDTH, height: DEFAULT_STAGE_HEIGHT });
      return;
    }
    let mounted = true;
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = imageSrc;
    image.onload = () => {
      if (!mounted) return;
      setImgEl(image);
      const naturalW = image.naturalWidth || DEFAULT_STAGE_WIDTH;
      const naturalH = image.naturalHeight || DEFAULT_STAGE_HEIGHT;
      const fit = fitImage(naturalW, naturalH, DEFAULT_STAGE_WIDTH, DEFAULT_STAGE_HEIGHT);
      setImageFit(fit);
      setStageSize({ width: fit.width, height: fit.height });
    };
    image.onerror = (err) => {
      console.error("Image load error", err);
      if (mounted) {
        setImgEl(null);
        setImageFit(null);
      }
    };
    return () => {
      mounted = false;
    };
  }, [imageSrc]);

  // create detector
  useEffect(() => {
    let mounted = true;
    let created: any = null;

    async function setup() {
      try {
        await tf.setBackend("webgl");
        await tf.ready();

        const modelConst = (faceLandmarksDetection as any).SupportedModels?.MediaPipeFaceMesh ??
                           (faceLandmarksDetection as any).MediaPipeFaceMesh;

        const detectorConfigTfjs = { runtime: "tfjs", maxFaces: 1, refineLandmarks: true };

        try {
          created = await (faceLandmarksDetection as any).createDetector(modelConst, detectorConfigTfjs);
          if (mounted) setDetector(created);
          else if (created?.dispose) created.dispose();
        } catch (errCreate) {
          console.warn("createDetector failed; trying legacy load():", errCreate);
          try {
            created = await (faceLandmarksDetection as any).load(
              (faceLandmarksDetection as any).SupportedPackages?.mediapipeFacemesh ??
                (faceLandmarksDetection as any).SupportedPackages,
              { maxFaces: 1 }
            );
            if (mounted) setDetector(created);
          } catch (errLegacy) {
            console.error("Both createDetector and legacy load() failed:", errLegacy);
            throw errLegacy;
          }
        }
      } catch (err) {
        console.error("Detector setup error:", err);
      }
    }

    setup();

    return () => {
      mounted = false;
      if (created && typeof created.dispose === "function") {
        try {
          created.dispose();
        } catch {}
      }
    };
  }, []);

  // detection (run after imageFit)
  useEffect(() => {
    if (!detector || !imgEl || !imageFit) return;
    let cancelled = false;

    async function detectOnce() {
      try {
        const off = document.createElement("canvas");
        off.width = imageFit.width;
        off.height = imageFit.height;
        const ctx = off.getContext("2d")!;
        ctx.clearRect(0, 0, off.width, off.height);
        ctx.drawImage(imgEl, 0, 0, imageFit.width, imageFit.height);

        let predictions: any = null;
        let lastErr: any = null;

        try {
          predictions = await (detector as any).estimateFaces(off);
        } catch (e1) {
          lastErr = e1;
          try {
            predictions = await (detector as any).estimateFaces({ input: off });
          } catch (e2) {
            lastErr = e2;
            try {
              predictions = await (detector as any).estimateFaces(off, { flipHorizontal: false });
            } catch (e3) {
              lastErr = e3;
            }
          }
        }

        if (!predictions) {
          console.error("All estimateFaces attempts failed:", lastErr);
          throw lastErr;
        }

        if (cancelled) return;

        if (!predictions || predictions.length === 0) {
          setLandmarks([]);
          return;
        }

        const face = predictions[0] as any;
        let mesh: Landmark[] = [];

        if (Array.isArray(face?.scaledMesh)) mesh = face.scaledMesh as Landmark[];
        else if (Array.isArray(face?.mesh)) mesh = face.mesh as Landmark[];
        else if (Array.isArray(face?.keypoints)) mesh = (face.keypoints as any[]).map(kp => [kp.x, kp.y, kp.z ?? 0]);
        else if (Array.isArray(face?.keypoints3D)) mesh = face.keypoints3D as Landmark[];
        else console.warn("Unknown prediction shape", face);

        setLandmarks(mesh || []);

        // populate some sensible default points (lips) if there are no points yet
        if ((!points || points.length === 0) && mesh && mesh.length > 0) {
          const lipsIndices = [13, 14, 78, 308];
          const defaultPoints = lipsIndices
            .map((i) => {
              const p = mesh && mesh[i];
              if (!p) return null;
              return { id: nanoid(), x: p[0], y: p[1], product: "", dose: "" };
            })
            .filter(Boolean) as AnnotationPoint[];
          setPoints(defaultPoints);
        }
      } catch (err) {
        console.error("detectOnce error:", err);
      }
    }

    detectOnce();

    return () => {
      cancelled = true;
    };
  }, [detector, imgEl, imageFit]); // re-run when imageFit is set

  // keep stage size in sync with imageFit whenever imageFit changes
  useEffect(() => {
    if (imageFit) {
      setStageSize({ width: imageFit.width, height: imageFit.height });
    }
  }, [imageFit]);

  function onAddPoint() {
    setPoints((p) => [...p, { id: nanoid(), x: (imageFit?.width ?? stageSize.width) / 2, y: (imageFit?.height ?? stageSize.height) / 2, product: "", dose: "" }]);
  }

  function onExport() {
    if (!stageRef.current) return;
    try {
      // @ts-ignore (kept intentionally)
      const uri = stageRef.current.toDataURL({ pixelRatio: 2 });
      const link = document.createElement("a");
      link.download = `face_map_${Date.now()}.png`;
      link.href = uri;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error("Export error:", err);
      alert("Export failed (canvas may be tainted). Make sure image is loaded from same origin or crossOrigin='anonymous'.");
    }
  }

  // reference calculations
  const midlineEndpoints =
    landmarks.length > 152 && landmarks[10] && landmarks[152]
      ? { a: { x: landmarks[10][0], y: landmarks[10][1] }, b: { x: landmarks[152][0], y: landmarks[152][1] } }
      : (() => {
          if (!landmarks || landmarks.length === 0) return null;
          let minY = Infinity,
            maxY = -Infinity;
          let minX = Infinity,
            maxX = -Infinity;
          for (const lm of landmarks) {
            if (!lm) continue;
            const x = lm[0],
              y = lm[1];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
          const cx = (minX + maxX) / 2;
          return { a: { x: cx, y: minY }, b: { x: cx, y: maxY } };
        })();

  const eyeLine =
    landmarks.length > 263 && landmarks[33] && landmarks[263]
      ? { l: { x: landmarks[33][0], y: landmarks[33][1] }, r: { x: landmarks[263][0], y: landmarks[263][1] } }
      : null;

  const thirdsYs = (() => {
    if (!landmarks || landmarks.length === 0) return null;
    let minY = Infinity,
      maxY = -Infinity;
    let minX = Infinity,
      maxX = -Infinity;
    for (const lm of landmarks) {
      if (!lm) continue;
      const x = lm[0],
        y = lm[1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const h = maxY - minY;
    if (!isFinite(h) || h <= 0) return null;
    const y1 = minY + h / 3;
    const y2 = minY + (2 * h) / 3;
    return { y1, y2, xmin: minX, xmax: maxX };
  })();

  // offsets
  const pointOffsets = points.map((pt) => {
    if (!midlineEndpoints) return { id: pt.id, dist: 0, side: 0 };
    const { a, b } = midlineEndpoints;
    const { dist, side } = perpDistanceToLine({ x: pt.x, y: pt.y }, a, b);
    return { id: pt.id, dist: Math.round(dist), side: side };
  });
  function sideLabel(side: number) {
    if (side === 0) return "C";
    return side > 0 ? "L" : "R";
  }

  // ---------- PRESETS ----------
  const presetTemplates = {
    lips: { indices: [13, 14, 78, 308], label: "Lips" },
    nasolabial: { indices: [50, 280, 146, 376], label: "Nasolabial" },
    marionette: { indices: [57, 287, 164, 393], label: "Marionette" },
  } as const;

  function addPreset(name: keyof typeof presetTemplates) {
    const tpl = presetTemplates[name];
    if (!landmarks || landmarks.length === 0) {
      const cx = stageSize.width / 2;
      const cy = stageSize.height / 2;
      const fallbackPoints: AnnotationPoint[] = [-1, 0, 1, 2].map((i) => ({
        id: nanoid(),
        x: cx + (i - 1.5) * 12,
        y: cy + (i - 1.5) * 10,
        product: "",
        dose: "",
      }));
      setPoints((p) => [...p, ...fallbackPoints]);
      return;
    }

    const newPts: AnnotationPoint[] = [];
    for (const idx of tpl.indices) {
      const lm = landmarks[idx];
      if (lm && typeof lm[0] === "number") {
        newPts.push({ id: nanoid(), x: lm[0], y: lm[1], product: "", dose: "" });
      } else {
        if (midlineEndpoints && thirdsYs) {
          const cx = (midlineEndpoints.a.x + midlineEndpoints.b.x) / 2;
          const dx = (newPts.length % 2 === 0) ? -12 : 12;
          const y = newPts.length < 2 ? thirdsYs.y1 : thirdsYs.y2;
          newPts.push({ id: nanoid(), x: cx + dx, y, product: "", dose: "" });
        } else {
          newPts.push({ id: nanoid(), x: stageSize.width / 2 + (newPts.length - 1) * 10, y: stageSize.height / 2, product: "", dose: "" });
        }
      }
    }
    setPoints((p) => [...p, ...newPts]);
  }

  // update point metadata helpers
  function updatePointMeta(id: string, data: Partial<Pick<AnnotationPoint, "product" | "dose" | "x" | "y">>) {
    setPoints((prev) => prev.map((pt) => (pt.id === id ? { ...pt, ...data } : pt)));
  }

  // delete
  function removePoint(id: string) {
    setPoints((ps) => ps.filter((p) => p.id !== id));
  }

  // snapping helper
  function snapPointToNearestLandmark(id: string, radius = 12) {
    const pt = points.find((p) => p.id === id);
    if (!pt || !landmarks || landmarks.length === 0) return;
    let best: { idx: number; dist: number } | null = null;
    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i];
      if (!lm) continue;
      const dx = lm[0] - pt.x;
      const dy = lm[1] - pt.y;
      const d = Math.hypot(dx, dy);
      if (d <= radius && (!best || d < best.dist)) best = { idx: i, dist: d };
    }
    if (best) {
      const lm = landmarks[best.idx];
      updatePointMeta(id, { x: lm[0], y: lm[1] });
    }
  }

  // ---------- MUSCLES (more accurate rendering) ----------
  // Each muscle defines an array of landmark indices that trace the muscle boundary or centerline.
  // We attempt to build a smooth outer boundary from those indices; then we draw:
  // - a smooth filled shape (tensioned Line closed)
  // - a faint outer glow (lighter fill)
  // - short fiber strokes sampled along the centerline to suggest fiber direction

  // muscle definitions (indices are approximate MediaPipe FaceMesh indices)
  const muscleDefs: { id: string; label: string; boundary: number[]; centerline?: number[]; color: string }[] = [
    {
      id: "orbicularis_oris",
      label: "Orbicularis Oris",
      boundary: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61], // outer ring
      centerline: [13, 78, 14], // approximate upper-mid-lower center points
      color: "#b4006a",
    },
    {
      id: "zygomaticus_major_l",
      label: "Zygomaticus Major (L)",
      boundary: [61, 226, 354, 334, 293], // cheek curve left
      centerline: [61, 226, 345],
      color: "#9b2a8a",
    },
    {
      id: "zygomaticus_major_r",
      label: "Zygomaticus Major (R)",
      boundary: [291, 446, 323, 361, 291], // cheek curve right
      centerline: [291, 446, 323],
      color: "#9b2a8a",
    },
    {
      id: "mentalis",
      label: "Mentalis",
      boundary: [152, 148, 176, 149, 150, 152],
      centerline: [152, 148, 176],
      color: "#7f2266",
    },
  ];

  // compute muscle render shapes (only when landmarks present)
  const musclesToRender = muscleDefs
    .map((m) => {
      const boundaryPts = indicesToXY(m.boundary, landmarks);
      if (!boundaryPts) return null;
      const centerPts = m.centerline ? indicesToXY(m.centerline, landmarks) : null;
      return { ...m, boundaryPts, centerPts };
    })
    .filter(Boolean) as Array<
    { id: string; label: string; boundaryPts: { x: number; y: number }[]; centerPts?: { x: number; y: number }[]; color: string }
  >;

  // ---------- VESSELS (unchanged approach) ----------
  const vesselDefs = [
    { id: "facial_artery_l", label: "Facial artery (L)", indices: [234, 93, 132, 58, 0, 1, 2] },
    { id: "facial_artery_r", label: "Facial artery (R)", indices: [454, 323, 361, 288, 267, 266, 265] },
    { id: "angular_l", label: "Angular (L)", indices: [2, 98, 4, 5] },
    { id: "angular_r", label: "Angular (R)", indices: [265, 362, 257, 256] },
    { id: "sup_labial", label: "Superior labial", indices: [61, 62, 63, 64, 65, 66] },
    { id: "inf_labial", label: "Inferior labial", indices: [291, 292, 293, 294, 295, 296] },
  ];

  const vesselShapes = vesselDefs
    .map((v) => {
      const pts = indicesToXY(v.indices, landmarks);
      if (!pts) return null;
      return { ...v, pts };
    })
    .filter(Boolean) as Array<{ id: string; label: string; pts: { x: number; y: number }[] }>;

  // ---------- RENDER ----------
  return (
    <div style={{ border: "1px solid #ddd", padding: 12, borderRadius: 8, maxWidth: 1300 }}>
      <div style={{ marginBottom: 12, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            if (!e.target.files?.[0]) return;
            setFile(e.target.files[0]);
            setPoints([]);
            setLandmarks([]);
            setImageFit(null);
          }}
        />
        <button onClick={onAddPoint}>Add point</button>
        <button onClick={onExport}>Export PNG</button>

        <div style={{ marginLeft: 8, display: "flex", gap: 6, alignItems: "center" }}>
          <strong>Presets:</strong>
          <button onClick={() => addPreset("lips")}>Lips</button>
          <button onClick={() => addPreset("nasolabial")}>Nasolabial</button>
          <button onClick={() => addPreset("marionette")}>Marionette</button>
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "center", marginLeft: 8 }}>
          <label>
            <input type="checkbox" checked={showLandmarks} onChange={(e) => setShowLandmarks(e.target.checked)} /> Landmarks
          </label>
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "center", marginLeft: 8 }}>
          <label>
            <input type="checkbox" checked={showMuscles} onChange={(e) => setShowMuscles(e.target.checked)} /> Muscles
          </label>
          <label>
            <input type="checkbox" checked={showVessels} onChange={(e) => setShowVessels(e.target.checked)} /> Vessels
          </label>
        </div>

        <div style={{ marginLeft: "auto", display: "flex", gap: 12 }}>
          <label>
            <input type="checkbox" checked={showMidline} onChange={(e) => setShowMidline(e.target.checked)} /> Midline
          </label>
          <label>
            <input type="checkbox" checked={showEyeLine} onChange={(e) => setShowEyeLine(e.target.checked)} /> Eye-line
          </label>
          <label>
            <input type="checkbox" checked={showThirds} onChange={(e) => setShowThirds(e.target.checked)} /> Thirds
          </label>
        </div>
      </div>

      <div style={{ display: "flex", gap: 12 }}>
        <div style={{ border: "1px solid #eee", padding: 6 }}>
          <Stage width={stageSize.width} height={stageSize.height} ref={stageRef}>
            <Layer>
              {imgEl ? (
                // @ts-ignore
                <KonvaImageElement image={imgEl as any} x={0} y={0} width={stageSize.width} height={stageSize.height} />
              ) : null}
            </Layer>

            <Layer>
              {/* landmarks */}
              {showLandmarks && landmarks.length > 0
                ? landmarks.map((lm, idx) => (
                    <Circle key={`lm-${idx}`} x={lm[0]} y={lm[1]} radius={1.2} fill="rgba(0,150,255,0.9)" />
                  ))
                : null}

              {/* MUSCLES: smooth tensioned shapes + fiber strokes */}
              {showMuscles &&
                musclesToRender.map((m) => {
                  const flat = flattenPts(m.boundaryPts);
                  // centerline sample for fibers
                  const centerPts = m.centerPts ?? null;
                  const fiberBasePts = centerPts ? centerPts : m.boundaryPts;
                  const fibreSamples = samplesAlong(fiberBasePts, 12);

                  return (
                    <React.Fragment key={m.id}>
                      {/* outer lighter glow (larger, more transparent) */}
                      <Line
                        points={flat}
                        closed
                        tension={0.6}
                        stroke={m.color}
                        strokeWidth={0.8}
                        fill={hexToRgba(m.color, 0.12)}
                      />
                      {/* main muscle body (slightly darker) */}
                      <Line
                        points={flat}
                        closed
                        tension={0.6}
                        stroke={darkenHex(m.color, 0.08)}
                        strokeWidth={1}
                        fill={hexToRgba(darkenHex(m.color, 0.06), 0.22)}
                      />

                      {/* fiber strokes */}
                      {fibreSamples.map((fs, i) => {
                        const len = 8 + (i % 3);
                        const x1 = fs.x - fs.nx * (len / 2);
                        const y1 = fs.y - fs.ny * (len / 2);
                        const x2 = fs.x + fs.nx * (len / 2);
                        const y2 = fs.y + fs.ny * (len / 2);
                        return (
                          <Line
                            key={`fiber-${m.id}-${i}`}
                            points={[x1, y1, x2, y2]}
                            stroke={hexToRgba(darkenHex(m.color, 0.25), 0.9)}
                            strokeWidth={1}
                            lineCap="round"
                            dash={undefined}
                          />
                        );
                      })}
                    </React.Fragment>
                  );
                })}

              {/* VESSELS */}
              {showVessels &&
                vesselShapes.map((v) => {
                  const flat = flattenPts(v.pts);
                  return (
                    <React.Fragment key={v.id}>
                      <Line points={flat} stroke="rgba(220,20,60,0.95)" strokeWidth={2} dash={[6, 3]} lineCap="round" />
                      <Text text={v.label} fontSize={11} fill="#c71b2b" x={v.pts[0].x + 6} y={v.pts[0].y - 12} />
                    </React.Fragment>
                  );
                })}

              {/* midline / eyeline / thirds */}
              {showMidline && midlineEndpoints ? (
                <>
                  <Line
                    points={[midlineEndpoints.a.x, midlineEndpoints.a.y, midlineEndpoints.b.x, midlineEndpoints.b.y]}
                    stroke="rgba(255,0,0,0.85)"
                    strokeWidth={1.5}
                    dash={[6, 4]}
                  />
                  <Text
                    text="Midline"
                    x={(midlineEndpoints.a.x + midlineEndpoints.b.x) / 2 + 6}
                    y={(midlineEndpoints.a.y + midlineEndpoints.b.y) / 2 + 6}
                    fontSize={12}
                    fill="rgba(255,0,0,0.9)"
                  />
                </>
              ) : null}

              {showEyeLine && eyeLine ? (
                <>
                  <Line
                    points={[eyeLine.l.x, eyeLine.l.y, eyeLine.r.x, eyeLine.r.y]}
                    stroke="rgba(0,200,0,0.9)"
                    strokeWidth={1.5}
                  />
                  <Text
                    text="Eye-line"
                    x={(eyeLine.l.x + eyeLine.r.x) / 2 + 6}
                    y={(eyeLine.l.y + eyeLine.r.y) / 2 - 16}
                    fontSize={12}
                    fill="rgba(0,160,0,0.9)"
                  />
                </>
              ) : null}

              {showThirds && thirdsYs ? (
                <>
                  <Line points={[thirdsYs.xmin, thirdsYs.y1, thirdsYs.xmax, thirdsYs.y1]} stroke="rgba(0,0,0,0.5)" strokeWidth={1} dash={[4, 4]} />
                  <Text text="1/3" x={thirdsYs.xmax + 6} y={thirdsYs.y1 - 6} fontSize={11} fill="#333" />
                  <Line points={[thirdsYs.xmin, thirdsYs.y2, thirdsYs.xmax, thirdsYs.y2]} stroke="rgba(0,0,0,0.5)" strokeWidth={1} dash={[4, 4]} />
                  <Text text="2/3" x={thirdsYs.xmax + 6} y={thirdsYs.y2 - 6} fontSize={11} fill="#333" />
                </>
              ) : null}

              {/* annotation points */}
              {points.map((p) => {
                const offset = pointOffsets.find((o) => o.id === p.id);
                const label = p.product ? `${p.product} ${p.dose ?? ""}`.trim() : `${offset ? sideLabel(offset.side) + " " + offset.dist + "px" : ""}`;
                return (
                  <Group
                    key={p.id}
                    x={p.x}
                    y={p.y}
                    draggable
                    onDragEnd={(e) => {
                      const nx = (e.target as any).x();
                      const ny = (e.target as any).y();
                      updatePointMeta(p.id, { x: nx, y: ny });
                    }}
                  >
                    <Circle radius={8} fill="rgba(255,165,0,0.95)" stroke="black" strokeWidth={1} />
                    <Text text="●" fontSize={12} offsetX={6} offsetY={6} />
                    <Text text={label} x={12} y={-10} fontSize={11} fill="#222" />
                  </Group>
                );
              })}
            </Layer>
          </Stage>
        </div>

        <div style={{ width: 420 }}>
          <h3>Session</h3>
          <div>
            <strong>Detected landmarks:</strong> {landmarks.length}
          </div>

          <div style={{ marginTop: 12 }}>
            <h4>Overlays</h4>
            <div>Muscles: {showMuscles ? "ON" : "OFF"}</div>
            <div>Vessels: {showVessels ? "ON" : "OFF"}</div>

            <div style={{ marginTop: 8 }}>
              <h5 style={{ marginBottom: 6 }}>Legend</h5>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <div style={{ width: 18, height: 12, background: "rgba(180,0,120,0.18)", border: "1px solid rgba(140,10,80,0.9)" }} />
                <div>Muscle region (approx)</div>
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 6 }}>
                <div style={{ width: 40, height: 4, background: "transparent", borderTop: "2px dashed rgba(220,20,60,0.95)" }} />
                <div>Vessel path (approx)</div>
              </div>
            </div>
          </div>

          <div style={{ marginTop: 12 }}>
            <h4>Injection points</h4>
            {points.length === 0 ? <div>No points yet</div> : null}
            {points.map((p) => {
              const off = pointOffsets.find((o) => o.id === p.id);
              return (
                <div key={p.id} style={{ marginBottom: 10, padding: 8, borderRadius: 6, background: "#0f0f0f", color: "#fff" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <small style={{ opacity: 0.9 }}>ID: {p.id}</small>
                    <div style={{ display: "flex", gap: 6 }}>
                      <button onClick={() => removePoint(p.id)}>remove</button>
                      <button onClick={() => snapPointToNearestLandmark(p.id)}>snap</button>
                    </div>
                  </div>

                  <div style={{ marginTop: 6 }}>
                    <div>
                      x: {Math.round(p.x)}, y: {Math.round(p.y)}
                      <button onClick={() => updatePointMeta(p.id, { x: Math.round(p.x), y: Math.round(p.y) })} style={{ marginLeft: 6 }}>
                        refresh
                      </button>
                    </div>

                    <div style={{ marginTop: 6 }}>
                      <label style={{ display: "block", fontSize: 12, marginBottom: 4 }}>Product</label>
                      <input
                        value={p.product ?? ""}
                        placeholder="Product name"
                        onChange={(e) => updatePointMeta(p.id, { product: e.target.value })}
                        style={{ width: "100%", padding: "6px 8px", borderRadius: 4 }}
                      />
                    </div>

                    <div style={{ marginTop: 6 }}>
                      <label style={{ display: "block", fontSize: 12, marginBottom: 4 }}>Dose</label>
                      <input
                        value={p.dose ?? ""}
                        placeholder="e.g. 0.1 mL (0.5 unit)"
                        onChange={(e) => updatePointMeta(p.id, { dose: e.target.value })}
                        style={{ width: "100%", padding: "6px 8px", borderRadius: 4 }}
                      />
                    </div>

                    <div style={{ marginTop: 6, color: "#ccc", fontSize: 13 }}>
                      Offset to midline: {off ? `${sideLabel(off.side)} ${off.dist}px` : "—"}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ marginTop: 8 }}>
            <small style={{ color: "#999" }}>
              Muscle overlays are improved: smooth shapes + fiber strokes to suggest direction. These are still approximations — tell me where
              a muscle looks off for your images and I’ll tweak landmark sets for that region.
            </small>
          </div>
        </div>
      </div>
    </div>
  );
}

// Small color helpers below

function hexToRgba(hex: string, alpha = 1) {
  // support shorthand & full hex (#abc or #aabbcc)
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const r = parseInt(full.substring(0, 2), 16);
  const g = parseInt(full.substring(2, 4), 16);
  const b = parseInt(full.substring(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function darkenHex(hex: string, amount = 0.1) {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const r = Math.max(0, Math.min(255, Math.floor(parseInt(full.substring(0, 2), 16) * (1 - amount))));
  const g = Math.max(0, Math.min(255, Math.floor(parseInt(full.substring(2, 4), 16) * (1 - amount))));
  const b = Math.max(0, Math.min(255, Math.floor(parseInt(full.substring(4, 6), 16) * (1 - amount))));
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}
function toHex(n: number) {
  const s = n.toString(16);
  return s.length === 1 ? "0" + s : s;
}

