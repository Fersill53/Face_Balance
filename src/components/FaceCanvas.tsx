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
              {/* cast to any for Konva image prop to avoid strict signature mismatch */}
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

