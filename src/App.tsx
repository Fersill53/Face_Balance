// src/App.tsx
import React from "react";
import FaceCanvas from "./components/FaceCanvas";

function App() {
  return (
    <div style={{ fontFamily: "Inter, system-ui, sans-serif", padding: 20 }}>
      <h1>Face Balance — Face Mapping Prototype</h1>
      <p>
        Upload a front-facing photo. The app will detect facial landmarks and let you add draggable injection points.
      </p>
      <FaceCanvas />
      <footer style={{ marginTop: 20, fontSize: 13, color: "#666" }}>
        Prototype — images are processed locally in your browser.
      </footer>
    </div>
  );
}

export default App;
