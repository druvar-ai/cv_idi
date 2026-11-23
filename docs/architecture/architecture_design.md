# Intelligent PCB Defect Inspection System — System Architecture

**Project:** Intelligent PCB Defect Inspection System Using Classical Computer Vision  
**Subject:** ACSC502 — Computer Vision (5th Semester)  
**Implementation Phase:** Phase 1 (Strictly Classical Computer Vision)  
**Document Classification:** Industrial Machine Vision Technical Specification  

---

## 1. Architectural Overview & Design Philosophy

The **Intelligent PCB Defect Inspection System** is engineered as a production-quality Automated Optical Inspection (AOI) product. In manufacturing facilities, optical inspection is deployed at two distinct physical checkpoints:
1. **Fabrication AOI (Bare PCB):** Inspecting raw copper patterns etched onto epoxy fiberglass substrate before solder mask, surface finishes, or electronic parts are placed.
2. **Assembly AOI (PCBA / SMT):** Inspecting component placement after surface-mount technology (SMT) pick-and-place and reflow soldering.

Because raw copper trace defects and component placement anomalies have completely different spatial characteristics, reference standards, and failure modes, this architecture enforces a **two-domain design pattern**:

```
                                  [ Input Images / Streams ]
                                              │
                                              ▼
                             [ Image Acquisition & Validator ]
                             (Size, Format, Integrity, Contrast)
                                              │
                                              ▼
                             [ Common Preprocessing Pipeline ]
                             (Gaussian / Median, CLAHE, Otsu)
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
       [ Module A: Bare PCB Inspection ]               [ Module B: PCBA Inspection ]
       • ORB-RANSAC Affine Registration                • Homography Board Alignment
       • Dual-Polarity Image Subtraction               • Component ROI Extraction (CAD / XML)
         - Copper Deficit (Open, Mousebite, Pinhole)   • Normalized Cross-Correlation (NCC)
         - Copper Excess (Short, Spur, Spurious Cu)    • Central Moment Orientation Analysis
       • Morphological 3×3 Filtering                   • Displacement Vector (dx, dy)
       • Connected Component Analysis (CCA)            • Missing Component Detection
       • DeepPCB Classical Rule Classifier             • Shift & Rotation Verification
                    │                                                   │
                    └─────────────────────────┬─────────────────────────┘
                                              ▼
                             [ Unified Decision Arbiter Engine ]
                             • Status: PASS / FAIL / REVIEW
                             • Severity Assignment: CRITICAL / MAJOR / MINOR
                             • Quality Verification (Registration inliers > 0.40)
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
          [ Multi-Format Reports ]   [ SQLite Audit Database ]  [ Streamlit AOI UI ]
          • Standardized JSON        • outputs/history.db       • Dashboard
          • Tabular Summary CSV      • Queryable Records        • Bare PCB Inspector
          • Official Audit PDF       • Latency & Yield Metrics  • PCBA Inspector
```

---

## 2. Subsystem Deconstruction

### 2.1 Image Acquisition & Input Validation Subsystem (`app.acquisition`)
- **`ImageLoader`**: Implements safe multi-source ingestion supporting file paths, raw byte streams (from web uploaders), and memory buffers. Handles Windows path encoding cleanly via `cv2.imdecode`.
- **`ImageValidator`**: Pre-flight verification ensuring:
  - Whitelisted extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`).
  - File size bounds ($0 < \text{size} \le 50\text{ MB}$).
  - Spatial resolution ($32 \times 32 \le H, W \le 8192 \times 8192$).
  - Photometric contrast check: Standard deviation of pixel intensities $\sigma \ge 8.0$ to catch blank, solid black, or completely washed-out sensor inputs.

### 2.2 Image Preprocessing Subsystem (`app.preprocessing`)
- **`filters.py`**: Spatial noise attenuation using calibrated $5 \times 5$ Gaussian kernels ($\sigma=1.0$) and non-linear median filtering for salt-and-pepper impulse noise.
- **`enhancement.py`**: Contrast Limited Adaptive Histogram Equalization (CLAHE) applied in LAB color space to equalize luminance while preserving color fidelity.
- **`thresholding.py`**: Otsu's intra-class variance minimization thresholding and adaptive local window thresholding.
- **`morphology.py`**: Structuring element operations (Opening, Closing, Top-Hat, Black-Hat) to isolate sub-pixel raster jitter.

### 2.3 Bare PCB Inspection Pipeline (`app.inspection.bare_pcb`)
- **Alignment:** ORB feature extraction (1,000 keypoints) paired with Hamming-distance BFMatcher and Lowe's ratio test ($0.75$), fitted using RANSAC affine transformations.
- **Difference Analysis:** Dual-polarity directional subtraction:
  $$\text{Deficit} = \max(0, I_{\text{template}} - I_{\text{test}})$$
  $$\text{Excess} = \max(0, I_{\text{test}} - I_{\text{template}})$$
- **Morphological Clean-up:** $3 \times 3$ rectangular opening removes 1-pixel raster jitter resulting from micro-misalignments.
- **Candidate Extraction:** 8-connectivity connected component labeling (`cv2.connectedComponentsWithStats`) extracting Area, Bounding Box, Perimeter, Centroid, and Circularity.
- **Classification Engine:** Deterministic rule engine mapping blobs to the 6 official DeepPCB classes (Open, Short, Mousebite, Spur, Copper, Pin-hole).

### 2.4 Assembled PCB Inspection Pipeline (`app.inspection.assembled_pcb`)
- **Alignment:** Multi-scale ORB with RANSAC homography estimation ($2,500$ keypoints).
- **Component Region Parsing:** Reads nominal placement coordinates from Pascal VOC XML or automatically discovers component clusters via morphological blackhat transforms.
- **Normalized Cross-Correlation (NCC):** Computes normalized correlation surface $\gamma(x, y)$ over a local search neighborhood ($\pm 24\text{ px}$).
- **Anomaly Detection Rules:**
  - **Missing Component:** Correlation drops below threshold ($\gamma < 0.42$) and edge density collapses ($E_{\text{test}}/E_{\text{tmpl}} < 0.40$).
  - **Shifted Component:** Peak correlation offset $\sqrt{\Delta x^2 + \Delta y^2} > 14\text{ px}$.
  - **Rotated Component:** Principal moment orientation deviation $|\Delta \theta| > 12^\circ$.

### 2.5 Unified Decision Engine (`app.inspection.decision_engine`)
Arbitrates final inspection outcome into three industrial categories:
- **`PASS`**: Board conforms to golden reference tolerances; zero defects detected.
- **`FAIL`**: One or more confirmed manufacturing defects detected.
- **`REVIEW`**: Registration inlier ratio $< 0.40$, alignment residual $> 25\text{ px}$, or image quality degraded. Unverified boards are never falsely released as PASS.

---

## 3. Storage & Multi-Format Reporting Subsystem
- **SQLite Database:** Stores complete audit trail in `outputs/history/inspection_history.db` with indexed timestamps, defect counts, and serialized telemetry.
- **Standardized JSON Report:** Machine-readable payload for MES (Manufacturing Execution Systems) and ERP automation.
- **Tabular CSV Export:** High-density spreadsheet logs for statistical process control (SPC) and yield tracking.
- **Official PDF Report:** Publication-grade document generated via ReportLab with pass/fail banner, inspection metadata, defect summary breakdown, and itemized defect register.
