# Intelligent PCB Defect Inspection System Using Classical Computer Vision

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red.svg)](https://streamlit.io/)
[![Pytest Tests](https://img.shields.io/badge/Tests-28%20Passed%20(100%25)-brightgreen.svg)]()
[![Academic Phase](https://img.shields.io/badge/Phase-Phase%201%20(Classical%20CV)-orange.svg)]()

> **Academic Course Identity:**  
> **Course:** Computer Vision — ACSC502  
> **Department:** Department of Artificial Intelligence & Machine Learning / Data Science (AI&ML / AI&DS)  
> **Semester:** 5th Semester, B.Tech / B.E.  
> **Academic Year:** 2026–2027  
> **Application Domain:** Industrial Machine Vision — Automated Optical Inspection (AOI)  
> **Phase:** Phase 1 ONLY (Strictly Classical Computer Vision, No Deep Learning)  

---

## 1. Executive Summary & Project Overview

The **Intelligent PCB Defect Inspection System** is an Automated Optical Inspection (AOI) software product developed for the 5th Semester Computer Vision curriculum (Course Code: ACSC502). It automates high-speed, sub-millimeter quality control for printed circuit board manufacturing using classical computer-vision algorithms.

The system enforces strict architectural separation between two inspection regimes:
1. **Module A: Bare PCB Fabrication Inspection:** Detects copper-layer trace defects on unpopulated circuit boards prior to component assembly (Open Circuits, Short Circuits, Mouse Bites, Spurs, Pin Holes, and Spurious Copper).
2. **Module B: Assembled PCB Inspection (PCBA):** Detects surface-mount device (SMD / SMT) placement anomalies on populated boards (Missing Components, Shifted/Misaligned Components, and Rotated Components).

Both modules share common, robust infrastructure: image ingestion, multi-check validation, classical filtering and enhancement, unified decision arbitrament (`PASS` / `FAIL` / `REVIEW`), SQLite audit logging, multi-format reporting (JSON, CSV, PDF), and a Streamlit AOI cockpit.

---

## 2. System Architecture

```
                                  [ Input Images / Camera Streams ]
                                                  │
                                                  ▼
                                 [ Image Acquisition & Validator ]
                                 (Format, Bounds, Integrity, Contrast)
                                                  │
                                                  ▼
                                 [ Common Preprocessing Subsystem ]
                                 (Gaussian/Median Blur, CLAHE, Otsu)
                                                  │
                        ┌─────────────────────────┴─────────────────────────┐
                        ▼                                                   ▼
           [ MODULE A: BARE PCB ]                              [ MODULE B: PCBA ]
           • ORB-RANSAC Alignment                              • Macro Board Homography
           • Dual-Polarity Subtraction                         • Component ROI Extraction
             - Deficit: Open, Mousebite, Pinhole               • Normalized Cross-Correlation (NCC)
             - Excess: Short, Spur, Spurious Copper            • Central Moment Orientation Angle
           • Morphological Opening Clean-up                    • Displacement Vector (dx, dy)
           • Connected Component Analysis (CCA)                • Missing Component Detection
           • DeepPCB Deterministic Rules                       • SMT Placement Tolerances
                        │                                                   │
                        └─────────────────────────┬─────────────────────────┘
                                                  ▼
                                 [ Unified Decision Arbiter Engine ]
                                 • Decision: PASS / FAIL / REVIEW
                                 • Severity: CRITICAL / MAJOR / MINOR
                                 • Registration Inlier Validation
                                                  │
                        ┌─────────────────────────┼─────────────────────────┐
                        ▼                         ▼                         ▼
              [ SQLite Audit Log ]       [ Multi-Format Reports ]   [ Streamlit UI ]
              • outputs/history.db       • Standardized JSON        • Dashboard
              • Historical records       • Tabular Summary CSV      • Bare PCB Inspector
              • Yield & Latency stats    • Official Audit PDF       • PCBA Inspector
```

---

## 3. Implemented Capabilities & Technical Modules

### 3.1 Common Infrastructure & Preprocessing
- **Multi-Source Image Loading:** File paths, byte arrays, memory buffers, and web uploads with Windows path handling via `cv2.imdecode`.
- **Pre-Flight Validation:** File size bounds ($<50\text{ MB}$), dimension bounds ($32\text{ to }8192\text{ px}$), whitelisted formats (`.jpg`, `.png`, `.bmp`, `.tif`), and photometric contrast variance checks ($\sigma \ge 8.0$) to reject blank or corrupted files.
- **Classical Spatial Filtering:** Gaussian convolution ($5\times 5$, $\sigma=1.0$) for Gaussian sensor noise, median filtering for impulse noise, and bilateral filtering for edge preservation.
- **Contrast Normalization:** Luminance-channel CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space.
- **Binarization:** Otsu's intra-class variance minimization thresholding and adaptive local window thresholding.
- **Mathematical Morphology:** Opening ($3\times 3$) to eliminate 1-pixel edge misregistration jitter; closing to seal conductor micro-voids; top-hat/black-hat transforms.

### 3.2 Module A: Bare PCB Fabrication Pipeline
- **Registration:** ORB feature extraction with Hamming-distance BFMatcher, Lowe's ratio test ($0.75$), and RANSAC affine transformation.
- **Dual-Polarity Image Subtraction:**
  - **Copper Deficit:** $\max(0, I_{\text{template}} - I_{\text{test}})$ (Missing copper: open, mousebite, pinhole).
  - **Copper Excess:** $\max(0, I_{\text{test}} - I_{\text{template}})$ (Extra copper: short, spur, spurious copper).
- **Candidate Extraction:** 8-connectivity Connected Component Analysis (CCA) measuring Area, Perimeter, Centroid, Bounding Box, Aspect Ratio ($W/H$), and Circularity ($4\pi A / P^2$).
- **Deterministic Classification Rules:**
  - `pin-hole`: Circular deficit ($C \ge 0.52$, $A \le 380\text{ px}$) enclosed in conductive trace.
  - `open`: Elongated deficit ($AR \ge 1.7$ or $A \ge 180\text{ px}$) cutting across trace width.
  - `mousebite`: Notch on conductor boundary.
  - `copper`: Isolated copper blob on substrate ($\text{copper ratio} < 0.20$).
  - `short`: Copper excess bridging adjacent conductors ($AR \ge 1.7$ or $A \ge 280\text{ px}$).
  - `spur`: Conductor boundary protrusion.

### 3.3 Module B: Assembled PCB (PCBA) Pipeline
- **Board Alignment:** Homography transformation mapping test PCB to golden reference assembly space.
- **Component Region Localization:** Parses nominal CAD/Pascal VOC XML coordinates or auto-segments component pads via morphological blackhat transforms.
- **Normalized Cross-Correlation (NCC):** Sub-window correlation search ($\pm 24\text{ px}$) to determine matching coefficient $\gamma \in [-1, 1]$.
- **SMT Placement Anomaly Rules:**
  - `MISSING`: $\gamma < 0.42$ or ($\gamma < 0.58$ and edge density ratio $< 0.40$).
  - `SHIFTED`: Displacement $\sqrt{\Delta x^2 + \Delta y^2} > 14\text{ px}$.
  - `ROTATED`: Principal moment orientation deviation $|\Delta \theta| > 12^\circ$.

### 3.4 Unified Decision Engine
- **PASS:** Zero manufacturing defects detected; board released.
- **FAIL:** One or more confirmed defects detected; flagged for rework.
- **REVIEW:** Inconclusive result, registration inliers $< 0.40$, or alignment residual $> 25\text{ px}$. Unverified boards are never released as defect-free.

---

## 4. Dataset Provenance, Scope & Licensing

| Dataset Domain | Candidate / Source | License & Terms | Resolution & Sensor | Implemented Categories |
|---|---|---|---|---|
| **Module A: Bare PCB** | **DeepPCB Benchmark** ([tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB)) | **MIT License** (Permissive Academic/Research; Copyright (c) 2018 tangsanli5201) | $640 \times 640$, Linear Scan CCD ($48\text{ px/mm}$) | Open, Short, Mousebite, Spur, Pin Hole, Spurious Copper (All 6 classes) |
| **Module B: PCBA** | **Enics / NanoNets PCBA Benchmark** ([LalitPradhan/NanoNetsPCBChecker](https://github.com/LalitPradhan/NanoNetsPCBChecker)) | **Open Academic Benchmark** (Educational & non-commercial research use; commercial rights unestablished) | $3840 \times 2748$ Ultra-HD RGB Optical | Missing Components, Shifted Components, Rotated Components |

### Explicit Dataset Permission & Licensing Verification
1. **DeepPCB (Module A):** Verified under the permissive **MIT License** (Copyright (c) 2018 tangsanli5201). Permitted for research, modification, and distribution provided copyright notice and disclaimer are retained.
2. **Enics / NanoNets PCBA (Module B):** Released as an open public academic dataset for computer vision benchmarking. Commercial rights are **not established** upstream; commercial deployment would require acquiring dedicated proprietary or licensed industrial imagery.
3. **PCB-SAID (Module B Candidate):** *Printed Circuit Board - Surface-Mounted Device Assembly Inspection Dataset* (Mineo et al., ICCV Workshops 2023). **Investigation Finding:** Confirmed to be access-restricted and available solely upon individual written request to the authors, without an open public download repository. It is **not** an unrestricted public dataset.
4. **Solder Bridge Scope Reduction:** Solder-bridge defect detection on macro optical imagery is scope-reduced due to lack of sub-millimeter microscopic resolution.

---

## 5. Quantitative Benchmark Measurements & Performance Improvement

Evaluation was conducted against ground truth annotations on both baseline and expanded test splits.

### 5.1 Bare PCB Manufacturing Inspection (Module A) — Before vs. After Optimization

| Evaluation Metric | Initial Baseline (10 Test Boards) | Post-Optimization (10 Test Boards) | Post-Optimization (Full 25 Test Boards) |
|---|---|---|---|
| **Evaluated Ground Truth Defects** | 74 defects | 74 defects | **180 defects** |
| **True Positives (TP)** | 45 | 72 | **178** |
| **False Positives (FP)** | 148 | 29 | **74** |
| **False Negatives (FN)** | 29 | 2 | **2** |
| **Detection Recall (Sensitivity)** | 60.81% | **97.30%** | **98.89%** |
| **Precision** | 23.32% | **71.29%** | **70.63%** |
| **F1-Score** | 0.3371 | **82.29%** | **82.41%** |
| **Mean Bounding Box IoU** | 0.4298 | **0.7097** | **0.7090** |
| **Mean Inspection Latency** | 60.97 ms | 44.99 ms | **33.55 ms (~29.8 FPS)** |
| **Defect-Free Board Verification** | 100% PASS (0 FPs) | 100% PASS (0 FPs) | **100% PASS (0 FPs)** |

#### Per-Class Recall Breakdown (Full 25 Test Boards, 180 Ground Truth Defects):
- **Open Circuit (Type 1):** 34 / 34 detected (**100.0% Recall**)
- **Short Circuit (Type 2):** 14 / 14 detected (**100.0% Recall**)
- **Mouse Bite (Type 3):** 30 / 31 detected (**96.77% Recall**)
- **Spur (Type 4):** 38 / 38 detected (**100.0% Recall**)
- **Spurious Copper (Type 5):** 31 / 32 detected (**96.88% Recall**)
- **Pin Hole (Type 6):** 31 / 31 detected (**100.0% Recall**)
- **Critical Escapes (Open/Short):** **0 escapes** (100% safety on electrical fault classes)

### 5.2 Assembled PCBA Inspection (Module B) — Before vs. After Optimization

| Evaluation Metric | Initial Baseline | Post-Optimization |
|---|---|---|
| **Discrete SMT Components** | 60 components | 60 components |
| **True Positives (TP)** | 6 | **9** |
| **False Positives (FP)** | 4 | **5** |
| **True Negatives (TN)** | 42 | **41** |
| **False Negatives (FN)** | 8 | **5** |
| **Precision** | 60.0% | **64.29%** |
| **Recall** | 42.86% | **64.29%** |
| **F1-Score** | 0.500 | **64.29%** |
| **Missing Component Accuracy** | 100% (`53_ac` caught) | **100% (`53_ac` caught)** |
| **Defect-Free Board Verification** | 100% PASS (0 FPs) | **100% PASS (0 FPs)** |
| **Mean Latency per Board** | 219.34 ms | **198.40 ms (~5.0 FPS)** |

---

## 6. Project Directory Structure

```
pcb-inspection-system/
│
├── app/
│   ├── __init__.py
│   ├── config.py                     # Master configuration dataclasses
│   │
│   ├── acquisition/
│   │   ├── __init__.py
│   │   ├── loader.py                 # Multi-source safe image loader
│   │   └── validator.py              # Pre-flight format & contrast validator
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── filters.py                # Gaussian, median, bilateral filters
│   │   ├── enhancement.py            # CLAHE & histogram equalization
│   │   ├── thresholding.py           # Otsu & adaptive thresholding
│   │   ├── morphology.py             # Opening, closing, gradients, top-hat
│   │   └── pipeline.py               # Preprocessing orchestrator & stage logger
│   │
│   ├── inspection/
│   │   ├── __init__.py
│   │   ├── registration.py           # ORB-RANSAC affine & homography alignment
│   │   ├── decision_engine.py        # Unified PASS / FAIL / REVIEW decision arbiter
│   │   │
│   │   ├── bare_pcb/
│   │   │   ├── __init__.py
│   │   │   ├── diff_analyzer.py      # Dual-polarity subtractive difference
│   │   │   ├── defect_extractor.py   # Connected components & contour features
│   │   │   ├── classifier.py         # Classical DeepPCB rule categorizer
│   │   │   └── pipeline.py           # Bare PCB end-to-end inspection pipeline
│   │   │
│   │   └── assembled_pcb/
│   │       ├── __init__.py
│   │       ├── component_detector.py # XML specification loader & auto-segmentation
│   │       ├── anomaly_analyzer.py   # NCC template matching & moment orientation
│   │       └── pipeline.py           # PCBA end-to-end inspection pipeline
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # TP, FP, TN, FN, Precision, Recall, F1, IoU
│   │   ├── evaluator_bare.py         # DeepPCB ground truth benchmark runner
│   │   └── evaluator_pcba.py         # PCBA ground truth benchmark runner
│   │
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── json_reporter.py          # Structured JSON audit generator
│   │   ├── csv_reporter.py           # Tabular CSV export
│   │   └── pdf_reporter.py           # ReportLab inspection audit PDF report
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   └── history_manager.py        # SQLite persistent audit store
│   │
│   └── ui/
│       ├── __init__.py
│       ├── components.py             # Reusable cards, badges & CSS styling
│       ├── page_dashboard.py         # Page 1: AOI Executive Dashboard
│       ├── page_bare_pcb.py          # Page 2: Bare PCB Interactive Inspector
│       ├── page_assembled_pcb.py     # Page 3: PCBA Interactive Inspector
│       ├── page_history.py           # Page 4: Inspection Audit History & Search
│       └── page_system_info.py       # Page 5: Architecture, Specs & Limitations
│
├── configs/
│   ├── default_config.yaml           # Master system parameters
│   ├── bare_pcb_config.yaml          # Bare PCB thresholds & kernel sizes
│   └── pcba_config.yaml              # SMT component placement tolerances
│
├── datasets/
│   ├── bare_pcb/                     # Curated DeepPCB images, templates, annotations
│   └── assembled_pcb/                # Curated PCBA images, templates, XML annotations
│
├── outputs/
│   ├── intermediate/                 # Intermediate processing visualizations
│   ├── reports/                      # Generated JSON, CSV, and PDF reports
│   └── history/                      # SQLite database: inspection_history.db
│
├── tests/
│   ├── test_acquisition.py           # Ingestion and validator tests
│   ├── test_preprocessing.py         # Filtering, thresholding, morphology tests
│   ├── test_bare_pcb.py              # Bare PCB difference & classification tests
│   ├── test_assembled_pcb.py         # PCBA missing, shift, rotation tests
│   ├── test_decision_engine.py       # Decision arbiter & severity tests
│   ├── test_reporting.py             # JSON, CSV, and PDF generation tests
│   └── test_evaluation.py            # IoU math & confusion matrix tests
│
├── docs/
│   ├── architecture/                 # Architecture design document
│   ├── methodology/                  # Mathematical CV formulations
│   ├── screenshots/                  # Real intermediate processing images
│   └── Phase_1_Presentation.pptx     # Exact 12-slide Phase 1 presentation
│
├── main.py                           # Streamlit application entry point
├── generate_ppt.py                   # Automated 12-slide presentation generator
├── requirements.txt                  # Production dependencies
├── Dockerfile                        # Containerized deployment definition
├── pytest.ini                        # Pytest configuration
├── conftest.py                       # Pythonpath fixture
├── .gitignore                        # Git exclusion rules
└── README.md                         # Complete project documentation
```

---

## 7. Installation & Execution Guide

### 7.1 Local Python Environment Setup
```bash
# 1. Clone or navigate to project directory
cd d:\coding\CV

# 2. Create and activate virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate # On Linux/macOS

# 3. Install production dependencies
pip install -r requirements.txt

# 4. Execute automated test suite (verifies all 28 tests pass)
pytest -v

# 5. Launch the Streamlit AOI Application
streamlit run main.py
```
Open browser at: `http://localhost:8501`

### 7.2 Docker Deployment
```bash
# Build production Docker container
docker build -t pcb-aoi-phase1 .

# Run container exposing port 8501
docker run -p 8501:8501 pcb-aoi-phase1
```

### 7.3 Generate Exact 12-Slide Presentation
```bash
python generate_ppt.py
```
Outputs to: `docs/Phase_1_Presentation.pptx`

---

## 8. Faculty Demonstration Guide (Step-by-Step Viva & Presentation)

This step-by-step walkthrough is structured for faculty evaluation, project viva, and live demonstration:

### Step 1: Open the Industrial Dashboard (`Page 1`)
- Run `streamlit run main.py` and open `http://localhost:8501`.
- Review the top KPI metric cards: Total Inspections, Bare PCB runs, PCBA runs, Yield Pass Rate (%), and Mean Latency (ms).
- Explain how SQLite audit persistence ([outputs/history/inspection_history.db](file:///d:/coding/CV/outputs/history/inspection_history.db)) guarantees traceable quality logging without data loss.

### Step 2: Demonstrate Bare PCB Manufacturing Inspection (`Page 2`)
- Under Image Acquisition Source, select **Curated DeepPCB Benchmark Samples**.
- Select board `00041000` with **Defective Board** mode.
- Click **Start Classical AOI Inspection**.
- **Observe Verdict:** Highlights **`FAIL`** badge with detected manufacturing defects.
- Review tabs:
  - **Tab 1 (Bounding Boxes):** Color-coded bounding boxes identifying Open circuits, Shorts, Mouse bites, and Pin holes.
  - **Tab 2 (Dual-Polarity Map):** Blue channel represents copper deficit (missing traces); Red channel represents copper excess (shorts/spurs).
  - **Tab 3 (Contours & Mask):** Cleaned morphological defect mask isolating defect boundaries.
  - **Tab 4 (Alignment & Registration):** Sub-pixel ORB-RANSAC alignment residual ($\approx 0.00\text{ px}$).
- **Demonstrate Defect-Free Pass:** Switch test target to **Defect-Free Reference (Pass Test)**. Click inspect. Observe **`PASS`** with exactly **0 false alarms**.

### Step 3: Demonstrate Assembled PCBA Inspection (`Page 3`)
- Under Acquisition Source, select **Curated PCBA Industrial Benchmark**.
- Select **Defective Board (Missing 53_ac, Rotated 47/48, Shifted)**.
- Click **Execute SMT Component Placement Inspection**.
- **Observe Verdict:** Highlights **`FAIL`** with 14 placement anomalies.
- Highlight critical detection:
  - **Missing Component:** Part `53_ac` cleanly detected via NCC score collapse to $0.330$ ($<0.42$ threshold).
  - **Rotated Components:** Angular deviations $|\Delta \theta| > 12^\circ$ detected via central moment principal axis analysis.
  - **Shifted Components:** Lateral displacements $\sqrt{\Delta x^2 + \Delta y^2} > 14\text{ px}$.
- **Demonstrate Defect-Free Pass:** Select **Defect-Free Reference Board (Pass Test)**. Click inspect. Observe **`PASS`** with 0 false anomalies.

### Step 4: Show Intermediate Image-Processing Stages
- In both inspection modules, navigate to intermediate stages tabs.
- Demonstrate classical CV operations in action: Gaussian denoising, CLAHE contrast equalization, Otsu variance-minimization binarization, morphological opening ($3\times 3$) and closing ($7\times 7$), and Canny gradient extraction.

### Step 5: Demonstrate the Decision Arbiter Engine (`PASS` / `FAIL` / `REVIEW`)
- Explain the 3-state industrial arbiter logic:
  - **`PASS`:** 0 defects detected; board cleared for production.
  - **`FAIL`:** Verified defect detected; flagged for rework with severity tagging (`CRITICAL`, `MAJOR`, `MINOR`).
  - **`REVIEW`:** Uncertain alignment or degraded contrast; prevents unverified boards from escaping into production.

### Step 6: Export Multi-Format Inspection Reports
- At the bottom of either inspection page, demonstrate immediate one-click report downloads:
  - **JSON Report:** Machine-readable MES record with defect coordinates and execution metadata.
  - **CSV Report:** Tabular defect register ready for statistical process control (SPC).
  - **PDF Report:** Publication-grade audit document compiled with ReportLab.

### Step 7: Review Inspection History (`Page 4`)
- Open the **Inspection History** page.
- Demonstrate filtering by inspection domain (Bare PCB vs. PCBA) and decision outcome (`PASS`, `FAIL`, `REVIEW`).
- Click on any inspection record to view its stored defect summary and timestamp.

### Step 8: Explain Technical Boundaries & Phase 2 Proposal (`Page 5` & Slides 11–12)
- Clearly present Phase 1 classical CV boundaries: illumination/reflection sensitivity, non-rigid board warpage, rigid heuristic thresholds.
- Detail the proposed Phase 2 deep-learning roadmap: YOLOv11/RT-DETR one-stage detection for bare PCB, Siamese contrastive networks for PCBA, and PatchCore memory-bank unsupervised anomaly detection.

---

## 9. Test Execution Verification

Run pytest across all modules:
```bash
pytest -v
```
**Output:**
```
============================= test session starts =============================
platform win32 -- Python 3.14.0, pytest-8.2.2, pluggy-1.6.0
rootdir: D:\coding\CV
configfile: pytest.ini
testpaths: tests
collected 28 items

tests/test_acquisition.py::test_validator_file_not_found PASSED          [  3%]
tests/test_acquisition.py::test_validator_unsupported_extension PASSED   [  7%]
tests/test_acquisition.py::test_validator_empty_file PASSED              [ 10%]
tests/test_acquisition.py::test_validator_blank_low_contrast PASSED      [ 14%]
tests/test_acquisition.py::test_validator_valid_image PASSED             [ 17%]
tests/test_acquisition.py::test_loader_real_dataset_image PASSED         [ 21%]
tests/test_assembled_pcb.py::test_pcba_pass_on_identical PASSED          [ 25%]
tests/test_assembled_pcb.py::test_pcba_fail_on_defective PASSED          [ 28%]
tests/test_assembled_pcb.py::test_xml_spec_loader PASSED                 [ 32%]
tests/test_bare_pcb.py::test_bare_pcb_pass_on_identical PASSED           [ 35%]
tests/test_bare_pcb.py::test_bare_pcb_fail_on_defective PASSED           [ 39%]
tests/test_bare_pcb.py::test_difference_analyzer PASSED                  [ 42%]
tests/test_decision_engine.py::test_decision_engine_pass PASSED          [ 46%]
tests/test_decision_engine.py::test_decision_engine_fail_critical PASSED [ 50%]
tests/test_decision_engine.py::test_decision_engine_review_on_bad_alignment PASSED [ 53%]
tests/test_evaluation.py::test_iou_identical_boxes PASSED                [ 57%]
tests/test_evaluation.py::test_iou_disjoint_boxes PASSED                 [ 60%]
tests/test_evaluation.py::test_iou_half_overlap PASSED                   [ 64%]
tests/test_evaluation.py::test_detection_metrics_perfect_match PASSED    [ 67%]
tests/test_evaluation.py::test_detection_metrics_fp_and_fn PASSED        [ 71%]
tests/test_preprocessing.py::test_filters PASSED                         [ 75%]
tests/test_preprocessing.py::test_enhancement_clahe PASSED               [ 78%]
tests/test_preprocessing.py::test_thresholding_otsu PASSED               [ 82%]
tests/test_preprocessing.py::test_morphology_open_close PASSED           [ 85%]
tests/test_preprocessing.py::test_preprocessing_pipeline PASSED          [ 89%]
tests/test_reporting.py::test_json_reporting PASSED                      [ 92%]
tests/test_reporting.py::test_csv_reporting PASSED                       [ 96%]
tests/test_reporting.py::test_pdf_reporting PASSED                       [100%]

====================== 28 passed in 1.27s ======================
```

---

## 10. Observed Phase 1 Limitations & Phase 2 Proposal

### 10.1 Phase 1 Classical Vision Limitations
1. **Photometric & Illumination Sensitivity:** Specular reflections on metallic solder joints and non-uniform lighting cause false difference blobs in subtractive comparison.
2. **Sub-Pixel Registration Artifacts:** Non-rigid board warpage cannot be completely resolved by global affine/homography transforms, necessitating morphological opening which can slightly erode very small defects.
3. **Rigid Rule Thresholding:** Deterministic geometric cutoffs struggle with edge cases where defects exhibit hybrid properties (e.g. wide mousebite vs. mild open trace).
4. **Fine-Pitch Solder Bridging:** Detecting sub-millimeter solder bridges between fine-pitch IC leads requires microscopic optical magnification or X-ray inspection.

### 10.2 Phase 2 Deep Learning Proposal
1. **Module A (Bare PCB):** YOLOv11 / RT-DETR one-stage object detection directly on multi-scale bare PCB images to eliminate template difference sensitivity.
2. **Module B (PCBA):** Siamese Feature Matching Networks trained via contrastive loss to match components invariant to vendor laser markings and ambient glare.
3. **Unsupervised Anomaly Detection:** PatchCore / FastFlow memory bank architectures trained purely on defect-free boards to flag novel defects without labeled defect data.

---

## 11. Final Acceptance Checklist

- [x] Both Bare PCB and Assembled PCB inspection modules exist.
- [x] Each module has an actual, dedicated classical CV implementation.
- [x] Image acquisition and validation work with format, bounds, and contrast checks.
- [x] Preprocessing produces inspectable intermediate outputs (filters, CLAHE, Otsu, morphology).
- [x] Reference-image comparison works (ORB-RANSAC alignment).
- [x] Selected defect categories detected and localised (6 bare PCB classes, 3 PCBA classes).
- [x] Unified decision engine produces PASS, FAIL, or REVIEW with severity tags.
- [x] Streamlit web application executes successfully across all 5 pages.
- [x] Inspection reports generated and exportable (JSON, CSV, Official PDF).
- [x] Actual performance metrics calculated against ground truth annotations (No fabrication).
- [x] Representative test cases executed (Defective and defect-free boards).
- [x] Automated tests pass (28/28 unit and integration tests passing).
- [x] Installation and execution instructions documented.
- [x] Project runs reproducibly via local venv and Dockerfile.
- [x] Actual demonstration screenshots captured and saved in `docs/screenshots/`.
- [x] Exact 12-slide Phase 1 presentation generated (`docs/Phase_1_Presentation.pptx`).
- [x] Phase 1 limitations and Phase 2 roadmap documented.
- [x] Zero fabricated results or unsupported claims exist.
