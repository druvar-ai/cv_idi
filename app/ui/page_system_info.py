"""Page 5: System Information, Architecture & Diagnostic Documentation."""

import streamlit as st


def render_system_info_page() -> None:
    """Render technical specifications, architecture diagrams, datasets, and limitations."""
    st.markdown('<div class="main-header">System Architecture & Technical Specifications</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Academic & Industrial Machine Vision Reference — ACSC502 Computer Vision (5th Semester)</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "System Architecture",
        "Classical CV Techniques",
        "Datasets & Licensing",
        "Phase 1 Limitations & Phase 2 Roadmap"
    ])

    with tab1:
        st.subheader("End-to-End System Architecture")
        st.markdown("""
        The **Intelligent PCB Defect Inspection System** is architected as an industrial Automated Optical Inspection (AOI) product.
        It enforces strict modular separation between bare board fabrication inspection and assembled board inspection:
        """)
        st.code("""
┌────────────────────────────────────────────────────────────────────────┐
│                   INDUSTRIAL AOI PIPELINE ARCHITECTURE                 │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                     [ 1. Image Acquisition & Loader ]
                     (Safe buffer ingestion, validation, contrast check)
                                    │
                                    ▼
                     [ 2. Image Preprocessing Subsystem ]
                     (Gaussian filter, Median blur, CLAHE, Otsu binarization)
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
 [ MODULE A: BARE PCB ]                              [ MODULE B: PCBA ]
 - ORB/RANSAC Alignment                              - Macro Board Homography
 - Dual-Polarity Subtraction                         - Component ROI Extraction
   * Deficit: Open, Mousebite, Pin-hole              - Normalized Cross Correlation
   * Excess: Short, Spur, Spurious Copper            - Principal Axis Moments (Rotation)
 - Connected Components Analysis                     - Missing Component Detection
 - Morphological Shape Metrics                       - Shift / Displacement Vector
 - DeepPCB Deterministic Rules                       - SMT Placement Tolerances
         │                                                     │
         └──────────────────────────┬──────────────────────────┘
                                    │
                                    ▼
                     [ 3. Unified Decision Engine ]
                     (Arbitrates PASS / FAIL / REVIEW with severity)
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
 [ SQLite History ]         [ Multi-Format Reports ]   [ Streamlit UI ]
 (Audit log & KPIs)         (JSON, CSV, Official PDF)  (Interactive 5 Pages)
        """, language="text")

    with tab2:
        st.subheader("Classical Computer Vision Techniques Applied")
        st.markdown("""
        In accordance with **Phase 1 academic boundaries**, no deep learning or neural network inference is used.
        The system relies on rigorous classical computer vision algorithms:

        1. **Image Representation & Photometry:**
           - BGR, RGB, Grayscale, and LAB color spaces.
           - Normalization, contrast stretching, and standard deviation contrast checks.
        2. **Spatial Domain Filtering:**
           - Gaussian smoothing ($5\\times 5$, $\\sigma=1.0$) for Gaussian sensor noise reduction.
           - Median filtering for impulse salt-and-pepper noise.
        3. **Contrast Enhancement:**
           - CLAHE (Contrast Limited Adaptive Histogram Equalization) in luminance channel.
        4. **Segmentation & Binarization:**
           - Otsu's intra-class variance minimization thresholding.
           - Adaptive Gaussian window thresholding.
        5. **Morphological Image Processing:**
           - Morphological Opening (erosion then dilation) for sub-pixel raster jitter removal.
           - Morphological Closing (dilation then erosion) for structural continuity.
           - Top-Hat and Black-Hat transforms for component region isolation.
        6. **Image Registration & Geometric Alignment:**
           - ORB (Oriented FAST and Rotated BRIEF) feature detection.
           - Hamming distance KNN matching with Lowe's ratio test ($0.75$).
           - RANSAC Affine and Homography transformation estimation with residual error tracking.
        7. **Dual-Polarity Image Subtraction:**
           - Copper Deficit: $I_{\\text{deficit}} = \\max(0, I_{\\text{template}} - I_{\\text{test}})$
           - Copper Excess: $I_{\\text{excess}} = \\max(0, I_{\\text{test}} - I_{\\text{template}})$
        8. **Connected Component Analysis (CCA) & Contour Moments:**
           - Area, Perimeter, Centroid, Bounding Box, Aspect Ratio ($W/H$), Circularity ($4\\pi A / P^2$), Solidity ($A / A_{\\text{hull}}$).
        9. **Normalized Cross-Correlation (NCC):**
           - $\\gamma(x, y) = \\frac{\\sum_{x', y'} (T(x', y') - \\bar{T}) (I(x+x', y+y') - \\bar{I})}{\\sqrt{\\sum (T - \\bar{T})^2 \\sum (I - \\bar{I})^2}}$ for component placement verification.
        10. **Orientation Deviation Analysis:**
            - Central moments $\\mu_{20}, \\mu_{02}, \\mu_{11}$ to compute principal orientation angle $\\theta = \\frac{1}{2} \\text{atan2}(2\\mu_{11}, \\mu_{20} - \\mu_{02})$.
        """)

    with tab3:
        st.subheader("Datasets, Provenance & Licensing")
        st.markdown("""
        ### 1. Bare PCB Benchmark: DeepPCB
        - **Source:** [tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB)
        - **Citation:** Tang, S., He, F., Huang, X., & Yang, J. (2019). *Online PCB Defect Detector on a New PCB Defect Dataset*. arXiv:1902.06197.
        - **License:** MIT License (Permissive commercial and academic research use).
        - **Resolution:** $640 \\times 640$ pixels, linear scan CCD sensor ($48\\text{ px/mm}$).
        - **Categories:** Open circuit, Short circuit, Mouse bite, Spur, Pin hole, Spurious copper.

        ### 2. Assembled PCB Benchmark: Enics / NanoNets Industrial PCBA
        - **Source:** [LalitPradhan/NanoNetsPCBChecker](https://github.com/LalitPradhan/NanoNetsPCBChecker)
        - **License:** Permissive Open Source / Academic Research Benchmark.
        - **Resolution:** $3840 \\times 2748$ high-resolution RGB optical macros.
        - **Categories Supported:** Missing components, Shifted components, Rotated components.

        ### 3. PCB-SAID Dataset Investigation Documentation
        - **Candidate:** PCB-SAID (*Printed Circuit Board - Surface-Mounted Device Assembly Inspection Dataset*, Mineo et al., ICCV Workshops 2023).
        - **Status & Finding:** Thorough investigation confirmed that PCB-SAID is access-restricted and available only *upon private email request to authors*, lacking an open public repository archive.
        - **Resolution:** In strict compliance with Master Prompt Section 5, this restriction is formally documented, and the legitimate open Enics/NanoNets PCBA benchmark is used.
        """)

    with tab4:
        st.subheader("Phase 1 Technical Limitations & Phase 2 Deep Learning Roadmap")
        st.markdown("""
        ### Observed Limitations of Classical Computer Vision (Phase 1)
        1. **Sensitivity to Illumination and Shadows:**
           - Optical glare, non-uniform specular reflections on metallic solder joints, and fluctuating ambient lighting cause spurious false-positive difference blobs.
        2. **Sub-Pixel Registration Artifacts:**
           - Non-rigid board warpage or board thermal flexing cannot be fully modeled by global affine or homography transforms, leaving 1–2 pixel edge misalignments that require aggressive morphological filtering.
        3. **Rigid Rule Thresholding:**
           - Fixed geometric thresholds (such as circularity $\\ge 0.52$ or aspect ratio $\\ge 1.7$) can struggle with boundary-case defects that share hybrid geometric traits (e.g., a wide mousebite resembling a mild open trace).
        4. **Component Surface Texture Variation:**
           - Passive SMD resistors from different component vendors possess varying laser-etched markings, causing Normalized Cross-Correlation to occasionally drop even when properly seated.

        ### Phase 2 Intelligent Machine Vision Proposal
        In Phase 2, the classical baseline will be complemented with deep learning:
        - **Module A Extension:** Fine-tuned YOLOv11 / RT-DETR for real-time bounding box detection directly on multi-scale bare PCB images.
        - **Module B Extension:** Siamese Feature Networks and Contrastive Learning to match assembled components invariant to vendor markings and ambient lighting.
        - **Anomaly Detection:** Reconstruction-based unsupervised autoencoders (e.g. PatchCore / FastFlow) trained on defect-free boards to detect novel or subtle solder bridging without requiring massive defect training sets.
        """)
