# Intelligent PCB Defect Inspection System — CV Methodology & Algorithms

**Subject:** ACSC502 — Computer Vision (5th Semester)  
**Academic Module:** Classical Computer Vision & Image Processing Formulations  
**Implementation Phase:** Phase 1 Only  

---

## 1. Classical Computer Vision Formulations

This system is built strictly upon mathematical foundations of classical image processing, without deep neural networks.

### 1.1 Spatial Domain Filtering
To eliminate high-frequency electronic sensor noise while preserving sharp copper trace edges, a 2D Gaussian kernel is applied:

$$G(x, y) = \frac{1}{2\pi \sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

For discrete implementation, a $5 \times 5$ kernel with $\sigma=1.0$ is convolved over the input image:
$$I_{\text{smooth}}(x, y) = I(x, y) * G(x, y)$$

For impulse noise (salt-and-pepper artifacts resulting from linear CCD scan dead pixels), non-linear median filtering is used:
$$I_{\text{med}}(x, y) = \text{median}\left(\{I(x+i, y+j) \mid (i, j) \in W\}\right)$$

---

### 1.2 Contrast Limited Adaptive Histogram Equalization (CLAHE)
Under variable industrial lighting, global histogram equalization over-amplifies background noise in homogeneous substrate regions. CLAHE restricts contrast amplification by clipping the local histogram at threshold $\beta = 2.0$ to $2.5$:

$$\tilde{h}(k) = \begin{cases} h(k) & \text{if } h(k) \le \beta \\ \beta & \text{if } h(k) > \beta \end{cases}$$

The clipped excess $\Delta = \sum \max(0, h(k) - \beta)$ is redistributed uniformly across all gray levels. Bilinear interpolation between contextual tiles ($8 \times 8$) eliminates artificial boundary artifacts.

---

### 1.3 Otsu Binarization (Intra-Class Variance Minimization)
To segment conductive copper tracks from non-conductive epoxy substrate, Otsu's method exhaustively searches for threshold $t^*$ that maximizes between-class variance $\sigma_B^2(t)$:

$$\sigma_B^2(t) = \omega_0(t) \omega_1(t) \left[\mu_0(t) - \mu_1(t)\right]^2$$

Where $\omega_0(t)$ and $\omega_1(t)$ are the cumulative class probabilities, and $\mu_0(t)$, $\mu_1(t)$ are the respective class means. The optimal threshold is:
$$t^* = \arg\max_{0 \le t < L} \sigma_B^2(t)$$

---

### 1.4 Feature-Based Registration (ORB & RANSAC)
To align test boards with reference templates, Oriented FAST and Rotated BRIEF (ORB) detects multi-scale corner keypoints. For each keypoint $(x, y)$, the local patch orientation is determined by intensity moments:

$$m_{pq} = \sum_{x, y} x^p y^q I(x, y)$$
$$\theta = \text{atan2}(m_{01}, m_{10})$$

Descriptor matching uses Hamming distance:
$$d_H(\mathbf{p}, \mathbf{q}) = \sum_{i=1}^{256} p_i \oplus q_i$$

Matches are filtered via Lowe's ratio test: $d_H(\mathbf{p}, \mathbf{q}_1) < 0.75 \cdot d_H(\mathbf{p}, \mathbf{q}_2)$.
The spatial mapping is estimated using RANSAC:
$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

---

### 1.5 Dual-Polarity Subtraction & Morphological Filtering
Pixel-level difference is decomposed directionally:
$$D_{\text{deficit}}(x, y) = \max\left(0, I_{\text{template}}(x, y) - I_{\text{aligned}}(x, y)\right)$$
$$D_{\text{excess}}(x, y) = \max\left(0, I_{\text{aligned}}(x, y) - I_{\text{template}}(x, y)\right)$$

To suppress sub-pixel raster jitter from minor registration residuals, morphological opening with structuring element $B_{3\times3}$ is executed:
$$D_{\text{clean}} = (D \ominus B) \oplus B$$

---

### 1.6 Geometric Feature Extraction & Classification Rules

For each connected component blob extracted via 8-connectivity:
1. **Area ($A$):** Number of pixels in the connected component.
2. **Perimeter ($P$):** Arc length of contour $\partial C$.
3. **Circularity ($C$):** $C = \frac{4\pi A}{P^2}$. (Ideal circle $C \to 1.0$).
4. **Aspect Ratio ($AR$):** $AR = \frac{\max(W, H)}{\min(W, H)}$.
5. **Solidity ($S$):** $S = \frac{A}{A_{\text{convex hull}}}$.

#### Deterministic Classification Rules (DeepPCB Standard):
- **Pin-Hole (Class 6):** $D_{\text{deficit}} \land C \ge 0.52 \land A \le 380\text{ px} \land \text{Template Copper Ratio} \ge 0.40$.
- **Open Circuit (Class 1):** $D_{\text{deficit}} \land (AR \ge 1.7 \lor A \ge 180\text{ px})$.
- **Mouse Bite (Class 3):** Default deficit along copper boundary.
- **Spurious Copper (Class 5):** $D_{\text{excess}} \land \text{Isolated Blob on Substrate} \land \text{Copper Ratio} < 0.20$.
- **Short Circuit (Class 2):** $D_{\text{excess}} \land (AR \ge 1.7 \lor A \ge 280\text{ px})$.
- **Spur (Class 4):** Default excess extending outward from copper boundary.

---

### 1.7 PCBA SMT Component Verification Formulations

For component template patch $T$ and search region $I$, Normalized Cross-Correlation (NCC) is computed:
$$\gamma(x, y) = \frac{\sum_{x', y'} \left[T(x', y') - \bar{T}\right] \left[I(x+x', y+y') - \bar{I}_{x, y}\right]}{\sqrt{\sum_{x', y'} \left[T(x', y') - \bar{T}\right]^2 \sum_{x', y'} \left[I(x+x', y+y') - \bar{I}_{x, y}\right]^2}}$$

- **Displacement Vector:** $\Delta \mathbf{r} = (\Delta x, \Delta y) = \arg\max_{(x, y)} \gamma(x, y) - \mathbf{r}_{\text{nominal}}$.
- **Shift Distance:** $D = \sqrt{\Delta x^2 + \Delta y^2}$. If $D > 14\text{ px} \to \text{SHIFTED}$.
- **Missing Check:** If $\max \gamma < 0.42 \lor (\max \gamma < 0.58 \land \text{EdgeRatio} < 0.40) \to \text{MISSING}$.
- **Rotation Check:** Using central second-order moments:
  $$\mu_{pq} = \sum_{x, y} (x - \bar{x})^p (y - \bar{y})^q I(x, y)$$
  $$\theta = \frac{1}{2} \text{atan2}(2\mu_{11}, \mu_{20} - \mu_{02})$$
  If $|\theta_{\text{test}} - \theta_{\text{tmpl}}| > 12^\circ \to \text{ROTATED}$.
