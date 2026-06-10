### 1. Physical Graph Construction and Pruning
NOMAD represents the proteomics landscape as a heterogeneous multi-partite graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, which encodes the physical and experimental relationships between proteins, peptides, precursors, and samples.

*   **Node Types:**
    *   **Protein ($P$):** Canonical sequences or isoforms extracted from FASTA databases.
    *   **Peptide ($E$):** Fragments resulting from proteolytic digestion.
    *   **Precursor ($Q$):** Specific $m/z$ and charge state observations from MS1.
    *   **Sample ($S$):** Experimental conditions (e.g., drug dose, replicate).
*   **Edge Relations:**
    *   **PRODUCES ($P \to E$):** Defined via **In-Silico Digestion**. NOMAD simulates cleavage (default Trypsin/P) allowing for missed cleavages and peptide length constraints (default 6–50 AA).
    *   **HAS_PRECURSOR ($E \to Q$):** Maps peptides to their observed precursor signatures, creating the physical topology mask $\mathbf{M}$.
    *   **DETECTED_IN ($Q \to S$):** Encodes the observed intensity $V_{ij}$ of precursor $j$ in sample $i$.

#### Graph Pruning
To focus on high-quality evidence, NOMAD applies a **pre-fitting pruning step**. Precursors with fewer than 3 experimental observations are removed. Subsequently, any peptides and proteins that lose all their supporting precursors are also pruned.

#### Sample Support Masking
To prevent unobserved conditions from biasing downstream analysis, NOMAD initializes a sample-level support mask $\mathbf{S}_{\text{mask}} \in \{0, 1\}^{N_S \times N_P}$ based on initial precursor detection counts. A sample is considered biologically supported for a target protein group if and only if it exhibits non-zero intensity for at least one structurally mapped precursor. This mask is recorded in the output results database to indicate sample-level confidence for each protein group.

---

### 2. Component Search and Problem Decomposition
To ensure computational efficiency, NOMAD decomposes the global graph into independent **Connected Components** before fitting.

1.  **Structural Isolation:** The graph is pruned of Sample nodes, and connected components are identified to find sets of proteins that share precursors.
2.  **Matrix Formulation:** For each component, NOMAD extracts the intensity matrix $\mathbf{V}$ and the binary topology mask $\mathbf{M}$. This ensures that the deconvolution of one protein group does not computationally interfere with another.
3.  **Adaptive Numerical Scaling:** To prevent hyper-amplification of polynomial regularization terms ($W^4$ scale factors) over high-abundance features, each isolated component matrix $\mathbf{V}$ is dynamically divided by its local maximum observation value. This normalizes the numerical range to $[0, 1]$ during iterative non-convex optimization passes, keeping scale-invariant fidelity residuals and structural regularization weights perfectly balanced. Optimized parameter tensors are scaled back to natural physical intensity units prior to storage.

---

### 3. Adaptive Outlier Pruning (Pass 1)
During the optimization of each component, NOMAD performs a **self-calibrating pruning pass** to remove catastrophic outliers.

1.  **Initial Reconstruction:** An initial fit is performed to establish a baseline.
2.  **Absolute Residual Analysis:** Residuals are calculated as absolute errors in square-root space: $r = |\sqrt{WH} - \sqrt{V}|$.
3.  **5-Sigma MAD Rule:** Observations exceeding $5.0 \times \sigma_{\text{MAD}}$ (where $\sigma_{\text{MAD}} = 1.4826 \times \text{MAD}$) and an absolute error > 0.5 are masked. This removes "ion spikes" and interference artifacts that would otherwise destabilize the deconvolution.

---

### 4. The Forward Model and Parameterization
NOMAD models the observed intensities $\mathbf{V}$ as a product of protein abundances $\mathbf{W}$ and a structured basis matrix $\mathbf{H}$.

*   **Quadratic-Mirror Parameterization (Q-domain):** To enforce non-negativity without numerical stalling, all intensity parameters are optimized in the **Q-domain**, where $P = Q^2$.
*   **Basis Matrix ($\mathbf{H}$):** Defined by the product of a shared signature vector $\mathbf{h}$ (precursor probabilities) and the fixed binary topology mask $\mathbf{M}$:
    $$\mathbf{H} = \mathbf{h} \odot \mathbf{M}, \quad \text{subject to } \max(\mathbf{h}) = 1.0$$
*   **Global Signature Sharing:** The model learns relative emission probabilities shared across all proteins in a component, pooling evidence to maximize signal-to-noise.

---

### 5. The Joint Objective Function
NOMAD minimizes a composite energy functional using a GPU-accelerated Adam optimizer, followed by an L-BFGS polish for high-precision convergence.

#### I. Freeman-Tukey Square Root Loss
Instead of standard squared error, NOMAD employs a **Freeman-Tukey transformation** for the data fidelity term:
$$r_{\text{nmf}, ij} = \frac{\sqrt{\sum_a W_{ia} H_{aj}} - \sqrt{V_{ij}}}{\sqrt{V_{ij} + 1.0}}$$
The square root transformation stabilizes Poisson-like shot noise, while the $\sqrt{V+1}$ denominator prioritizes relative accuracy across the instrument's dynamic range.

#### II. Monotonicity Constraint ($\mathcal{R}_{\text{DR}}$)
For dose-response experiments, NOMAD enforces biological plausibility using a **Sign-Flip Penalty**. It penalizes instances where the protein intensity trend changes direction multiple times across dose levels, encouraging monotonic or sigmoidal profiles.

#### III. Replicate Consistency (MBCR)
This term minimizes within-group variance between technical replicates of the same condition:
$$\mathcal{R}_{\text{rep}} = \beta \sum_{g} \sum_{i \in G_g} (W_{ia} - \bar{W}_{ga})^2$$
By pooling signal across replicates, the model reduces the impact of sample-specific noise.

#### IV. L2 Regularization ($\mathcal{R}_{\text{L2}}$)
To prevent overfitting and handle unobserved parameters, a global L2 penalty is applied to the protein abundances:
$$\mathcal{R}_{\text{L2}} = \lambda_{\text{reg}} \sum W_{ia}^2$$

---

### 6. Indifferentiability and Merging
When two protein isoforms are structurally collinear (share the same precursor set) and biological priors cannot separate them, they become mathematically indifferentiable.

1.  **Correlation Detection:** During uncertainty calculation, the model computes the correlation between protein intensity estimates based on the standard error covariance matrix.
2.  **Agglomerative Merging:** If the correlation between two proteins is extremely positive ($r > 0.9$, indicating the model is unable to distinguish their signals), they are merged into a single **"Meta-Isoform"** by unioning their topology masks. This ensures that every reported intensity is supported by unique evidence.

---

### 7. Parameter Uncertainty and Inferential Statistics
Uncertainty is estimated via the **Gauss-Newton (Jacobian) Approximation** of the Hessian matrix.

#### I. Covariance and Standard Error
The covariance matrix $\mathbf{\Sigma}$ is calculated as:
$$\mathbf{\Sigma} = \sigma^2 \left( \mathbf{J}^\top \mathbf{J} + \epsilon \mathbf{I} \right)^{-1}$$
Standard errors are projected from the Q-domain back to the physical P-domain using the chain rule: $\text{SE}(P) = \text{SE}(Q) \cdot 2|Q|$.

#### II. Cross-Validation and Replicate Correlation Metrics
NOMAD evaluates NMF deconvolution engine performance through two complementary validation strategies:
1.  **Leave-One-Out (LOO) Cross-Validation**: Masks one random observation per component and evaluates the reconstruction accuracy against the true left-out values.
2.  **Technical Replicate Correlation**: Measures the statistical consistency between identical replicate samples.

To ensure parity and statistical fidelity across all numeric intensity ranges, both validations compute the following correlation metrics:
*   **Pearson $r$ (Raw)**: Measures linear correlation in natural physical intensity space.
*   **Pearson $r$ (Log10)**: Measures linear correlation in $\log_{10}$-transformed space, reflecting accuracy across the instrument's dynamic range.
*   **Pearson $r$ (Sqrt)**: Measures linear correlation in $\sqrt{\vphantom{x}\ \ \ }$-transformed space, stabilizing Poisson shot-noise.
*   **Spearman $\rho$**: Non-parametric rank-order correlation, ensuring robust resistance to outlier spikes.