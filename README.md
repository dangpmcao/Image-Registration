# Image-Registration

A study project on **deformable image registration**. We compare a classical optimisation method (B-spline Free-Form Deformation) with a learned method (CNN-based DLIR) by:

1. **Understanding** both methods well enough to implement them from scratch.
2. **Replicating** the key results of the two foundational papers in [paper/](paper/).
3. **Extending** the comparison to several datasets beyond the ones used in the papers.
4. **Demoing** both methods in an interactive app: pick a random fixed/moving pair, choose a method, tune its parameters, and inspect the result.

---

## 1. Papers

### 1.1 B-Spline FFD: Rueckert et al., 1999

> D. Rueckert, L. I. Sonoda, C. Hayes, D. L. G. Hill, M. O. Leach, D. J. Hawkes. *Nonrigid Registration Using Free-Form Deformations: Application to Breast MR Images.* IEEE TMI 18(8):712–721, 1999.

**Problem.** Correct patient motion between pre- and post-contrast 3D breast MRI. Contrast uptake changes intensities, so SSD/CC cannot be the optimisation metric.

**Method.**

| Component | Choice |
|---|---|
| Transformation | `T(x) = T_global(x) + T_local(x)`. Global = affine (12 DOF in 3D). Local = cubic B-spline FFD on a uniform control-point grid of spacing δ: `T_local(x) = Σ_l Σ_m Σ_n B_l(u) B_m(v) B_n(w) φ_{i+l, j+m, k+n}` |
| Multi-resolution | Coarse-to-fine control grids. Spacing is halved at each level, and new control points come from B-spline subdivision. Image resolution increases in step. |
| Similarity | Normalised Mutual Information `NMI = (H(A) + H(B)) / H(A,B)` (Studholme), computed from the joint histogram |
| Regulariser | Bending energy (3D thin-plate): integral of all squared 2nd derivatives of T over the volume. It is zero for affine transforms. |
| Cost | `C = −C_similarity + λ · C_smooth`, with λ = 0.01 |
| Optimiser | Stage 1: affine only (maximise NMI). Stage 2: FFD parameters. Plain gradient descent with a fixed step, stopping when the change in cost is below ε. |

**Experiments to replicate.**
- 8 volunteers: two 3D scans each (256×256×64, 1.33×1.33×2.5 mm), with motion between them and no contrast agent. Methods compared: rigid, affine, and FFD at 20 / 15 / 10 mm spacing. Metrics: SSD and CC (Table I).
- 2 volunteers × 6 scans with graded motion (none, cough, head, arm, lift out of coil) (Figs. 4–5).
- 8 patients with Gd-DTPA contrast, ranked blind by 2 radiologists. FFD was ranked best in 94% of cases (Table II).
- Expected trend: **none < rigid ≈ affine < FFD-20 < FFD-15 < FFD-10**.

**Reproducibility.** The breast data is private, so the numbers cannot be reproduced exactly. We reproduce the **qualitative trend** on a proxy dataset (see §3.1).

### 1.2 CNN / DLIR: de Vos et al., 2019

> B. D. de Vos, F. F. Berendsen, M. A. Viergever, H. Sokooti, M. Staring, I. Išgum. *A Deep Learning Framework for Unsupervised Affine and Deformable Image Registration.* Medical Image Analysis 52:128–143, 2019. (arXiv:1809.06130)

**Idea.** The pipeline is the same as conventional registration (metric → transform → resample), but a ConvNet predicts the transform parameters. It is trained **unsupervised** on image similarity, with no ground-truth deformations. Once trained, it registers a pair in one forward pass.

| Component | Choice |
|---|---|
| Affine net | Two weight-shared pipelines (fixed and moving), each 5 × [3³ conv → 2³ downsample]. Then global average pooling, concatenation, FC layers, and 12 outputs (translation, rotation, scale, shear). Constraints: rotation/shear in [−π, π], scale in [0.5, 1.5]. |
| Deformable net | Input is the fixed and moving images concatenated. Fully convolutional: [3³ conv → 2³ downsample] repeated until the output matches the B-spline grid, then 2 extra 3³ convs, 2 × 1³ convs, and 3 output channels (control-point displacements). |
| B-spline → dense field | **Transposed convolution with a fixed cubic B-spline kernel** |
| Multi-stage | Affine → coarse B-spline → fine B-spline. Stages are trained one after another, with earlier stages frozen. |
| Loss | `L = −NCC + α · P_bending`, with α = 0 for affine and α = 0.05 for deformable stages |
| Training | Adam, Glorot init, ReLU (linear output), bilinear warping in training, nearest-neighbour for warping labels |
| Baseline | SimpleElastix (B-spline + NCC, adaptive SGD, 500 iterations, 2000 samples per iteration) |
| Metrics | Dice, Hausdorff distance, average symmetric surface distance, landmark TRE, % folding (det J ≤ 0), std of det J, runtime |

**Experiments to replicate.**

| Experiment | Data | Key result |
|---|---|---|
| Intra-patient cardiac cine MRI, ES→ED (§6) | **Sunnybrook**: 45 scans × 20 timepoints. 3-fold CV (30 train / 15 test). Intensity rescaled to [0, 1] using min and 99th percentile. Augmentation: 90° rotations and ±16-voxel crops. 10k iterations per stage. | Dice 0.70 → **0.89** (multi-stage DLIR+BP), folding ≈ 0.002%, ~39 ms on GPU (Table 2) |
| Inter-patient low-dose chest CT (§7) | **NLST**: 2,000 training scans. Affine stage plus 3 B-spline stages (64 / 32 / 16 mm grids). | Aorta Dice 0.31 → **0.75**, ~0.43 s on GPU (Table 4) |
| Intra-patient 4D lung CT (§8) | **DIR-Lab**: 10 cases, leave-one-out. HU clipped to [−1000, −200]. 3 B-spline stages (32 / 16 / 8 mm grids). | Mean TRE **2.64 mm** (1.63 mm without the worst 10% of landmarks) (Table 6) |

**Reproducibility.** Sunnybrook and DIR-Lab are public, so those results can be reproduced. The NLST images are public on TCIA, but the aorta and landmark annotations are not. The Learn2Reg NLST task is the practical substitute.

### 1.3 How the two papers connect

DLIR **reuses Rueckert's B-spline transform and bending-energy penalty**. What changes is how the parameters are found:

| | B-Spline FFD (Rueckert) | DLIR (de Vos) |
|---|---|---|
| Parameters found by | Per-pair iterative optimisation | Trained network, one forward pass |
| Metric | NMI (handles multi-modal pairs) | NCC (mono-modal) |
| Training data needed | None | Hundreds to thousands of pairs |
| Runtime per pair | Seconds to minutes (15–30 min in 1999) | Milliseconds |
| Failure mode | Local minima, sensitive to initialisation | Domain shift, large unseen deformations |

This gives us a fair comparison: **same transformation model and regulariser, different optimiser**.

---

## 2. Project Structure

```
Image-Registration/
├── run.py             # Single-pair experiment and result export
├── common/
│   ├── data.py        # OASIS 2D loader and subject splits
│   └── geometry.py    # Warp, Dice, and Jacobian shared by methods
├── b_spline/
│   └── ffd.py         # 2D B-spline FFD and optimisation
├── tests/
│   └── test_registration.py
├── paper/             # Source papers (PDFs)
├── data/download.md   # Download notes; image data is outside this repo
├── cnn_based/         # Planned DLIR implementation
├── demo/              # Planned interactive inference app
└── results/           # Generated outputs, ignored by Git
```

The downloaded datasets are in `../dataset/`, outside the repository.

### Run the current 2D baseline (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python run.py
```

`run.py` uses two subjects from a deterministic, subject-disjoint OASIS 2D validation split unless `--fixed` and `--moving` are supplied together from the same split. It writes `metrics.json`, `field.npz`, and `preview.png` under `results/oasis_2d/<pair>/<settings>/`. Use `--root` if the OASIS folder is elsewhere. This first baseline has cubic B-spline FFD, NCC/SSD, and a 2D bending penalty; affine registration, NMI, a true image pyramid, CNN, and 3D registration remain to be implemented. The default bending weight is for this code's mean-squared discrete penalty and is not numerically comparable to the 1999 paper's λ.

---

## 3. Datasets

Download the 2D or small datasets first (Tier 0–1) so the whole pipeline works end-to-end on a laptop before moving to 3D.

### 3.1 Tier 1: Paper replication

| Dataset | Used for | Size / access | Link |
|---|---|---|---|
| **Sunnybrook Cardiac Data** | DLIR §6 replication (exact) | 45 cine MRI with LV contours. Public (CC0). Get the package *with contours*. | [Cardiac Atlas Project](https://cardiacatlasproject.isd.kcl.ac.uk) · [SourceForge mirror](https://sourceforge.net/projects/cardiac-mr/files) |
| **DIR-Lab 4DCT** | DLIR §8 replication (exact), and B-spline on lung | 10 cases × 10 phases, 300 landmarks at T00/T50. Free, but you must fill in a request form. | [DIR-Lab (Emory)](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html) |
| **Learn2Reg NLST / Lung CT** | Substitute for DLIR §7 (NLST chest CT) | Preprocessed NLST pairs with keypoints. Free. | [Learn2Reg datasets](https://learn2reg.grand-challenge.org/Datasets/) · [Lung CT (Zenodo)](https://doi.org/10.5281/zenodo.3835682) |
| **Breast DCE-MRI (TCIA)**, e.g. Duke-Breast-Cancer-MRI | Proxy for Rueckert (pre- vs post-contrast) | Very large. Download **only 5–10 patients**. Free after registering with TCIA. | [TCIA](https://www.cancerimagingarchive.net/) |

### 3.2 Tier 2: Extending the comparison

These cover different anatomy, deformation size, and modality, so they test where each method wins.

| Dataset | Why include it | Labels for evaluation |
|---|---|---|
| **OASIS brain MRI** (Learn2Reg / neurite-oasis) | The standard inter-subject benchmark (VoxelMorph). A **2D slice version** exists, which makes it ideal for laptops and the demo. | 24 foreground labels in 2D; 35 anatomical labels in 3D |
| **ACDC cardiac MRI** | Larger, more modern alternative to Sunnybrook (150 patients) | ED/ES segmentations of LV, RV, myocardium |
| **Learn2Reg Abdomen CT-CT** | Large inter-patient deformations, a hard case for both methods | 13 organ labels |
| **IXI brain MRI** (T1 / T2 / PD) | **Multi-modal** pairs (T1↔T2) to test NMI (Rueckert) against NCC (DLIR) | None, but can be paired with synthetic deformations |
| **FIRE retinal fundus** | 2D, non-volumetric, small dataset (134 pairs): stresses DLIR's data hunger | Ground-truth correspondence points |
| **ANHIR histology** | 2D, very large images and deformations, stain changes | Landmarks |
| **MNIST / synthetic warps** | Toy sanity checks and unit tests; ground-truth deformation known exactly | Exact ground-truth displacement field |

### 3.3 Recommended download order

1. **MNIST + neurite-oasis 2D**: small, no registration needed. Use these to build and debug both methods and the demo.
2. **Sunnybrook**: first real replication (DLIR Table 2) and a cardiac comparison against B-spline.
3. **DIR-Lab** (submit the request form early, since approval takes time): landmark-based TRE (DLIR Table 6).
4. **Learn2Reg (OASIS 3D, Lung/NLST, Abdomen)**: wider comparison.
5. **IXI, FIRE, a few TCIA breast cases**: multi-modal, 2D-photo, and Rueckert-proxy experiments.

Downloaded datasets live under `../dataset/<name>/`, outside this repository. Put derived versions in `../dataset/<name>/processed/`; do not commit image data.

---

## 4. Roadmap

- [ ] **Phase 0: Foundations.** Image I/O, resampling, warping, metrics (SSD, NCC, NMI, Dice, HD, ASD, TRE, det J).
- [ ] **Phase 1: B-Spline FFD.** Affine, then multi-level B-spline with NMI/NCC/SSD and bending energy, in 2D then 3D. Check against SimpleElastix.
- [ ] **Phase 2: Rueckert replication.** Rigid vs affine vs FFD (20/15/10 mm) on breast-MRI proxy data, reporting SSD/CC and difference images.
- [ ] **Phase 3: DLIR.** Affine net, B-spline net (transposed-conv upsampling), multi-stage training, in 2D then 3D.
- [ ] **Phase 4: de Vos replication.** Sunnybrook (Table 2) and DIR-Lab (Table 6).
- [ ] **Phase 5: Cross-dataset comparison.** Same metrics on every Tier-2 dataset: accuracy, smoothness (folding), runtime, and robustness to deformation size, modality, and training-set size.
- [ ] **Phase 6: Demo.** Interactive app (see below).

## 5. Demo (target)

An interactive app (e.g. Gradio or Streamlit) where the user:

1. Picks a **dataset**. A random **fixed** and **moving** image is sampled, or the user picks them.
2. Picks a **method**: Affine only · B-Spline FFD · DLIR (single- or multi-stage).
3. Adjusts **parameters**:
   - B-Spline: grid spacing, number of levels, metric (SSD / NCC / NMI), λ (bending energy), iterations, step size
   - DLIR: checkpoint / stage count, (optional) test-time fine-tuning steps
4. Sees **results**: warped image, difference images (before and after), deformation grid overlay, det J heat-map with folding highlighted, metric values, and runtime.

## 6. References

- Rueckert et al., *Nonrigid Registration Using Free-Form Deformations*, IEEE TMI 1999.
- de Vos et al., *A Deep Learning Framework for Unsupervised Affine and Deformable Image Registration*, MedIA 2019.
- Balakrishnan et al., *VoxelMorph*, IEEE TMI 2019. Useful as a second CNN baseline.
- Hering et al., *Learn2Reg*, IEEE TMI 2022. Benchmark datasets and evaluation protocol.
- Klein et al., *elastix*, IEEE TMI 2010; Marstal et al., *SimpleElastix*, 2016. Reference B-spline implementation.
