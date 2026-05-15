# 🧬 Protein Folding Research Lab & Structural Dynamics Suite
A professional-grade, unified platform for computational structural biology, molecular dynamics (MD), and AI-driven protein analysis. This suite integrates over **100 research-grade features** across two specialized applications, transforming raw sequence data into deep structural and evolutionary insights.
---
## 🚀 Overview
The platform consists of two master modules designed to handle everything from primary sequence analytics to complex 4D structural simulations.
### 🔬 [Module 1: The Research Lab (app.py)](file:///c:/Users/Admin/protein_lab/app.py)
A high-density dashboard featuring the **50-Metric Deep Analytics Engine** and the **50-Graph Advanced Visualization Suite**. It is optimized for interpreting AI-generated folds (AlphaFold/ESMFold) and validating structural integrity.
### 🧪 [Module 2: The Advanced Structural Engine (app2.py)](file:///c:/Users/Admin/protein_lab/app2.py)
A heavy-duty computational suite powered by **Biopython**, **ProDy**, and **MDTraj**. It enables large-scale Normal Mode Analysis (NMA), trajectory analytics, and evolutionary phylogeny.
---
## 📊 Feature Highlights: The 100-Feature Suite
### 1. The 50-Metric Deep Analytics Engine
*   **Physicochemical**: MW, pI, Aliphatic Index, Instability Index, GRAVY, and Solubility.
*   **Structural**: Helix/Sheet/Coil fractions, Buried/Exposed residue %, and Contact Density.
*   **Stability**: ΔG estimation, Aggregation propensity, and Phase Separation scores.
*   **AI Diagnostics**: Sequence Novelty, Embedding Norms, and pLDDT Confidence stats (P25/P50/P75).
### 2. The 50-Graph Advanced Visualization Suite
*   **Structural & Folding**: PAE Heatmaps, Contact Maps, Distance Histograms, and Ramachandran Density.
*   **Dynamics & Motion**: Free Energy Landscapes, RMSD/RMSF trajectories, and Dynamic Cross-Correlation (DCCM).
*   **AI Interpretation**: Transformer Attention Heatmaps, t-SNE/UMAP Latent Spaces, and AI Residue Importance maps.
*   **Mutation Analytics**: ΔΔG Stability Matrices and Mutation Impact Heatmaps.
##
### 3. Molecular Dynamics Analytics (MDTraj Integration)
*   **Trajectory Loading**: Native support for `.xtc`, `.dcd`, and `.trr` formats.
*   **Simulation Quality**: SASA (Surface Area) dynamics, H-bond persistence, and salt-bridge lifetimes.
*   **Ensemble Analysis**: Structural clustering and time-resolved DSSP tracking.
### 4. Evolutionary & Synthesis Lab (Biopython Integration)
*   **Alignment**: Pairwise and Multiple Sequence Alignment (MSA) using Clustal/MUSCLE interfaces.
*   **Synthesis Engineering**: Codon optimization for E. coli/Yeast/Mammalian and restriction site analysis.
*   **Phylogeny**: Construction of evolutionary trees and motif scanning (Zinc fingers, Kinases).
---

```
### Requirements
*   `streamlit`
*   `biopython`
*  
*   `mdtraj`
*   `plotly`
*   `fpdf`
*   `scikit-learn`
---
## 🖥️ Usage
### To launch the Sequence & AI Research Lab:
```bash
streamlit run app.py
```
### To launch the Advanced Structural Dynamics & MD Suite:
```bash
streamlit run app2.py
```
---
## 📜 Documentation
For a deep dive into the underlying physics and AI models used in this platform, refer to the [Research Paper (research_paper.md)](file:///c:/Users/Admin/protein_lab/research_paper.md).
---
## ⚖️ License
Sayantan Roy
Distributed under the MIT License.
