This repository contains the source code and pre-trained machine learning model developed for predicting melt pool dimensions and defect types in Laser Powder Bed Fusion (LPBF).

The proposed approach combines Finite Element Method (FEM) simulations with an Artificial Neural Network (ANN) and transfer learning. The ANN is first trained using FEM-generated data and subsequently fine-tuned using experimental data.

The model predicts:

- Melt pool depth
- Melt pool width
- Defect type:
  - Lack of Fusion (LoF)
  - Full-dense
  - Keyhole

## Methodology

The overall workflow is:

**FEM simulations → Source ANN → Pre-trained Source Model → Experimental data → Transfer learning / Fine-tuning → Melt pool and defect prediction**

## Publication

The methodology presented in this repository is described in our paper:

**Hybrid FEM-multitask neural network approach for processing map prediction in LPBF of novel AlFeCrX alloys**, *Additive Manufacturing*, 2026.

DOI: [10.1016/j.addma.2026.105352](https://doi.org/10.1016/j.addma.2026.105352)
