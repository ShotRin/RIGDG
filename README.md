# RIGDG

> Reference Image Guided Defect Generation Model for Robust Vision Inspection

[![Paper](https://img.shields.io/badge/Paper-DBpia-blue)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12318489)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official PyTorch implementation of the conference paper presented at the **2026 Korea Institute of Information Technology (KIIT) Conference**.

<p align="center">
  <img width="2587" height="628" alt="image" src="https://github.com/user-attachments/assets/7ded701f-9554-4f00-9059-c00cf7a0a020" />
</p>

---

## Overview

Industrial visual inspection systems require large numbers of defective samples for training deep learning models. However, collecting sufficient defect images is expensive and time-consuming because defects occur infrequently in real manufacturing environments.

RIGDG (Reference Image Guided Defect Generation) is a GAN-based defect image generation framework that synthesizes realistic defect images from paired **Reference** and **Inspection** images.

Instead of relying on pixel-level defect annotations, RIGDG extracts feature differences between paired images, emphasizes defect-related information using CBAM attention, and generates realistic defect images conditioned on reference features.

---

## Method

The proposed framework consists of the following stages.

### 1. Feature Extraction

Both Reference and Inspection images are encoded using a shared Featuremap Reconstructor.

```
Reference Image
Inspection Image
        ↓
Featuremap Reconstructor
        │
        ├── z_ref
        └── z_ins
```

---

### 2. Difference Feature Extraction

Feature differences are computed as

```
ΔF = z_ins − z_ref
```

The difference feature contains defect-related information while suppressing common background characteristics.

---

### 3. CBAM Attention

CBAM is applied to emphasize informative defect regions.

```
ΔF
        ↓
CBAM
        ↓
Attention Feature
```

---

### 4. Conditional Feature Construction

The attended defect feature is combined with the reference feature.

```
z_cond = z_ref + CBAM(ΔF)
```

This conditional feature preserves normal PCB structure while injecting defect information.

---

### 5. Defect Image Generation

The conditional feature map is decoded by the generator to synthesize realistic defect images.

```
Reference + Inspection
        ↓
Feature Extraction
        ↓
Difference Feature (ΔF)
        ↓
CBAM
        ↓
Conditional Feature
        ↓
Generator
        ↓
Generated Defect Image
```
| Stage | Description |
|-------|-------------|
| Feature Extraction | Extract reference and inspection features using the shared encoder. |
| Difference Feature | Compute feature difference between paired images. |
| CBAM | Enhance defect-related responses. |
| Conditional Feature | Fuse the refined difference feature with the reference feature. |
| Generator | Decode the conditional feature into a defect image. |
---

## Key Features

- Reference-guided defect image generation
- No pixel-level defect annotation required
- Difference feature-based conditioning
- CBAM attention for defect enhancement
- GAN-based image synthesis
- PCB inspection dataset support

---

## Repository Structure

```
RIGDG
│
├── Encoders/
│   └── encoders.py
│
├── Methods/
│   └── ProposedMethod0421.py
│
├── train.py
├── generate_defects.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset consists of paired inspection images.

```
dataset/

False Alarm/
    sample001_Inspection.jpg
    sample001_Reference.jpg

Pollution/
    sample002_Inspection.jpg
    sample002_Reference.jpg

```

Each **Inspection** image must have a corresponding **Reference** image.

---

## Installation

```bash
git clone https://github.com/ShotRin/RIGDG.git

cd RIGDG

pip install -r requirements.txt
```

---

## Training

Train the proposed GAN model.

```bash
python train.py
```

Default settings

| Parameter | Value |
|-----------|------:|
| Epochs | 100 |
| Batch Size | 40 |
| Learning Rate | 5e-5 |

---

## Inference

Generate defect images using trained weights.

```bash
python generate_defects.py
```

Generated images will be saved in

```
results/
```

---

## Results

Example generated images.

| Reference | Generated |
|-----------|------------|
| <img width="100" alt="gen_defect_166_origin" src="https://github.com/user-attachments/assets/6e09cb36-6bb8-4eed-bb25-b17572e62464" /> | <img width="100" alt="gen_defect_166" src="https://github.com/user-attachments/assets/632af738-32bb-4f05-98ef-f9008717c12b" />  |

### Comparison with Existing Methods

Qualitative comparison of generated defect images with existing image generation methods.

| BAGAN | cGAN | SMOTified-GAN | DG2GAN | **RIGDG (Ours)** |
|-----------|----------|---------|-----------|------------------|
| <img width="100" alt="image" src="https://github.com/user-attachments/assets/979037ab-16f5-4e1f-bc67-4bb144a3c634" /> | <img width="100" alt="image" src="https://github.com/user-attachments/assets/9c2c385e-f37b-4418-878d-fbc3e528c432" /> | <img width="100" alt="image" src="https://github.com/user-attachments/assets/f34dd4f5-f173-47a9-aa4a-a5a21bfe7e11" /> | <img width="100" alt="image" src="https://github.com/user-attachments/assets/9ec55f41-5e3e-4d34-aee1-065b5b0d7359" /> | <img width="100" alt="image" src="https://github.com/user-attachments/assets/3108c9ec-117d-4e3f-b331-d92dd27b112a" /> |
| <img width="100" alt="image" src="https://github.com/user-attachments/assets/4a5dcfe4-1959-4cf1-ae08-7505ca3086fc" /> | <img width="100" alt="image" src="https://github.com/user-attachments/assets/6fe20b78-ce3b-4b58-ab74-063000808456" /> | <img width="100" alt="image" src="https://github.com/user-attachments/assets/9054a82c-7ffa-4d3a-90bc-3cc93d9b54f4" /> | <img width="100" alt="image" src="https://github.com/user-attachments/assets/7952b0a1-39e4-4dbf-b2d5-4b5b5a5e61a4" /> | <img width="100" alt="image" src="https://github.com/user-attachments/assets/40d36d79-49b3-4ee1-b85c-94a7dffc9fc9" /> |

### Quantitative Evaluation
| Method | FID ↓ | Classification Accuracy (F1-Score) ↑ |
|--------|------:|--------------------------:|
| Baseline | - | 0.82 |
| BAGAN | 372.60 | 0.81 |
| cGAN | 376.76 | 0.81 |
| SMOTified-GAN | 267.89 | 0.81 |
| DG2GAN | 273.36 | 0.81 |
| **RIGDG (Ours)** | **165.66** | **0.84** |
---

## Publication

**Reference Image Guided Defect Generation Model for Robust Vision Inspection**

Proceedings of the Korea Institute of Information Technology (KIIT) Conference, 2026.

DBpia:
[Paper](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12318489)

---

## Citation

```bibtex
@inproceedings{lee2026rigdg,
  title={Reference Image Guided Defect Generation Model for Robust Vision Inspection},
  author={Lee, Seunghun and ...},
  booktitle={Proceedings of the Korea Institute of Information Technology Conference},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, please contact:

Seunghun Lee

Email: oscm9@tukorea.ac.kr
