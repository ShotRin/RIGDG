# RIGDG

> Reference Image Guided Defect Generation Model for Robust Vision Inspection

[![Paper](https://img.shields.io/badge/Paper-DBpia-blue)](...)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](...)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red)](...)
[![License](https://img.shields.io/badge/License-MIT-green)](...)




## Paper

This repository contains the official implementation of the paper

**RIGDG: Reference Image Guided Defect Generation Model for Robust Vision Inspection**

presented at the **2026 KIIT Conference**.

Paper:
https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12318489

# RIGDG

**Reference Image Guided Defect Generation Model for Robust Vision Inspection**

Official implementation of **RIGDG**, a reference image-guided defect generation framework for industrial visual inspection.

The proposed method synthesizes realistic defect images from paired reference and inspection images without requiring pixel-level defect annotations, improving the robustness of vision inspection systems through realistic defect augmentation.

---

## Overview

Industrial defect inspection often suffers from insufficient defective samples, making it difficult to train robust deep learning models.

RIGDG addresses this limitation by generating realistic defect images using paired **Reference (Master)** and **Inspection (Current)** images. Instead of relying on manually annotated defect masks, the proposed framework automatically extracts defect-aware features from image pairs and synthesizes realistic defects while preserving normal PCB structures.

---

## Method

The overall framework consists of four stages.

1. Feature Encoding
2. Difference Feature Extraction (ΔF)
3. Defect-aware Localization
4. Reference-guided Defect Generation

<p align="center">
<img width="2587" height="628" alt="image" src="https://github.com/user-attachments/assets/ec8b283c-3ef9-47c0-9890-1eda1faeca01" />
</p>

The generated images can be utilized to augment industrial inspection datasets and improve downstream defect detection performance.

---

## Key Features

- Reference image-guided defect generation
- No pixel-level defect annotations required
- Difference feature-based defect localization
- Realistic synthetic defect generation
- Industrial PCB inspection framework

---

## Repository Structure

```
RIGDG
│
├── datasets/          Dataset loader
├── models/            Network architectures
├── utils/             Utility functions
├── train.py           Training
├── test.py            Inference
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset consists of paired PCB images.

```
dataset/

    train/

        Master/

        Current/

    test/

        Master/

        Current/
```

- **Master** : Reference PCB image
- **Current** : Inspection PCB image

Each inspection image must have a corresponding reference image.

---

## Installation

```bash
git clone https://github.com/ShotRin/RIGDG.git

cd RIGDG

pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Inference

```bash
python eval.py
```

---

## Results

Example of generated defect images per model.
<img width="1258" height="611" alt="image" src="https://github.com/user-attachments/assets/dbdbdbaf-8808-4c9c-bf06-0684e6781213" />


<img width="1333" height="553" alt="image" src="https://github.com/user-attachments/assets/dcc2bb7a-6887-463a-bc9b-456338c67128" />

<img width="1328" height="369" alt="image" src="https://github.com/user-attachments/assets/29b90784-db93-4d57-94dd-d42c131ac211" />


---

## Publication

This repository accompanies the conference paper:

**Reference Image Guided Defect Generation Model for Robust Vision Inspection**

Proceedings of the Korea Institute of Information Technology (KIIT) Conference, 2026.

DBpia:
https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12318489

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{lee2026rigdg,
  title={Reference Image Guided Defect Generation Model for Robust Vision Inspection},
  author={...},
  booktitle={Proceedings of the Korea Institute of Information Technology Conference},
  year={2026}
}
```

---

## License

This project is intended for academic and research purposes.

---

## Contact

If you have any questions, please open an Issue or contact the authors.
