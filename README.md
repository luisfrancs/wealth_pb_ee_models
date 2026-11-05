# wealth_pb_ee_models

Machine learning models and utilities for predicting **Physical Behaviour (PB)** and **Energy Expenditure (EE)** from wearable sensor data within the WEALTH project framework.

This repository provides pretrained models and processing pipelines for analysing data collected using **activPAL, ActiGraph, and wrist-worn wearable sensors**, supporting physical activity monitoring under free-living conditions.

---

## Overview

`wealth_pb_ee_models` is a Python package developed within the WEALTH project to support the automated analysis of wearable accelerometer data for:

- **Physical Behaviour (PB) classification**
- **Energy Expenditure (EE) estimation**

The package implements machine learning and deep learning pipelines for large-scale, real-world monitoring of physical activity and sedentary behaviour.

It supports different wearable placements, including thigh-, waist-, and wrist-worn sensors, enabling the assessment and comparison of physical activity predictions across devices and sensor locations.

The models and methods are based on data collected from multi-centre European cohorts and evaluated under free-living conditions.

---

## Supported Sensors and Formats

The package supports multiple wearable devices and input formats.

### activPAL (Thigh-worn)

- Compressed files: `.datx`
- Uncompressed CSV files: `.csv`

### ActiGraph (Waist-worn)

- Raw files: `.gt3x`
- Native GT3X data processing and preprocessing pipeline.

### Wrist-worn Watch Sensors

- Standard CSV files: `.csv`

Wrist-worn data should be provided in the expected three-axis accelerometer format and meet the sampling frequency and preprocessing requirements of the corresponding pretrained model.

### Combined Data (Dual Sensor)

- Synchronized activPAL + ActiGraph CSV files.
- Synchronized data should be provided at a sampling frequency of 20 Hz.

---

## Processing Pipeline

The implemented pipeline comprises:

1. **Data loading and preprocessing:** Device-specific data extraction, signal formatting, and preprocessing.
2. **Signal synchronisation:** Temporal alignment of dual-sensor recordings.
3. **Sliding-window segmentation:** Segmentation of accelerometer signals into fixed-length windows.
4. **Feature extraction / raw-signal handling:** Preparation of input signals for machine learning and deep learning models.
5. **Model-based inference:** Prediction using pretrained models adapted to the corresponding sensor placement and configuration.
6. **Post-processing:** Processing of model probabilities and class predictions.
7. **Label decoding and formatting:** Conversion of numerical predictions into interpretable PB and EE categories.

The output consists of time-resolved predictions for physical behaviour and energy expenditure.

---

## Predicted Outputs

### Physical Behaviour (PB)

Seven activity classes:

- Sitting
- Standing
- Walking
- Running
- Cycling
- Sports
- Lying

### Energy Expenditure (EE)

Three intensity levels:

- Sedentary
- Light Physical Activity (LPA)
- Moderate-to-Vigorous Physical Activity (MVPA)

---

## Installation

### Requirements

- Python ≥ 3.9

### Install from Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/luisfrancs/wealth_pb_ee_models.git
cd wealth_pb_ee_models
pip install -e .
```

### Dependencies

Main dependencies include:

- NumPy
- Pandas
- Scikit-learn
- Joblib
- TensorFlow / Keras
- Additional device-specific data processing dependencies.

All dependencies are defined in `pyproject.toml`.

---

## Package Structure

The repository follows a standard `src`-based layout:

```text
wealth_pb_ee_models/
├── src/
│   └── wealth_pb_ee_models/
│       ├── models/          # Pretrained ML/DL models
│       ├── utils/           # Data loading and processing utilities
│       ├── pipeline/        # Inference workflows
│       ├── sample_data/     # Example datasets
│       └── config/          # Configuration files
│
├── notebooks/               # Example Jupyter/Colab notebooks
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Example Notebooks (Google Colab)

Example notebooks are provided in the `notebooks/` directory.

They are designed to be executed in **Google Colab**.

### Recommended Workflow

1. Open the notebook in Colab.
2. Select **File → Save a copy in Drive**.
3. Run and modify your private copy.

[![Run in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luisfrancs/wealth_pb_ee_models/blob/main/examples/ActivePAL_Daily_prediction_GITHUB_FINAL.ipynb)

---

## Scientific Background

The models are based on multitask learning approaches combining:

- Multi-Head Convolutional Neural Networks (MH-CNN).

They were developed and validated using long-term free-living data from the WEALTH project.

The methodological foundations are described in peer-reviewed and preprint publications associated with the WEALTH consortium.

---

## Publications

If you use this software, please cite the following publications:

1. Sigcha L, et al.  
   **Data Labelling for Free-Living Physical Activity Recognition using Thigh-Worn Wearables and Event-based Ecological Momentary Assessment.**  
   *Research Square*, 2025 (Preprint).  
   https://www.researchsquare.com/article/rs-6835979/v1

2. Sigcha L, et al.  
   **Robust Assessment of Free-Living Physical Behaviors and Activity Intensity Using Dual-Wearable Multitask Learning: Development and Evaluation Study From the Multicenter WEALTH Project.**  
   *JMIR mHealth and uHealth*. 2026;14:e94302.  
   https://doi.org/10.2196/94302

3. Hayes G, et al.  
   **Standardized Methods for Evaluating Physical and Eating Behaviors: The WEALTH Cross-Sectional Study Protocol.**  
   *JMIR Research Protocols*. 2026;15:e70186.  
   https://doi.org/10.2196/70186

---

## Data Availability

Raw participant data from the WEALTH project are not publicly distributed due to ethical and regulatory constraints.

Access may be granted upon reasonable request and in accordance with institutional approvals.

This repository provides:

- Pretrained models for different wearable sensor placements.
- Configuration files.
- Demonstration datasets in `sample_data/`.

These resources support reproducibility and methodological validation.

---

## License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

## Authors and Contributors

**Luis Sigcha, PhD**  
University of Limerick  
Email: luisfrancs@gmail.com

WEALTH Consortium

Contributions from partner institutions in Czechia, France, Germany, and Ireland.

---

## Citation

If you use this software in academic work, please cite both this repository and the associated WEALTH publications.

### Software Citation

```bibtex
@software{wealth_pb_ee_models,
  author  = {Sigcha, Luis},
  title   = {wealth\_pb\_ee\_models: Machine Learning Models for Physical Behaviour and Energy Expenditure Estimation},
  year    = {2026},
  url     = {https://github.com/luisfrancs/wealth_pb_ee_models}
}
```
