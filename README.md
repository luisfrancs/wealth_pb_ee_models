# wealth_pb_ee_models

Machine learning models and utilities for predicting **Physical Behaviour (PB)** and **Energy Expenditure (EE)** from wearable sensor data within the WEALTH project framework.

This repository provides pretrained models and processing pipelines for analysing data collected with **activPAL** and **ActiGraph** devices in free-living conditions.

---

## Overview

`wealth_pb_ee_models` is a Python package developed within the WEALTH project to support the automated analysis of wearable accelerometer data for:

- **Physical Behaviour (PB) classification**
- **Energy Expenditure (EE) estimation**

The package implements machine learning and deep learning pipelines for large-scale, real-world monitoring of physical activity and sedentary behaviour.

The models and methods are based on data collected from multi-centre European cohorts and evaluated under free-living conditions.

---

## Supported Sensors and Formats

The package supports multiple input formats:

### activPAL
- Compressed files: `.datx`
- Uncompressed CSV files: `.csv`

### ActiGraph
- Raw files: `.gt3x`

### Combined Data
- Synchronized activPAL + ActiGraph CSV files (Synchronized data should be provided at a sampling frequency of 20 Hz): `.csv`

---

## Processing Pipeline

The implemented pipeline comprises:

1. **Data loading and preprocessing**
2. **Signal synchronisation (dual-sensor)**
3. **Sliding-window segmentation**
4. **Feature extraction / raw-signal handling**
5. **Model-based inference**
6. **Post-processing**
7. **Label decoding and formatting**

The output consists of time-resolved and aggregated predictions for PB and EE.

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
- pygt3x  

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
├── notebooks/              # Example Jupyter/Colab notebooks
├── pyproject.toml
├── README.md
└── LICENSE
```
## Example Notebooks (Google Colab)

Example notebooks are provided in the `notebooks/` directory.

They are designed to be executed in **Google Colab**.

### Recommended Workflow

1. Open the notebook in Colab  
2. Select **File → Save a copy in Drive**  
3. Run and modify your private copy  

[![Run in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luisfrancs/wealth_pb_ee_models/blob/main/examples/ActivePAL_Daily_prediction_GITHUB_FINAL.ipynb)

---

## Scientific Background

The models are based on multitask learning approaches combining:

- Multi-Head Convolutional Neural Networks (MH-CNN)  

They were developed and validated using long-term free-living data from the WEALTH project.

The methodological foundations are described in peer-reviewed and preprint publications associated with the WEALTH consortium.

---

## Publications

If you use this software, please cite the following publications:

1. Sigcha L, et al.  
   **Data Labelling for Free-Living Physical Activity Recognition using Thigh-Worn Wearables and Event-based Ecological Momentary Assessment.**  
   *Research Square*, 2025 (Preprint).  
   (https://www.researchsquare.com/article/rs-6835979/v1)

2. Hayes G, et al.  
   **Standardized Methods for Evaluating Physical and Eating Behaviours: The WEALTH Project.**  
   *JMIR Research Protocols*, 2024 (Preprint).  
   https://preprints.jmir.org/preprint/70186
   
---

## Data Availability

Raw participant data from the WEALTH project are not publicly distributed due to ethical and regulatory constraints.

Access may be granted upon reasonable request and in accordance with institutional approvals.

This repository provides:

- Pretrained models  
- Configuration files  
- Demonstration datasets in `sample_data/`  

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

