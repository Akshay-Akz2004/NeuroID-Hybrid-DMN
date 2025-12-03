# Robust EEG Biometric Identification using Hybrid Spectral-Connectivity Features

![Status](https://img.shields.io/badge/Status-Work_in_Progress-yellow)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Abstract
This project proposes a novel biometric identification system utilizing resting-state Electroencephalogram (EEG) signals. While recent literature demonstrates the efficacy of Frequency-Weighted Power (FWP) features from the Default Mode Network (DMN), existing approaches primarily rely on univariate analysis, neglecting the functional dependencies between brain regions. This work aims to address this limitation by introducing a hybrid feature extraction framework. The proposed method fuses spectral FWP features with Pearson Correlation-based connectivity metrics using a lightweight 12-channel setup. The system is evaluated on the PhysioNet Motor Movement/Imagery Dataset (Eyes-Open Baseline).

## 1. Introduction
Biometric systems based on EEG signals offer significant security advantages due to their resistance to spoofing and non-volitional nature. This project specifically targets the Default Mode Network (DMN), a resting-state network active during wakeful rest, to establish unique subject-specific signatures.

## 2. Methodology
The proposed system pipeline consists of four stages:

1.  **Data Acquisition:** Utilization of the publicly available PhysioNet database (109 subjects, 64 channels, 160Hz sampling rate).
2.  **Preprocessing:** * Band-pass filtering (0.5 Hz - 50 Hz).
    * Artifact removal and segmentation into 10-second non-overlapping epochs.
    * Channel selection: 12 DMN-specific electrodes (C1, TP8, FC5, F8, FT8, AF3, T8, P3, Iz, O2, T9, TP7).
3.  **Feature Extraction:**
    * **Spectral Features:** Frequency Weighted Power (FWP) extracted from Beta (13-30 Hz) and Gamma (30-50 Hz) bands.
    * **Connectivity Features:** Functional connectivity assessment using Pearson Correlation Coefficients (PCC) to generate a correlation matrix representing network topology.
4.  **Classification:** Implementation of a Support Vector Machine (SVM) with a linear kernel for subject identification.

## 3. Dataset Information
* **Name:** PhysioNet EEG Motor Movement/Imagery Dataset
* **Protocol:** Resting-state, Eyes Open (Task R01)
* **Subjects:** 109 Volunteers
* **Source:** [PhysioNet EEG MMIDB](https://physionet.org/content/eegmmidb/1.0.0/)

