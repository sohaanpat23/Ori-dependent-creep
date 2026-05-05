# Ori-dependent-creep: Orientation-Dependent Creep Life Evaluation of Ni- based singel crystal superalloys.

This repository contains a specialized **Physics-Informed Neural Network (PINN)** framework designed for evaluating and predicting the creep rupture life of single-crystal Nickel-based superalloys, with a specific focus on orientation dependency.

## Overview
Predicting the creep life of superalloys is a critical challenge in aerospace and power generation. This project implements a machine learning approach that incorporates metallurgical physics into the neural network architecture to improve prediction accuracy across different crystallographic orientations and stress-temperature regimes.

## 🛠 Key Features
*   **Machine Learning Baselines:** Used to evaluate the performance of our model compared to other baselines.
*   **Physics-Informed Architecture:** Integrates physical constraints (like Larson-Miller Parameters and Resolved Shear Stress) into the loss function.
*   **Orientation Sensitivity:** Handles anisotropic behavior by projecting stress tensors onto slip systems.
*   **SHAP Explainability:** Includes SHAP (SHapley Additive exPlanations) analysis to interpret model decisions and feature importance.
*   **Robust Validation:** Features k-fold cross-validation and internal validation on unseen experimental datasets.

## 📂 Project Structure
*   `src/`: Core Python scripts for baseline run('allbaselines.py'), model training (`codePINN.py`), SHAP analysis (`SHAPPINN.py`), and validation (`validation.py`).
*   `data/`: Processed datasets and experimental creep data.
*   `output/`: Model performance metrics, parity plots, and feature importance charts.
*   `papers/`: Key research literature on Ni-based superalloys and creep modeling.

## 💻 Getting Started
1. **Prerequisites:** Ensure you have Python 3.8+ installed with the following libraries:
   - `torch`
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `matplotlib`
   - `shap`

2. **Run Training:**
   ```bash
   python src/codePINN.py
   ```

3. **Run Feature Analysis:**
   ```bash
   python src/SHAPPINN.py
   ```


