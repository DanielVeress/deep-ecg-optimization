# Running a Deep ECG Model on Low-Cost Hardware

> **Master's Thesis Project** | Uppsala University  
> **Student:** Dániel Veress

## Project Overview

Deep Neural Network (DNN) models achieve high accuracy but often require significant computational resources. In emergency medical settings, rapid inference on constrained hardware is critical.

This project focuses on optimizing a Deep ECG Model to significantly reduce its resource demands (memory and inference time) while preserving its accuracy. The goal is to enable the deployment of advanced diagnostic models on low-cost hardware or edge devices.

## Objectives

- **Minimize Resource Usage:** Reduce memory footprint and FLOPs.
- **Accelerate Inference:** Achieve faster runtimes for time-critical analysis.
- **Maintain Accuracy:** Ensure the compressed model performs comparably to the baseline "true" model.

## Background

This work builds upon the paper _"A deep learning ECG model for localization of occlusion myocardial infarction"_ (Gustafsson et al., 2025). We utilize the fully trained model from this research as our baseline and aim to compress it using various optimization strategies.
