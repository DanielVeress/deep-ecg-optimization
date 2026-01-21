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

## Setup

Follow the steps below to prepare your environment and fetch the necessary data.

### 1. Install Dependencies

This project uses uv for package management. Run the following to install requirements:

```bash
    uv pip install -r requirements.txt
    # OR, if you use uv sync:
    # uv sync
```

### 2. Download Data

Download from [PTB-XL, a large publicly available electrocardiography dataset](https://physionet.org/content/ptb-xl/1.0.3/), the fastest way is using AWS CLI:

```bash
    aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ data
```

The code expects the dataset to be in the **data** folder.
