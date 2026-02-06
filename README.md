# Cucumber Disease Diagnostic Lab: Guide

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://deep-learning-image-classification-by-jayanth-konanki.streamlit.app/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/jayanthkonanki/deep-learning-image-classification)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


https://github.com/user-attachments/assets/86ad05f5-7a5f-4f98-ad2c-502e6623cc6e


This repository contains the **Cucumber Disease Diagnostic Lab**, a deep learning-powered platform designed to identify common diseases in cucumber leaves. By utilizing state-of-the-art Convolutional Neural Networks (CNNs) and Hybrid architectures, the system provides real-time diagnostic reports and actionable treatment plans to assist in crop management.

### Core AI Solution
* **Disease Classification**: Identifying conditions such as Anthracnose, Bacterial Wilt, Downy Mildew, and Gummy Stem Blight.
* **Hybrid Architectures**: Combining features from multiple backbones (e.g., ResNet50 + InceptionV3) for enhanced diagnostic accuracy.
* **Real-time Inference**: Fast image processing and probability estimation using PyTorch.
* **Actionable Insights**: Providing specific treatment recommendations based on the detected disease.

## 🚀 Overview

The application provides a comprehensive diagnostic workflow:

1.  **Architecture Selection**: Users can choose from standard models (ResNet50, InceptionV3, EfficientNetB0, MobileNet) or custom Hybrid models.
2.  **Diagnostic Report**: Generates a high-confidence prediction, a probability distribution chart across all categories, and a guided "Action Plan" for recovery.

---

## 🛠️ Setup Instructions

### 1. Prerequisites

Ensure you have Python 3.11+ installed. The system requires a `models/` directory containing `.pth` weight files for the chosen architectures.

### 2. Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
