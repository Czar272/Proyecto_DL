# Gesture Recognition with Camera (Deep Learning)

This repo implements real-time **hand gesture recognition** using:
- **MediaPipe Hands** for fast hand detection & cropping
- **MobileNetV2 (Transfer Learning)** to classify hand crops into gestures
- **OpenCV** for webcam streaming

## Overview
This project allows real-time gesture recognition through your webcam.
It uses a pre-trained CNN (MobileNetV2) to classify hand gestures such as:
`open`, `fist`, `thumbs_up`, `thumbs_down`, `heart`,  and `peace`.

## Steps to Use
1. Collect dataset with your webcam (`collect_dataset.py`).
2. Split the dataset into train/validation (`split_dataset.py`).
3. Train a CNN model (`train.py`).
4. Run real-time inference (`inference_webcam.py`).

See `docs/report_template.md` and `docs/slides_outline.md` for templates.
