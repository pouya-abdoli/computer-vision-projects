# FoodVision Mini – Image Classification Demo

A lightweight image-classification project built with **PyTorch** and deployed through a **Gradio** web interface.
The model classifies food images into three categories:

* Pizza
* Steak
* Sushi

This repository demonstrates how to load a trained deep-learning model, process images for inference, and build an interactive browser-based UI.

---

## Features

* Pretrained model loading (EfficientNet-based feature extractor)
* Image preprocessing for inference
* Top-3 prediction output with confidence scores
* Real-time inference time measurement
* Gradio demo interface with example images
* Easy to modify and extend for other datasets or models

---

## Online Demo (Hugging Face Space)

The model is also deployed publicly on **Hugging Face Spaces**:

**Live Demo:**
[https://huggingface.co/spaces/pouya-8367/foodvision_mini](https://huggingface.co/spaces/pouya-8367/foodvision_mini)

This allows anyone to test the classifier directly in the browser without running the notebook locally.

---

## Demo

Running the notebook will automatically launch a **Gradio interface** where you can:

* Upload an image
* View model predictions
* Inspect inference latency
* Try built-in sample images

The interface runs locally in your browser.

---

## Installation

```bash
pip install torch torchvision gradio pillow numpy
```

---

## How to Use

1. Open the notebook `extra.ipynb`.
2. Ensure the model weights file is available in the specified path.
3. Ensure an `examples/` folder exists with sample test images.
4. Run all cells.
5. A local Gradio link will appear in the output — open it in your browser.

---

## Project Structure

```
.
├── extra.ipynb        # Notebook containing model loading, inference, UI
├── models/            # (Optional) directory for .pth weight files
├── examples/          # Sample images for the Gradio interface
└── README.md
```

---

## Requirements

* Python 3.8+
* PyTorch
* Torchvision
* Gradio
* Pillow
* NumPy
* Pandas

---

## Purpose of the Project

This project serves as a minimal example for:

* Deploying ML models with Gradio
* Running image classification with PyTorch
* Integrating a pre-trained feature extractor
* Building reproducible ML demos for sharing or testing

---

## Customization

You can easily modify:

* The dataset
* The model architecture
* Prediction labels
* The Gradio interface layout
