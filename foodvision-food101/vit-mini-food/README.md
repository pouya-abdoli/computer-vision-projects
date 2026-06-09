# vit-mini-food 🍕🥩🍣

A lightweight Vision Transformer (ViT) implementation for food image classification using PyTorch, achieving **92% accuracy** on a 3-class food dataset.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview

This project demonstrates transfer learning with Vision Transformers (ViT) for image classification on a subset of the Food-101 dataset. The model classifies images into three categories: **pizza, steak, and sushi**.

**Key Features:**
- Vision Transformer (ViT-B/16) with pre-trained weights
- Modular PyTorch code structure
- Transfer learning with frozen backbone
- Training monitoring and visualization
- Model checkpointing and evaluation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/pouya-8367/vit-mini-food.git
cd vit-mini-food

# Install dependencies
pip install torch torchvision torchinfo requests
