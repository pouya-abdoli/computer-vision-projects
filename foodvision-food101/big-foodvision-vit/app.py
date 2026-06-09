import gradio as gr
import os
import torch
from demos.foodvision_big.model import create_vit_b16_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
with open(class_name_path, "r") as f:
    class_names = [food_name.strip() for food_name in f.readlines()]

# Create model and transforms
vit, vit_transforms = create_vit_b16_model(num_classes=101)

# Load saved weights
vit.load_state_dict(
    torch.load("demos/foodvision_big/food101_vit_features_extraction.pth",
               map_location=torch.device("cpu"))
)

def predict(img) -> Tuple[Dict, float]:
    # Start a timer
    start_time = timer()

    # Transform the input image for use with ViT-16
    img = vit_transforms(img).unsqueeze(0)

    vit.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(vit(img), dim=1)

        pred_labels_and_probs = {}

        for i in range(len(class_names)):
            label = class_names[i]
            prob = float(pred_probs[0][i])
            pred_labels_and_probs[label] = prob

    # Calculate pred time
    end_time = timer()
    pred_time = round(end_time - start_time, 4)

    # Return pred dict and pred time
    return pred_labels_and_probs, pred_time

# Create title, description and article
title = "FoodVision BIG 🍔👁💪"
description = "A [Vision Transformer (ViT-B/16)](https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html) computer vision model to classify images [101 classes of food from the Food101 dataset](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/food101_class_names.txt)."  # FIXED: Updated description to ViT
article = "Thanks to [Daniel Bourke (mrdbourke)](https://github.com/mrdbourke/pytorch-deep-learning/)"

# Create example list
example_list = [["demos/foodvision_big/examples/" + example] for example in os.listdir("demos/foodvision_big/examples/")]

# Create the Gradio demo
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=5, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

demo.launch(debug=False, share=True)
