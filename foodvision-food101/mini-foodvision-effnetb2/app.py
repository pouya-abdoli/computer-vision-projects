
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Dict

class_names = ['pizza', 'steak', 'sushi']

effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=3,
                                                      seed=42)
effnetb2.load_state_dict(
    torch.load(
        fr'effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth',
                         map_location=torch.device("cpu")))

# predict function
def predict(img):

    start_time = timer()

    img = effnetb2_transforms(img).unsqueeze(0)

    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    # pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    pred_labels_and_probs = {}

    for i in range(len(class_names)):
        label = class_names[i]
        prob = float(pred_probs[0][i])
        pred_labels_and_probs[label] = prob
    
    end_time = timer()
    pred_time = round(end_time - start_time, 4)

    return pred_labels_and_probs, pred_time

title = "FoodVision Mini 🍕🥩🍣"
description = "An [EfficientNetB2 feature extractor](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2) computer vision model to classify images as pizza, steak or sushi."
article = "Thanks to [Daniel Bourke (mrdbourke)](https://github.com/mrdbourke/pytorch-deep-learning/)"

example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

demo.launch()
