
from torch import nn
import torch
import torchvision

def create_effnetb2_model(num_classes: int=3,
                          seed: int=42):
                          
    # Create EffNetB2 pretrained weights, transforms and model
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transformer = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    torch.manual_seed(seed=seed)

    # Change classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes)
    )

    return model, transformer
