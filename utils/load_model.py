import torch
from torchvision.models import efficientnet_v2_s

# Function to load pre-trained EfficientNetV2-S model
def load_model():
    model = efficientnet_v2_s(weights='DEFAULT')
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model