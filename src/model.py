import torch.nn as nn
import torchvision.models as models

def load_inception_v3(num_classes=2):
    model = models.inception_v3(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model