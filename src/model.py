import torch.nn as nn
import torchvision.models as models

def load_inception_v3(num_classes=2):
    """
    Loads to Inception V3 model as a binary classifier by default.
    :param num_classes: The number of classes if not default.
    :return: The model.
    """
    model = models.inception_v3(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model