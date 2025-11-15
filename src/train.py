import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import Inception3
from typing import Optional
from .image_dataset_loader import MelanomaImageDatasetLoader, LoaderType
from .model import load_inception_v3
from . import config

def _run_epoch(model: Inception3,
               data_loader: DataLoader,
               device: torch.device,
               loss_function: nn.Module,
               epoch: int,
               num_epochs: int,
               is_training: bool,
               optimizer: Optional[optim.Optimizer] = None):
    """
    Perform one training or validation epoch: Forward pass and (if training) backward pass.
    :param model: The Inception V3 model to train.
    :param data_loader: The data loader on training or validation indices.
    :param device: The device to run the computations on (e.g. GPU).
    :param loss_function: The loss function (e.g. Cross Entropy Loss).
    :param epoch: The current epoch.
    :param num_epochs: The total number of epochs.
    :param is_training: True if this is a training epoch (not a validation epoch)
    :param optimizer: The gradient optimizer (if training).
    """
    epoch_loss = 0.0
    num_correct = 0
    total_labels = 0
    phase = "Training" if is_training else "Validation"

    # set model and gradients
    if is_training:
        model.train()
        context = torch.enable_grad()
    else:
        model.eval()
        context = torch.no_grad()

    # run epoch
    with context:
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            # forward pass
            outputs = model(images)

            if is_training and model.aux_logits:
                main_outputs = outputs.logits
                # weighted auxilary loss
                loss = loss_function(main_outputs, labels) + 0.4 * loss_function(outputs.aux_logits, labels)
            elif hasattr(outputs, 'logits'):
                main_outputs = outputs.logits
                loss = loss_function(main_outputs, labels)
            else:
                main_outputs = outputs
                loss = loss_function(main_outputs, labels)

            # backward pass (training)
            if is_training:
                loss.backward()
                optimizer.step()

            # update validation metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(main_outputs.data, 1)
            num_correct += predicted.eq(labels).sum().item()
            total_labels += labels.size(0)

    # emit metrics
    normalized_epoch_loss = epoch_loss / len(data_loader)
    epoch_accuracy = 100 * num_correct / total_labels
    print(f'Epoch [{epoch}/{num_epochs}] | {phase} Loss: {normalized_epoch_loss:.4f} | {phase} Accuracy: {epoch_accuracy:.2f}%')

def _train_on_images_internal(model: Inception3,
                              train_loader: DataLoader,
                              validation_loader: DataLoader,
                              num_epochs: Optional[int] = 10,
                              learning_rate: Optional[float] = 0.001):
    """
    Perform training of Inception V3 model on Melanoma image dataset.
    :param model: The model to train.
    :param train_loader: The dataset loader for training indices.
    :param validation_loader: The dataset loader for validation indices.
    :param num_epochs: The number of times to update weights.
    :param learning_rate: The speed of updating weights.
    :return: The trained model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use GPU if available
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # training
        _run_epoch(model, train_loader, device, loss_function, epoch + 1, num_epochs, True, optimizer)
        # validation
        _run_epoch(model, validation_loader, device, loss_function, epoch + 1, num_epochs, False)

        print('-' * 60)

    print('Finished training')
    return model

def train_on_images():
    """
    API to call in main to train the model on Melanoma image dataset.
    :return: The trained model.
    """
    loader = MelanomaImageDatasetLoader(config.METADATA_FILE, config.IMAGE_DIRECTORY_STR)
    training_loader = loader.get_melanoma_image_dataset_loader(LoaderType.TRAINING)
    validation_loader = loader.get_melanoma_image_dataset_loader(LoaderType.VALIDATION)
    model = load_inception_v3()

    print('Beginning training')
    return _train_on_images_internal(model, training_loader, validation_loader)
