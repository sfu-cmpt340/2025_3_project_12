import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import Inception3
from typing import Optional

def _run_epoch(model: Inception3,
               data_loader: DataLoader,
               device: torch.device,
               loss_function: nn.Module,
               epoch: int,
               num_epochs: int,
               is_training: bool,
               optimizer: Optional[optim.Optimizer] = None):
    epoch_loss = 0.0
    num_correct = 0
    total_labels = 0

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
            loss = loss_function(outputs, labels)

            # backward pass (training)
            if is_training:
                loss.backward()
                optimizer.step()

            # update validation metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            num_correct += predicted.eq(labels).sum().item()
            total_labels += labels.size(0)

    # emit metrics
    normalized_epoch_loss = epoch_loss / len(data_loader)
    epoch_accuracy = 100 * num_correct / total_labels
    phase = "Training" if is_training else "Validation"
    print(f'Epoch [{epoch}/{num_epochs}] | {phase} Loss: {normalized_epoch_loss:.4f} | {phase} Accuracy: {epoch_accuracy:.2f}%')

def train_on_images(model: Inception3,
                    train_loader: DataLoader,
                    validation_loader: DataLoader,
                    num_epochs: Optional[int] = 10,
                    learning_rate: Optional[float] = 0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
