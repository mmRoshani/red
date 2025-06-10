import torch
from src.utils import device_checker

def model_evaluation(model, loader, device: str):
    device = device_checker(device)

    model.eval()
    criterion = torch.nn.CrossEntropyLoss().to(device, non_blocking=True)
    correct, total = 0, 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels.long())

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / total
    accuracy = correct / total

    return loss, accuracy
