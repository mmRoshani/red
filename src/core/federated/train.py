from src.utils.log import Log
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader


def train(model: Module, loader: DataLoader, optimizer, epochs, device: str, log: 'Log'):
    criterion = CrossEntropyLoss().to(device, non_blocking=True)
    model.train()

    running_loss = 0.0
    # RvQ: The fuck is running_loss?

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels.long())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch > 1:
            log.info(f"[{epoch + 1}] loss: {running_loss / len(loader):.3f}")

    return model, running_loss / len(loader)
    # RvQ: Why devide running_loss with the lentgh of the loader