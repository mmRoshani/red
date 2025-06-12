from src.utils.log import Log
from src.utils.checker import device_checker
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader
import torch
from typing import List
from src.validators.config_validator import ConfigValidator
from src.constants.models_constants import MODEL_BERT, MODEL_SHAKESPEARE_HYPER, MODEL_ALBERT
from tqdm import tqdm


def train(
    model: Module,
    loader: DataLoader,
    loader_mask: List[int],
    optimizer,
    epochs,
    device: str,
    config: "ConfigValidator",
    log: "Log",
    track_gradients=False,
    val_loader=None,
):
    if config.MODEL_TYPE in [MODEL_BERT, MODEL_SHAKESPEARE_HYPER, MODEL_ALBERT]:
        device = device_checker(device=device)
        model.to(device)
        
        accumulated_grads = None
        if track_gradients:
            accumulated_grads = []
            for param in model.parameters():
                if param.requires_grad:
                    accumulated_grads.append(torch.zeros_like(param, device=device))
                else:
                    accumulated_grads.append(None)

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs["loss"]
                logits = outputs["logits"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if track_gradients:
                    for i, param in enumerate(model.parameters()):
                        if param.requires_grad and param.grad is not None:
                            accumulated_grads[i] += param.grad.detach().abs()

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            avg_loss = total_loss / len(loader)
            accuracy = correct / total
            log.info(f"[{epoch+1}] Train Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")

        if track_gradients:
            return model, avg_loss, accumulated_grads
        else:
            return model, avg_loss
    else:
        criterion = CrossEntropyLoss()
        model.train()

        running_loss = 0.0

        accumulated_grads = None
        if track_gradients:
            accumulated_grads = []
            for param in model.parameters():
                if param.requires_grad:
                    accumulated_grads.append(torch.zeros_like(param, device=device))
                else:
                    accumulated_grads.append(None)

        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels.long())
                loss.backward()

                if track_gradients:
                    for i, param in enumerate(model.parameters()):
                        if param.requires_grad and param.grad is not None:
                            accumulated_grads[i] += param.grad.detach().abs()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()

            if epoch > 1:
                log.info(f"[{epoch+1}] loss: {running_loss / len(loader):.3f}")

        if track_gradients:
            return model, running_loss / len(loader), accumulated_grads
        else:
            return model, running_loss / len(loader)