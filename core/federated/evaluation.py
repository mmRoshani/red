import torch

from optimizer.loss_factory import loss_factory
from utils.checker import device_checker
from typing import List
from sklearn.metrics import f1_score

from validators.config_validator import ConfigValidator
from validators.validator_exporters import model_constant_exporter_nlp


def model_evaluation(model, config: 'ConfigValidator', loader, loader_mask: List[int], device: str):
    device = device_checker(device)

    model.eval()

    criterion = loss_factory(loss_func=config.LOSS_FUNCTION, loader_mask=loader_mask)

    correct_global, total_global = 0, 0
    running_loss = 0.0

    all_preds = []
    all_labels = []

    if config.MODEL_TYPE in model_constant_exporter_nlp():
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                loss = criterion(outputs, labels)
                running_loss += loss.item() * input_ids.size(0)

                _, predicted = torch.max(outputs, dim=1)
                total_global += labels.size(0)
                correct_global += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    else:
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)  
                
                loss = criterion(outputs, labels.long())  # Loss calculated on masked classes

                running_loss += loss.item() * images.size(0)

                _, predicted_global = torch.max(outputs.data, 1)
                total_global += labels.size(0)
                correct_global += (predicted_global == labels).sum().item()

                all_preds.extend(predicted_global.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total_global if total_global > 0 else 0.0
    global_accuracy = correct_global / total_global if total_global > 0 else 0.0

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, global_accuracy, macro_f1
