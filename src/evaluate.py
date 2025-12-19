import torch
import torch.nn as nn

def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.BCELoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for _, input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            probs = model(input_ids)
            loss = criterion(probs, labels)

            total_loss += loss.item()

            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy
