import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import mlflow
import mlflow.pytorch

from data_loading import load_imdb, IMDbDataset, get_dataloader
from text_processing import build_vocab
from models import TextCNN
from bert import BertTeacher
from evaluate import evaluate


# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# MLflow setup
# -------------------------
mlflow.set_experiment("imdb_distillation")


# -------------------------
# Load IMDb data
# -------------------------
train_data, _ = load_imdb()
train_data = train_data.shuffle(seed=42).select(range(3000))

texts = [item["text"] for item in train_data]
vocab = build_vocab(texts)

dataset = IMDbDataset(train_data, vocab)

# train / validation split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = get_dataloader(train_dataset, batch_size=16, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size=16, shuffle=False)


# -------------------------
# Models
# -------------------------
student = TextCNN(vocab_size=len(vocab)).to(device)
teacher = BertTeacher(device)


# -------------------------
# Loss + Optimizer
# -------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(student.parameters(), lr=1e-3)

alpha = 0.5   # balance between hard labels and teacher


# -------------------------
# Training
# -------------------------
num_epochs = 3
os.makedirs("models", exist_ok=True)

with mlflow.start_run():

    # log parameters
    mlflow.log_param("embedding_dim", 128)
    mlflow.log_param("num_filters", 100)
    mlflow.log_param("kernel_sizes", "3,4,5")
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("epochs", num_epochs)

    for epoch in range(num_epochs):
        student.train()
        total_loss = 0.0

        for texts, input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # ---- Teacher (frozen)
            with torch.no_grad():
                teacher_probs = teacher.predict_proba(texts)

            # ---- Student
            student_probs = student(input_ids)

            # ---- Losses
            label_loss = criterion(student_probs, labels)
            distill_loss = criterion(student_probs, teacher_probs)

            loss = alpha * label_loss + (1 - alpha) * distill_loss

            # ---- Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ---- Validation
        val_loss, val_acc = evaluate(student, val_loader, device)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # ---- MLflow metrics
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # ---- Save model
        torch.save(
            student.state_dict(),
            f"models/student_epoch_{epoch+1}.pt"
        )

    # ---- Log final model to MLflow
    mlflow.pytorch.log_model(student, "student_model")
