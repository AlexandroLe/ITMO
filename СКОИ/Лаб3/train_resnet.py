import os
import time
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
USER_PHOTOS_DIR = BASE_DIR / "Фото_руки"

def get_resnet_model(num_classes=3, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
                
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

def get_data_loaders(batch_size=32, img_size=(224, 224)):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    return train_loader, val_loader, class_names

def train_and_eval(epochs=4, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    train_loader, val_loader, class_names = get_data_loaders(batch_size=batch_size)
    print(f"Классы: {class_names}")

    model = get_resnet_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_weights_path = RESULTS_DIR / "resnet18_best.pth"

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\n--- Запуск Transfer Learning (ResNet-18) ---")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Валидация
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(f"Эпоха [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% "
              f"|| Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%", flush=True)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_weights_path)

    print(f"\nДообучение завершено за {time.time() - start_time:.1f} сек. Лучшая Val Acc: {best_val_acc * 100:.2f}%")

    # Тестирование на пользовательских фото в 'Фото_руки'
    model.load_state_dict(torch.load(best_weights_path))
    evaluate_on_user_photos(model, class_names, device)

def evaluate_on_user_photos(model, class_names, device):
    if not USER_PHOTOS_DIR.exists():
        print(f"Папка {USER_PHOTOS_DIR} не найдена!")
        return

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(USER_PHOTOS_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
    print("\n========================================================")
    print(" РЕЗУЛЬТАТЫ RESNET-18")
    print("========================================================")
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Матрица ошибок ResNet-18 на ваших 50 фото')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.tight_layout()
    cm_path = RESULTS_DIR / "resnet_user_photos_cm.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Матрица ошибок сохранена: {cm_path}")

    # Сетка примеров
    plt.figure(figsize=(12, 12))
    indices = np.random.choice(len(test_dataset), min(16, len(test_dataset)), replace=False)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, idx in enumerate(indices):
        tensor, label = test_dataset[idx]
        with torch.no_grad():
            pred = model(tensor.unsqueeze(0).to(device)).argmax(1).item()
        img_np = tensor.permute(1, 2, 0).numpy() * std + mean
        img_np = np.clip(img_np, 0, 1)

        plt.subplot(4, 4, i + 1)
        plt.imshow(img_np)
        color = "green" if pred == label else "red"
        plt.title(f"Истина: {class_names[label]}\nПредсказано: {class_names[pred]}", color=color, fontsize=9)
        plt.axis("off")

    samples_path = RESULTS_DIR / "resnet_user_photos_samples.png"
    plt.tight_layout()
    plt.savefig(samples_path, dpi=300)
    plt.close()
    print(f"Сетка с предсказаниями сохранена: {samples_path}")

if __name__ == "__main__":
    train_and_eval(epochs=3, batch_size=32, lr=0.001)
