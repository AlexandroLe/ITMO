import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from train import SimpleRPSCNN, BASE_DIR, DATA_DIR, RESULTS_DIR

def evaluate_custom_test(test_path=None, batch_size=16, img_size=(128, 128)):
    if test_path is not None:
        test_dir = Path(test_path)
    elif (BASE_DIR / "Фото_руки").exists():
        test_dir = BASE_DIR / "Фото_руки"
    else:
        test_dir = DATA_DIR / "test_custom"

    weights_path = RESULTS_DIR / "best_model.pth"

    if not weights_path.exists():
        print(f"Ошибка: файл весов модели {weights_path} не найден! Сначала запустите train.py.")
        return

    # Проверка наличия изображений в test_custom
    classes = ["paper", "rock", "scissors"] # ImageFolder сортирует по алфавиту
    total_imgs = 0
    class_counts = {}
    for cls in ["rock", "paper", "scissors"]:
        folder = test_dir / cls
        if folder.exists():
            imgs = [f for f in folder.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            class_counts[cls] = len(imgs)
            total_imgs += len(imgs)
        else:
            class_counts[cls] = 0

    print(f"Найдено изображений в {test_dir.name}:")
    for cls, count in class_counts.items():
        print(f"  - {cls}: {count} фото")
    print(f"Всего тестовых фото: {total_imgs} (по заданию требуется >= 50)")

    if total_imgs == 0:
        print("\n[Внимание] В папке data/test_custom пока нет ваших фотографий!")
        print("Пожалуйста, добавьте туда свои фото (по подпапкам rock, paper, scissors) и запустите этот скрипт снова.")
        return

    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    class_names = test_dataset.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleRPSCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Отчет классификации
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
    print("\n--- Результаты на вашей тестовой выборке (Custom Test) ---")
    print(report)

    with open(RESULTS_DIR / "custom_test_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # Матрица ошибок
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Матрица ошибок на тестовой выборке (≥50 фото)')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.tight_layout()
    cm_path = RESULTS_DIR / "custom_test_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Матрица ошибок сохранена в: {cm_path}")

    # Визуализация примеров предсказаний (до 16 случайных картинок)
    samples_indices = np.random.choice(len(test_dataset), min(16, len(test_dataset)), replace=False)
    plt.figure(figsize=(12, 12))
    
    # Денормализация для корректного отображения
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, idx in enumerate(samples_indices):
        img_tensor, true_label = test_dataset[idx]
        img_input = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_input)
            _, pred = torch.max(output, 1)
            pred_label = pred.item()

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        plt.subplot(4, 4, i + 1)
        plt.imshow(img_np)
        is_correct = (pred_label == true_label)
        color = 'green' if is_correct else 'red'
        plt.title(f"Истина: {class_names[true_label]}\nПредсказание: {class_names[pred_label]}",
                  color=color, fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    samples_path = RESULTS_DIR / "custom_test_samples.png"
    plt.savefig(samples_path, dpi=300)
    plt.close()
    print(f"Примеры классификации сохранены в: {samples_path}")

if __name__ == "__main__":
    evaluate_custom_test()
