# train.py
import os
import random
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(
    root_dir: Path,
    batch_size: int = 32,
    val_frac: float = 0.2,
    max_per_class: int = 1000,
):
    """
    root_dir — папка, где лежат подпапки классов:
        aeroplane/, car/, Ship/

    max_per_class — максимум изображений на каждый класс
    (чтобы не тренироваться на 64k машин и не ждать вечность).
    """

    image_root = root_dir

    # Аугментации для обучения
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Для валидации только ресайз + нормализация
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Полный датасет (но позже возьмём подмножество)
    full_train = datasets.ImageFolder(root=image_root, transform=train_transform)
    full_val = datasets.ImageFolder(root=image_root, transform=val_transform)

    class_names = full_train.classes
    print("Найденные классы:", class_names)
    # например: ['Ship', 'aeroplane', 'car']

    # Собираем индексы по каждому классу
    indices_by_class = {cls_idx: [] for cls_idx in range(len(class_names))}
    for idx, target in enumerate(full_train.targets):
        indices_by_class[target].append(idx)

    # Выбираем не больше max_per_class случайных индексов из каждого класса
    selected_indices = []
    for cls_idx, idx_list in indices_by_class.items():
        random.shuffle(idx_list)
        cut = idx_list[: min(max_per_class, len(idx_list))]
        selected_indices.extend(cut)

    random.shuffle(selected_indices)
    print("Всего выбрано изображений:", len(selected_indices))

    # Делим выбранные индексы на train / val
    n_total = len(selected_indices)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val

    train_indices = selected_indices[:n_train]
    val_indices = selected_indices[n_train:]

    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_val, val_indices)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, class_names


def build_model(num_classes: int):
    # ВАЖНО: weights=None → ничего не скачивается из интернета
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def main():
    set_seed(42)

    # Папка проекта, где лежат aeroplane/, car/, Ship/
    root_dir = Path(__file__).parent

    num_epochs = 10
    batch_size = 32
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используем устройство:", device)

    train_loader, val_loader, class_names = get_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        val_frac=0.2,      # 20% на валидацию
        max_per_class=1000 # максимум 1000 картинок на класс
    )

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/car_boat_plane_local.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
        },
        checkpoint_path,
    )
    print("Модель сохранена в", checkpoint_path)


if __name__ == "__main__":
    main()
