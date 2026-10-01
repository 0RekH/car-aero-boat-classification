# predict.py
from pathlib import Path
import torch
from torchvision import transforms, models
from PIL import Image


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)


def predict(img_path, checkpoint_path="checkpoints/car_boat_plane_local.pth"):
    model, class_names = load_model(checkpoint_path)
    x = preprocess_image(img_path)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_class = class_names[pred_idx]

    return pred_class, probs.tolist(), class_names


if __name__ == "__main__":
    img_path = "test.png"  # замени на свою картинку
    pred_class, probs, class_names = predict(img_path)
    print("Класс:", pred_class)
    print("Классы:", class_names)
    print("Вероятности:", probs)
