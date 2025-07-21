import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import gradio as gr
import numpy as np
import cv2

# Classes
CLASSES = ["normal", "benign", "malignant"]

# Prétraitement
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

#  modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("breast_model.pth", map_location=device))
model.eval().to(device)

# Fonction d’analyse
def analyser_image(image):
    if image is None:
        return None, "❌ No image received."

    img_rgb = image.convert("RGB")
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)

    label = CLASSES[pred.item()]
    msg = {
        "normal": " Result: Normal image. No suspicious signs detected.",
        "benign": " Result: Benign lesion detected. A medical check is recommended.",
        "malignant": " Result: Suspicion of malignant tumor. See a doctor immediately."
    }[label]

    # Détection 
    img_cv = np.array(img_rgb.convert("L"))
    blurred = cv2.GaussianBlur(img_cv, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
        img_cv_color = np.array(img_rgb)
        cv2.circle(img_cv_color, center, radius, (255, 0, 0), 3)
        size_estimation = f"~{radius*2}px"
        cv2.putText(img_cv_color, f"Size: {size_estimation}", (center[0]-40, center[1]-radius-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        final_img = Image.fromarray(img_cv_color)
        return final_img, msg

    return img_rgb, msg

# Interface Gradio
interface = gr.Interface(
    fn=analyser_image,
    inputs=gr.Image(type="pil", label="Upload breast ultrasound image"),
    outputs=[
        gr.Image(label="Analyzed Image"),
        gr.Textbox(label="AI Result")
    ],
    title="Breast Cancer AI Assistant ",
    description="Upload a breast ultrasound image to get an AI-based prediction (normal, benign, malignant), with automatic highlight of the suspicious region."
)

if __name__ == "__main__":
    interface.launch()
