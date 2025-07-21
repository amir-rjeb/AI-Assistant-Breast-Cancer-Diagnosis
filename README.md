# Breast Cancer AI Assistant 

This project uses a deep learning model (ResNet18) to classify breast ultrasound images into:

-  Normal
-  Benign tumor
-  Malignant tumor (cancer)

It also highlights the suspicious region with a simple circle and estimates its size.

## Model

The model used is a fine-tuned `ResNet18` trained on breast ultrasound images. You can upload your own ultrasound to test the classifier.

## Features

- Automatic classification: normal, benign, or malignant
- Visual highlight of potential tumor zone
- Gradio-based UI, ready for deployment on Hugging Face Spaces

## How to use

1. Upload a breast ultrasound image
2. Get instant AI feedback and tumor zone highlight
3. Consult a medical expert based on the result

---

 **Disclaimer**: This tool is for educational/demo purposes only. Always consult a qualified radiologist or physician for diagnosis.

