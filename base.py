import torch
from PIL import Image
from ultralytics import YOLO

def predict_and_save(image_path, model_path, output_path):
    # Load the YOLOv8 model
    model = YOLO(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load the image
    image = Image.open(image_path)

    # Perform object detection
    results = model.predict(source=image, save=True)

    # Save the annotated image
    results[0].save(output_path)

# Example usage
image_path = 'D:\Project\OSTEOARTHRITIS\Flask-Knee-Osteoarthritis-Classification\Anno_MR.KUMARAGURUBARAN625516122023134350_Rej.jpg'
model_path = 'best.pt'
output_path = 'annotated_image.jpg'

predict_and_save(image_path, model_path, output_path)