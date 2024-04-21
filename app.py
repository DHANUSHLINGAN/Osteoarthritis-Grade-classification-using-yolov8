from flask import Flask, render_template, request, send_file
from PIL import Image
from ultralytics import YOLO
import torch
import os

app = Flask(__name__)

def predict_image(image_file, model_path):
    # Load the YOLOv8 model
    model = YOLO(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Perform object detection
    results = model.predict(source=image_file)

    # Get the base filename
    base_filename = os.path.splitext(os.path.basename(image_file))[0]

    # Save the annotated image
    annotated_image_path = os.path.join('static', 'uploads', f"{base_filename}_annotated.jpg")
    results[0].save(annotated_image_path)

    return annotated_image_path

# Routes
@app.route("/", methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        # Get the uploaded image file
        img = request.files['file']
        img_path = os.path.join('static', 'uploads', img.filename)
        img.save(img_path)

        # Perform prediction
        model_path = 'best.pt'
        annotated_image_path = predict_image(img_path, model_path)

        # Render the result page with uploaded and predicted images
        return render_template("result.html", original_image=img_path, annotated_image=annotated_image_path)

    return render_template("index.html")

if __name__ == '__main__':
    # Create the uploads directory if it doesn't exist
    upload_dir = os.path.join(app.root_path, 'static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    app.run(debug=True)