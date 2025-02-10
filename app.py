from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load food classes and nutrition info
with open("nutrition.json", "r") as f:
    nutrition_data = json.load(f)

food_classes = list(nutrition_data.keys())

# Load model validation metrics separately
with open("model_metrics.json", "r") as f:
    model_metrics = json.load(f)

# Pre-load models (you can modify this to load on demand)
models = {
    "custom_cnn": load_model("models/custom_cnn.h5"),
    "vgg16": load_model("models/vgg16_model.h5"),
    "resnet50": load_model("models/resnet50_model.h5"),
}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize as per model
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def index():
    return render_template("index.html", food_classes=food_classes)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or "model" not in request.form:
        return jsonify({"error": "Missing file or model selection"}), 400

    file = request.files["file"]
    model_choice = request.form["model"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded image
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # Preprocess image and make prediction
    img_array = preprocess_image(img_path)
    model = models.get(model_choice)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = food_classes[predicted_class_index]

    # Fetch nutrition info
    nutrition_info = nutrition_data.get(predicted_class_name, "No data available")

    # Fetch model validation report from the separate JSON file
    model_report = model_metrics.get(model_choice, {})

    return jsonify({
        "class_name": predicted_class_name,
        "nutrition": nutrition_info,
        "model_report": model_report
    })

if __name__ == "__main__":
    app.run(debug=True)
