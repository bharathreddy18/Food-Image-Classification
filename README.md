# 🍔 Food Classification System with Flask

This project is a **Food Classification Web App** built using **Flask**. Users can upload food images, select a deep learning model (Custom CNN, VGG16, or ResNet50), and get predictions with nutritional information.

## 🚀 Features
- 📸 Upload food images for classification which are listed in the web app interface
- 🧠 Choose from 3 models: Custom CNN, VGG16, ResNet50
- 📊 View model performance metrics (Accuracy, Precision, Recall, F1-Score)
- 🥦 Get nutritional information for the predicted food class

---

## 🛠️ Installation
### 1. Clone the Repository
```bash
git clone https://github.com/bharathreddy18/Food-Image-Classification.git
cd Food-Image-Classification
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate    # For Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚦 Usage
### 1. Run the Flask App
```bash
python app.py
```
Access the app at: `http://127.0.0.1:5000`

### 2. Upload an Image and Select a Model
- Upload a food image
- Select a model (CNN, VGG16, or ResNet50)
- View the predicted class, nutritional information and selected model metrics

---

## 🧪 Model Information
- **Custom CNN:** Trained on 34 food classes with high accuracy.
- **VGG16 & ResNet50:** Transfer learning models pre-trained on ImageNet.

### 📊 Model Metrics
- Accuracy, Precision, Recall, F1-Score (from `metrics.json`)
- Displays True Positives, False Positives, etc.

---

## 📂 Project Structure
```
food-classification-app/
├── app.py              # Flask backend
├── templates/
│   └── index.html      # Frontend HTML
├── static/
│   └── uploads/        # Uploaded images
├── models/
│   └── model_cnn.h5   # Trained models
└── requirements.txt     # Dependencies
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

If you find any mistakes or any doubts please post your comment so that I or anyone here can help you [Discussions Page](https://github.com/bharathreddy18/Food-Image-Classification/discussions/1)

---

## ⭐ Acknowledgments
- TensorFlow for model building
- Flask for the web framework

---

## 📞 Contact Me
- GitHub: [Jay](https://github.com/bharathreddy18)
- LinkedIn: [Jaya Bharath Reddy](https://linkedin.com/in/jaya-bharath-reddy-iska-7a3844210)
- Mail: bharathreddy.iska@gmail.com
