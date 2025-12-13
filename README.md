# 🚗 Vehicle Damage Detection App

This project is a **vehicle damage detection system** that predicts the type of damage on a car image. The app allows users to **drag and drop an image of a car**, and the system will classify the damage based on the **third quarter front or rear view** of the car. The model is trained using **ResNet50 transfer learning** for high accuracy and fast predictions.

---

![App Screenshot](app.jpg)

---

## 📌 Key Features

- ✅ Predicts **Front/Rear damage** and type (**Normal, Crushed, Breakage**)
- ✅ Supports **third quarter front and rear car views**
- ✅ Lightweight and interactive frontend built with **Streamlit**
- ✅ Uses pre-trained **ResNet50** model for transfer learning
- ✅ Quick predictions with ~80% validation accuracy

---

## ⚙️ Tech Stack

| Layer        | Tools & Libraries            |
|--------------|------------------------------|
| Frontend     | Streamlit                    |
| Backend      | Python                       |
| ML Libraries | PyTorch, NumPy, PIL          |
| Deployment   | Local / Streamlit Sharing    |

---

## 🧪 Sample Output

- **Uploaded Image:** `car_front.jpg`  
- **Prediction:** Front Crushed  

- **Uploaded Image:** `car_rear.jpg`  
- **Prediction:** Rear Breakage  

---

## 🗂️ Project Structure

vehicle-damage-detection/
├── app.py # Main Streamlit app
├── model_helper.py # Model loading and prediction logic
├── trained_model.pth 
├── dataset
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── app_screenshot.jpg # App screenshot


---
