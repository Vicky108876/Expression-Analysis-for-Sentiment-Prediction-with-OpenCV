# 😄 Expression Analysis for Sentiment Prediction with OpenCV

This project performs **expression analysis** using **OpenCV** for real-time face detection and a **CNN model** for **sentiment prediction**. It can detect human faces from a webcam feed and predict facial expressions like happy, sad, angry, and more.

---

## ✨ Features

- 🎥 Real-time face detection using Haar Cascade (OpenCV)

- 🧠 Expression classification using a CNN (Trained on FER-2013)

- 📦 Bounding box and emotion label on live video

- 🖼️ Easy to train and test on your own dataset

- 🗂️ FER-2013 dataset support (7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)

---

## 🗃️ Dataset

This project uses the **FER-2013** dataset from Kaggle:  
👉 [https://www.kaggle.com/datasets/msambare/fer2013]

---

##  📌 How it Works

1. The model is trained on FER-2013 dataset with 7 emotion categories.

2. Haar Cascade detects faces from live webcam feed.

3. Each detected face is preprocessed (grayscale, resized to 48x48).

4. The trained CNN model predicts the emotion.

5. Bounding box and label are drawn on the video frame.

---

## 🧠 Technologies Used

OpenCV (for face detection & live camera)

TensorFlow / Keras (for CNN model)

NumPy

FER-2013 dataset

---

## Install Dependencies

numpy==1.24.3

opencv-python

tensorflow

keras

---

##  How to Runs Locally

👉Step-1: Download the datset using above link.

🎥 Run the Real-Time Emotion Detection

👉Step-2: Run Detect_expression file

---
