import cv2
import numpy as np
from keras.models import load_model

model = load_model("model/emotion_model.keras")
classes = np.load("Classes/classes.npy")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (48, 48)).reshape(1, 48, 48, 1) / 255.0
        prediction = model.predict(roi_resized)
        label = classes[np.argmax(prediction)]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.9, (255, 0, 0), 2)
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()