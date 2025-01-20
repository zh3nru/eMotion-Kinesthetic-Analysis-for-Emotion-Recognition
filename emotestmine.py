import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load MobileNet model for person detection
net = cv2.dnn.readNetFromCaffe(r"E:\YOLO\deploy.prototxt", r"E:\YOLO\mobilenet_iter_73000.caffemodel")

model = load_model(r"D:\THESIS 2.0\models\eMotion.h5")
emotions = ["Aversion", "Anger", "Happiness", "Fear", "Sadness", "Surprise", "Peace"]

# Path to the input video
video_path = r"D:\My Files\Downloads(Edge - 2)\12.mp4"


cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('8.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Get frame dimensions once to use for clipping
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])

            # Check for person
            if class_id == 15:  # Person class in COCO is 15 for MobileNet
                # Get coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                if startX >= endX or startY >= endY:
                    continue 

                face = frame[startY:endY, startX:endX]
                if face.size == 0:
                    continue  

                try:
                    face = cv2.resize(face, (64, 64))
                except cv2.error as e:
                    print(f"Error resizing ROI: {e}")
                    continue
                
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                # Normalize the image
                face = face.astype("float32") / 255.0

                # Expand dimensions to match model's input shape (1, 64, 64, 3)
                face = np.expand_dims(face, axis=0)

                # Inference
                preds = model.predict(face)
                emotion_idx = np.argmax(preds)
                emotion = emotions[emotion_idx]

                # Draw the bounding box and the emotion label
                label = f"{emotion}: {confidence*100:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    out.write(frame)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
