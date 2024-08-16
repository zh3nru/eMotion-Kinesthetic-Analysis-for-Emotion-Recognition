import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load MobileNet model for person detection
net = cv2.dnn.readNetFromCaffe(r"E:\YOLO\deploy.prototxt", r"E:\YOLO\mobilenet_iter_73000.caffemodel")

# Load the trained emotion recognition model
emotion_model = load_model(r"D:\THESIS 2.0\eMotion.h5")

# Define the emotions
emotions = ["Aversion", "Anger", "Happiness", "Fear", "Sadness", "Surprise", "Peace"]

# Path to the input video
video_path = r"D:\THESIS 2.0\JAK.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video writer initialized to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('JAK_output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for faster processing
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # Get the class label for the detection
            class_id = int(detections[0, 0, i, 1])

            # If the detection is of a person (COCO class id 15)
            if class_id == 15:  
                # Get the coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Extract the ROI of the person and resize to the model's input size
                face = frame[startY:endY, startX:endX]
                face = cv2.resize(face, (64, 64))
                face = face.astype("float32") / 255.0
                face = np.expand_dims(face, axis=0)

                # Predict the emotion
                preds = emotion_model.predict(face)[0]
                emotion_idx = np.argmax(preds)
                emotion = emotions[emotion_idx]

                # Draw the bounding box and the emotion label on the frame
                label = f"{emotion}: {confidence*100:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

    # Display the output frame (optional)
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
