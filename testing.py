import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=r"D:\My Files\Downloads - V\trackingobject\assets\posenet_mobilenet.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess input image
def preprocess_image(image, input_size):
    input_image = cv2.resize(image, (input_size[1], input_size[2]))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = (input_image.astype(np.float32) / 127.5) - 1.0
    return input_image

# Function to draw keypoints on the image
def draw_keypoints(image, keypoints, confidence_threshold=0.5):
    height, width, _ = image.shape
    for keypoint in keypoints:
        y, x, confidence = keypoint
        if confidence > confidence_threshold:
            cv2.circle(image, (int(x * width), int(y * height)), 4, (0, 255, 0), -1)

# Function to postprocess the output from the model
def postprocess_output(output_data, image_shape):
    print(f"Output shape: {output_data.shape}")
    print(f"Output data: {output_data}")

    height, width, _ = image_shape
    keypoints = []
    
    # Assuming output_data is a 3D array where the third dimension holds y, x, and confidence values
    for i in range(output_data.shape[2]):  # Adjust this based on actual output shape
        y = output_data[0, 0, i, 0]
        x = output_data[0, 0, i, 1]
        confidence = output_data[0, 0, i, 2]
        keypoints.append((y, x, confidence))
        
    return keypoints

# Open video file
cap = cv2.VideoCapture(r"D:\THESIS 2.0\JAK.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the input frame
    input_image = preprocess_image(frame, input_details[0]['shape'])

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor (keypoints)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Postprocess the output
    keypoints = postprocess_output(output_data, frame.shape)

    # Draw keypoints on the frame
    draw_keypoints(frame, keypoints)

    # Display the frame
    cv2.imshow('Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
