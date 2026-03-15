import cv2
import tensorflow as tf
import numpy as np
import time

# Load MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for model
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    # Prediction
    preds = model.predict(img)
    label = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0][1]

    # FPS calculation
    end = time.time()
    fps = 1 / (end - start)

    # Display label
    cv2.putText(frame, "Object: " + label, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Display FPS
    cv2.putText(frame, "FPS: " + str(int(fps)), (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # Show webcam
    cv2.imshow("Webcam Object Detection", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()