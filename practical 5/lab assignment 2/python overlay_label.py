import cv2
import tensorflow as tf
import numpy as np

# Load MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    # Predict object
    preds = model.predict(img)
    label = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0][1]

    # Overlay label on video feed
    cv2.putText(frame, "Object: " + label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Show video
    cv2.imshow("Object Detection", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()