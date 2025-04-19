import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Load pre-trained object detection model from TensorFlow Hub
print("Loading model...")
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
print("Model loaded successfully!")

# Load label map for COCO dataset
labels_path = tf.keras.utils.get_file("mscoco_label_map.txt", 
    "https://storage.googleapis.com/download.tensorflow.org/data/mscoco_label_map.txt")
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines() if line.strip()]

# Function to run detection
def detect_objects(frame):
    img = cv2.resize(frame, (320, 320))
    img_tensor = tf.convert_to_tensor([img], dtype=tf.uint8)
    results = model(img_tensor)
    return results

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detect_objects(frame)

    boxes = results["detection_boxes"][0].numpy()
    class_ids = results["detection_classes"][0].numpy().astype(int)
    scores = results["detection_scores"][0].numpy()

    h, w, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > 0.5:
            box = boxes[i]
            y1, x1, y2, x2 = box
            x1, x2, y1, y2 = int(x1 * w), int(x2 * w), int(y1 * h), int(y2 * h)
            class_name = labels[class_ids[i] - 1] if class_ids[i] <= len(labels) else "N/A"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}: {int(scores[i]*100)}%", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("Real-Time Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()