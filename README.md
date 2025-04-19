# 🎯 Real-Time Object Detection App using TensorFlow & OpenCV

This project demonstrates real-time object detection using TensorFlow and OpenCV. It uses a pre-trained SSD MobileNet model from TensorFlow Hub to identify and label objects via webcam feed.

## 📌 Features

- Real-time object detection using webcam
- Uses TensorFlow Hub pre-trained SSD MobileNet V2
- Object labeling with bounding boxes and confidence scores

## 📦 Dependencies

```bash
pip install tensorflow opencv-python tensorflow-hub numpy
```

## 🚀 How to Run

1. Plug in your webcam.
2. Run the script:
```bash
python real_time_object_detection.py
```
3. Press `q` to quit.

## 📁 Files

- `real_time_object_detection.py`: Main script for webcam object detection
- `README.md`: Project overview and instructions

## 💡 Notes

- Uses CPU or GPU automatically if available
- Works with most common USB/web cameras

## 📷 Output

- Live video with real-time detection and labeled bounding boxes