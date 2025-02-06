import cv2
import numpy as np
import time
import argparse

# Argument parser to get video source
parser = argparse.ArgumentParser(description="YOLO Motion Detection with RTMP/Video Source")
parser.add_argument("video_source", type=str, help="Path to video file, RTMP URL, or camera index (e.g., 0 for webcam)")
args = parser.parse_args()
video_source = args.video_source

# Load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open the video source
if video_source.startswith(("rtmp://", "http://", "https://")):
    print(f"Connecting to RTMP stream: {video_source}")
cap = cv2.VideoCapture(video_source)

# FPS Calculation
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or cannot be accessed.")
        break

    # Skip every other frame for faster processing
    if frame_count % 60 == 0:
        boxes, confidences, class_ids = [], [], []
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        # Process YOLO outputs
        person_detected = False  # Flag for person detection
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, w, h) = box.astype("int")
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    # Check if the detected object is a person (COCO class index 0)
                    if classes[class_id] == "person":
                        person_detected = True

        # Print warning if a person is detected
        if person_detected:
            print("⚠️ WARNING: Person detected!")

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {int(confidences[i] * 100)}%"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    # Display FPS with black background and white border
    fps_text = f"FPS: {fps:.2f}"
    text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_w, text_h = text_size
    cv2.rectangle(frame, (10, 30), (10 + text_w + 10, 30 + text_h + 10), (0, 0, 0), -1)  # Black background
    cv2.rectangle(frame, (10, 30), (10 + text_w + 10, 30 + text_h + 10), (255, 255, 255), 2)  # White border
    cv2.putText(frame, fps_text, (15, 30 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the video with detections
    cv2.imshow("YOLO Motion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
