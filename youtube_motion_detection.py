import cv2
import numpy as np
import time
import os
from datetime import timedelta

# Video source (change as needed: file path, RTMP stream, or webcam index)
video_source = "camera2.mp4"  # Example: "rtmp://your-stream-url" or 0 for webcam

# Create the 'detections' folder if it doesn't exist
if not os.path.exists("detections"):
    os.makedirs("detections")

# Load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open video source
cap = cv2.VideoCapture(video_source)

# Check if video source is opened
if not cap.isOpened():
    print(f"Error: Could not open video source {video_source}")
    exit()

# Get video name from the file path
video_name = video_source.split("/")[-1].split(".")[0]

frame_count = 0
processed_frames = 0
start_time = time.time()  # Start timing

# Track previously detected persons (unique bounding boxes)
detected_persons = []

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video or stream error

    # Skip 14 out of every 15 frames
    if frame_count % 15 != 0:
        frame_count += 1
        continue

    processed_frames += 1
    frame_start_time = time.time()  # Start FPS timer

    # Get frame dimensions
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Process YOLO outputs
    person_detected = False
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # If a person is detected (class_id == 0 in COCO dataset)
            if confidence > 0.5 and class_id == 0:
                # Get bounding box coordinates
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Check for duplicate person by comparing bounding boxes
                is_duplicate = False
                for idx, prev_box in enumerate(detected_persons):
                    prev_x, prev_y, prev_w, prev_h = prev_box
                    if (abs(x - prev_x) < 200) and (abs(y - prev_y) < 200):  # Tolerance for slight movement
                        is_duplicate = True
                        # Update the bounding box position (update stored coordinates)
                        detected_persons[idx] = (x, y, w, h)
                        break

                # If not a duplicate, save the frame and add to the list
                if not is_duplicate:
                    person_detected = True
                    detected_persons.append((x, y, w, h))  # Store the bounding box to avoid duplicates
                    
                    # Draw bounding box on the frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{classes[class_id]}: {int(confidence * 100)}%"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Get timestamp from the video (in milliseconds)
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)  # Time in milliseconds
                    timestamp = str(timedelta(milliseconds=timestamp_ms))  # Convert to HH:MM:SS format
                    
                    # Save the frame with bounding box and timestamp
                    filename = f"detections/{video_name}_{timestamp}.png"
                    cv2.imwrite(filename, frame)

    if person_detected:
        print(f"[WARNING] Person detected in frame {frame_count}")

    # Calculate FPS
    frame_time = time.time() - frame_start_time
    fps = 1 / frame_time if frame_time > 0 else 0
    print(f"FPS: {fps:.2f}")

    frame_count += 1

cap.release()

# Calculate and print elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total elapsed time: {elapsed_time:.2f} seconds")
print(f"Processed frames: {processed_frames}")
print("Video processing complete.")
