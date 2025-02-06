import cv2
import numpy as np
import time
import threading

# Function to load YOLO model
def load_yolo_model(weights_path, config_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, output_layers

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define target resolution for resizing
target_width = 640
target_height = 480

# Function to process each frame
def process_frame(frame, net, output_layers, classes):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {int(confidences[i] * 100)}%"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Function to capture and process frames for each video
def process_video(video_path, net, output_layers, classes, frame_queue, skip_frames=15):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to reduce processing load
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        frame_count += 1

        # Process the frame
        processed_frame = process_frame(frame, net, output_layers, classes)
        processed_frame = cv2.resize(processed_frame, (target_width, target_height))

        # Add the processed frame to the queue
        frame_queue.append(processed_frame)

    cap.release()

# Load separate YOLO models for each thread
net1, output_layers1 = load_yolo_model("yolov4.weights", "yolov4.cfg")
net2, output_layers2 = load_yolo_model("yolov4.weights", "yolov4.cfg")
net3, output_layers3 = load_yolo_model("yolov4.weights", "yolov4.cfg")

# Queues to store processed frames from each video
frame_queue1 = []
frame_queue2 = []
frame_queue3 = []

# Start threads for each video
thread1 = threading.Thread(target=process_video, args=("video1.mp4", net1, output_layers1, classes, frame_queue1))
thread2 = threading.Thread(target=process_video, args=("video2.mp4", net2, output_layers2, classes, frame_queue2))
thread3 = threading.Thread(target=process_video, args=("youtube1.mp4", net3, output_layers3, classes, frame_queue3))

thread1.start()
thread2.start()
thread3.start()

frame_count = 0
start_time = time.time()

while True:
    # Check if all queues have frames
    if frame_queue1 and frame_queue2 and frame_queue3:
        frame1 = frame_queue1.pop(0)
        frame2 = frame_queue2.pop(0)
        frame3 = frame_queue3.pop(0)

        # Create a 2x2 grid
        top_row = np.hstack((frame1, frame2))
        bottom_row = np.hstack((frame3, np.zeros_like(frame3)))  # Empty frame
        grid = np.vstack((top_row, bottom_row))

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        fps_text = f"FPS: {fps:.2f}"
        start_time = time.time()
        cv2.putText(grid, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the grid
        cv2.imshow("Monitoring System", grid)

        frame_count += 1

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Wait for threads to finish
thread1.join()
thread2.join()
thread3.join()

cv2.destroyAllWindows()