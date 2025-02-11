# YOLO Motion Detection

This project uses the YOLO (You Only Look Once) object detection model to perform motion detection on a video stream. The algorithm detects and draws bounding boxes around objects in the video stream, prints a warning when a person is detected, and saves a snapshot of the frame when a person is detected. 

## Dependencies

The following dependencies are required for the proper execution of this algorithm:

- **OpenCV**: For video capture and processing, as well as object detection.
- **NumPy**: For matrix and numerical operations.
- **argparse**: For command-line argument parsing.

To install the necessary dependencies, you can use the following commands:

```bash
pip install opencv-python
pip install numpy
pip install yt-dlp
```

### Additional Dependencies

You will need the following files to run the YOLO model:

1. **yolov4.weights**  
   The YOLOv4 weights file, which contains the pre-trained weights of the YOLOv4 model.  
   You can download the weights file from the official YOLO website:  
   [YOLOv4 weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)

2. **yolov4.cfg**  
   The YOLOv4 configuration file, which defines the model architecture.  
   Download it from the official YOLO repository:  
   [YOLOv4 config file](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg)

3. **coco.names**  
   A file that contains the names of the 80 classes used by YOLO. This is required for interpreting the detection results.  
   You can get the `coco.names` file here:  
   [COCO names file](https://github.com/pjreddie/darknet/blob/master/data/coco.names)


## Directory Structure

The project structure should look like this:
```
/motion-detection
    /yolov4.weights
    /yolov4.cfg
    /coco.names
    /video_source.mp4    # Or an RTMP stream URL or camera index
    /motion_detection.py # The main Python script
```

## How to Use

### Running the Motion Detection Algorithm

To run the motion detection script, use the following command:

```bash
python motion_detection.py [video_source]
```

#### Parameters:

- `video_source` (required):  
  Specify the path to your video file, an RTMP stream URL, or a camera index.  
  **Examples:**
  - Video file: `"video1.mp4"`
  - RTMP stream: `"rtmp://your-stream-url"` or `"http://your-stream-url"`
  - Webcam: `0` (for the default webcam)

#### Example:
```bash
python motion_detection.py video1.mp4
```

This command will load the video from video1.mp4, and the program will perform motion detection on it using the YOLO model. If a person is detected, it will print a warning in the terminal and save a snapshot of the frame.

## How it Works

### YOLO (You Only Look Once)
YOLO is a real-time object detection algorithm that detects objects in images and videos. Unlike other object detection methods, YOLO divides an image into a grid and directly predicts bounding boxes and class probabilities for each grid cell. This makes YOLO incredibly fast and efficient for real-time object detection.

1. **Model Loading**:
The script loads the YOLOv4 model using the `cv2.dnn.readNet` function, which reads the pre-trained YOLO model weights (`yolov4.weights`) and the configuration file (`yolov4.cfg`).

2. **Frame Capture**:
The video stream is captured using `cv2.VideoCapture` with the video source (file, RTMP URL, or webcam) provided as an argument.

3. **Frame Processing**:
Each frame from the video is processed using the YOLO model to detect objects. The frame is converted into a blob format that YOLO understands using `cv2.dnn.blobFromImage`. The model then runs inference on the frame using `net.forward(output_layers)`.

4. **Bounding Boxes**:
For each object detected, the model calculates bounding boxes and class IDs. If the class ID corresponds to a "person" (class ID = 0 in the COCO dataset), a bounding box is drawn around the detected person.

5. **Saving Frames**:
When a person is detected, the program saves the current frame using `cv2.imwrite()` with a timestamp appended to the filename.

6. **FPS Display**:
The program calculates and displays the frames per second (FPS) of the video processing in the top-left corner of the frame.

## Troubleshooting

1. **Error: 'Cannot open video source'**: 
Ensure that the video source is valid and accessible. If you're using an RTMP stream, verify that the URL is correct. If using a webcam, try specifying a different index (e.g., 1 instead of 0).

2. **Error: 'yolov4.weights not found'**: 
Ensure that the `yolov4.weights`, `yolov4.cfg`, and `coco.names` files are in the same directory as the script, or specify the correct paths to them.


