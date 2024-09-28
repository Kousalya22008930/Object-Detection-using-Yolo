# Object-Detection-using-Yolo

## Aim:
To implement real-time object detection using a webcam and the YOLOv4 (You Only Look Once) model. The program captures frames from the webcam, detects objects in each frame using the YOLOv4 model, and displays the detected objects with bounding boxes and class labels.

## Procedure:

1. Install necessary libraries and download YOLOv4 weights, config, and class labels.
2. Load YOLOv4 model using cv2.dnn.readNet with weights and config files.
3. Read class labels from the coco.names file.
4. Initialize webcam with cv2.VideoCapture(0).
5. Capture frames and preprocess using cv2.dnn.blobFromImage.
6. Pass the frame through YOLOv4 model and extract detections.
7. Apply Non-Maximum Suppression (NMS) to filter overlapping boxes.
8. Draw bounding boxes with labels and display; exit on 'q' key press.

## Developed BY:
### Name: KOUSALYA A.
### Reg No: 212222230068

## Program:

### Imports and Model Loading
```python
import cv2
import numpy as np

# Load YOLOv4 network
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
```
### Webcam Initialization
```python
# Set up video capture for webcam
cap = cv2.VideoCapture(0)
```
### Real-Time Object Detection Loop
``` python
while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Prepare the image for YOLOv4
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get YOLO output
    outputs = net.forward(output_layers)
    
    # Initialize lists to store detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top-left corner of the box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
```
### Post-Processing and Drawing
```python
    # Apply Non-Max Suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
```
### Displaying the Output and Exiting
``` python
    # Show the image with detected objects
    cv2.imshow("YOLOv4 Real-Time Object Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
```
## Output:



## Result:
Thus to implement real-time object detection using a webcam and the YOLOv4 (You Only Look Once) model is successfully completed.
