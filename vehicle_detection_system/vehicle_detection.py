import cv2
import numpy as np
import time
import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from utils.tracker import CentroidTracker
from utils.speed_estimator import SpeedEstimator
from config.settings import Settings

# Import DeepSORT (assumes deepsort module is available in your project)
from deep_sort_realtime.deepsort_tracker import DeepSort

def check_yolo_files():
    """
    Check if YOLO files exist and provide instructions if they don't
    """
    cfg_path = os.path.join("config", "yolo", "yolov4.cfg")
    names_path = os.path.join("config", "yolo", "coco.names")
    weights_path = os.path.join("config", "yolo", "yolov4.weights")
    
    files_needed = {
        cfg_path: "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
        names_path: "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
    }
    
    weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
    
    missing_files = []
    for file, url in files_needed.items():
        if not os.path.exists(file):
            os.makedirs(os.path.dirname(file), exist_ok=True)
            missing_files.append(file)
            print(f"Downloading {file}...")
            try:
                import urllib.request
                urllib.request.urlretrieve(url, file)
                print(f"Downloaded {file} successfully!")
            except Exception as e:
                print(f"Error downloading {file}: {e}")
                print(f"Please download manually from: {url}")
    
    if not os.path.exists(weights_path):
        print("\nYOLO weights file (yolov4.weights) is required but not found.")
        print(f"Please download it manually from: {weights_url}")
        print(f"And place it in: {weights_path}")
        print("This is a large file (~250MB).")
        return False
    
    return len(missing_files) == 0 and os.path.exists(weights_path)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Vehicle Detection and Counting System")
    parser.add_argument("--source", type=int, default=0, help="Camera source (default: 0)")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--output", type=str, default="output.avi", help="Output video path (default: output.avi)")
    args = parser.parse_args()
    
    # Check for YOLO files
    if not check_yolo_files():
        print("Missing required YOLO files. Please download them as instructed above.")
        return
    
    # Load settings
    settings = Settings()
    
    # Load YOLO network
    print("Loading YOLO model...")
    cfg_path = os.path.join("config", "yolo", "yolov4.cfg")
    weights_path = os.path.join("config", "yolo", "yolov4.weights")
    names_path = os.path.join("config", "yolo", "coco.names")
    
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    
    # Set preferred backend and target
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    try:
        # OpenCV 4.5.4+
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        # Older OpenCV versions
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # Load class names
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Get class IDs for vehicles
    vehicle_class_ids = [classes.index(vehicle) for vehicle in settings.VEHICLE_CLASSES if vehicle in classes]
    
    # Initialize video capture
    print(f"Opening camera source: {args.source}...")
    cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera source {args.source}.")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default FPS if not available
    
    # Initialize video writer (if saving)
    out = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
        print(f"Saving output to: {args.output}")
    
    # Initialize DeepSORT tracker
    deepsort = DeepSort(max_age=40, n_init=3, nms_max_overlap=1.0, embedder="mobilenet")
    speed_estimator = SpeedEstimator(fps=fps, ppm=settings.PIXELS_PER_METER)
    vehicle_count = 0
    counted_objects = set()
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    # Calculate line position
    line_y = int(frame_height * settings.LINE_Y_POSITION)
    
    print("Press 'q' to quit")
    
    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Calculate FPS
        if frame_count % 10 == 0:
            elapsed_time = current_time - start_time
            if elapsed_time > 0:
                fps_display = frame_count / elapsed_time
        
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Draw counting line
        cv2.line(display_frame, (0, line_y), (frame_width, line_y), (255, 0, 255), 2)
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        
        # Run forward pass
        layer_outputs = net.forward(output_layers)
        
        # Initialize lists for detected objects
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each output layer
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for vehicle classes with sufficient confidence
                if class_id in vehicle_class_ids and confidence > settings.CONFIDENCE_THRESHOLD:
                    # YOLO returns bounding box coordinates as center, width, height
                    # relative to the image dimensions
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    w = int(detection[2] * frame_width)
                    h = int(detection[3] * frame_height)
                    
                    # Calculate top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, settings.CONFIDENCE_THRESHOLD, settings.NMS_THRESHOLD)
        
        # Prepare detections for DeepSORT: (xyxy, confidence, class_id)
        detections_for_deepsort = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box
                x1, y1, x2, y2 = x, y, x + w, y + h
                detections_for_deepsort.append(([x1, y1, x2, y2], confidences[i], class_ids[i]))
        
        # Update DeepSORT tracker
        tracks = deepsort.update_tracks(detections_for_deepsort, frame=frame)
        
        # --- Improved vehicle counting logic with DeepSORT ---
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            class_id = track.det_class if hasattr(track, 'det_class') else 0
            confidence = track.det_conf if hasattr(track, 'det_conf') else 1.0
            
            # Update speed estimator with new position
            speed_estimator.update(track_id, centroid)
            
            # Get class name
            class_name = classes[class_id] if class_id < len(classes) else 'vehicle'
            
            # Improved: Use a small region around the line to avoid double counting due to jitter
            if track_id not in counted_objects:
                prev_centroid = getattr(track, 'prev_centroid', None)
                # Only count if the object crosses the line from above to below and is not too far from the line
                if prev_centroid is not None:
                    crossed = prev_centroid[1] < line_y and centroid[1] >= line_y and abs(centroid[1] - line_y) < 20
                    if crossed:
                        vehicle_count += 1
                        counted_objects.add(track_id)
                track.prev_centroid = centroid
            
            # Calculate speed
            speed = speed_estimator.get_speed(track_id)
            speed_status = "FAST" if speed > settings.SPEED_THRESHOLD else "SLOW"
            
            # Determine color based on vehicle type
            color_idx = settings.VEHICLE_CLASSES.index(class_name) if class_name in settings.VEHICLE_CLASSES else 0
            color = settings.COLORS[color_idx % len(settings.COLORS)]
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw centroid
            cv2.circle(display_frame, centroid, 4, (0, 0, 255), -1)
            
            # Prepare label with class, confidence, and speed
            speed_text = f"{speed:.1f} km/h"
            try:
                conf_val = float(confidence)
                label = f"ID {track_id} {class_name} {conf_val:.2f} {speed_text} {speed_status}"
            except Exception:
                label = f"ID {track_id} {class_name} {speed_text} {speed_status}"
            
            # Draw label background
            cv2.rectangle(display_frame, (x1, y1 - 30), (x1 + len(label) * 9, y1), color, -1)
            
            # Draw label text
            cv2.putText(display_frame, label, (x1, y1 - 10), settings.FONT, 0.5, (255, 255, 255), 2)
        
        # Display count and FPS information
        info_text = f"Vehicle Count: {vehicle_count} | FPS: {fps_display:.1f}"
        cv2.putText(display_frame, info_text, (10, 30), settings.FONT, 0.7, (0, 0, 255), 2)
        
        # Show frame (move fullscreen setup outside the loop for efficiency)
        if frame_count == 1:
            cv2.namedWindow("Vehicle Detection", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Vehicle Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Vehicle Detection", display_frame)
        
        # Save frame if requested
        if args.save and out is not None:
            out.write(display_frame)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Total vehicles detected: {vehicle_count}")

if __name__ == "__main__":
    main()