import cv2
import numpy as np

# Define the car control function
def control_car(lane_center, frame_width, left_lines_count, right_lines_count, car_positions):
    car_center = frame_width // 2
    deviation = lane_center - car_center

    if car_positions:
        # Check the position of detected cars
        for (x, y) in car_positions:
            if x < frame_width // 2:
                # Car on the left side
                if y < frame_width // 3:
                    # Car is approaching from the top left
                    label = "Car approaching from top left"
                else:
                    # Car is approaching from the left
                    label = "Car approaching from left"
            else:
                # Car on the right side
                if y < frame_width // 3:
                    # Car is approaching from the top right
                    label = "Car approaching from top right"
                else:
                    # Car is approaching from the right
                    label = "Car approaching from right"
    elif left_lines_count >= 2 and right_lines_count >= 2:
        # Move forward
        # Implement the logic to keep the car moving forward
        label = "Move forward"
    elif deviation > -50:
        # Turn left
        # Implement the logic to steer the car left
        label = "Turn left"
    elif deviation < 50:
        # Turn right
        # Implement the logic to steer the car right
        label = "Turn right"
    else:
        # Move forward (fallback option)
        label = "Move forward"

    return label

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def estimate_lanes(lines):
    # Implement the logic to estimate lanes based on previous lane lines
    # Return the estimated lane lines

    # Placeholder implementation: return the input lines as is
    return lines

def detect_lanes(video_path):
    cap = cv2.VideoCapture(video_path)

    # Initialize previous lanes as None
    previous_lanes = None

    # Load YOLOv3-320 model
    net = cv2.dnn.readNetFromDarknet('/Users/mbn/Documents/Programmering/python3/LaneDetectionPython/yolov3.cfg', '/Users/mbn/Documents/Programmering/python3/LaneDetectionPython/yolov3.weights')

    # Load car class labels
    with open('/Users/mbn/Documents/Programmering/python3/LaneDetectionPython/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Set input size for YOLOv3-320 model
    input_size = (320, 320)

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, input_size, swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        car_positions = []
        class_ids = []
        confidences = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id] == 'car':
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    car_positions.append((center_x, center_y))
                    class_ids.append(class_id)
                    confidences.append(float(confidence))

                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

        bboxes = np.array([(*pos, 1, 1) for pos in car_positions], dtype=np.float32)
        confidences = np.array(confidences, dtype=np.float32)
        indexes = cv2.dnn.NMSBoxes(bboxes, confidences, 0.5, 0.4)


        car_positions_filtered = []
        for i in range(len(car_positions)):
            if i in indexes:
                car_positions_filtered.append(car_positions[i])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        height, width = frame.shape[:2]
        roi_bottom_width = width * 0.52
        roi_top_width = width * 0.25
        roi_height = height * 0.34
        roi_vertices = np.array([
            [(0, height), (roi_bottom_width, roi_height), (width - roi_bottom_width, roi_height), (width, height)]
        ], dtype=np.int32)
        roi_edges = region_of_interest(edges, roi_vertices)

        lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

        if lines is not None:
            lane_center = 0
            left_lines_count = 0
            right_lines_count = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), thickness=2)

                # Accumulate the x-coordinate values of the lines to calculate the average lane center
                lane_center += (x1 + x2) // 2

                # Count the lines on the left and right side of the lane center
                if (x1 + x2) // 2 < width // 2:
                    left_lines_count += 1
                else:
                    right_lines_count += 1

            if left_lines_count >= 2 and right_lines_count >= 2:
                lane_center //= (left_lines_count + right_lines_count)
                # Update the previous lanes with the current lanes
                previous_lanes = lines
            elif previous_lanes is not None:
                # Estimate the lanes based on previous lanes
                estimated_lanes = estimate_lanes(previous_lanes)
                if estimated_lanes is not None:
                    lane_center = np.mean([((line[0][0] + line[0][2]) // 2) for line in estimated_lanes])
                    # Update the previous lanes with the estimated lanes
                    previous_lanes = estimated_lanes
                else:
                    # Reset the lane center to the car center if no estimated lanes are available
                    lane_center = width // 2
            else:
                # Reset the lane center to the car center if there are not enough lines on both sides and no previous lanes
                lane_center = width // 2

            # Control the car based on the lane center, line counts, and car positions
            control_label = control_car(lane_center, width, left_lines_count, right_lines_count, car_positions_filtered)

            # Draw control label on the frame
            cv2.putText(frame, control_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Render ROI on the frame
        roi_overlay = np.zeros_like(frame)
        cv2.fillPoly(roi_overlay, roi_vertices, (0, 255, 0))
        frame_with_roi = cv2.addWeighted(frame, 1, roi_overlay, 0.3, 0)

        # Draw car label on the frame
        car_label = "Car Detected" if len(car_positions_filtered) > 0 else "No Car Detected"
        cv2.putText(frame_with_roi, car_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Lane Detection', frame_with_roi)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

video_path = "/Users/mbn/Documents/Programmering/python3/LaneDetectionPython/Better test.mp4"
detect_lanes(video_path)

"""
import cv2
import numpy as np
import threading
import queue

# Define the car control function
def control_car(lane_center, frame_width, left_lines_count, right_lines_count, car_positions):
    car_center = frame_width // 2
    deviation = lane_center - car_center

    if car_positions:
        # Check the position of detected cars
        for (x, y) in car_positions:
            if x < frame_width // 2:
                # Car on the left side
                if y < frame_width // 3:
                    # Car is approaching from the top left
                    label = "Car approaching from top left"
                else:
                    # Car is approaching from the left
                    label = "Car approaching from left"
            else:
                # Car on the right side
                if y < frame_width // 3:
                    # Car is approaching from the top right
                    label = "Car approaching from top right"
                else:
                    # Car is approaching from the right
                    label = "Car approaching from right"
    elif left_lines_count >= 2 and right_lines_count >= 2:
        # Move forward
        # Implement the logic to keep the car moving forward
        label = "Move forward"
    elif deviation > -50:
        # Turn left
        # Implement the logic to steer the car left
        label = "Turn left"
    elif deviation < 50:
        # Turn right
        # Implement the logic to steer the car right
        label = "Turn right"
    else:
        # Move forward (fallback option)
        label = "Move forward"

    return label

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def estimate_lanes(lines):
    # Implement the logic to estimate lanes based on previous lane lines
    # Return the estimated lane lines

    # Placeholder implementation: return the input lines as is
    return lines

def detect_objects(frame, net, output_layers, classes, input_size, detection_results):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, input_size, swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    car_positions = []
    class_ids = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == 'car':
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                car_positions.append((center_x, center_y))
                class_ids.append(class_id)
                confidences.append(float(confidence))

                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

    bboxes = np.array([(*pos, 1, 1) for pos in car_positions], dtype=np.float32)
    confidences = np.array(confidences, dtype=np.float32)
    indexes = cv2.dnn.NMSBoxes(bboxes, confidences, 0.5, 0.4)

    car_positions_filtered = []
    for i in range(len(car_positions)):
        if i in indexes:
            car_positions_filtered.append(car_positions[i])

    detection_results.put(car_positions_filtered)

def process_frame(frame, detection_results, previous_lanes):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = frame.shape[:2]
    roi_bottom_width = width * 0.52
    roi_top_width = width * 0.25
    roi_height = height * 0.34
    roi_vertices = np.array([
        [(0, height), (roi_bottom_width, roi_height), (width - roi_bottom_width, roi_height), (width, height)]
    ], dtype=np.int32)
    roi_edges = region_of_interest(edges, roi_vertices)

    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        lane_center = 0
        left_lines_count = 0
        right_lines_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), thickness=2)

            # Accumulate the x-coordinate values of the lines to calculate the average lane center
            lane_center += (x1 + x2) // 2

            # Count the lines on the left and right side of the lane center
            if (x1 + x2) // 2 < width // 2:
                left_lines_count += 1
            else:
                right_lines_count += 1

        if left_lines_count >= 2 and right_lines_count >= 2:
            lane_center //= (left_lines_count + right_lines_count)
            # Update the previous lanes with the current lanes
            previous_lanes.put(lines)
        elif not previous_lanes.empty():
            # Estimate the lanes based on previous lanes
            estimated_lanes = estimate_lanes(previous_lanes.get())
            if estimated_lanes is not None:
                lane_center = np.mean([((line[0][0] + line[0][2]) // 2) for line in estimated_lanes])
                # Update the previous lanes with the estimated lanes
                previous_lanes.put(estimated_lanes)
            else:
                # Reset the lane center to the car center if no estimated lanes are available
                lane_center = width // 2
        else:
            # Reset the lane center to the car center if there are not enough lines on both sides and no previous lanes
            lane_center = width // 2

        # Control the car based on the lane center, line counts, and car positions
        control_label = control_car(lane_center, width, left_lines_count, right_lines_count, detection_results.get())

        # Draw control label on the frame
        cv2.putText(frame, control_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Render ROI on the frame
    roi_overlay = np.zeros_like(frame)
    cv2.fillPoly(roi_overlay, roi_vertices, (0, 255, 0))
    frame_with_roi = cv2.addWeighted(frame, 1, roi_overlay, 0.3, 0)

    # Draw car label on the frame
    car_label = "Car Detected" if len(detection_results.get()) > 0 else "No Car Detected"
    cv2.putText(frame_with_roi, car_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Lane Detection', frame_with_roi)

def detect_lanes(video_path):
    cap = cv2.VideoCapture(video_path)

    # Initialize previous lanes as None
    previous_lanes = queue.Queue()

    # Load YOLOv3-320 model
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

    # Load car class labels
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Set input size for YOLOv3-320 model
    input_size = (320, 320)

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    detection_results = queue.Queue()

    def process_video_frames():
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            detect_objects(frame, net, output_layers, classes, input_size, detection_results)
            process_frame(frame, detection_results, previous_lanes)

            if cv2.waitKey(1) == ord('q'):
                break

    video_thread = threading.Thread(target=process_video_frames)
    video_thread.start()

    video_thread.join()

    cap.release()
    cv2.destroyAllWindows()

video_path = "Better test.mp4"
detect_lanes(video_path)
"""