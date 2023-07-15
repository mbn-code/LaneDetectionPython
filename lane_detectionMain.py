import cv2
import numpy as np
import time

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_lines(image, lines, color=(0, 0, 255), thickness=2):
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def detect_lanes(image, edges):
    height, width = edges.shape[:2]
    region_of_interest_vertices = [
        (0, height),
        (width // 2, height // 2),
        (width, height)
    ]
    masked_edges = region_of_interest(edges, np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=100
    )

    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        if slope < 0:
            left_lines.append(line[0])
        else:
            right_lines.append(line[0])

    left_lines = np.array(left_lines)
    right_lines = np.array(right_lines)

    left_lane = np.mean(left_lines, axis=0, dtype=np.int32) if len(left_lines) > 0 else np.zeros(4, dtype=np.int32)
    right_lane = np.mean(right_lines, axis=0, dtype=np.int32) if len(right_lines) > 0 else np.zeros(4, dtype=np.int32)

    return np.array([left_lane, right_lane])

# Define the behavior labels
behavior_labels = {
    "Left": "Turn Left",
    "Right": "Turn Right",
    "Center": "Stay Center",
    "Small Correction Left": "Small Correction Left",
    "Small Correction Right": "Small Correction Right"
}

# Video capture
cap = cv2.VideoCapture("/Users/mbn/Documents/Programmering/python3/LaneDetectionPython/Lane detect test data.mp4")

# Initialize behavior update variables
behavior_update_interval = 0.5  # 0.5 seconds
last_behavior_update_time = time.time()
current_behavior = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    edges = preprocess_image(frame)
    lanes = detect_lanes(frame, edges)
    
    # Create a blank image to draw the detected lanes
    lane_image = np.zeros_like(frame)
    draw_lines(lane_image, lanes)
    
    # Combine the original frame with the lane image
    result = cv2.addWeighted(frame, 0.8, lane_image, 1, 0)
    
    # Update behavior label if the interval has passed
    current_time = time.time()
    if current_time - last_behavior_update_time >= behavior_update_interval:
        if len(lanes) > 0:
            x1_left, _, x2_left, _ = lanes[0]
            x1_right, _, x2_right, _ = lanes[1]

            center = frame.shape[1] // 2
            if x1_left < center and x2_left < center and x1_right < center and x2_right < center:
                behavior = behavior_labels["Left"]
            elif x1_left > center and x2_left > center and x1_right > center and x2_right > center:
                behavior = behavior_labels["Right"]
            elif center - 50 < x1_left < center + 50 and center - 50 < x2_left < center + 50 and center - 50 < x1_right < center + 50 and center - 50 < x2_right < center + 50:
                behavior = behavior_labels["Center"]
            elif x1_left < center and x2_left > center and x1_right < center and x2_right > center:
                behavior = behavior_labels["Small Correction Left"]
            elif x1_left > center and x2_left < center and x1_right > center and x2_right < center:
                behavior = behavior_labels["Small Correction Right"]
        else:
            behavior = "No Lanes Detected"

        current_behavior = current_behavior
        last_behavior_update_time = current_time


    # Display the suggested behavior in the top left corner
    cv2.putText(result, current_behavior, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Lane Detection", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
