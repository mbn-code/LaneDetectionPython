import cv2
import numpy as np
import time

# Read the video
video_path = "Better test.mp4"
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object to save the output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter("line_detection_output.mp4", fourcc, fps, (frame_width, frame_height))

# Define the region of interest (ROI) polygon vertices
roi_vertices = np.array([[(0, frame_height), (frame_width * 0.7, frame_height * 0.1),
                          (frame_width * 0.6, frame_height * 0.30), (frame_width, frame_height)]],
                        dtype=np.int32)

# Define the number of dots to draw
num_dots = 5

# Define the dot radius
dot_radius = 3

# Define smoothing parameters
smoothing_factor = 0.03
prev_dot_position = None

# Define steering parameters
steering_threshold = 250

# Define reset duration (in seconds)
reset_duration = 2

# Initialize the timer variables
start_time = time.time()
reset_timer = 0

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Apply ROI mask
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, roi_mask)

    # Perform Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Calculate the average dot position
    dot_positions = []
    if lines is not None:
        if lines.shape[0] > 1:
            avg_x = np.mean(lines[:, :, [0, 2]])
            avg_y = np.mean(lines[:, :, [1, 3]])
            dot_positions = np.column_stack((avg_x, avg_y))
            dot_positions = dot_positions.reshape((-1, 2))
        else:
            line = lines[0]
            x1, y1, x2, y2 = line[0]
            dx = (x2 - x1) / (num_dots - 1)
            dy = (y2 - y1) / (num_dots - 1)
            dot_positions = [(x1 + i * dx, y1 + i * dy) for i in range(num_dots)]
            dot_positions = np.array(dot_positions)

    # Reset average dot position if timer exceeds reset duration
    elapsed_time = time.time() - start_time
    if elapsed_time >= reset_timer:
        prev_dot_position = None
        start_time = time.time()
        reset_timer = np.random.uniform(low=reset_duration, high=reset_duration + 1)

        # Add label indicating the average reset
        cv2.putText(frame, "Average Reset", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Smoothing the dot movement
    if prev_dot_position is None:
        smoothed_dot_positions = dot_positions
    else:
        smoothed_dot_positions = smoothing_factor * dot_positions + (1 - smoothing_factor) * prev_dot_position

    # Update the previous dot position
    prev_dot_position = smoothed_dot_positions

    # Draw the smoothed dot positions
    for dot_position in smoothed_dot_positions:
        cv2.circle(frame, tuple(map(int, dot_position)), radius=dot_radius, color=(0, 255, 0), thickness=-1)

    # Calculate car positioning based on the average dot position
    car_position = np.mean(smoothed_dot_positions, axis=0)[0]
    frame_center = frame_width // 2

    # Calculate steering angle and small correction
    steering_angle = car_position - frame_center

    if abs(steering_angle) > steering_threshold:
        if steering_angle < 0:
            car_direction = "Turn Right"
        else:
            car_direction = "Turn Left"
    else:
        if steering_angle < 0:
            car_direction = "Small Correction Right"
        else:
            car_direction = "Small Correction Left"

    # Write the car positioning information on the frame
    cv2.putText(frame, car_direction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # Display the frame with the detected dots
    cv2.imshow("Line Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and output video objects
cap.release()
output_video.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
