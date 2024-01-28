import cv2
import numpy as np

# Function to make the car stay in the center and render a dynamic 3D dotted line# Function to make the car stay in the center and render a dynamic 3D dotted line
def stay_in_center(frame, smoothed_dot_positions):
    height, width = frame.shape[:2]
    target_position = width // 2

    # Check if smoothed_dot_positions is empty
    if len(smoothed_dot_positions) == 0:
        car_direction = "No Lane Detected"
        curvature = 0
    else:
        # Calculate the car positioning based on the average dot position
        car_position = np.mean(smoothed_dot_positions[:, 0])
        frame_center = width // 2

        # Calculate the steering angle and determine the car movement label
        steering_angle = car_position - frame_center

        # Adjust the steering angle threshold for better predictions
        if abs(steering_angle) > 80:
            if steering_angle < 0:
                car_direction = "Turn Right"
            else:
                car_direction = "Turn Left"
        elif abs(steering_angle) > 20:
            if steering_angle < 0:
                car_direction = "Small Correction Right"
            else:
                car_direction = "Small Correction Left"
        else:
            # Check if both lanes are detected
            if len(smoothed_dot_positions) > 50:
                car_direction = "Go Straight"
            else:
                car_direction = "Stay Center"

        # Calculate the curvature of the line
        curvature = car_position - target_position

    # Write the car movement label on the frame
    cv2.putText(frame, car_direction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame, curvature


def main():
    # Read the video
    video_path = "Better test.mp4"
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object to save the output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter("line_detection_output.mp4", fourcc, fps, (frame_width, frame_height))

    # Define smoothing parameters
    smoothing_factor = 1
    prev_dot_positions = None

    # Define the region of interest (ROI) polygon vertices
    roi_vertices = np.array([[(0, frame_height), (frame_width * 0.7, frame_height * 0.1),
                              (frame_width * 0.7, frame_height * 0.1), (frame_width, frame_height)]],
                            dtype=np.int32)

    while True:
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

        # Calculate the dot positions
        dot_positions = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                if abs(slope) < 1.0 and abs(slope) > 0.1:
                    dx = (x2 - x1) / 10
                    dy = (y2 - y1) / 10
                    dot_positions.extend([(x1 + i * dx, y1 + i * dy) for i in range(10)])
            dot_positions = np.array(dot_positions)

        # Smoothing the dot movement
        if prev_dot_positions is None or prev_dot_positions.shape[0] != dot_positions.shape[0]:
            smoothed_dot_positions = dot_positions
        else:
            smoothed_dot_positions = smoothing_factor * dot_positions + (1 - smoothing_factor) * prev_dot_positions

        # Update the previous dot positions
        prev_dot_positions = smoothed_dot_positions

        # Make the car stay in the center and render the dynamic 3D dotted line
        processed_frame, curvature = stay_in_center(frame, smoothed_dot_positions)

        # Render lines between the most dominant detections
        if smoothed_dot_positions.shape[0] > 1:
            sorted_indices = np.argsort(smoothed_dot_positions[:, 1])
            sorted_positions = smoothed_dot_positions[sorted_indices]
            for i in range(sorted_positions.shape[0] - 1):
                start_point = (int(sorted_positions[i][0]), int(sorted_positions[i][1]))
                end_point = (int(sorted_positions[i+1][0]), int(sorted_positions[i+1][1]))
                cv2.line(processed_frame, start_point, end_point, (255, 0, 0), 2)


        # Display the frame with the dynamic 3D dotted line
        cv2.imshow("Line Detection", processed_frame)

        # Display the calculated curvature
        print("Curvature:", curvature)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
