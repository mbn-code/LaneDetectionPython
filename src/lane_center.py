import cv2
import numpy as np
import os

# Constants
SMOOTHING_FACTOR = 1
THRESHOLD = 100
MIN_LINE_LENGTH = 100
MAX_LINE_GAP = 10
ROI_VERTICES_RATIO = [(0, 1), (0.7, 0.1), (0.7, 0.1), (1, 1)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_POSITION = (20, 50)
TEXT_COLOR = (0, 0, 255)
LINE_COLOR = (255, 0, 0)
LINE_THICKNESS = 2

# Add new constants
OVERLAY_ALPHA = 0.3
DASHBOARD_HEIGHT = 100
DASHBOARD_COLOR = (20, 20, 20)
DASHBOARD_ALPHA = 0.8
SUCCESS_COLOR = (0, 255, 0)
WARNING_COLOR = (0, 165, 255)
DANGER_COLOR = (0, 0, 255)

def create_dashboard(frame, car_direction, curvature):
    """Create a semi-transparent dashboard overlay"""
    height, width = frame.shape[:2]
    dashboard = np.zeros((DASHBOARD_HEIGHT, width, 3), dtype=np.uint8)
    dashboard[:] = DASHBOARD_COLOR
    
    # Add direction indicator
    direction_color = SUCCESS_COLOR if abs(curvature) < 10 else WARNING_COLOR if abs(curvature) < 50 else DANGER_COLOR
    cv2.putText(dashboard, car_direction, (20, 60), FONT, 1, direction_color, 2)
    
    # Add curvature meter
    meter_width = 200
    meter_height = 20
    meter_x = width - meter_width - 20
    meter_y = 40
    cv2.rectangle(dashboard, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), (50, 50, 50), -1)
    center_x = meter_x + meter_width // 2
    indicator_x = int(center_x + (curvature / 100) * (meter_width // 2))
    indicator_x = max(meter_x, min(meter_x + meter_width, indicator_x))
    cv2.circle(dashboard, (indicator_x, meter_y + meter_height // 2), 10, direction_color, -1)
    
    return dashboard

def enhance_visualization(frame, smoothed_dot_positions):
    """Create enhanced visualization with lane overlay"""
    height, width = frame.shape[:2]
    overlay = frame.copy()
    
    if len(smoothed_dot_positions) > 1:
        # Create lane polygon
        sorted_indices = np.argsort(smoothed_dot_positions[:, 1])
        sorted_positions = smoothed_dot_positions[sorted_indices]
        points = np.int32(sorted_positions)
        
        # Add mirrored points to create a polygon
        center_line = np.mean(points, axis=0)[0]
        left_points = points[points[:, 0] < center_line]
        right_points = points[points[:, 0] >= center_line]
        
        if len(left_points) > 0 and len(right_points) > 0:
            lane_points = np.vstack((left_points, right_points[::-1]))
            cv2.fillPoly(overlay, [np.int32(lane_points)], (0, 255, 0))
            
    # Blend overlay with original frame
    frame = cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0)
    return frame

def stay_in_center(frame, smoothed_dot_positions):
    """
    Make the car stay in the center and render a dynamic 3D dotted line.
    """
    height, width = frame.shape[:2]
    target_position = width // 2

    if len(smoothed_dot_positions) == 0:
        car_direction = "No Lane Detected"
        curvature = 0
    else:
        car_position = np.mean(smoothed_dot_positions[:, 0])
        frame_center = width // 2
        steering_angle = car_position - frame_center

        if abs(steering_angle) > 50:
            car_direction = "↺ Turn Right" if steering_angle < 0 else "Turn Left ↻"
        elif abs(steering_angle) > 10:
            car_direction = "→ Small Right" if steering_angle < 0 else "Small Left ←"
        else:
            car_direction = "▲ Straight Ahead" if len(smoothed_dot_positions) > 50 else "● Center"

        curvature = car_position - target_position

    # Enhanced visualization
    frame = enhance_visualization(frame, smoothed_dot_positions)
    dashboard = create_dashboard(frame, car_direction, curvature)

    # Add dashboard to frame
    frame[-DASHBOARD_HEIGHT:] = cv2.addWeighted(
        frame[-DASHBOARD_HEIGHT:], 1 - DASHBOARD_ALPHA,
        dashboard, DASHBOARD_ALPHA, 0
    )

    return frame, curvature

def preprocess_frame(frame, roi_vertices):
    """
    Preprocess the frame to detect edges and apply ROI mask.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, roi_mask)
    return masked_edges

def calculate_dot_positions(lines):
    """
    Calculate the dot positions from the detected lines.
    """
    dot_positions = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if 0.1 < abs(slope) < 1.0:
                dx = (x2 - x1) / 10
                dy = (y2 - y1) / 10
                dot_positions.extend([(x1 + i * dx, y1 + i * dy) for i in range(10)])
        dot_positions = np.array(dot_positions)
    return dot_positions

def main():
    video_path = "videos/Better test.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter("line_detection_output.mp4", fourcc, fps, (frame_width, frame_height))

    prev_dot_positions = None
    roi_vertices = np.array([[(int(x * frame_width), int(y * frame_height)) for x, y in ROI_VERTICES_RATIO]], dtype=np.int32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        masked_edges = preprocess_frame(frame, roi_vertices)
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, THRESHOLD, minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
        dot_positions = calculate_dot_positions(lines)

        if prev_dot_positions is None or prev_dot_positions.shape[0] != dot_positions.shape[0]:
            smoothed_dot_positions = dot_positions
        else:
            smoothed_dot_positions = SMOOTHING_FACTOR * dot_positions + (1 - SMOOTHING_FACTOR) * prev_dot_positions

        prev_dot_positions = smoothed_dot_positions
        processed_frame, curvature = stay_in_center(frame, smoothed_dot_positions)

        if smoothed_dot_positions.shape[0] > 1:
            sorted_indices = np.argsort(smoothed_dot_positions[:, 1])
            sorted_positions = smoothed_dot_positions[sorted_indices]
            for i in range(sorted_positions.shape[0] - 1):
                start_point = (int(sorted_positions[i][0]), int(sorted_positions[i][1]))
                end_point = (int(sorted_positions[i+1][0]), int(sorted_positions[i+1][1]))
                cv2.line(processed_frame, start_point, end_point, LINE_COLOR, LINE_THICKNESS)

        output_video.write(processed_frame)
        cv2.imshow("Lane Detection System", processed_frame)
        print("Curvature:", curvature)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
