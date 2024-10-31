import cv2
import numpy as np
import time
from typing import Optional, Tuple

VIDEO_PATH = "videos/Better test.mp4"
OUTPUT_PATH = "videos/line_detection_output.mp4"
FRAME_WIDTH_OFFSET = 0.7
FRAME_HEIGHT_OFFSET_1 = 0.1
FRAME_HEIGHT_OFFSET_2 = 0.30
DOT_RADIUS = 3
NUM_DOTS = 5
SMOOTHING_FACTOR = 0.03
STEERING_THRESHOLD = 250
RESET_DURATION = 1


def initialize_video_capture(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {path}")
    return cap


def initialize_video_writer(cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


def get_roi_vertices(frame_width: int, frame_height: int) -> np.ndarray:
    return np.array([[
        (0, frame_height),
        (int(frame_width * FRAME_WIDTH_OFFSET), int(frame_height * FRAME_HEIGHT_OFFSET_1)),
        (int(frame_width * 0.6), int(frame_height * FRAME_HEIGHT_OFFSET_2)),
        (frame_width, frame_height)
    ]], dtype=np.int32)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)


def apply_roi(edges: np.ndarray, roi_vertices: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    return cv2.bitwise_and(edges, mask)


def draw_dots(frame: np.ndarray, lines: Optional[np.ndarray], color: Tuple[int, int, int]):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.circle(frame, (x1, y1), DOT_RADIUS, color, -1)
            cv2.circle(frame, (x2, y2), DOT_RADIUS, color, -1)


def calculate_dot_positions(lines: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if lines is None:
        return None
    if lines.shape[0] >= 1:
        avg_x = np.mean(lines[:, :, [0, 2]], axis=1)
        avg_y = np.mean(lines[:, :, [1, 3]], axis=1)
        dot_x = np.linspace(avg_x.min(), avg_x.max(), NUM_DOTS)
        dot_y = np.linspace(avg_y.min(), avg_y.max(), NUM_DOTS)
        return np.column_stack((dot_x, dot_y))


def main():
    cap = initialize_video_capture(VIDEO_PATH)
    output_video = initialize_video_writer(cap, OUTPUT_PATH)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_vertices = get_roi_vertices(frame_width, frame_height)

    prev_dot_position: Optional[np.ndarray] = None
    start_time = time.time()
    reset_timer = RESET_DURATION

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        edges = preprocess_frame(frame)
        masked_edges = apply_roi(edges, roi_vertices)
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

        draw_dots(frame, lines, color=(0, 0, 255))

        dot_positions = calculate_dot_positions(lines)

        elapsed_time = time.time() - start_time
        if elapsed_time >= reset_timer:
            prev_dot_position = None
            start_time = time.time()
            reset_timer = np.random.uniform(RESET_DURATION, RESET_DURATION + 1)
            cv2.putText(frame, "Average Reset", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if dot_positions is not None:
            if prev_dot_position is not None:
                dot_positions = SMOOTHING_FACTOR * dot_positions + (1 - SMOOTHING_FACTOR) * prev_dot_position
            smoothed_dot_positions = dot_positions
            prev_dot_position = smoothed_dot_positions
            for pos in smoothed_dot_positions:
                cv2.circle(frame, tuple(map(int, pos)), DOT_RADIUS, (0, 255, 0), -1)

            car_position = np.mean(smoothed_dot_positions, axis=0)[0]
            frame_center = frame_width // 2
            steering_angle = car_position - frame_center

            if abs(steering_angle) > STEERING_THRESHOLD:
                direction = "Turn Right" if steering_angle < 0 else "Turn Left"
            else:
                direction = "Small Correction Right" if steering_angle < 0 else "Small Correction Left"

            cv2.putText(frame, direction, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.polylines(frame, [roi_vertices], isClosed=True, color=(255, 255, 255), thickness=2)
        cv2.imshow("Line Detection", frame)
        output_video.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
