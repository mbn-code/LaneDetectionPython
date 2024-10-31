import cv2
import numpy as np

def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Yellow mask
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # White mask
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Define ROI
    height, width = frame.shape[:2]
    mask_roi = np.zeros_like(mask_yellow)
    polygon = np.array([[
        (int(0.1 * width), height),
        (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ]], np.int32)
    cv2.fillPoly(mask_roi, polygon, 255)
    
    # Apply ROI to masks
    mask_yellow = cv2.bitwise_and(mask_yellow, mask_roi)
    mask_white = cv2.bitwise_and(mask_white, mask_roi)
    
    # Edge detection for yellow
    edges_yellow = cv2.Canny(mask_yellow, 50, 150)
    lines_yellow = cv2.HoughLinesP(edges_yellow, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    
    # Edge detection for white
    edges_white = cv2.Canny(mask_white, 50, 150)
    lines_white = cv2.HoughLinesP(edges_white, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    
    line_image = np.zeros_like(frame)
    
    if lines_yellow is not None:
        for line in lines_yellow:
            x1, y1, x2, y2 = line[0]
            color = (0, 255, 255)  # Yellow
            cv2.line(line_image, (x1, y1), (x2, y2), color, 5)
    
    if lines_white is not None:
        for line in lines_white:
            x1, y1, x2, y2 = line[0]
            color = (255, 255, 255)  # White
            cv2.line(line_image, (x1, y1), (x2, y2), color, 5)
    
    combined = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    return combined

def main():
    cap = cv2.VideoCapture('videos/lane_detect_2.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_lane_detection.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed = process_frame(frame)
        out.write(processed)
        cv2.imshow('Lane Detection', processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    def calculate_steering_angle(frame, lines):
        height, width, _ = frame.shape
        if lines is None:
            return 0  # No lines detected, go straight

        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

        if len(left_lines) == 0 or len(right_lines) == 0:
            return 0  # Not enough lines to determine direction

        left_line = np.mean(left_lines, axis=0).astype(int)
        right_line = np.mean(right_lines, axis=0).astype(int)

        mid_x = width // 2
        left_x2 = left_line[0][2]
        right_x2 = right_line[0][2]
        mid_lane_x = (left_x2 + right_x2) // 2

        steering_angle = np.arctan2(mid_lane_x - mid_x, height) * 180 / np.pi
        return steering_angle

    def draw_steering_line(frame, angle):
        height, width, _ = frame.shape
        mid_x = width // 2
        length = height // 2

        end_x = int(mid_x + length * np.tan(angle * np.pi / 180))
        end_y = height // 2

        cv2.line(frame, (mid_x, height), (end_x, end_y), (0, 0, 255), 5)
        return frame

    def main():
        cap = cv2.VideoCapture('videos/lane_detect_2.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_lane_detection.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed = process_frame(frame)
            lines = cv2.HoughLinesP(cv2.Canny(cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY), 50, 150), 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
            angle = calculate_steering_angle(frame, lines)
            processed = draw_steering_line(processed, angle)
            
            out.write(processed)
            cv2.imshow('Lane Detection', processed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    if __name__ == '__main__':
        main()
        def add_speedometer(frame, speed):
            height, width = frame.shape[:2]
            # Create speedometer position
            pos = (width - 150, height - 50)
            cv2.putText(frame, f'Speed: {speed:.1f} km/h', pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame

        def add_lane_departure_warning(frame, steering_angle):
            if abs(steering_angle) > 20:
                cv2.putText(frame, 'LANE DEPARTURE WARNING!', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

        def simulate_speed(steering_angle):
            # Simulate lower speed in sharp turns
            base_speed = 60
            return base_speed * (1 - abs(steering_angle) / 45)

        def night_vision_mode(frame):
            # Enhance frame visibility in dark conditions
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge((l,a,b))
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        def process_frame(frame):
            # Check if it's dark
            is_dark = np.mean(frame) < 100
            if is_dark:
                frame = night_vision_mode(frame)
            
            processed = process_frame(frame)
            lines = cv2.HoughLinesP(cv2.Canny(cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY), 
                                   50, 150), 1, np.pi/180, threshold=50, 
                                   minLineLength=100, maxLineGap=50)
            
            angle = calculate_steering_angle(frame, lines)
            speed = simulate_speed(angle)
            
            processed = draw_steering_line(processed, angle)
            processed = add_speedometer(processed, speed)
            processed = add_lane_departure_warning(processed, angle)
            
            return processed