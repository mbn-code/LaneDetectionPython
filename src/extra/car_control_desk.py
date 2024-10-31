import cv2
import numpy as np
import mss
from PIL import Image
import pyautogui
import time

# Define the car control function
def control_car(lane_center, frame_width, left_lines_count, right_lines_count, speed, car_detected):
    car_center = frame_width // 2
    deviation = lane_center - car_center

    if car_detected:
        if left_lines_count < right_lines_count:
            direction = "Right"
            key = 'd'  # Press 'D' for right
            key = 'd'  # Press 'D' for right

            delay = 0.05  # Decrease delay for faster turns
        else:
            direction = "Left"
            key = 'a'  # Press 'A' for left
            key = 'a'  # Press 'A' for left

            delay = 0.05  # Decrease delay for faster turns
    else:
        if left_lines_count >= 2 and right_lines_count >= 2:
            direction = "Forward"
            key = 'w'  # Press 'W' for forward
        elif deviation > -50:
            direction = "Left"
            key = 'a'  # Press 'A' for left
        elif deviation < 50:
            direction = "Right"
            key = 'd'  # Press 'D' for right
        else:
            direction = "Forward"
            key = 'w'  # Press 'W' for forward
        delay = 0.1  # Set a higher delay for smoother straight driving

    # Press the corresponding key
    pyautogui.keyDown(key)

    # Adjust delay between key presses based on speed
    if speed > 0:
        if speed != float('inf'):  # Check for infinite speed
            delay /= speed if speed != 0 else float('inf')
        else:
            delay = 0.01  # Set a small delay for infinite speed
    time.sleep(delay)

    return direction, key


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def detect_lanes():
    with mss.mss() as sct:
        # Define monitor coordinates to capture the top left corner of the screen
        monitor = {"top": 0, "left": 0, "width": 800, "height": 600}

        speed = 1  # Initial car speed (adjust as needed)
        car_cascade = cv2.CascadeClassifier('/Users/mbn/Documents/Programmering/python3/LaneDetectionPython/car_cas.xml')

        while True:
            # Capture the screen image
            img = np.array(sct.grab(monitor))

            # Convert PIL image to OpenCV format
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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

            # Detect cars using the Haar cascade classifier
            cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

            if len(cars) > 0:
                car_detected = True
            else:
                car_detected = False

            if car_detected:
                cv2.putText(frame, "Car Detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

            if lines is not None:
                lane_center = 0
                left_lines_count = 0
                right_lines_count = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]

                    # Exclude horizontal lines (lines with slopes close to 0)
                    slope = (y2 - y1) / (x2 - x1)
                    if abs(slope) < 0.3:  # Adjust the slope threshold as needed
                        continue

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
                else:
                    # Reset the lane center to the car center if there are not enough lines on both sides
                    lane_center = width // 2

                # Control the car based on the lane center, line counts, and car detection
                direction, key = control_car(lane_center, width, left_lines_count, right_lines_count, speed, car_detected)
            
                cv2.putText(frame, f"Direction: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Render ROI on the frame
            roi_overlay = np.zeros_like(frame)
            cv2.fillPoly(roi_overlay, roi_vertices, (0, 255, 0))
            frame_with_roi = cv2.addWeighted(frame, 1, roi_overlay, 0.3, 0)

            cv2.imshow('Lane Detection', frame_with_roi)

            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()

detect_lanes()
