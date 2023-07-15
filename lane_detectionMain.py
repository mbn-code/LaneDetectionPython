import cv2
import numpy as np
import mss

def detect_lanes(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Define a region of interest (ROI) mask
    height, width = frame.shape[:2]
    roi_vertices = [(0, height), (width // 2, height // 2), (width, height)]
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([roi_vertices], np.int32), 255)

    # Apply the mask to the edges image
    masked_edges = cv2.bitwise_and(edges, mask)

    # Apply Hough line detection
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

    # Draw detected lanes on the frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return frame

while True:
    # Capture the desktop screen
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Change index if needed
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)

    # Convert RGB to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the frame
    cv2.imshow('Lane Detection', frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('k'):
        # Detect lanes
        frame = detect_lanes(frame)
        cv2.imshow('Lane Detection', frame)

    # Exit loop if 'q' key is pressed
    if key == ord('q'):
        break

# Destroy windows
cv2.destroyAllWindows()
