import cv2
import numpy as np

def lane_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define a region of interest (ROI)
    height, width = edges.shape
    roi_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height)
    ]
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Apply Hough line transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

    # Check if lines are detected
    if lines is not None:
        # Calculate the slope and intercept of each line
        left_lines = []
        right_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

        # Calculate the average slope and intercept for left and right lanes
        if len(left_lines) > 0:
            left_slope, left_intercept = np.mean(left_lines, axis=0)
        else:
            left_slope, left_intercept = np.nan, np.nan

        if len(right_lines) > 0:
            right_slope, right_intercept = np.mean(right_lines, axis=0)
        else:
            right_slope, right_intercept = np.nan, np.nan

        # Calculate the coordinates of the lane lines
        y1 = height
        y2 = int(height / 2)
        left_x1 = int((y1 - left_intercept) / left_slope)
        left_x2 = int((y2 - left_intercept) / left_slope)
        right_x1 = int((y1 - right_intercept) / right_slope)
        right_x2 = int((y2 - right_intercept) / right_slope)

        # Draw the lane lines on the image
        lane_image = np.zeros_like(image)
        cv2.line(lane_image, (left_x1, y1), (left_x2, y2), (0, 255, 0), 5)
        cv2.line(lane_image, (right_x1, y1), (right_x2, y2), (0, 255, 0), 5)

        # Overlay the lane lines on the original image
        result = cv2.addWeighted(image, 0.8, lane_image, 1, 0)

        return result
    else:
        return image

# Read the input video
video = cv2.VideoCapture("Better test.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Perform lane detection on each frame
    result = lane_detection(frame)

    # Display the result
    cv2.imshow("Lane Detection", result)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()
