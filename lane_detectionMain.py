import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 0, 255), thickness=3):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def calculate_center_line(lines, height, width):
    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                if slope < -0.3 and intercept > width / 2:
                    left_lines.append(line)
                elif slope > 0.3 and intercept < width / 2:
                    right_lines.append(line)

    left_slope, left_intercept = calculate_average_slope_intercept(left_lines)
    right_slope, right_intercept = calculate_average_slope_intercept(right_lines)

    y1 = height
    y2 = int(height * 0.6)

    left_x1 = int((y1 - left_intercept) / left_slope) if left_slope != 0 else int(width / 2)
    left_x2 = int((y2 - left_intercept) / left_slope) if left_slope != 0 else int(width / 2)
    right_x1 = int((y1 - right_intercept) / right_slope) if right_slope != 0 else int(width / 2)
    right_x2 = int((y2 - right_intercept) / right_slope) if right_slope != 0 else int(width / 2)

    return [(left_x1, y1, left_x2, y2), (right_x1, y1, right_x2, y2)]

def calculate_average_slope_intercept(lines):
    slopes = []
    intercepts = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            slopes.append(slope)
            intercepts.append(intercept)

    average_slope = np.mean(slopes) if slopes else 0
    average_intercept = np.mean(intercepts) if intercepts else 0

    return average_slope, average_intercept

def lane_detection(image):
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    edges = cv2.Canny(gray, 100, 200)

    height, width = edges.shape[:2]
    vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=150, maxLineGap=50)

    line_image = np.zeros_like(image)
    draw_lines(line_image, lines)

    center_line = calculate_center_line(lines, height, width)
    cv2.line(line_image, center_line[0][:2], center_line[0][2:], (0, 255, 0), 5)
    cv2.line(line_image, center_line[1][:2], center_line[1][2:], (0, 255, 0), 5)

    lane_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return lane_image

# Read the input video
cap = cv2.VideoCapture('/Users/mbn/Documents/Programmering/python3/LaneDetectionPython/Better test.mp4')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    lane_frame = lane_detection(frame)
    cv2.imshow('Lane Detection', lane_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
