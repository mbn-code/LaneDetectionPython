import cv2
import numpy as np

# Number of frames to wait before updating the lane line
DELAY_FRAMES = 10

# Initialize empty lists for left and right lane lines
left_lines = []
right_lines = []

# Counter to keep track of consecutive frames with similar lane line positions
frame_counter = 0

# Variable to store the latest lane mask
latest_lane_mask = None

def check_overlap(left_line, right_line, threshold=100):
    """
    Check if the left and right lines overlap by comparing the x-coordinates at a specific y-coordinate.
    If the difference between the x-coordinates is less than the threshold, the lines are considered overlapping.
    """
    y = left_line[0, 1]  # Use a specific y-coordinate to check for overlapping
    left_x = left_line[0, 0]
    right_x = right_line[0, 0]
    return abs(left_x - right_x) < threshold, y

def estimate_lines(y, prev_left_line, prev_right_line):
    """
    Estimate the left and right lines using the previous non-overlapping lines and the given y-coordinate.
    The estimated lines are created by copying and flipping the previous lines.
    """
    left_x = prev_left_line[0, 0]
    right_x = prev_right_line[0, 0]
    estimated_left_line = np.array([[left_x, y, left_x, y]], dtype=np.int32)
    estimated_right_line = np.array([[right_x, y, right_x, y]], dtype=np.int32)
    return estimated_left_line, estimated_right_line

def lane_detection(image):
    global left_lines, right_lines, frame_counter, latest_lane_mask

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
        (width // 2, height // 2),
        (width, height)
    ]
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Apply Hough line transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

    # Check if lines are detected
    if lines is not None:
        # Separate the left and right lane points
        left_points = []
        right_points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.5:
                continue
            if slope < 0:
                left_points.append((x1, y1))
                left_points.append((x2, y2))
            else:
                right_points.append((x1, y1))
                right_points.append((x2, y2))

        # Add new lane lines to the list
        left_lines.append(left_points)
        right_lines.append(right_points)

        # Limit the list size to avoid memory issues
        if len(left_lines) > 100:
            left_lines = left_lines[-100:]
        if len(right_lines) > 100:
            right_lines = right_lines[-100:]

        # Combine the lane points from multiple frames
        combined_left_points = [point for sublist in left_lines for point in sublist]
        combined_right_points = [point for sublist in right_lines for point in sublist]

        # Fit polynomial curves to the combined left and right lane points
        if len(combined_left_points) > 0:
            combined_left_points = np.array(combined_left_points)
            left_coeffs = np.polyfit(combined_left_points[:, 1], combined_left_points[:, 0], deg=2)
            left_line = np.poly1d(left_coeffs)
        else:
            return image

        if len(combined_right_points) > 0:
            combined_right_points = np.array(combined_right_points)
            right_coeffs = np.polyfit(combined_right_points[:, 1], combined_right_points[:, 0], deg=2)
            right_line = np.poly1d(right_coeffs)
        else:
            return image

        # Generate x-coordinates for the curves
        plot_y = np.linspace(height, height // 2, num=100)
        left_fit_x = left_line(plot_y)
        right_fit_x = right_line(plot_y)

        # Delay the update of the lane line if the positions change too quickly
        if frame_counter >= DELAY_FRAMES or len(left_lines) < DELAY_FRAMES:
            # Check if the left and right lines overlap
            overlap, y = check_overlap(np.array([[left_fit_x[0], plot_y[0], left_fit_x[-1], plot_y[-1]]], dtype=np.int32),
                                       np.array([[right_fit_x[0], plot_y[0], right_fit_x[-1], plot_y[-1]]], dtype=np.int32))
            if overlap:
                # Determine the overlapping region
                left_end_x = left_fit_x[-1]
                right_end_x = right_fit_x[-1]
                overlap_x = max(0, min(left_end_x, right_end_x))
                overlap_y = y
                overlap_index = np.where(plot_y >= overlap_y)[0][0]

                # Cut off the lines at the overlapping region
                left_fit_x[overlap_index:] = overlap_x
                right_fit_x[overlap_index:] = overlap_x

            # Create a mask for the lane area
            lane_mask = np.zeros_like(image)
            left_pts = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
            right_pts = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
            pts = np.hstack((left_pts, right_pts))
            cv2.fillPoly(lane_mask, np.int_([pts]), (0, 255, 0))

            # Overlay the lane area on the original image
            result = cv2.addWeighted(image, 0.8, lane_mask, 0.4, 0)

            # Reset the frame counter
            frame_counter = 0

            # Store the latest lane mask
            latest_lane_mask = lane_mask

            return result
        else:
            # Increment the frame counter
            frame_counter += 1

            # Display the latest lane mask during the delay period
            result = cv2.addWeighted(image, 0.8, latest_lane_mask, 0.4, 0)

            return result
    else:
        # Reset the frame counter
        frame_counter = 0

        return image


# Read the input video
video = cv2.VideoCapture("/Users/mbn/Documents/Programmering/python3/LaneDetectionPython/Better test.mp4")

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
