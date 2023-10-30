import cv2
import numpy as np

# Define the car control function
def control_car(lane_center, frame_width, left_lines_count, right_lines_count, car_positions):
    car_center = frame_width // 2
    deviation = lane_center - car_center

    if car_positions:
        # Check the position of detected cars
        for (x, y) in car_positions:
            if x < frame_width // 2:
                # Car on the left side
                if y < frame_width // 3:
                    # Car is approaching from the top left
                    label = "Car approaching from top left"
                else:
                    # Car is approaching from the left
                    label = "Car approaching from left"
            else:
                # Car on the right side
                if y < frame_width // 3:
                    # Car is approaching from the top right
                    label = "Car approaching from top right"
                else:
                    # Car is approaching from the right
                    label = "Car approaching from right"
    elif left_lines_count >= 2 and right_lines_count >= 2:
        # Move forward
        # Implement the logic to keep the car moving forward
        label = "Move forward"
    elif deviation > -50:
        # Turn left
        # Implement the logic to steer the car left
        label = "Turn left"
    elif deviation < 50:
        # Turn right
        # Implement the logic to steer the car right
        label = "Turn right"
    else:
        # Move forward (fallback option)
        label = "Move forward"

    return label

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def estimate_lanes(lines):
    # Implement the logic to estimate lanes based on previous lane lines
    # Return the estimated lane lines

    # Placeholder implementation: return the input lines as is
    return lines

def detect_lanes(video_path):
    cap = cv2.VideoCapture(video_path)

    # Initialize previous lanes as None
    previous_lanes = None

    # Load car detection cascade classifier
    car_cascade = cv2.CascadeClassifier("car_cas.xml")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

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

        # Expand ROI area for car detection
        expanded_roi_bottom_width = roi_bottom_width * 1.2
        expanded_roi_top_width = roi_top_width * 1.2
        expanded_roi_height = roi_height * 1.2
        expanded_roi_vertices = np.array([
            [(roi_bottom_width - expanded_roi_bottom_width / 2, height),
             (roi_bottom_width + expanded_roi_bottom_width / 2, expanded_roi_height),
             (width - roi_bottom_width - expanded_roi_bottom_width / 2, expanded_roi_height),
             (width - roi_bottom_width + expanded_roi_bottom_width / 2, height)]
        ], dtype=np.int32)
        expanded_roi_gray_frame = gray[expanded_roi_vertices[0][1][1]:expanded_roi_vertices[0][0][1],
                                       expanded_roi_vertices[0][0][0]:expanded_roi_vertices[0][3][0]]
        cars = car_cascade.detectMultiScale(expanded_roi_gray_frame, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

        # Extract car positions
        car_positions = []
        for (x, y, w, h) in cars:
            car_x = x + expanded_roi_vertices[0][0][0] + w // 2
            car_y = y + expanded_roi_vertices[0][1][1] + h // 2
            car_positions.append((car_x, car_y))

        lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

        if lines is not None:
            lane_center = 0
            left_lines_count = 0
            right_lines_count = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
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
                # Update the previous lanes with the current lanes
                previous_lanes = lines
            elif previous_lanes is not None:
                # Estimate the lanes based on previous lanes
                estimated_lanes = estimate_lanes(previous_lanes)
                if estimated_lanes is not None:
                    lane_center = np.mean([((line[0][0] + line[0][2]) // 2) for line in estimated_lanes])
                    # Update the previous lanes with the estimated lanes
                    previous_lanes = estimated_lanes
                else:
                    # Reset the lane center to the car center if no estimated lanes are available
                    lane_center = width // 2
            else:
                # Reset the lane center to the car center if there are not enough lines on both sides and no previous lanes
                lane_center = width // 2

            # Control the car based on the lane center, line counts, and car positions
            control_label = control_car(lane_center, width, left_lines_count, right_lines_count, car_positions)

            # Draw control label on the frame
            cv2.putText(frame, control_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Render ROI on the frame
        roi_overlay = np.zeros_like(frame)
        cv2.fillPoly(roi_overlay, roi_vertices, (0, 255, 0))
        frame_with_roi = cv2.addWeighted(frame, 1, roi_overlay, 0.3, 0)

        # Draw car label on the frame
        car_label = "Car Detected" if len(car_positions) > 0 else "No Car Detected"
        cv2.putText(frame_with_roi, car_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Lane Detection', frame_with_roi)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

video_path = "Better test.mp4"
detect_lanes(video_path)
