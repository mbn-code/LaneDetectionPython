# LaneDetectionPython

![Lane Detection Example](lane_detection_sample1.mp4)

LaneDetectionPython is a Python project for lane line detection and estimation. It uses popular libraries such as OpenCV (cv2), NumPy, and Matplotlib to process video data and detect lane lines in road scenes. This repository is a simple but effective way to get started with lane detection using computer vision techniques.

## Features

- Lane line detection in video data.
- Estimation of lane center and position.
- Sample video file (Lane detect test data.mp4) for testing the lane detection performance.

## Getting Started

These instructions will help you set up and run the project on your local machine.

### Prerequisites

Before you begin, make sure you have the following libraries installed:

- OpenCV (cv2)
- NumPy
- Matplotlib

You can install these libraries using pip:

```
pip install opencv-python numpy matplotlib
```

### Running the Lane Detection

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/LaneDetectionPython.git
   ```

2. Navigate to the project directory:

   ```bash
   cd LaneDetectionPython
   ```

3. Run the lane detection script:

   ```bash
   python lane_detectionMain.py
   ```

This will process the sample video file "Lane detect test data.mp4" and create an output video "lane_detection_output.mp4" with detected lane lines.

You can also use your own video files for lane detection by modifying the input file in the `lane_detectionMain.py` script.

## Testing

You can use the provided "Lane detect test data.mp4" video file to test the lane detection performance. Feel free to replace it with your own video files for testing.

## Contributing

If you'd like to contribute to this project, please open an issue or a pull request with your suggestions and improvements.

## Authors

- Your Name

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the OpenCV, NumPy, and Matplotlib communities for their fantastic libraries.

**Note:** Please update the "Authors" section with your name and any other relevant information.
