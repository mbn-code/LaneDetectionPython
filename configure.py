import os
import shutil

def create_directory_structure():
    # Define the directory structure
    directories = [
        'src',
        'models',
        'includes',
        'data',
        'config',
        'tests'
    ]
    
    # Create base directories
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}/")

    # Define file movements
    file_structure = {
        'src/': [
            'car_control_car_detect.py',
            'car_control_desk.py',
            'car_control_high_def.py',
            'Car_Control.py',
            'car_detection.py',
            'lane_center.py',
            'lane_detectionMain.py',
            'LaneDetectionPython/lane_center.py'
        ],
        'models/': [
            'frozen_inference_graph.pb',
            'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
        ],
        'config/': [
            'yolov2.cfg',
            'yolov3-tiny.cfg',
            'coco.names'
        ],
        'data/': [
            'car_cas.xml'
        ]
    }

    # Move files to their new locations
    for directory, files in file_structure.items():
        for file in files:
            if os.path.exists(file):
                try:
                    shutil.move(file, os.path.join(directory, os.path.basename(file)))
                    print(f"Moved {file} to {directory}")
                except Exception as e:
                    print(f"Error moving {file}: {str(e)}")

    # Remove empty LaneDetectionPython directory if it exists
    if os.path.exists('LaneDetectionPython') and not os.listdir('LaneDetectionPython'):
        os.rmdir('LaneDetectionPython')
        print("Removed empty LaneDetectionPython directory")

if __name__ == "__main__":
    print("Starting workspace reorganization...")
    create_directory_structure()
    print("\nNew directory structure created:")
    print("""
    project/
    ├── src/                 # Source code files
    ├── models/             # Model files (.pb, .pbtxt)
    ├── includes/           # Header files or included libraries
    ├── data/              # Data files (xml, etc.)
    ├── config/            # Configuration files
    ├── tests/             # Test files
    └── README.md
    """)