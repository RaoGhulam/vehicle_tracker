# Vehicle Tracker

A simple vehicle tracking project using YOLO and ByteTrack. Detects vehicles (cars, trucks, buses) in a video, assigns IDs, and counts them.

## Features

- Detects vehicles in video
- Assigns consistent track IDs across frames
- Counts vehicles
- Shows ID, class, and confidence on video
- Colorful bounding boxes

## Requirements

- Python 3.8+
- Packages listed in `requirements.txt`

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/RaoGhulam/vehicle_tracker.git
    cd vehicle_tracker
    ```

2. (Recommended) Create and activate a virtual environment:
    ```bash
    # Linux / Mac
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python3 -m venv venv
    venv\Scripts\activate
    ```
    You can skip this step if you want to install globally, but using a virtual environment is safer.

3. Install CPU-only PyTorch first (this avoids downloading large GPU packages):
    ```bash
    pip install torch==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    ```


4. Install the rest of the dependencies and package:
    ```bash
    pip install .
    ```

5. Run the tracker on a video:
    ```bash
    vehicle-tracker --video "path_or_url_to_video.mp4"
    ```

6. Optional: Uninstall the package later
    ```bash
    pip uninstall vehicle_tracker
    ```
