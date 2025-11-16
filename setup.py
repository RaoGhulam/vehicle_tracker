from setuptools import setup, find_packages

setup(
    name="vehicle_tracker",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "ultralytics",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            "vehicle-tracker = vehicle_tracker.tracker:main",
        ],
    },
)