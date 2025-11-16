import argparse
import cv2
from ultralytics import YOLO
import numpy as np

class VehicleTracker:
    """A class for tracking and counting vehicles in video streams using YOLOv8 and ByteTrack."""
    
    def __init__(self, model_path='yolov8m.pt', video_path=None):
        """
        Initialize the VehicleTracker with model and video paths.
        
        Args:
            model_path (str): Path to YOLO model weights
            video_path (str): Path to input video file
        """
        if video_path is None:
            raise ValueError("Video path must be provided")
            
        # Fixed color palette for tracking visualization (BGR format)
        self.colors = [
            (0, 0, 255),      # Red
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue
            (0, 165, 255),    # Orange
            (128, 0, 128)     # Purple
        ]
        
        # Vehicle classes to track and count
        self.vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']
        
        # Initialize counters
        self.counted_ids = set()
        self.vehicle_counts = {vehicle: 0 for vehicle in self.vehicle_classes}
        
        # Load YOLOv8 model
        try:
            self.model = YOLO(model_path)
            print(f"Successfully loaded model: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Initialize video capture
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Successfully opened video: {video_path}")
            print(f"Video resolution: {self.frame_width}x{self.frame_height}")
            print(f"FPS: {self.fps:.2f}")
            
        except Exception as e:
            print(f"Error opening video: {e}")
            raise
        
        # Create display window
        cv2.namedWindow('Vehicle Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Vehicle Tracking', 640, 480)
    
    def process_frame(self, frame):
        """
        Process a single frame: detect, track, and annotate vehicles.
        
        Args:
            frame: Input video frame
            
        Returns:
            numpy.ndarray: Annotated frame with bounding boxes and labels
        """
        try:
            # Perform object tracking with ByteTrack
            results = self.model.track(frame, tracker='bytetrack.yaml', persist=True)
            
            # Check if results are available
            if not results or len(results) == 0:
                return frame
                
            r = results[0]
            
            # Create annotated frame copy
            annotated_frame = frame.copy()
            
            # Check if boxes are available
            if r.boxes is None or len(r.boxes) == 0:
                return annotated_frame
            
            # Process each detected object
            for box in r.boxes:
                # Filter by confidence threshold
                conf = float(box.conf[0])
                if conf < 0.75:
                    continue
                
                # Extract bounding box coordinates and metadata
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                class_name = self.model.names[class_id]
                
                # Process only vehicle classes of interest
                if class_name in self.vehicle_classes:
                    # Assign color based on track ID for consistent visualization
                    color = self.colors[track_id % len(self.colors)]
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Create annotation label
                    label = f"ID {track_id} | {class_name} | {conf:.2f}"
                    
                    # Draw label above bounding box
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
                    
                    # Count vehicle if not already counted
                    if track_id not in self.counted_ids and track_id != -1:
                        self.vehicle_counts[class_name] += 1
                        self.counted_ids.add(track_id)
                        print(f"New vehicle counted - ID: {track_id}, Type: {class_name}")
            
            return annotated_frame
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame
    
    def create_counter_strip(self, width):
        """
        Create a strip displaying vehicle counts.
        
        Args:
            width (int): Width of the strip
            
        Returns:
            numpy.ndarray: Counter strip image
        """
        strip_height = 50
        strip = np.zeros((strip_height, width, 3), dtype=np.uint8)
        
        # Prepare counter text
        counter_text = '   '.join([f"{k}: {v}" for k, v in self.vehicle_counts.items()])
        
        # Draw counter text on strip
        cv2.putText(strip, counter_text, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return strip
    
    def run(self):
        """Main method to run the vehicle tracking pipeline."""
        try:
            frame_count = 0
            
            while True:
                # Read frame from video
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                frame_count += 1
                
                # Process frame for vehicle tracking
                annotated_frame = self.process_frame(frame)
                
                # Create counter strip
                counter_strip = self.create_counter_strip(self.frame_width)
                
                # Combine frame with counter strip
                combined_frame = np.vstack((annotated_frame, counter_strip))
                
                # Display result
                cv2.imshow('Vehicle Tracking', combined_frame)
                
                # Exit on 'q' press or window close
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty('Vehicle Tracking', cv2.WND_PROP_VISIBLE) < 1:
                    print("⏹️ User interrupted playback")
                    break
                    
        except Exception as e:
            print(f"Error in main loop: {e}")
        
        finally:
            # Cleanup resources
            self.cap.release()
            cv2.destroyAllWindows()
            print("\nVehicle tracking completed!")
            print("Final counts:")
            for vehicle, count in self.vehicle_counts.items():
                print(f"   {vehicle}: {count}")
            print(f"Total vehicles: {sum(self.vehicle_counts.values())}")

def main():
    """
    Main function to handle command line arguments and initialize vehicle tracking.
    """
    parser = argparse.ArgumentParser(description="Vehicle Tracking and Counting System")
    parser.add_argument('--video', type=str, required=True, 
                       help="Path to input video file")
    parser.add_argument('--model', type=str, default='yolov8m.pt',
                       help="Path to YOLO model weights (default: yolov8m.pt)")
    
    args = parser.parse_args()
    
    print("Starting Vehicle Tracking System")
    print(f"Video source: {args.video}")
    print(f"Model: {args.model}")
    print("-" * 50)
    
    try:
        # Initialize and run vehicle tracker
        tracker = VehicleTracker(model_path=args.model, video_path=args.video)
        tracker.run()
    except Exception as e:
        print(f"Failed to initialize vehicle tracker: {e}")
        return 1
    
    return 0

# Main execution
if __name__ == "__main__":
    exit(main())