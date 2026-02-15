import cv2
import sys
import numpy as np
from lego_processor import LegoProcessor

def main():
    print("Initializing LEGO Generation System...", flush=True)
    print(f"OpenCV Version: {cv2.__version__}", flush=True)
    
    # Initialize Camera (Task 1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera. Switching to Test Mode (Static Image).", file=sys.stderr)
        # Create a dummy image for testing if camera fails
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw something
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), -1)
        cv2.putText(frame, "No Camera", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        use_camera = False
    else:
        use_camera = True
        
    processor = LegoProcessor()
    
    print("\nControls:", flush=True)
    print(" 'q' - Quit application", flush=True)
    print(" 's' - Save current frame snapshot", flush=True)
    print(" '2' - Generate Task 2 (1x1 bricks, 3 colors) from current frame", flush=True)
    print(" '3' - Generate Task 3 (Multi-size bricks, multi-color) from current frame", flush=True)
    print(" '4' - Toggle Real-time LEGO mode (Task 4)", flush=True)
    
    real_time_mode = False
    
    while True:
        if use_camera:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...", file=sys.stderr)
                break
        else:
            # Create a dynamic dummy image for testing
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), -1)
            cv2.circle(frame, (320, 240), 50, (0, 255, 0), -1)
            # Add some movement or noise to simulate video?
            import time
            t = time.time()
            cv2.putText(frame, f"Time: {t:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # Add a slight delay to mimic 30fps
            cv2.waitKey(30)

        # Display original frame (Task 1)
        cv2.imshow('Camera Feed (Task 1)', frame)
        
        # Real-time processing (Task 4)
        if real_time_mode:
            # Downscale for performance during live view
            preview_frame, _ = processor.process_task3(frame, size=10)
            cv2.imshow('Real-time LEGO (Task 4)', preview_frame)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = "snapshot.png"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
        elif key == ord('2'):
            print("Processing Task 2...")
            lego_img = processor.process_task2(frame, size=15)
            cv2.imshow('Task 2 Output', lego_img)
            print("Task 2 Complete.")
        elif key == ord('3'):
            print("Processing Task 3...")
            lego_img, stats = processor.process_task3(frame, size=15)
            cv2.imshow('Task 3 Output', lego_img)
            print("Task 3 Brick Summary:")
            total_bricks = 0
            for k, v in sorted(stats.items()):
                print(f"  {k}: {v}")
                total_bricks += v
            print(f"  Total Bricks: {total_bricks}")
        elif key == ord('4'):
            real_time_mode = not real_time_mode
            if real_time_mode:
                print("Real-time mode ON")
            else:
                print("Real-time mode OFF")
                cv2.destroyWindow('Real-time LEGO (Task 4)')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
