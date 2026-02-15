import cv2
import sys
import numpy as np
from lego_processor import LegoProcessor

def main():
    print("Initializing LEGO Generation System...")
    print("OpenCV Version:", cv2.__version__)
    
    # Initialize Camera (Task 1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)
        
    processor = LegoProcessor()
    
    print("\nControls:")
    print(" 'q' - Quit application")
    print(" 's' - Save current frame snapshot")
    print(" '2' - Generate Task 2 (1x1 bricks, 3 colors) from current frame")
    print(" '3' - Generate Task 3 (Multi-size bricks, multi-color) from current frame")
    print(" '4' - Toggle Real-time LEGO mode (Task 4)")
    
    real_time_mode = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break
            
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
