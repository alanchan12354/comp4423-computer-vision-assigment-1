# Assignment 1: LEGO Image Generation - Method Report Draft

## Task 1: Camera Capture
Implemented using OpenCV `VideoCapture`. The program captures frames from the default camera (index 0).

## Task 2: 1x1 Brick Generation (3 Colors)
**Algorithm:**
1.  **Preprocessing**: The input image is resized using `cv2.resize`. To satisfy the constraint of "no more than 100x100 bricks", we calculate a scaling factor so that the maximum dimension is 100 pixels.
2.  **Quantization**: We define a specific palette (White, Gray, Black). 
    -   We reshape the image into a list of pixels.
    -   We calculate the Euclidean distance between each pixel and the 3 palette colors.
    -   Each pixel is assigned the color of the nearest palette entry (Vector quantization).
3.  **Rendering**: Each "pixel" in the quantized low-res grid is drawn as a LEGO brick.
    -   To simulate the brick look, we draw a rectangle.
    -   We add a border for separation.
    -   We draw a "stud" (circle) on top to mimic the physical LEGO geometry.

## Task 3: Multi-size Brick Generation
**Algorithm:**
1.  **Preprocessing**: Similar to Task 2, but resized to a slightly smaller grid (e.g., 60x60) to allow for larger bricks without losing too much detail, or kept at 100x100.
2.  **Extended Palette**: Used 8 common LEGO colors (White, Gray, Black, Red, Green, Blue, Yellow, Brown).
3.  **Greedy Packing Algorithm**:
    -   We iterate through the grid pixels from top-left to bottom-right.
    -   For each unvisited pixel, we attempt to place the largest possible brick that fits.
    -   **Fitting criteria**: 
        1. The brick must stay within image boundaries.
        2. All pixels covered by the brick must have the exact same quantized color.
        3. No part of the brick can overlap with already placed bricks.
    -   **Brick Sizes**: We check sizes in descending order: 2x4, 2x4 (rotated), 2x2, 2x1, 1x2, 1x1.
    -   Once a fit is found, we mark those grid positions as 'visited' and record the brick type.
4.  **Rendering**: The recorded bricks are drawn on a fresh canvas. 2x4 bricks get 8 studs, 1x1 get 1 stud, etc.

## Task 4: Real-World Scenario
-   The system runs a continuous loop reading from the camera.
-   A "Real-time Mode" flag allows toggling the LEGO generation on the live feed.
-   To maintain performance (FPS), we process the real-time view at a lower resolution or with a simpler quantization step if needed, though the current efficient numpy implementation handles frame-by-frame processing well on modern CPUs.

## Findings & Limitation
-   **Lighting**: Camera noise can cause flickering bricks.
-   **Quantization**: Simple Euclidean distance works well for distinct colors but might fail under colored lighting (e.g., yellow light makes everything look yellow/brown).
-   **Greedy Approach**: It doesn't find the *optimal* (minimum) number of bricks, just a valid covering. It works well visually but might result in many small bricks at boundaries.
