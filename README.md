# How to Run

1.  **Install Prerequisites**:
    Open a terminal and run the following commands to set up a virtual environment (avoids system conflicts):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Run the Application**:
    Make sure your virtual environment is activated (`source venv/bin/activate`), then run:
    ```bash
    python main.py
    ```

3.  **Controls**:
    -   `q`: Quit the program.
    -   `s`: Save a snapshot of the current view.
    -   `2`: Generate Task 2 (1x1 Bricks, 3 Colors) from the current camera frame.
    -   `3`: Generate Task 3 (Multi-size Bricks, High Color) from the current camera frame.
    -   `4`: Toggle Real-time LEGO mode (Task 4).

4.  **Files**:
    -   `main.py`: Entry point.
    -   `lego_processor.py`: Contains the logic for converting images to LEGO style.
    -   `REPORT_DRAFT.md`: Explanation of methods for your assignment report.
