
# Fingertip Volume Controller

A real-time hand gesture-based volume control application built with Python, OpenCV, and MediaPipe. This project allows users to adjust their system's volume by drawing gestures in the air with their index finger, detected via webcam.

## Features
- **Gesture Recognition**: Detects upward or downward finger movements to increase or decrease volume.
- **Smoothed Tracking**: Uses exponential moving average (EMA) and buffering for smoother finger trail visualization.
- **Volume Bar Display**: Real-time on-screen volume level indicator.
- **Optimized Performance**: Includes multi-threading, frame skipping, and efficient processing to reduce lag (though some optimizations are still in progress).
- **Hand Landmark Detection**: Powered by MediaPipe for accurate fingertip tracking.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/codebyadhyanth/fingertip-volume-controller.git
   cd fingertip-volume-controller
   ```
2. Install dependencies (requires Python 3.x):
   ```
   pip install opencv-python numpy mediapipe
   ```
   Note: Ensure you have a webcam connected for input.

## Usage
1. Run the main script:
   ```
   python main.py
   ```
2. Point your webcam at your hand.
3. Use your index finger to draw upward (for volume up) or downward (for volume down) gestures.
4. Press ESC to exit the application.

A window will display the camera feed with hand landmarks, a green finger trail, gesture info, and a volume bar.

## Known Issues and Limitations
- **Performance**: The application still experiences some lag and slowness, especially on lower-end hardware or with high-resolution cameras. This can make interactions feel less responsive. Ongoing optimizations are planned to improve FPS and reduce latency.
- **Gesture Sensitivity**: Detection may vary based on lighting, hand distance, or background noise—ensure good lighting for best results.
- **Platform Compatibility**: Tested on Windows; volume control functionality may require additional setup on macOS or Linux (e.g., via `utils/volume_control.py` adjustments).

## Contributing
Feel free to fork the repo and submit pull requests for improvements, such as better smoothing algorithms or GPU acceleration. If you encounter bugs, open an issue on GitHub.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Created by codebyadhyanth. Last updated: July 2025.*
