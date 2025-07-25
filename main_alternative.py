# main.py
import cv2
import numpy as np
import threading
import queue
import mediapipe as mp
import time
from utils.volume_control import set_volume, get_volume_level

# Enable OpenCV optimizations
cv2.setUseOptimized(True)

# MediaPipe Hands with GPU support (if available) and simpler model for speed
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,            # simpler/faster model
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    # use_gpu=True                   # enable GPU inference
)
mp_draw = mp.solutions.drawing_utils

# Parameters
BUFFER_SIZE = 64
EMA_ALPHA = 0.3
FRAME_SKIP = 2                     # process every 2nd frame
GESTURE_COOLDOWN = 0.3            # seconds

# Shared queues for threading
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

# Rolling buffers for raw and smoothed points
pts = np.zeros((BUFFER_SIZE, 2), dtype=np.int32)
smoothed = np.zeros((2,), dtype=np.float32)  # current EMA point
pts_idx = 0

gesture_info = "Detecting..."
last_volume_change = 0.0

def capture_thread(cap):
    """Continuously read frames from camera."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)

def inference_thread():
    """Continuously process frames through MediaPipe."""
    while True:
        frame = frame_queue.get()
        # Resize & convert once per frame
        small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LINEAR)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_small)
        result_queue.put((frame, small, result))

def main():
    global pts_idx, smoothed, gesture_info, last_volume_change , pts , pts_idx, pts_list

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # faster grab
    print("\n🟢 Real-Time Finger Volume Controller Running")

    # Start threads
    threading.Thread(target=capture_thread, args=(cap,), daemon=True).start()
    threading.Thread(target=inference_thread, daemon=True).start()

    frame_count = 0

    while True:
        frame_count += 1
        # Skip frames to reduce load
        if frame_count % FRAME_SKIP != 0:
            # Still consume inference queue to keep threads synced
            if not result_queue.empty():
                _ = result_queue.get()
            continue

        # Get latest processed result
        frame, small, result = result_queue.get()
        h, w, _ = frame.shape
        cx = cy = None

        # Landmark detection
        if result.multi_hand_landmarks:
            handLms = result.multi_hand_landmarks[0]
            lm = handLms.landmark[8]  # index fingertip
            cx = int(lm.x * w)
            cy = int(lm.y * h)

            # Rolling buffer update
            pts = np.roll(pts, 1, axis=0)
            pts[0] = (cx, cy)
            pts_idx = min(pts_idx + 1, BUFFER_SIZE)

            # EMA smoothing
            if pts_idx == 1:
                smoothed = np.array([cx, cy], dtype=np.float32)
            else:
                smoothed = EMA_ALPHA * np.array([cx, cy], dtype=np.float32) + (1 - EMA_ALPHA) * smoothed

            # Draw hand connections on full frame
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # Draw smoothed trail
        if pts_idx > 1:
            pts_list = pts[:pts_idx].astype(int)
            for i in range(1, len(pts_list)):
                cv2.line(frame,
                         tuple(pts_list[i - 1]),
                         tuple(pts_list[i]),
                         (0, 255, 0), 2)

        # Gesture estimation when enough points
        if pts_idx >= 6:
            x = pts[ :6, 0]
            y = pts[ :6, 1]
            vx1 = int(x[3] - x[5]); vy1 = int(y[3] - y[5])
            vx2 = int(x[1] - x[3]); vy2 = int(y[1] - y[3])
            live_cross = vx1 * vy2 - vy1 * vx2

            now = time.time()
            if now - last_volume_change > GESTURE_COOLDOWN:
                if live_cross > 1000:
                    set_volume("up")
                    gesture_info = "🔊 Volume Up"
                    last_volume_change = now
                elif live_cross < -1000:
                    set_volume("down")
                    gesture_info = "🔉 Volume Down"
                    last_volume_change = now

        # Volume bar (use full frame for UI)
        vol = get_volume_level()
        vol_bar = int(np.interp(vol, [0.0, 1.0], [400, 150]))
        cv2.rectangle(frame, (50, 150), (85, 400), (0, 0, 0), 2)
        cv2.rectangle(frame, (50, vol_bar), (85, 400), (0, 255, 0), -1)
        cv2.putText(frame, f'{int(vol * 100)} %', (40, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Display gesture info
        cv2.putText(frame, gesture_info, (150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Show result
        cv2.imshow("Finger Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
