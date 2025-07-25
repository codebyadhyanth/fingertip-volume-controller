# main.py
import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import time
from utils.volume_control import set_volume, get_volume_level

cv2.setUseOptimized(True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Track fingertip positions
pts = deque(maxlen=64)
smoothed_pts = deque(maxlen=64)
frame_count = 0

# Finger tip index
INDEX_TIP_ID = 8

EMA_ALPHA = 0.3
last_gesture_time = 0
cooldown_secs = 1.5

cap = cv2.VideoCapture(0)
print("\n🟢 Fast & Optimized Finger Volume Controller Running")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    cx, cy = None, None

    # Run MediaPipe only every 3rd frame
    if frame_count % 2 == 0:
        small_frame = cv2.resize(frame, (320, 240))
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_small)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                lm_list = handLms.landmark
                index_tip = lm_list[INDEX_TIP_ID]
                cx = int(index_tip.x * w)
                cy = int(index_tip.y * h)

                if len(pts) > 0:
                    px, py = pts[0]
                    if np.hypot(cx - px, cy - py) > 100:
                        continue

                pts.appendleft((cx, cy))

                if len(smoothed_pts) == 0:
                    smoothed_pts.appendleft((cx, cy))
                else:
                    prev_x, prev_y = smoothed_pts[0]
                    if abs(prev_x - cx) > 2 or abs(prev_y - cy) > 2:
                        new_x = int(EMA_ALPHA * cx + (1 - EMA_ALPHA) * prev_x)
                        new_y = int(EMA_ALPHA * cy + (1 - EMA_ALPHA) * prev_y)
                        smoothed_pts.appendleft((new_x, new_y))

                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Draw smoothed trail
    for i in range(1, len(smoothed_pts)):
        if smoothed_pts[i - 1] and smoothed_pts[i]:
            cv2.line(frame, smoothed_pts[i - 1], smoothed_pts[i], (0, 255, 0), 2)

    # Gesture detection every 6th frame
    gesture_info = "Detecting..."
    if frame_count % 6 == 0 and len(smoothed_pts) >= 20:
        x = np.array([p[0] for p in smoothed_pts if p])
        y = np.array([p[1] for p in smoothed_pts if p])
        dx = np.diff(x)
        dy = np.diff(y)
        cross = np.sum(dx[:-1] * dy[1:] - dy[:-1] * dx[1:])

        now = time.time()
        if now - last_gesture_time > cooldown_secs:
            if cross < 1000:
                set_volume("up")
                gesture_info = "🔊 Volume Up"
                last_gesture_time = now
                smoothed_pts.clear()
            elif cross > -1000:
                set_volume("down")
                gesture_info = "🔉 Volume Down"
                last_gesture_time = now
                smoothed_pts.clear()

    # Volume bar
    vol_level = get_volume_level()
    vol_bar = int(np.interp(vol_level, [0.0, 1.0], [400, 150]))
    cv2.rectangle(frame, (50, 150), (85, 400), (0, 0, 0), 2)
    cv2.rectangle(frame, (50, vol_bar), (85, 400), (0, 255, 0), -1)
    cv2.putText(frame, f'{int(vol_level * 100)} %', (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display info
    cv2.putText(frame, gesture_info, (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Finger Volume Control", frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()