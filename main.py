from logger import append_event, append_marker

import time
import cv2
import numpy as np
import mediapipe as mp
import json
import os


mp_face_mesh = mp.solutions.face_mesh

# MediaPipe FaceMesh landmark indices for eyes and mouth.
# These sets are commonly used and work well in practice.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14, 78, 308]  # corners + inner top/bottom + outer points

def _dist(a, b):
    return np.linalg.norm(a - b)

def eye_aspect_ratio(pts):
    # pts: 6 points in order [p1, p2, p3, p4, p5, p6]
    p1, p2, p3, p4, p5, p6 = pts
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4) + 1e-6)

def mouth_aspect_ratio(pts):
    # pts: [left_corner, right_corner, inner_top, inner_bottom, outer_left, outer_right]
    left, right, inner_top, inner_bottom, outer_left, outer_right = pts
    horiz = _dist(left, right)
    vert = _dist(inner_top, inner_bottom)
    return vert / (horiz + 1e-6)

def landmarks_to_np(face_landmarks, w, h):
    pts = []
    for lm in face_landmarks.landmark:
        pts.append((lm.x * w, lm.y * h))
    return np.array(pts, dtype=np.float32)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check macOS camera permissions.")

    # Thresholds, tune later with calibration
    cal = load_calibration()
    EAR_THRESH = float(cal["ear_thresh"]) if cal and "ear_thresh" in cal else 0.20

    MAR_THRESH = 0.60

    # Time windows
    EYE_CLOSED_SECONDS = 1.5
    YAWN_SECONDS = 0.5
    COOLDOWN_SECONDS = 6.0

    eye_low_start = None
    mouth_high_start = None
    last_alert_time = 0.0

    prev_t = time.time()
    fps = 0.0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            status = "NO FACE"
            ear = None
            mar = None

            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0]
                pts = landmarks_to_np(face, w, h)

                left_eye_pts = pts[LEFT_EYE]
                right_eye_pts = pts[RIGHT_EYE]
                mouth_pts = pts[MOUTH]

                ear_left = eye_aspect_ratio(left_eye_pts)
                ear_right = eye_aspect_ratio(right_eye_pts)
                ear = (ear_left + ear_right) / 2.0
                mar = mouth_aspect_ratio(mouth_pts)

                now = time.time()

                # Eye closure timer
                if ear < EAR_THRESH:
                    if eye_low_start is None:
                        eye_low_start = now
                else:
                    eye_low_start = None

                # Yawn timer
                if mar > MAR_THRESH:
                    if mouth_high_start is None:
                        mouth_high_start = now
                else:
                    mouth_high_start = None

                eye_closed = eye_low_start is not None and (now - eye_low_start) >= EYE_CLOSED_SECONDS
                yawn = mouth_high_start is not None and (now - mouth_high_start) >= YAWN_SECONDS

                # Cooldown
                can_alert = (now - last_alert_time) >= COOLDOWN_SECONDS

                reason = None

                if eye_closed and yawn:
                    reason = "eyes_closed_and_yawn"
                elif eye_closed:
                    reason = "eyes_closed"
                elif yawn:
                    reason = "yawn"

                if reason and can_alert:
                    last_alert_time = now
                    status = "DROWSY ALERT"

                    append_event("logs.csv", ear, mar, reason, fps)

                    cv2.putText(frame, "ALERT", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 4)
                else:
                    if eye_closed and yawn:
                        status = "VERY DROWSY"
                    elif eye_closed:
                        status = "EYES CLOSED"
                    elif yawn:
                        status = "YAWN"
                    else:
                        status = "OK"


                # Draw a few landmark points for sanity
                for idx in LEFT_EYE + RIGHT_EYE + MOUTH:
                    x, y = pts[idx]
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            # FPS calc
            t = time.time()
            dt = t - prev_t
            prev_t = t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Overlay text
            cv2.putText(frame, f"Status: {status}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            if ear is not None and mar is not None:
                cv2.putText(frame, f"EAR: {ear:.3f}  MAR: {mar:.3f}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Drowsiness Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                append_marker("logs.csv", "session_start")
            if key == ord("e"):
                append_marker("logs.csv", "session_end")


    cap.release()
    cv2.destroyAllWindows()

def append_marker(path, label):
    ts = datetime.now(timezone.utc)
    row = [ts.isoformat(), f"{ts.timestamp():.3f}", "", "", f"marker:{label}", ""]
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(HEADER)
        w.writerow(row)


def load_calibration(path="calibration.json"):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    main()

