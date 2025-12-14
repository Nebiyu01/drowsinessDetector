import time
import json
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def _dist(a, b):
    return np.linalg.norm(a - b)

def eye_aspect_ratio(pts):
    p1, p2, p3, p4, p5, p6 = pts
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4) + 1e-6)

def landmarks_to_np(face_landmarks, w, h):
    pts = []
    for lm in face_landmarks.landmark:
        pts.append((lm.x * w, lm.y * h))
    return np.array(pts, dtype=np.float32)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check macOS camera permissions.")

    duration_sec = 10.0
    start = time.time()
    ears = []

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

            now = time.time()
            remaining = max(0.0, duration_sec - (now - start))

            msg = f"Calibration: look normal, keep eyes open. {remaining:.1f}s left"
            cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0]
                pts = landmarks_to_np(face, w, h)

                ear_left = eye_aspect_ratio(pts[LEFT_EYE])
                ear_right = eye_aspect_ratio(pts[RIGHT_EYE])
                ear = (ear_left + ear_right) / 2.0
                ears.append(float(ear))

                cv2.putText(frame, f"EAR: {ear:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No face detected", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if (now - start) >= duration_sec:
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(ears) < 30:
        raise RuntimeError("Not enough samples. Try again with better lighting and keep your face in frame.")

    avg_ear = float(np.mean(ears))
    ear_thresh = avg_ear * 0.75

    data = {
        "avg_ear": avg_ear,
        "ear_thresh": ear_thresh,
        "ear_multiplier": 0.75,
        "duration_sec": duration_sec,
        "samples": len(ears),
        "created_at_unix": time.time(),
    }

    with open("calibration.json", "w") as f:
        json.dump(data, f, indent=2)

    print("Saved calibration.json")
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
