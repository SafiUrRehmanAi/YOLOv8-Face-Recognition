import cv2
import numpy as np
import face_recognition
from pathlib import Path
from ultralytics import YOLO
import threading
import time
from datetime import datetime

# ========================= CONFIG =========================
MODEL_PATH = "best.pt"
KNOWN_FACES_DIR = Path("known_faces")
YOLO_IMGSZ = 320               # 320 = fast, 480 = balanced, 640 = accurate
CONFIDENCE = 0.5
RECOGNITION_INTERVAL = 10      # Run expensive face_recognition every N frames
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
# ==========================================================


class FaceRecognitionApp:
    def __init__(self):
        KNOWN_FACES_DIR.mkdir(exist_ok=True)

        print("[INFO] Loading YOLO model...")
        self.model = YOLO(MODEL_PATH)

        print("[INFO] Loading known faces...")
        self.known_encodings, self.known_names = self._load_known_faces()
        if self.known_names:
            print(f"[INFO] Registered identities: {', '.join(self.known_names)}")
        else:
            print("[WARN] No identities found in 'known_faces/'. All faces will show as 'Unknown'.")

        print("[INFO] Starting webcam...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # CRITICAL: stops old frames from piling up

        # Thread-safe shared state
        self.frame_lock = threading.Lock()
        self.result_lock = threading.Lock()

        self.latest_frame = None          # Raw BGR from camera
        self.latest_annotated = None      # Frame with boxes drawn
        self.tracked_identities = []      # For proximity tracking between recognition frames
        self.running = True

        self.fps_display = 0
        self.fps_inference = 0
        self.recognition_counter = 0

    # ------------------------------------------------------------------
    # Load reference encodings
    # ------------------------------------------------------------------
    def _load_known_faces(self):
        encodings, names = [], []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for file_path in KNOWN_FACES_DIR.glob(ext):
                try:
                    image = face_recognition.load_image_file(str(file_path))
                    face_encs = face_recognition.face_encodings(image)
                    if face_encs:
                        encodings.append(face_encs[0])
                        names.append(file_path.stem)
                except Exception as e:
                    print(f"[WARN] Could not load {file_path.name}: {e}")
        return encodings, names

    # ------------------------------------------------------------------
    # Detection + Identification
    # ------------------------------------------------------------------
    def _detect(self, image_bgr: np.ndarray, run_recognition: bool, prev_identities: list):
        h, w, _ = image_bgr.shape

        # YOLO inference (small imgsz = much faster on CPU)
        results = self.model.predict(
            source=image_bgr,
            conf=CONFIDENCE,
            verbose=False,
            stream=False,
            imgsz=YOLO_IMGSZ,
        )
        result = results[0] if results else None
        output = image_bgr.copy()
        current_identities = []

        if result is None or result.boxes is None or len(result.boxes) == 0:
            return output, current_identities

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for xyxy, score in zip(boxes, confs):
            x1, y1, x2, y2 = xyxy
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            face_crop = image_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            name = "Unknown"

            # --- Full face recognition (expensive) ---
            if run_recognition and face_crop.size > 0 and len(self.known_encodings) > 0:
                rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                unknown_encs = face_recognition.face_encodings(rgb_crop)
                if unknown_encs:
                    matches = face_recognition.compare_faces(
                        self.known_encodings, unknown_encs[0], tolerance=0.6
                    )
                    distances = face_recognition.face_distance(
                        self.known_encodings, unknown_encs[0]
                    )
                    if True in matches:
                        best = np.argmin(distances)
                        if matches[best]:
                            name = self.known_names[best]

            # --- Proximity tracking (cheap) ---
            elif not run_recognition and prev_identities:
                min_dist = float("inf")
                best_name = "Unknown"
                for prev in prev_identities:
                    d = np.hypot(prev["center"][0] - cx, prev["center"][1] - cy)
                    if d < min_dist:
                        min_dist = d
                        best_name = prev["name"]
                if min_dist < (w * 0.25):          # Only reuse if face hasn't jumped too far
                    name = best_name

            current_identities.append({"center": (cx, cy), "name": name})

            # Draw box + label
            label = f"{name} ({score:.2f})"
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
            cv2.putText(output, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return output, current_identities

    # ------------------------------------------------------------------
    # Thread 1: Frame grabber (never blocks)
    # ------------------------------------------------------------------
    def _grab_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            time.sleep(0.001)

    # ------------------------------------------------------------------
    # Thread 2: Inference worker (heavy lifting happens here)
    # ------------------------------------------------------------------
    def _infer(self):
        last_time = time.time()
        while self.running:
            # Pull latest frame (drop old ones automatically)
            frame = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()

            if frame is None:
                time.sleep(0.005)
                continue

            self.recognition_counter += 1
            run_rec = (self.recognition_counter % RECOGNITION_INTERVAL == 0)

            annotated, identities = self._detect(
                frame,
                run_recognition=run_rec,
                prev_identities=self.tracked_identities,
            )

            # Inference FPS
            now = time.time()
            self.fps_inference = 1.0 / (now - last_time + 1e-6)
            last_time = now

            with self.result_lock:
                self.latest_annotated = annotated
                self.tracked_identities = identities

    # ------------------------------------------------------------------
    # Main loop: Display + keyboard controls
    # ------------------------------------------------------------------
    def run(self):
        threading.Thread(target=self._grab_frames, daemon=True).start()
        threading.Thread(target=self._infer, daemon=True).start()

        print("\n[INFO] Stream started.")
        print("       Q = Quit | S = Screenshot | R = Register face\n")

        disp_time = time.time()
        disp_frames = 0

        while True:
            # Get latest processed frame (never wait for inference)
            display = None
            with self.result_lock:
                if self.latest_annotated is not None:
                    display = self.latest_annotated.copy()

            if display is not None:
                disp_frames += 1
                elapsed = time.time() - disp_time
                if elapsed >= 1.0:
                    self.fps_display = int(disp_frames / elapsed)
                    disp_frames = 0
                    disp_time = time.time()

                # HUD overlay
                cv2.putText(display, f"Display: {self.fps_display} FPS", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display, f"Inference: {self.fps_inference:.1f} FPS", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display, "Q:Quit  S:Save  R:Register", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow("YOLOv8 Face Recognition", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                self._save_screenshot(display)
            elif key == ord("r"):
                self._register_face()

        # Cleanup
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown complete.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _save_screenshot(self, frame):
        if frame is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"screenshot_{ts}.jpg"
        cv2.imwrite(path, frame)
        print(f"[INFO] Screenshot saved: {path}")

    def _register_face(self):
        """Grab the latest raw frame, detect a face, and save the crop."""
        raw = None
        with self.frame_lock:
            if self.latest_frame is not None:
                raw = self.latest_frame.copy()

        if raw is None:
            print("[WARN] No frame available to register.")
            return

        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        if not locations:
            print("[WARN] No face detected in current frame. Try again.")
            return

        top, right, bottom, left = locations[0]
        crop = raw[top:bottom, left:right]

        idx = len(list(KNOWN_FACES_DIR.glob("registered_*.jpg")))
        path = KNOWN_FACES_DIR / f"registered_{idx:03d}.jpg"
        cv2.imwrite(str(path), crop)
        print(f"[INFO] Face crop saved: {path}")
        print("[INFO] Rename this file to the person's name and restart the script to load it.")


# ===================================================================
if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()