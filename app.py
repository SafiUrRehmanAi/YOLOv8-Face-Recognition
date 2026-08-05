import av
import cv2
import numpy as np
import streamlit as st
import face_recognition
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import threading
import time

# =========================
# Optional Webcam Support
# =========================
try:
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    WEBRTC_AVAILABLE = True
except ImportError:
    VideoProcessorBase = object
    webrtc_streamer = None
    WEBRTC_AVAILABLE = False

# =========================
# Streamlit Configuration
# =========================
st.set_page_config(
    page_title="YOLOv8 Face Recognition",
    page_icon="🎯",
    layout="centered",
)

st.title("🎯 YOLOv8 Face Recognition App")
st.write(
    "Upload an image or use your webcam to detect and identify faces using YOLOv8 and embeddings."
)

# =========================
# Paths Configuration
# =========================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best.pt"
FACES_DIR = BASE_DIR / "known_faces"

FACES_DIR.mkdir(exist_ok=True)

# =========================
# Model & Face Database Loading
# =========================
@st.cache_resource
def load_model():
    try:
        # If you have a GPU, YOLO will auto-use it. For CPU, consider using yolov8n.pt (nano).
        return YOLO(str(MODEL_PATH))
    except Exception as e:
        st.error(f"❌ Failed to load model:\n\n{e}")
        st.stop()

model = load_model()

@st.cache_resource
def load_known_faces():
    known_encodings = []
    known_names = []
    valid_extensions = ("*.jpg", "*.jpeg", "*.png")
    file_list = []
    for ext in valid_extensions:
        file_list.extend(FACES_DIR.glob(ext))

    for file_path in file_list:
        name = file_path.stem
        try:
            image = face_recognition.load_image_file(str(file_path))
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
        except Exception as e:
            st.sidebar.warning(f"Could not load image {file_path.name}: {e}")

    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

# =========================
# Sidebar & Admin Panel
# =========================
st.sidebar.header("⚙️ Configuration")
confidence = st.sidebar.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.05
)

# --- Performance Tuning ---
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Performance Tuning")
YOLO_IMGSZ = st.sidebar.selectbox("YOLO Input Size (lower = faster)", [320, 480, 640], index=0)
RECOGNITION_INTERVAL = st.sidebar.slider(
    "Face Recognition Interval (frames)", min_value=1, max_value=30, value=15, step=1,
    help="Run expensive face_recognition every N processed frames. Identities are cached in between."
)
WEBCAM_W = st.sidebar.selectbox("Webcam Width", [640, 1280], index=0)
WEBCAM_H = st.sidebar.selectbox("Webcam Height", [480, 720], index=0)

# --- ADMIN PANEL ---
st.sidebar.markdown("---")
with st.sidebar.expander("👤 Admin: Dynamic Face Registration", expanded=False):
    st.write("Register a new person into the identity database.")
    new_person_name = st.text_input("Enter Person's Name", placeholder="e.g. Person A")
    new_person_image = st.file_uploader("Upload Clear Face Image", type=["jpg", "jpeg", "png"])

    if st.button("Save Profile"):
        if not new_person_name.strip():
            st.error("Please enter a valid name.")
        elif new_person_image is None:
            st.error("Please upload an image file.")
        else:
            safe_name = "".join(c for c in new_person_name.strip() if c.isalnum() or c in (" ", "_", "-")).rstrip()
            target_path = FACES_DIR / f"{safe_name}.jpg"
            try:
                img_to_save = Image.open(new_person_image).convert("RGB")
                img_to_save.save(target_path, "JPEG")
                st.success(f"Successfully registered '{safe_name}'!")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error saving image: {e}")

# Database status
st.sidebar.markdown("---")
if not known_names:
    st.sidebar.warning("⚠️ No images found in 'known_faces/'. All faces will display as 'Unknown'.")
else:
    st.sidebar.success(f"👥 Loaded {len(known_names)} registered profile(s).")
    with st.sidebar.expander("View Registered Names"):
        for name in known_names:
            st.text(f"• {name}")

MAX_IMAGE_SIZE = 1280

input_mode = st.radio(
    "Choose Input Source", ["Upload Image", "Live Camera"], horizontal=True
)

# =========================
# Detection Logic (Optimized)
# =========================
def detect_and_identify_faces(image_bgr: np.ndarray, run_recognition: bool = True, tracked_identities: list = None):
    """
    image_bgr: BGR numpy array
    run_recognition: if False, skip face_recognition and reuse cached names by proximity
    tracked_identities: list of dicts [{'center': (x,y), 'name': str}, ...] from previous frame
    Returns: annotated image, list of current identities for tracking
    """
    # YOLO predict with smaller imgsz for speed
    results = model.predict(
        source=image_bgr,
        conf=confidence,
        verbose=False,
        stream=False,          # stream=False is slightly faster for single images
        imgsz=YOLO_IMGSZ,      # KEY SPEEDUP: 320 is much faster than 640
    )
    result = results[0] if results else None

    output_image = image_bgr.copy()
    h, w, _ = image_bgr.shape
    current_identities = []

    if result is None or result.boxes is None:
        return output_image, current_identities

    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    for i, (xyxy, score) in enumerate(zip(boxes, confs)):
        x1, y1, x2, y2 = xyxy
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Safety crop
        face_crop = image_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        name = "Unknown"

        if run_recognition and face_crop.size > 0 and len(known_encodings) > 0:
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            unknown_encodings = face_recognition.face_encodings(face_crop_rgb)
            if unknown_encodings:
                matches = face_recognition.compare_faces(known_encodings, unknown_encodings[0], tolerance=0.6)
                distances = face_recognition.face_distance(known_encodings, unknown_encodings[0])
                if True in matches:
                    best = np.argmin(distances)
                    if matches[best]:
                        name = known_names[best]

        elif not run_recognition and tracked_identities:
            # Reuse the name from the closest face in the previous frame (simple proximity tracker)
            min_dist = float('inf')
            best_name = "Unknown"
            for prev in tracked_identities:
                d = np.hypot(prev['center'][0] - cx, prev['center'][1] - cy)
                if d < min_dist:
                    min_dist = d
                    best_name = prev['name']
            # Only reuse if the face hasn't jumped too far (e.g., half the frame width)
            if min_dist < (w * 0.25):
                name = best_name

        current_identities.append({'center': (cx, cy), 'name': name})

        label = f"{name} ({score:.2f})"
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output_image, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
        cv2.putText(output_image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return output_image, current_identities


# =========================
# Webcam Processor (Async / Non-Blocking)
# =========================
class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.recognition_counter = 0

        # Thread-safe shared state
        self._lock = threading.Lock()
        self._latest_input = None
        self._latest_output = None
        self._tracked_identities = []
        self._running = True

        # Start the inference worker thread
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def _inference_loop(self):
        """Background thread: runs detection without blocking the video stream."""
        while self._running:
            img = None
            with self._lock:
                if self._latest_input is not None:
                    img = self._latest_input.copy()
                    self._latest_input = None  # Mark consumed

            if img is None:
                time.sleep(0.001)
                continue

            self.recognition_counter += 1
            run_rec = (self.recognition_counter % RECOGNITION_INTERVAL == 0)

            annotated, identities = detect_and_identify_faces(
                img, run_recognition=run_rec, tracked_identities=self._tracked_identities
            )

            with self._lock:
                self._latest_output = annotated
                self._tracked_identities = identities

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Called for every camera frame. Must return FAST — never block."""
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")

        # Feed the latest frame to the background thread (drop frames if inference is behind)
        with self._lock:
            self._latest_input = img
            output = self._latest_output

        # Return annotated frame if available, otherwise raw frame (zero lag)
        if output is not None:
            return av.VideoFrame.from_ndarray(output, format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def on_ended(self):
        self._running = False


# =========================
# Upload Image Mode
# =========================
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Identify Faces"):
            with st.spinner("Processing pipeline..."):
                img_rgb = np.array(image)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                plotted_bgr, identities = detect_and_identify_faces(img_bgr, run_recognition=True)
                plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

                st.image(plotted_rgb, caption="Recognition Result", use_container_width=True)
                if not identities:
                    st.warning("No faces detected.")
                else:
                    st.success(f"Detected {len(identities)} face(s) in image.")

# =========================
# Webcam Mode
# =========================
else:
    st.write("Allow camera permission in your browser to enable live face identification.")

    if not WEBRTC_AVAILABLE:
        st.error(
            "Live camera requires the `streamlit-webrtc` package.\n\n"
            "Install it using:\n"
            "`pip install streamlit-webrtc av`"
        )
    else:
        webrtc_streamer(
            key="face-recognition",
            video_processor_factory=FaceDetectionProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": WEBCAM_W},
                    "height": {"ideal": WEBCAM_H},
                },
                "audio": False,
            },
            async_processing=True,
        )