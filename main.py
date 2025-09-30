import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from functools import lru_cache

# YOLO
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

# Browser webcam (WebRTC)
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoTransformerBase,
    RTCConfiguration,
)
import av

# Video decoding fallback
import imageio.v3 as iio

# Optional: for runtime model download if you add a URL later
import requests

# ------------------------------------------------------------
# Streamlit page setup
# ------------------------------------------------------------
st.set_page_config(page_title="YOLOv8 Human & Luggage", layout="wide")
st.title("YOLOv8 Detection: Human & Luggage 🚀")

# ------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------
COCO_PATH = "yolov8n.pt"          # keep at repo root (already in your repo)
CUSTOM_PATH = "models/bestV3.pt"  # your custom model (may not exist online)

ALLOWED_COCO_REMAP = {"person": "human", "suitcase": "luggage"}

# WebRTC ICE servers (needed online)
RTC_CFG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        # Add TURN for stricter networks:
        # {"urls": ["turn:YOUR_TURN:3478"], "username": "USER", "credential": "PASS"},
    ]
})

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
options = ["COCO (filtered)", "Custom (bestV3.pt)"]
mode = st.sidebar.radio("Select Model:", options, index=0)
CONF = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# ------------------------------------------------------------
# Helpers: detect Git LFS pointer & optional downloader
# ------------------------------------------------------------
def is_lfs_pointer(path: str) -> bool:
    try:
        # LFS pointer files are tiny and contain "git-lfs" text
        if not os.path.exists(path) or os.path.getsize(path) < 2048:
            with open(path, "rb") as f:
                head = f.read(512).decode("utf-8", "ignore")
            return "git-lfs" in head.lower()
    except Exception:
        pass
    return False

def download_if_missing(url: str, path: str):
    """Optional helper: download a file at runtime if missing/LFS pointer."""
    need = (not os.path.exists(path)) or is_lfs_pointer(path)
    if not need:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    st.info(f"Downloading model to `{path}` …")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

# ------------------------------------------------------------
# Lazy model loader (prevents startup crash if custom model missing)
# ------------------------------------------------------------
@lru_cache(maxsize=2)
def load_model(selected_mode: str) -> YOLO:
    if selected_mode.startswith("Custom"):
        # If hosting externally, uncomment and set your URL:
        # download_if_missing("https://YOUR_HOST/bestV3.pt", CUSTOM_PATH)
        if not os.path.exists(CUSTOM_PATH) or is_lfs_pointer(CUSTOM_PATH):
            raise FileNotFoundError(
                "Custom model `models/bestV3.pt` is missing or is a Git LFS pointer.\n"
                "Use COCO (filtered) for demo, or host the file and enable runtime download."
            )
        return YOLO(CUSTOM_PATH)
    else:
        return YOLO(COCO_PATH)

# Acquire active model (show friendly error if custom is unavailable)
try:
    active_model = load_model(mode)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ------------------------------------------------------------
# Inference / drawing
# ------------------------------------------------------------
def process_frame(frame_bgr: np.ndarray, mode: str, conf: float, active_model: YOLO) -> np.ndarray:
    """Run YOLO and return an annotated BGR frame."""
    res = active_model(frame_bgr, conf=conf)[0]

    # If using COCO model, filter to person + suitcase and relabel
    if mode.startswith("COCO"):
        mask = [active_model.names[int(c)] in ALLOWED_COCO_REMAP for c in res.boxes.cls]
        if any(mask):
            res.boxes = Boxes(res.boxes.data[np.array(mask)], res.orig_shape)
            for c in res.boxes.cls:
                old = active_model.names[int(c)]
                res.names[int(c)] = ALLOWED_COCO_REMAP[old]
        else:
            res.boxes = Boxes(res.boxes.data[:0], res.orig_shape)

    # res.plot() returns BGR image with rendered boxes/labels
    return res.plot()

# ------------------------------------------------------------
# Video annotation (OpenCV with imageio fallback)
# ------------------------------------------------------------
def annotate_video(input_path: str, output_path: str, mode: str, conf: float, active_model: YOLO) -> bool:
    cap = cv2.VideoCapture(input_path)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_placeholder = st.empty()
        prog = st.progress(0.0)
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1, 1)
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated = process_frame(frame, mode, conf, active_model)
            out.write(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            if idx % 3 == 0:
                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Video Detection ({mode})",
                    use_container_width=True,
                )
                prog.progress(min(idx / total, 0.999))
            idx += 1

        cap.release()
        out.release()
        prog.progress(1.0)
        return True

    # Fallback: imageio (handles tricky codecs on cloud)
    try:
        reader = iio.imiter(input_path, plugin="FFMPEG", dtype="uint8")
        meta = iio.improps(input_path, plugin="FFMPEG")
        fps = getattr(meta, "fps", 24) or 24
        w, h = getattr(meta, "size", (None, None))

        if w is None or h is None:
            first = next(reader)
            h, w = first.shape[:2]
            frames_iter = (f for f in ([first] + list(reader)))
        else:
            frames_iter = reader

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        frame_placeholder = st.empty()
        prog = st.progress(0.0)

        processed = 0
        for f in frames_iter:
            frame_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            annotated = process_frame(frame_bgr, mode, conf, active_model)
            out.write(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            if processed % 3 == 0:
                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Video Detection ({mode})",
                    use_container_width=True,
                )
            processed += 1

        out.release()
        prog.progress(1.0)
        return True
    except Exception as e:
        st.error(f"Video decoding failed: {e}")
        return False

# ------------------------------------------------------------
# WebRTC transformer (browser webcam)
# ------------------------------------------------------------
class YOLOTransformer(VideoTransformerBase):
    def __init__(self, mode: str, conf: float, active_model: YOLO):
        self.mode = mode
        self.conf = conf
        self.active_model = active_model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        annotated = process_frame(img_bgr, self.mode, self.conf, self.active_model)
        # WebRTC expects RGB frames
        return av.VideoFrame.from_ndarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), format="rgb24")

# ------------------------------------------------------------
# Tabs UI
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📷 Image", "🎞️ Video File", "🎥 Live Camera"])

# ---------- Tab 1: Image ----------
with tab1:
    st.subheader("Test Image or Upload")
    use_sample = st.checkbox("Use sample image from /samples/test_image.jpg")
    img_file = None
    if not use_sample:
        img_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if use_sample:
        frame = cv2.imread("samples/test_image.jpg")
    elif img_file:
        file_bytes = np.frombuffer(img_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        frame = None

    if frame is not None:
        annotated = process_frame(frame, mode, CONF, active_model)
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption=f"Detections ({mode})",
            use_container_width=True,
        )

# ---------- Tab 2: Video ----------
with tab2:
    st.subheader("Test Video or Upload")
    use_sample_vid = st.checkbox("Use sample video from /samples/test_video.mp4")
    vid_file = None
    if not use_sample_vid:
        vid_file = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"])

    if use_sample_vid:
        video_path = "samples/test_video.mp4"
    elif vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_file.read())
        video_path = tfile.name
    else:
        video_path = None

    if video_path:
        t_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        success = annotate_video(video_path, t_out.name, mode, CONF, active_model)
        if success:
            st.success("Processing complete ✅")
            st.video(t_out.name)
            with open(t_out.name, "rb") as f:
                st.download_button(
                    "Download Annotated Video",
                    data=f,
                    file_name="annotated_output.mp4",
                    mime="video/mp4",
                )

# ---------- Tab 3: Webcam ----------
with tab3:
    st.subheader("Webcam Stream (Browser)")
    st.caption("Runs in your browser via WebRTC; grant camera permission to start.")
    ctx = webrtc_streamer(
        key="yolo-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CFG,
        media_stream_constraints={"video": True, "audio": False},
        video_transformer_factory=lambda: YOLOTransformer(mode, CONF, active_model),
    )
    if ctx and ctx.state.playing:
        st.success("Webcam connected.")
