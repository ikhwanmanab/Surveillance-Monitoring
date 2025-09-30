# main.py
import os
import tempfile
from functools import lru_cache

import av
import cv2
import imageio.v3 as iio
import numpy as np
import streamlit as st
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

# WebRTC
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
)

# ------------------------------------------------------------
# Streamlit page setup
# ------------------------------------------------------------
st.set_page_config(page_title="YOLOv8 Human & Luggage", layout="wide")
st.title("YOLOv8 Detection: Human & Luggage 🚀")

# ------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------
COCO_PATH = "yolov8n.pt"          # keep at repo root
CUSTOM_PATH = "models/bestV3.pt"  # your custom model path

ALLOWED_COCO_REMAP = {"person": "human", "suitcase": "luggage"}

# WebRTC ICE servers (HTTPS page is required for getUserMedia)
RTC_CFG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        # Add TURN here if your network needs it.
    ]
})

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
mode_options = ["COCO (filtered)", "Custom (bestV3.pt)"]
mode = st.sidebar.radio("Select Model:", mode_options, index=0)
CONF = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def is_lfs_pointer(path: str) -> bool:
    """Detect Git LFS pointer files (tiny text with 'git-lfs')."""
    try:
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) > 2048:
            return False
        with open(path, "rb") as f:
            head = f.read(512).decode("utf-8", "ignore")
        return "git-lfs" in head.lower()
    except Exception:
        return False

@lru_cache(maxsize=2)
def load_model(selected_mode: str) -> YOLO:
    """
    Lazy-loads the model. On macOS we force CPU to avoid GPU/Metal issues
    unless you know your environment supports it.
    """
    if selected_mode.startswith("Custom"):
        if (not os.path.exists(CUSTOM_PATH)) or is_lfs_pointer(CUSTOM_PATH):
            raise FileNotFoundError(
                "Custom model `models/bestV3.pt` is missing or is a Git LFS pointer.\n"
                "Use COCO (filtered) for now, or add the file."
            )
        model = YOLO(CUSTOM_PATH)
    else:
        model = YOLO(COCO_PATH)

    # Force CPU (reliable on MacBooks)
    try:
        model.to("cpu")
    except Exception:
        pass
    return model

# Acquire active model or show a friendly error and stop
try:
    active_model = load_model(mode)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

def process_frame(frame_bgr: np.ndarray, mode: str, conf: float, model: YOLO) -> np.ndarray:
    """
    Run YOLO on a BGR frame and return a BGR frame with drawn boxes/labels.
    - If using COCO, we filter down to 'person' and 'suitcase' and relabel.
    """
    # Perform inference
    res = model(frame_bgr, conf=conf)[0]

    # COCO filtering
    if mode.startswith("COCO"):
        if res.boxes is not None and len(res.boxes) > 0:
            # Build a mask for allowed classes
            names = model.names  # dict id->name
            cls = res.boxes.cls
            allowed_mask = [names[int(c)] in ALLOWED_COCO_REMAP for c in cls]
            if any(allowed_mask):
                # Keep only allowed boxes
                res.boxes = Boxes(res.boxes.data[np.array(allowed_mask)], res.orig_shape)
                # Remap names for labeling
                for c in res.boxes.cls:
                    old = names[int(c)]
                    res.names[int(c)] = ALLOWED_COCO_REMAP[old]
            else:
                # No allowed classes found
                res.boxes = Boxes(res.boxes.data[:0], res.orig_shape)

    # Draw results; plot() returns BGR image with overlays
    annotated_bgr = res.plot()
    return annotated_bgr

def annotate_video(input_path: str, output_path: str, mode: str, conf: float, model: YOLO) -> bool:
    """
    Annotate a video file and save to output_path (MP4). Shows progress in UI.
    Uses OpenCV first; falls back to imageio if needed.
    """
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
            annotated = process_frame(frame, mode, conf, model)

            # annotated is BGR already; write BGR into OpenCV VideoWriter
            out.write(annotated)

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

    # Fallback: imageio (some codecs on cloud)
    try:
        reader = iio.imiter(input_path, plugin="FFMPEG", dtype="uint8")
        meta = iio.improps(input_path, plugin="FFMPEG")
        fps = getattr(meta, "fps", 24) or 24
        w, h = getattr(meta, "size", (None, None))

        first = None
        if w is None or h is None:
            first = next(reader)
            h, w = first.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        frame_placeholder = st.empty()
        prog = st.progress(0.0)

        processed = 0

        def _iter_frames():
            if first is not None:
                yield first
            for f in reader:
                yield f

        for f in _iter_frames():
            frame_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            annotated = process_frame(frame_bgr, mode, conf, model)
            out.write(annotated)
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

    frame = None
    if use_sample:
        if os.path.exists("samples/test_image.jpg"):
            frame = cv2.imread("samples/test_image.jpg")
        else:
            st.warning("samples/test_image.jpg not found.")
    elif img_file:
        file_bytes = np.frombuffer(img_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

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

    video_path = None
    if use_sample_vid:
        if os.path.exists("samples/test_video.mp4"):
            video_path = "samples/test_video.mp4"
        else:
            st.warning("samples/test_video.mp4 not found.")
    elif vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_file.read())
        video_path = tfile.name

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

# ---------- Tab 3: Webcam (macOS-friendly) ----------
with tab3:
    st.subheader("Webcam Stream (Browser)")
    st.caption("Click ▶ Start in the widget to begin. Use Chrome on macOS for best results.")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        annotated = process_frame(img_bgr, mode, CONF, active_model)
        # WebRTC expects RGB (uint8)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return av.VideoFrame.from_ndarray(rgb, format="rgb24")

    ctx = webrtc_streamer(
        key="yolo-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CFG,
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=video_frame_callback,   # callback style = reliable on Mac
        async_transform=True,                        # keeps UI smooth on CPU
    )

    if ctx and ctx.state.playing:
        st.success("Webcam connected ✅ — detecting...")
