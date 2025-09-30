import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

# for browser webcam
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
import av
import imageio.v3 as iio

# ------------------------------------------------------------
# Streamlit page setup
# ------------------------------------------------------------
st.set_page_config(page_title="YOLOv8 Human & Luggage", layout="wide")
st.title("YOLOv8 Detection: Human & Luggage 🚀")

# ------------------------------------------------------------
# Sidebar Settings
# ------------------------------------------------------------
st.sidebar.header("Settings")
mode = st.sidebar.radio("Select Model:", ["Custom (bestV3.pt)", "COCO (filtered)"], index=0)
CONF = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Allowed class remap for COCO
allowed_classes = {"person": "human", "suitcase": "luggage"}

# ------------------------------------------------------------
# Load models once
# ------------------------------------------------------------
@st.cache_resource
def load_custom():
    return YOLO("models/bestV3.pt")

@st.cache_resource
def load_coco():
    return YOLO("yolov8n.pt")

custom_model = load_custom()
coco_model = load_coco()

# ------------------------------------------------------------
# Detection logic
# ------------------------------------------------------------
def process_frame(frame, mode, conf, custom_model, coco_model):
    if mode == "Custom (bestV3.pt)":
        res = custom_model(frame, conf=conf)[0]
    else:  # COCO (filtered)
        res = coco_model(frame, conf=conf)[0]
        mask = [coco_model.names[int(c)] in allowed_classes for c in res.boxes.cls]
        if any(mask):
            res.boxes = Boxes(res.boxes.data[np.array(mask)], res.orig_shape)
            for c in res.boxes.cls:
                old = coco_model.names[int(c)]
                res.names[int(c)] = allowed_classes[old]
        else:
            res.boxes = Boxes(res.boxes.data[:0], res.orig_shape)  # empty
    return res.plot()

# ------------------------------------------------------------
# Video annotation (OpenCV + fallback to imageio)
# ------------------------------------------------------------
def annotate_video(input_path, output_path, mode, conf, custom_model, coco_model):
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
            annotated = process_frame(frame, mode, conf, custom_model, coco_model)
            out.write(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            if idx % 3 == 0:
                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Video Detection ({mode})", use_container_width=True
                )
                prog.progress(min(idx / total, 0.999))
            idx += 1
        cap.release()
        out.release()
        prog.progress(1.0)
        return True

    # Fallback: imageio
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
            annotated = process_frame(frame_bgr, mode, conf, custom_model, coco_model)
            out.write(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            if processed % 3 == 0:
                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Video Detection ({mode})", use_container_width=True
                )
            processed += 1
        out.release()
        prog.progress(1.0)
        return True
    except Exception as e:
        st.error(f"Video decoding failed: {e}")
        return False

# ------------------------------------------------------------
# WebRTC transformer for browser webcam
# ------------------------------------------------------------
class YOLOTransformer(VideoTransformerBase):
    def __init__(self, mode, conf, custom_model, coco_model):
        self.mode = mode
        self.conf = conf
        self.custom_model = custom_model
        self.coco_model = coco_model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        annotated = process_frame(img, self.mode, self.conf, self.custom_model, self.coco_model)
        return av.VideoFrame.from_ndarray(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), format="rgb24"
        )

# ------------------------------------------------------------
# Tabs
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
        annotated = process_frame(frame, mode, CONF, custom_model, coco_model)
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption=f"Detections ({mode})",
            use_container_width=True
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
        success = annotate_video(video_path, t_out.name, mode, CONF, custom_model, coco_model)
        if success:
            st.success("Processing complete ✅")
            st.video(t_out.name)
            with open(t_out.name, "rb") as f:
                st.download_button(
                    "Download Annotated Video",
                    data=f,
                    file_name="annotated_output.mp4",
                    mime="video/mp4"
                )

# ---------- Tab 3: Webcam ----------
with tab3:
    st.subheader("Webcam Stream (Browser)")
    st.caption("This captures from your browser, not the server.")
    webrtc_streamer(
        key="yolo-webrtc",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_transformer_factory=lambda: YOLOTransformer(mode, CONF, custom_model, coco_model),
    )
