import json
import threading
import time
from collections import Counter, deque
from pathlib import Path
from textwrap import dedent

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


# ============================================================
# Signa — Accessibility Purple
# ============================================================

st.set_page_config(
    page_title="Signa — Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sign_language_final_41class.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "sign_language_classes_41.json"

# Preserve the optimized camera/inference settings.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 20
INFERENCE_INTERVAL = 0.08       # target up to ~12 predictions/sec
CONFIDENCE_THRESHOLD = 0.40
SMOOTHING_WINDOW = 3


# ============================================================
# Accessibility Purple UI
# ============================================================

st.markdown(
    dedent("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #15114a;
        --muted: #66658a;
        --purple: #6847f5;
        --purple-2: #8b6cff;
        --lavender: #f2edff;
        --line: rgba(104,71,245,.13);
        --white: rgba(255,255,255,.86);
    }

    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp {
        color: var(--ink);
        background:
            radial-gradient(circle at 4% 28%, rgba(221,189,255,.55), transparent 20%),
            radial-gradient(circle at 96% 18%, rgba(150,123,255,.32), transparent 22%),
            radial-gradient(circle at 55% 100%, rgba(211,197,255,.55), transparent 28%),
            linear-gradient(135deg, #fbf9ff 0%, #f3efff 48%, #ece8ff 100%);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 1440px;
        padding: 1.35rem 2rem 2rem;
    }

    /* ---------- top navigation ---------- */

    .nav {
        min-height: 72px;
        padding: 0 25px;
        border-radius: 26px;
        background: rgba(255,255,255,.73);
        border: 1px solid rgba(255,255,255,.9);
        box-shadow: 0 18px 55px rgba(80,57,150,.10);
        backdrop-filter: blur(18px);
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.55rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-mark {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 23px;
        background: linear-gradient(145deg, #7957ff, #5e3ee8);
        box-shadow: 0 9px 22px rgba(104,71,245,.25);
    }

    .brand-name {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -.04em;
        color: #171044;
    }

    .camera-ready {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        border-radius: 999px;
        padding: 10px 15px;
        background: rgba(110,77,246,.08);
        color: #30265d;
        font-size: .82rem;
        font-weight: 700;
    }

    .ready-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #22cf83;
        box-shadow: 0 0 12px rgba(34,207,131,.42);
    }

    /* ---------- hero ---------- */

    .hero {
        padding: 1.15rem .45rem .4rem .2rem;
    }

    .eyebrow {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(104,71,245,.08);
        color: #6847f5;
        font-size: .71rem;
        font-weight: 800;
        letter-spacing: .07em;
        margin-bottom: 1rem;
    }

    .hero-title {
        color: #171044;
        font-size: clamp(2.8rem, 5vw, 5rem);
        line-height: .98;
        letter-spacing: -.068em;
        font-weight: 800;
        max-width: 560px;
        margin-bottom: 1.15rem;
    }

    .hero-title .accent {
        color: #6847f5;
    }

    .hero-copy {
        max-width: 520px;
        color: #69688c;
        font-size: 1rem;
        line-height: 1.65;
        margin-bottom: 1.55rem;
    }

    .features {
        display: grid;
        gap: 12px;
        max-width: 500px;
    }

    .feature {
        display: flex;
        gap: 13px;
        align-items: center;
        padding: 11px 13px;
        border-radius: 17px;
        background: rgba(255,255,255,.46);
        border: 1px solid rgba(255,255,255,.72);
    }

    .feature-icon {
        flex: 0 0 42px;
        height: 42px;
        border-radius: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        background: #eee7ff;
    }

    .feature-title {
        color: #171044;
        font-size: .86rem;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .feature-copy {
        color: #777493;
        font-size: .74rem;
    }

    /* ---------- camera card ---------- */

    .camera-title-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 2px 0 10px;
        color: #171044;
        font-size: 1rem;
        font-weight: 800;
    }

    .camera-icon {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #ede6ff;
    }

    .camera-card {
        padding: 15px;
        border-radius: 26px;
        background: rgba(255,255,255,.70);
        border: 1px solid rgba(255,255,255,.9);
        box-shadow: 0 22px 65px rgba(80,57,150,.13);
        backdrop-filter: blur(18px);
    }

    .camera-help {
        margin-top: 11px;
        padding: 11px 13px;
        border-radius: 14px;
        color: #5264a3;
        background: rgba(220,231,255,.62);
        border: 1px solid rgba(120,151,255,.12);
        font-size: .74rem;
        line-height: 1.45;
    }

    .footer-note {
        color: #8b88a7;
        font-size: .7rem;
        text-align: center;
        margin-top: 1.1rem;
    }

    /* Let the WebRTC video occupy the camera card. */
    [data-testid="stCustomComponentV1"] {
        width: 100%;
    }

    @media (max-width: 900px) {
        .block-container {
            padding: 1rem;
        }
        .nav {
            border-radius: 20px;
        }
        .camera-ready {
            display: none;
        }
        .hero-title {
            font-size: 3rem;
        }
    }
    </style>
    """),
    unsafe_allow_html=True,
)


# ============================================================
# Model and optimized inference
# ============================================================
@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class labels file not found: {CLASS_NAMES_PATH}")

    keras_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if isinstance(labels, dict):
        try:
            labels = [labels[str(i)] for i in range(len(labels))]
        except KeyError:
            labels = list(labels.values())

    # Warm the model once. Without this, the first live prediction can
    # feel much slower because TensorFlow initializes kernels lazily.
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    _ = keras_model(dummy, training=False)

    # Build a tf.function once so repeated inference avoids Python/Keras
    # tracing overhead as much as possible.
    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=(1, 224, 224, 3),
                dtype=tf.float32,
            )
        ],
        reduce_retracing=True,
    )
    def fast_inference(x):
        return keras_model(x, training=False)

    # Warm the compiled inference path as well.
    _ = fast_inference(tf.convert_to_tensor(dummy))

    return fast_inference, list(labels)


try:
    fast_inference, class_names = load_assets()
except Exception as exc:
    st.error("The recognition engine could not be loaded.")
    st.code(str(exc))
    st.stop()


# ============================================================
# Non-blocking processor
# ============================================================

class SignLanguageProcessor(VideoProcessorBase):

    def __init__(self):
        self.lock = threading.Lock()
        self.frame_event = threading.Event()
        self.running = True

        self.latest_frame = None
        self.prediction = "READY"
        self.confidence = 0.0
        self.history = deque(maxlen=SMOOTHING_WINDOW)

        self.worker = threading.Thread(
            target=self._inference_loop,
            daemon=True,
        )
        self.worker.start()

    def _inference_loop(self):
        last_run = 0.0

        while self.running:
            self.frame_event.wait(timeout=0.15)

            if not self.running:
                return

            # Take only the newest frame and immediately release the lock.
            with self.lock:
                frame = self.latest_frame
                self.latest_frame = None
                self.frame_event.clear()

            if frame is None:
                continue

            # Throttle inference, never the video callback.
            elapsed = time.perf_counter() - last_run
            if elapsed < INFERENCE_INTERVAL:
                time.sleep(INFERENCE_INTERVAL - elapsed)

            try:
                # Fast BGR -> RGB conversion + resize.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(
                    rgb,
                    (224, 224),
                    interpolation=cv2.INTER_LINEAR,
                )

                batch = resized.astype(
                    np.float32,
                    copy=False,
                )[None, ...]

                output = fast_inference(
                    tf.convert_to_tensor(batch)
                )

                scores = output.numpy()[0]
                index = int(np.argmax(scores))
                confidence = float(scores[index])
                label = str(class_names[index])

                if confidence >= CONFIDENCE_THRESHOLD:
                    self.history.append(label)
                    stable_label = Counter(
                        self.history
                    ).most_common(1)[0][0]

                    with self.lock:
                        self.prediction = stable_label
                        self.confidence = confidence
                else:
                    with self.lock:
                        self.prediction = "—"
                        self.confidence = confidence

            except Exception:
                # Never interrupt the video stream because of one bad frame.
                pass

            last_run = time.perf_counter()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # recv() deliberately performs NO TensorFlow inference.
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        # Do not copy/queue every frame if inference is already working.
        # We simply replace the pending frame with the newest one.
        with self.lock:
            self.latest_frame = image.copy()
            self.frame_event.set()
            label = self.prediction
            confidence = self.confidence

        # Minimal overlay to keep per-frame processing extremely cheap.
        height, width = image.shape[:2]

        cv2.rectangle(
            image,
            (12, 12),
            (min(width - 12, 330), 91),
            (12, 14, 20),
            -1,
        )

        cv2.putText(
            image,
            "RECOGNIZED",
            (25, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (185, 195, 210),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            label,
            (25, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.92,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if label not in ("READY", "—"):
            cv2.putText(
                image,
                f"{confidence:.0%}",
                (max(160, width - 88), 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (205, 214, 228),
                1,
                cv2.LINE_AA,
            )

        cv2.circle(
            image,
            (width - 18, 18),
            5,
            (70, 220, 125),
            -1,
        )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )

    def stop(self):
        self.running = False
        self.frame_event.set()


# ============================================================
# Page
# ============================================================

st.markdown(
    '<div class="nav"><div class="brand"><div class="brand-mark">🤟</div><div class="brand-name">Signa</div></div><div class="camera-ready"><span class="ready-dot"></span>Camera ready</div></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([0.78, 1.32], gap="large")

with left:
    st.markdown(
        '<div class="eyebrow">REAL-TIME SIGN LANGUAGE RECOGNITION</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-title">Make every<br><span class="accent">gesture</span><br>understood.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-copy">Turn on your camera, make a sign, and see the recognized gesture instantly.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="feature"><div class="feature-icon">⚡</div><div><div class="feature-title">Real-time Recognition</div><div class="feature-copy">Signs are detected while you use the camera.</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="height:10px"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="feature"><div class="feature-icon">◎</div><div><div class="feature-title">Easy to Use</div><div class="feature-copy">Turn on your camera and start signing.</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="height:10px"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="feature"><div class="feature-icon">♡</div><div><div class="feature-title">For Everyone</div><div class="feature-copy">Designed around accessible communication.</div></div></div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="camera-card">', unsafe_allow_html=True)

    st.markdown(
        dedent("""
        <div class="camera-title-row">
            <span class="camera-icon">📷</span>
            Live Camera
        </div>
        """),
        unsafe_allow_html=True,
    )

    webrtc_streamer(
        key="signa-accessibility-purple",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": CAMERA_WIDTH, "max": CAMERA_WIDTH},
                "height": {"ideal": CAMERA_HEIGHT, "max": CAMERA_HEIGHT},
                "frameRate": {"ideal": CAMERA_FPS, "max": CAMERA_FPS},
            },
            "audio": False,
        },
        async_processing=True,
    )

    st.markdown(
        dedent("""
        <div class="camera-help">
            💡 Keep your hand clearly visible, use steady lighting,
            and hold each sign briefly for the best result.
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    dedent("""
    <div class="footer-note">
        Signa · Real-time gesture recognition for more accessible communication
    </div>
    """),
    unsafe_allow_html=True,
)
