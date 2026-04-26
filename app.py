import time
import threading
import av
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from scipy.spatial import distance
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25
DROWSY_TIME = 2.0


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.closed_start_time = None
        self.lock = threading.Lock()
        self.status = "Normal"
        self.ear = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24")
        img = np.fliplr(img).copy()

        h, w, _ = img.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        timestamp_ms = int(time.time() * 1000)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        status = "No face detected"
        ear = 0.0

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_eye = []
            right_eye = []

            for idx in LEFT_EYE:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                left_eye.append((x, y))
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 0))

            for idx in RIGHT_EYE:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                right_eye.append((x, y))
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 0))

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

            draw.text((30, 30), f"EAR: {ear:.2f}", fill=(255, 255, 255))

            if ear < EAR_THRESHOLD:
                if self.closed_start_time is None:
                    self.closed_start_time = time.time()

                if time.time() - self.closed_start_time >= DROWSY_TIME:
                    status = "DROWSINESS ALERT!"
                    draw.text((30, 70), "DROWSINESS ALERT!", fill=(255, 0, 0))
                else:
                    status = "Eyes closed"
            else:
                self.closed_start_time = None
                status = "Normal"
        else:
            self.closed_start_time = None
            draw.text((30, 30), "No face detected", fill=(255, 0, 0))

        with self.lock:
            self.status = status
            self.ear = ear

        output = np.array(pil_img)
        return av.VideoFrame.from_ndarray(output, format="rgb24")


st.set_page_config(page_title="Drowsiness Detection", page_icon="😴")

st.title("Drowsiness Detection using Webcam")
st.write("Click **START**, allow camera permission, and keep your face visible.")

ctx = webrtc_streamer(
    key="drowsiness-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    status_placeholder = st.empty()
    ear_placeholder = st.empty()

    while ctx.state.playing:
        with ctx.video_processor.lock:
            status = ctx.video_processor.status
            ear = ctx.video_processor.ear

        if status == "DROWSINESS ALERT!":
            status_placeholder.error("🚨 DROWSINESS ALERT!")
        elif status == "Eyes closed":
            status_placeholder.warning("⚠️ Eyes closed")
        elif status == "No face detected":
            status_placeholder.warning("No face detected")
        else:
            status_placeholder.success("✅ Normal")

        ear_placeholder.write(f"EAR: `{ear:.2f}`")
        time.sleep(0.5)
