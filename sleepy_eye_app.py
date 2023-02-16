import streamlit as st
from streamlit_webrtc import webrtc_streamer
import VideoProcessor_se as VideoProcessor_se
from PIL import Image


st.title("Aren't you asleep?")
image = Image.open('judge_eye.jpeg')

video_processor_factory = VideoProcessor_se.face_mesh_VideoProcessor

ctx = webrtc_streamer(
    key="example",
    video_processor_factory=video_processor_factory,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

if ctx.video_processor:
    st.write("目を閉じている連続フレーム数の判定基準")
    ctx.video_processor.judge_time = int(st.slider("フレーム数", min_value=1.0, max_value=50.0, step=1.0, value=20.0))
    st.write("　α　：目を閉じているかの判定基準")
    ctx.video_processor.judge_eye = st.slider("　α　", min_value=0.1, max_value=0.5, step=0.1, value=0.3)
    st.image(image, caption='')



