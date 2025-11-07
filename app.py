import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import os
from gtts import gTTS
import speech_recognition as sr
import base64

st.set_page_config(page_title="Sign ↔ Speech Translator", layout="wide")

# Load model and labels
model = tf.keras.models.load_model("sign_model.h5")
with open("sign_labels.pkl", "rb") as f:
    label_dict = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

st.title("🤖 Sign ↔ Speech Translator")

col1, col2 = st.columns(2)

# 1️⃣ Sign → Text/Voice
with col1:
    st.header("🖐 Sign → Text/Voice")
    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    if run:
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
            while run:
                ret, frame = camera.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks = []
                    for lm in results.multi_hand_landmarks[0].landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)

                    prediction = model.predict([np.asarray(landmarks)])
                    pred_word = label_dict[np.argmax(prediction)]
                    cv2.putText(frame, pred_word, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Text + Voice Output
                    st.write(f"Detected Word: **{pred_word}**")
                    tts = gTTS(pred_word)
                    tts.save("output.mp3")
                    audio_file = open("output.mp3", "rb")
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3")

                FRAME_WINDOW.image(frame)

# 2️⃣ Voice/Text → Sign
with col2:
    st.header("🎤 Voice/Text → Sign")

    text_input = st.text_input("Type something (or use mic):")
    if st.button("🎙️ Speak"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            audio = r.listen(source)
            try:
                text_input = r.recognize_google(audio)
                st.write(f"You said: {text_input}")
            except:
                st.write("Sorry, couldn’t understand.")

    if text_input:
        st.write(f"Showing sign for: {text_input}")
        gif_path = f"sign_videos/{text_input.lower()}.gif"
        if os.path.exists(gif_path):
            file_ = open(gif_path, "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            st.markdown(f'<img src="data:image/gif;base64,{data_url}" width="300" />', unsafe_allow_html=True)
        else:
            st.warning("No sign video found for this word.")
