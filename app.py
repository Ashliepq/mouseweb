import cv2
import numpy as np
import os
import math
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ─── تنظیمات ظاهر وب‌سایت ───────────────────────────────────
st.set_page_config(page_title="Gesture Match Game", layout="centered")
st.title("🎯 پروژه آنلاین تشخیص ژست دست با هوش مصنوعی")
st.write("نسخه فوق پایدار وب: دستت رو جلوی وب‌کم بگیر تا افکت‌ها ظاهر بشن!")

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

# ─── بارگذاری عکس‌ها (کد اصلی شما) ───────────────────────────
@st.cache_data
def load_images():
    imgs = {}
    mapping = {
        "quiet":      "quiet.jpg",
        "middle":     "middle.jpg",
        "tongue":     "tongue.jpg",
        "heart":      "heart.jpg",
        "silhouette": "silhouette.jpg",
        "victory":    "victory.jpg",
        "vape":       "vape.jpg",
        "thumbsup":   "thumbsup.jpg",
        "angry":      "angry.jpg",
        "thinking":   "thinking.jpg",
    }
    for key, fname in mapping.items():
        path = os.path.join(IMAGES_DIR, fname)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                imgs[key] = img
    return imgs

images = load_images()

# ─── منطق تشخیص انگشتان و ژست‌ها (کد اصلی شما) ─────────────────
def fingers_up(lm, is_right):
    tips  = [4, 8, 12, 16, 20]
    bases = [3, 6, 10, 14, 18]
    up = []
    if is_right:
        up.append(lm[4].x < lm[3].x)
    else:
        up.append(lm[4].x > lm[3].x)
    for i in range(1, 5):
        up.append(lm[tips[i]].y < lm[bases[i]].y)
    return up

def detect_gesture(results, face_results=None):
    if not results.multi_hand_landmarks:
        return "silhouette"

    hand = results.multi_hand_landmarks[0]
    label = results.multi_handedness[0].classification[0].label
    lm = hand.landmark
    is_right = (label == "Right")
    up = fingers_up(lm, is_right)
    thumb, index, middle, ring, pinky = up

    if face_results and face_results.multi_face_landmarks:
        fl = face_results.multi_face_landmarks[0].landmark
        lip_x = (fl[13].x + fl[14].x) / 2
        lip_y = (fl[13].y + fl[14].y) / 2
        for tip in [4, 8, 12, 16, 20]:
            d = math.sqrt((lm[tip].x - lip_x)**2 + (lm[tip].y - lip_y)**2)
            if d < 0.07:
                return "vape"

    if thumb and not index and not middle and not ring and not pinky: return "thumbsup"
    if index and middle and not ring and not pinky and not thumb: return "victory"
    if middle and not index and not ring and not pinky: return "middle"
    if index and not middle and not ring and not pinky and not thumb: return "quiet"

    if len(results.multi_hand_landmarks) >= 2:
        up2 = fingers_up(results.multi_hand_landmarks[1].landmark,
                         results.multi_handedness[1].classification[0].label == "Right")
        if thumb and index and not middle and up2[0] and up2[1] and not up2[2]:
            return "heart"

    if index and middle and ring and pinky: return "tongue"
    if not index and not middle and not ring and not pinky and not thumb: return "angry"

    return "thinking"

# ─── کلاس پردازش زنده ویدیو در مرورگر ──────────────────────────
class GestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)
        self.face = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.6)
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        fh, fw = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res_hands = self.hands.process(rgb)
        res_face  = self.face.process(rgb)

        gesture = detect_gesture(res_hands, res_face)
        
        # ساخت پنل صورتی شما
        pink = np.full((fh, fw, 3), (220, 195, 255), dtype=np.uint8)

        if gesture in images:
            ov_img = images[gesture]
            scale = min(fw / ov_img.shape[1], fh / ov_img.shape[0])
            nw = int(ov_img.shape[1] * scale)
            nh = int(ov_img.shape[0] * scale)
            img_r = cv2.resize(ov_img, (nw, nh))
            ox = (fw - nw) // 2
            oy = (fh - nh) // 2
            if img_r.shape[2] == 4:
                a = img_r[:,:,3:4] / 255.0
                pink[oy:oy+nh, ox:ox+nw] = (img_r[:,:,:3] * a + pink[oy:oy+nh, ox:ox+nw] * (1-a)).astype(np.uint8)
            else:
                pink[oy:oy+nh, ox:ox+nw] = img_r

        # چسباندن دوربین و پنل صورتی کنار هم
        combined = np.hstack([img, pink])
        return combined

# اجرای سیستم استریم زنده ویدیو روی سایت
webrtc_streamer(key="gesture-mouse", video_transformer_factory=GestureTransformer)
