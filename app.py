import cv2
import numpy as np
import os
import math
import streamlit as st
import mediapipe as mp

# ─── تنظیمات ظاهر وب‌سایت ───────────────────────────────────
st.set_page_config(page_title="Gesture Match Game", layout="centered")
st.title("🎯 پروژه آنلاین تشخیص ژست دست با هوش مصنوعی")
st.write("بدون نیاز به نصب هیچ ابزاری، وب‌کم خودت رو روشن کن و ژست‌ها رو امتحان کن!")

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

# ─── منطق تشخیص انگشتان (کد اصلی شما) ─────────────────────────
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

# ─── تشخیص ژست‌ها (کد اصلی شما) ──────────────────────────────
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

    if thumb and not index and not middle and not ring and not pinky:
        return "thumbsup"
    if index and middle and not ring and not pinky and not thumb:
        return "victory"
    if middle and not index and not ring and not pinky:
        return "middle"
    if index and not middle and not ring and not pinky and not thumb:
        return "quiet"

    if len(results.multi_hand_landmarks) >= 2:
        up2 = fingers_up(results.multi_hand_landmarks[1].landmark,
                         results.multi_handedness[1].classification[0].label == "Right")
        if thumb and index and not middle and up2[0] and up2[1] and not up2[2]:
            return "heart"

    if index and middle and ring and pinky:
        return "tongue"
    if not index and not middle and not ring and not pinky and not thumb:
        return "angry"

    return "thinking"

# ─── آماده‌سازی دیتکتورها ────────────────────────────────────
@st.cache_resource
def get_detectors():
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6)
    face = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.6)
    return hands, face

hands_detector, face_detector = get_detectors()
images = load_images()

GESTURE_NAMES = {
    "quiet":      "Shhh 🤫",
    "middle":     "😤",
    "tongue":     "Bleh! 😛",
    "heart":      "Love ❤️",
    "silhouette": "Back 🔄",
    "victory":    "Victory ✌️",
    "vape":       "Vape 💨",
    "thumbsup":   "Nice! 👍",
    "angry":      "Angry 😡",
    "thinking":   "Hmm 🤔",
}

# دریافت وب‌کم در محیط وب
img_file = st.camera_input("دوربینت رو روشن کن و ژست بگیر 👇")

if img_file is not None:
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    frame = cv2.flip(cv2_img, 1)
    fh, fw = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    res_hands = hands_detector.process(rgb)
    res_face  = face_detector.process(rgb)

    gesture = detect_gesture(res_hands, res_face)
    gesture_label = GESTURE_NAMES.get(gesture, gesture)

    # ساخت پنل صورتی اصلی شما
    panel_w = fw
    pink = np.full((fh, panel_w, 3), (220, 195, 255), dtype=np.uint8)

    if gesture in images:
        img = images[gesture]
        scale = min(panel_w / img.shape[1], fh / img.shape[0])
        nw = int(img.shape[1] * scale)
        nh = int(img.shape[0] * scale)
        img_r = cv2.resize(img, (nw, nh))
        ox = (panel_w - nw) // 2
        oy = (fh - nh) // 2
        if img_r.shape[2] == 4:
            a = img_r[:,:,3:4] / 255.0
            pink[oy:oy+nh, ox:ox+nw] = (img_r[:,:,:3] * a + pink[oy:oy+nh, ox:ox+nw] * (1-a)).astype(np.uint8)
        else:
            pink[oy:oy+nh, ox:ox+nw] = img_r

    if gesture_label:
        cv2.putText(pink, gesture_label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 30, 150), 2)

    # چسباندن فریم دوربین و پنل صورتی کنار هم (دقیقاً مثل خروجی دسکتاپت)
    combined = np.hstack([frame, pink])
    
    # نمایش خروجی نهایی در سایت
    st.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)