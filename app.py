import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import tflite_runtime.interpreter as tflite
from av import VideoFrame

# --- 1. Resources Loading (Using tflite-runtime to avoid crash) ---
# Mediapipe ko is tarah import karein taaki 'solutions' missing na ho
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh

# --- 1. Resources Loading ---
@st.cache_resource
def load_assets():
    # TFLite Interpreter (No TensorFlow for stability)
    interpreter = tflite.Interpreter(model_path="iris_pure_float32.tflite")
    interpreter.allocate_tensors()
    
    # Lens texture loading
    lens_img = cv2.imread("images/1.png", cv2.IMREAD_UNCHANGED)
    return interpreter, lens_img

# Initialize interpreter and assets
interpreter, lens_img = load_assets()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- 2. Mediapipe Face Mesh Setup ---
# Direct class call karein taaki attribute error na aaye
face_mesh_tool = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 2. Model Prediction Logic ---
def predict_mask_with_model(crop):
    # UNet input processing
    img = cv2.resize(crop, (384, 384)).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Mask creation
    mask = (np.squeeze(pred) > 0.3).astype(np.uint8) * 255
    return cv2.resize(mask, (crop.shape[1], crop.shape[0]))

# --- 3. Lens Application (Hybrid: Model + Landmarks) ---
def apply_hybrid_lens(frame, landmarks, lens_texture):
    h, w = frame.shape[:2]
    # Eye indices for masking
    EYE_INDICES = [
        (468, 469, [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]), # Left
        (473, 474, [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]) # Right
    ]

    for iris_center, iris_edge, eye_pts in EYE_INDICES:
        try:
            cx, cy = int(landmarks[iris_center].x * w), int(landmarks[iris_center].y * h)
            ex, ey = int(landmarks[iris_edge].x * w), int(landmarks[iris_edge].y * h)
            r = int(np.sqrt((cx-ex)**2 + (cy-ey)**2) * 1.3)
            
            y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0: continue
            
            # --- MODEL INFERENCE ---
            model_mask = predict_mask_with_model(crop)
            
            # Boundary mask (Eyelids)
            eye_poly = np.array([[(landmarks[p].x*w - x1), (landmarks[p].y*h - y1)] for p in eye_pts], dtype=np.int32)
            occ_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            cv2.fillPoly(occ_mask, [eye_poly], 255)
            
            final_mask = cv2.bitwise_and(model_mask, occ_mask)
            final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)

            # Blending
            lens_res = cv2.resize(lens_texture, (x2-x1, y2-y1))
            if lens_res.shape[2] == 4:
                alpha = (lens_res[:, :, 3] / 255.0) * (final_mask / 255.0)
                for c in range(3):
                    crop_c = crop[:, :, c].astype(float)
                    lens_c = lens_res[:, :, c].astype(float)
                    crop[:, :, c] = (alpha * lens_c + (1 - alpha) * crop_c).astype(np.uint8)
                frame[y1:y2, x1:x2] = crop
        except: continue
    return frame

class VideoProcessor(VideoTransformerBase):
    # 'transform' ki jagah 'recv' likhein
    def recv(self, frame):
        # Frame ko ndarray mein badlein
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        h_orig, w_orig = img.shape[:2]
        img_proc = cv2.resize(img, (640, 480))
        
        rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
        results = face_mesh_tool.process(rgb)
        
        if results.multi_face_landmarks:
            img_proc = apply_hybrid_lens(img_proc, results.multi_face_landmarks[0].landmark, lens_img)
            
        img_final = cv2.resize(img_proc, (w_orig, h_orig))

        # Naye tareeke mein VideoFrame object wapas bhejna hota hai
        return VideoFrame.from_ndarray(img_final, format="bgr24")

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
# --- Streamer Settings (Mobile Optimized) ---
webrtc_streamer(
    key="ttdeye-v3",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIG,
    # Async processing on karne se UI freeze nahi hoti
    async_processing=True,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "frameRate": {"ideal": 20}
        },
        "audio": False
    }
)