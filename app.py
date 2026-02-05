# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import tflite_runtime.interpreter as tflite
# # import streamlit as st
# # from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# # # ==========================
# # # 1️⃣ SETUP & CACHING
# # # ==========================
# # st.set_page_config(page_title="TTDEye AR Filter", layout="wide")
# # st.title("👁️ TTDEye Ultra-Realistic Lens Filter")

# # # Model aur Lens ko cache karna taaki baar baar load na ho
# # @st.cache_resource
# # def load_assets():
# #     interpreter = tflite.Interpreter(model_path="iris_pure_float32.tflite")
# #     interpreter.allocate_tensors()
# #     lens_img = cv2.imread("images/1.png", cv2.IMREAD_UNCHANGED)
# #     return interpreter, lens_img

# # interpreter, lens_img = load_assets()
# # input_details = interpreter.get_input_details()
# # output_details = interpreter.get_output_details()

# # mp_face_mesh = mp.solutions.face_mesh
# # face_mesh = mp_face_mesh.FaceMesh(
# #     refine_landmarks=True, 
# #     min_detection_confidence=0.6, 
# #     min_tracking_confidence=0.6
# # )

# # # ==========================
# # # 2️⃣ CORE LOGIC (Aapka Function)
# # # ==========================

# # def predict_mask(crop):
# #     # Mobile ke liye input size handling
# #     img = cv2.resize(crop, (384, 384)).astype(np.float32) / 255.0
# #     interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
# #     interpreter.invoke()
# #     pred = interpreter.get_tensor(output_details[0]['index'])[0]
# #     mask = (np.squeeze(pred) > 0.3).astype(np.uint8) * 255
# #     return cv2.resize(mask, (crop.shape[1], crop.shape[0]))

# # def apply_hybrid_lens(frame, landmarks, lens_texture):
# #     h, w = frame.shape[:2]
# #     LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# #     RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# #     eye_configs = [(468, 159, 145, 469, LEFT_EYE_POINTS), (473, 386, 374, 474, RIGHT_EYE_POINTS)]

# #     for iris_idx, top_idx, bot_idx, edge_idx, eye_pts in eye_configs:
# #         try:
# #             cx, cy = int(landmarks[iris_idx].x * w), int(landmarks[iris_idx].y * h)
# #             t_y, b_y = landmarks[top_idx].y * h, landmarks[bot_idx].y * h
# #             if abs(t_y - b_y) < (h * 0.012): continue 

# #             ex, ey = int(landmarks[edge_idx].x * w), int(landmarks[edge_idx].y * h)
# #             r = int(np.sqrt((cx - ex)**2 + (cy - ey)**2) * 1.25)
            
# #             y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
# #             crop = frame[y1:y2, x1:x2].copy()
# #             if crop.size == 0: continue
            
# #             # Masking & Realism Logic
# #             eye_poly = np.array([[(landmarks[p].x*w - x1), (landmarks[p].y*h - y1)] for p in eye_pts], dtype=np.int32)
# #             occ_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
# #             cv2.fillPoly(occ_mask, [eye_poly], 255)

# #             model_mask = predict_mask(crop)
# #             geo_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
# #             cv2.circle(geo_mask, (geo_mask.shape[1]//2, geo_mask.shape[0]//2), int(r * 0.95), 255, -1)
            
# #             final_mask = cv2.bitwise_and(cv2.bitwise_or(geo_mask, model_mask), occ_mask)
# #             final_mask = cv2.GaussianBlur(final_mask, (7, 7), 0)

# #             lens_res = cv2.resize(lens_texture, (x2-x1, y2-y1), interpolation=cv2.INTER_LANCZOS4)
# #             lens_res = cv2.GaussianBlur(lens_res, (3, 3), 0)

# #             if lens_res.shape[2] == 4:
# #                 gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
# #                 _, highlights = cv2.threshold(gray_crop, 210, 255, cv2.THRESH_BINARY)
# #                 hi_3d = cv2.merge([highlights]*3).astype(float) / 255.0

# #                 sh_map = np.ones(crop.shape[:2], dtype=np.float32)
# #                 sh_map[:int(sh_map.shape[0]*0.5), :] = 0.82
# #                 sh_3d = cv2.merge([sh_map]*3)

# #                 lens_bgr = cv2.convertScaleAbs(lens_res[:,:,:3].astype(float), alpha=1.2, beta=-10)
# #                 alpha = (lens_res[:,:,3]/255.0) * (final_mask/255.0)
# #                 alpha_3d = cv2.merge([alpha]*3)

# #                 fg = (lens_bgr * sh_3d) * alpha_3d
# #                 bg = crop.astype(float) * (1.0 - alpha_3d)
# #                 res = cv2.addWeighted(cv2.add(fg, bg), 1.0, crop.astype(float)*hi_3d, 0.45, 0)
# #                 frame[y1:y2, x1:x2] = np.clip(res, 0, 255).astype(np.uint8)
# #         except: continue
# #     return frame

# # # ==========================
# # # 3️⃣ STREAMLIT WEBRTC HANDLER
# # # ==========================

# # class VideoProcessor(VideoTransformerBase):
# #     def __init__(self):
# #         self.frame_count = 0

# #     def transform(self, frame):
# #         self.frame_count += 1
# #         img = frame.to_ndarray(format="bgr24")
        
# #         # --- Speed Fix 1: Downsize for Processing ---
# #         # 1080p ko 640p par le aayin taaki lag khatam ho jaye
# #         h_orig, w_orig = img.shape[:2]
# #         img_small = cv2.resize(img, (640, int(640 * h_orig / w_orig)))
# #         img_small = cv2.flip(img_small, 1)
        
# #         # --- Speed Fix 2: Process only Alternate Frames ---
# #         # Agar lag zyada ho, toh har doosra frame process karein
# #         if self.frame_count % 1 == 0: 
# #             rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
# #             results = face_mesh.process(rgb)
            
# #             if results.multi_face_landmarks:
# #                 img_small = apply_hybrid_lens(img_small, results.multi_face_landmarks[0].landmark, lens_img)

# #         # --- Quality Fix: Scale back to Original if needed ---
# #         # Display ke liye wapas original size par le jayin
# #         return cv2.resize(img_small, (w_orig, h_orig))

# # # Mobile browsers usually need this RTC Config
# # RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# # webrtc_streamer(
# #     key="lens-filter",
# #     video_processor_factory=VideoProcessor,
# #     rtc_configuration=RTC_CONFIGURATION,
# #     media_stream_constraints={"video": True, "audio": False},
# #     async_processing=True
# # )






# import cv2
# import mediapipe as mp
# import numpy as np
# import tflite_runtime.interpreter as tflite
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# # --- Assets Loading ---
# @st.cache_resource
# def load_resources():
#     interpreter = tflite.Interpreter(model_path="iris_pure_float32.tflite")
#     interpreter.allocate_tensors()
#     lens_img = cv2.imread("images/1.png", cv2.IMREAD_UNCHANGED)
#     return interpreter, lens_img

# interpreter, lens_img = load_resources()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     refine_landmarks=True, 
#     min_detection_confidence=0.5, 
#     min_tracking_confidence=0.5
# )

# # --- Processing Functions ---
# def predict_mask(crop):
#     img = cv2.resize(crop, (384, 384)).astype(np.float32) / 255.0
#     interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
#     interpreter.invoke()
#     pred = interpreter.get_tensor(output_details[0]['index'])[0]
#     mask = (np.squeeze(pred) > 0.3).astype(np.uint8) * 255
#     return cv2.resize(mask, (crop.shape[1], crop.shape[0]))

# # (Yahan aapka apply_hybrid_lens function aayega jo pehle diya tha)
# def apply_hybrid_lens(frame, landmarks, lens_texture):
#     h, w = frame.shape[:2]
#     LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
#     RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

#     eye_configs = [(468, 159, 145, 469, LEFT_EYE_POINTS), (473, 386, 374, 474, RIGHT_EYE_POINTS)]

#     for iris_idx, top_idx, bot_idx, edge_idx, eye_pts in eye_configs:
#         try:
#             cx, cy = int(landmarks[iris_idx].x * w), int(landmarks[iris_idx].y * h)
#             t_y, b_y = landmarks[top_idx].y * h, landmarks[bot_idx].y * h
#             if abs(t_y - b_y) < (h * 0.012): continue 

#             ex, ey = int(landmarks[edge_idx].x * w), int(landmarks[edge_idx].y * h)
#             r = int(np.sqrt((cx - ex)**2 + (cy - ey)**2) * 1.25)
            
#             y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
#             crop = frame[y1:y2, x1:x2].copy()
#             if crop.size == 0: continue
            
#             # Masking & Realism Logic
#             eye_poly = np.array([[(landmarks[p].x*w - x1), (landmarks[p].y*h - y1)] for p in eye_pts], dtype=np.int32)
#             occ_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
#             cv2.fillPoly(occ_mask, [eye_poly], 255)

#             model_mask = predict_mask(crop)
#             geo_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
#             cv2.circle(geo_mask, (geo_mask.shape[1]//2, geo_mask.shape[0]//2), int(r * 0.95), 255, -1)
            
#             final_mask = cv2.bitwise_and(cv2.bitwise_or(geo_mask, model_mask), occ_mask)
#             final_mask = cv2.GaussianBlur(final_mask, (7, 7), 0)

#             lens_res = cv2.resize(lens_texture, (x2-x1, y2-y1), interpolation=cv2.INTER_LANCZOS4)
#             lens_res = cv2.GaussianBlur(lens_res, (3, 3), 0)

#             if lens_res.shape[2] == 4:
#                 gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#                 _, highlights = cv2.threshold(gray_crop, 210, 255, cv2.THRESH_BINARY)
#                 hi_3d = cv2.merge([highlights]*3).astype(float) / 255.0

#                 sh_map = np.ones(crop.shape[:2], dtype=np.float32)
#                 sh_map[:int(sh_map.shape[0]*0.5), :] = 0.82
#                 sh_3d = cv2.merge([sh_map]*3)

#                 lens_bgr = cv2.convertScaleAbs(lens_res[:,:,:3].astype(float), alpha=1.2, beta=-10)
#                 alpha = (lens_res[:,:,3]/255.0) * (final_mask/255.0)
#                 alpha_3d = cv2.merge([alpha]*3)

#                 fg = (lens_bgr * sh_3d) * alpha_3d
#                 bg = crop.astype(float) * (1.0 - alpha_3d)
#                 res = cv2.addWeighted(cv2.add(fg, bg), 1.0, crop.astype(float)*hi_3d, 0.45, 0)
#                 frame[y1:y2, x1:x2] = np.clip(res, 0, 255).astype(np.uint8)
#         except: continue
#     return frame

# class VideoProcessor(VideoTransformerBase):
#     def __init__(self):
#         self.frame_count = 0

#     def transform(self, frame):
#         self.frame_count += 1
#         img = frame.to_ndarray(format="bgr24")
        
#         # Performance Fix: Process at 480p, then scale up
#         h_orig, w_orig = img.shape[:2]
#         img_proc = cv2.resize(img, (640, 480))
#         img_proc = cv2.flip(img_proc, 1)

#         # Process every 2nd frame for maximum speed on mobile
#         if self.frame_count % 2 == 0:
#             rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
#             results = face_mesh.process(rgb)
#             if results.multi_face_landmarks:
#                 img_proc = apply_hybrid_lens(img_proc, results.multi_face_landmarks[0].landmark, lens_img)

#         return cv2.resize(img_proc, (w_orig, h_orig))

# # RTC for Mobile Browsers
# RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# webrtc_streamer(
#     key="ttdeye-filter",
#     video_processor_factory=VideoProcessor,
#     rtc_configuration=RTC_CONFIG,
#     media_stream_constraints={"video": True, "audio": False},
#     async_processing=True
# )

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# --- Model Loading (The Professional Way to avoid Memory Crash) ---
@st.cache_resource
def load_tflite_model():
    # Mediapipe ki internal TFLite wrapper use kar rahe hain taaki RAM kam use ho
    from mediapipe.tasks.python import core
    from mediapipe.tasks.python import vision
    
    # Simple Interpreter for TFLite (Low memory footprint)
    import tensorflow.lite as tflite # Streamlit Cloud supports this inside mediapipe
    interpreter = tflite.Interpreter(model_path="iris_pure_float32.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@st.cache_resource
def get_lens():
    return cv2.imread("images/1.png", cv2.IMREAD_UNCHANGED)

lens_img = get_lens()
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# --- Aapki Model Prediction Logic ---
def predict_mask_with_your_model(crop):
    # Model input handling
    img = cv2.resize(crop, (384, 384)).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    mask = (np.squeeze(pred) > 0.3).astype(np.uint8) * 255
    return cv2.resize(mask, (crop.shape[1], crop.shape[0]))

# (Yahan aapka apply_hybrid_lens function aayega jo model use karta hai)
def apply_hybrid_lens(frame, landmarks, lens_texture):
    h, w = frame.shape[:2]
    LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    eye_configs = [(468, 159, 145, 469, LEFT_EYE_POINTS), (473, 386, 374, 474, RIGHT_EYE_POINTS)]

    for iris_idx, top_idx, bot_idx, edge_idx, eye_pts in eye_configs:
        try:
            cx, cy = int(landmarks[iris_idx].x * w), int(landmarks[iris_idx].y * h)
            t_y, b_y = landmarks[top_idx].y * h, landmarks[bot_idx].y * h
            if abs(t_y - b_y) < (h * 0.012): continue 

            ex, ey = int(landmarks[edge_idx].x * w), int(landmarks[edge_idx].y * h)
            r = int(np.sqrt((cx - ex)**2 + (cy - ey)**2) * 1.25)
            
            y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0: continue
            
            # Masking & Realism Logic
            eye_poly = np.array([[(landmarks[p].x*w - x1), (landmarks[p].y*h - y1)] for p in eye_pts], dtype=np.int32)
            occ_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
            cv2.fillPoly(occ_mask, [eye_poly], 255)

            model_mask = predict_mask(crop)
            geo_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            cv2.circle(geo_mask, (geo_mask.shape[1]//2, geo_mask.shape[0]//2), int(r * 0.95), 255, -1)
            
            final_mask = cv2.bitwise_and(cv2.bitwise_or(geo_mask, model_mask), occ_mask)
            final_mask = cv2.GaussianBlur(final_mask, (7, 7), 0)

            lens_res = cv2.resize(lens_texture, (x2-x1, y2-y1), interpolation=cv2.INTER_LANCZOS4)
            lens_res = cv2.GaussianBlur(lens_res, (3, 3), 0)

            if lens_res.shape[2] == 4:
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, highlights = cv2.threshold(gray_crop, 210, 255, cv2.THRESH_BINARY)
                hi_3d = cv2.merge([highlights]*3).astype(float) / 255.0

                sh_map = np.ones(crop.shape[:2], dtype=np.float32)
                sh_map[:int(sh_map.shape[0]*0.5), :] = 0.82
                sh_3d = cv2.merge([sh_map]*3)

                lens_bgr = cv2.convertScaleAbs(lens_res[:,:,:3].astype(float), alpha=1.2, beta=-10)
                alpha = (lens_res[:,:,3]/255.0) * (final_mask/255.0)
                alpha_3d = cv2.merge([alpha]*3)

                fg = (lens_bgr * sh_3d) * alpha_3d
                bg = crop.astype(float) * (1.0 - alpha_3d)
                res = cv2.addWeighted(cv2.add(fg, bg), 1.0, crop.astype(float)*hi_3d, 0.45, 0)
                frame[y1:y2, x1:x2] = np.clip(res, 0, 255).astype(np.uint8)
        except: continue
    return frame
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Resolution scaling for Speed
        h_orig, w_orig = img.shape[:2]
        img_proc = cv2.resize(img, (640, 480))
        
        results = mp_face_mesh.process(cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            # Model use karke lens lagana
            img_proc = apply_hybrid_lens(img_proc, results.multi_face_landmarks[0].landmark, lens_img)
            
        return cv2.resize(img_proc, (w_orig, h_orig))

# Start Streamer
webrtc_streamer(key="ttdeye", video_processor_factory=VideoProcessor, 
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))