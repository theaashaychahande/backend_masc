import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
from collections import deque

import threading

st.set_page_config(page_title="Vision Detection App", layout="wide")
st.title("Smart Vision Detection System")

# Shared state for detections
class SharedState:
    def __init__(self):
        self.current_detections = []
        self.lock = threading.Lock()

    def update(self, detections):
        with self.lock:
            self.current_detections = detections

    def get(self):
        with self.lock:
            return self.current_detections

if "shared_state" not in st.session_state:
    st.session_state.shared_state = SharedState()

class DetectionSmoother:
    def __init__(self, buffer_size=5):
        self.buffer = deque(maxlen=buffer_size)

    def smooth(self, current_detections):
        self.buffer.append(current_detections)
        # Simple smoothing: if a label appears in >50% of recent frames, keep it
        final_detections = {}
        all_labels = [label for frame in self.buffer for label in frame]
        for label in set(all_labels):
            if all_labels.count(label) > len(self.buffer) / 2:
                # Average position (mocked here for simplicity)
                for frame in reversed(self.buffer):
                    if label in frame:
                        final_detections[label] = frame[label]
                        break
        return final_detections


@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo()

OBJECT_MAPPING = {
    67: "Pen",
    76: "Eraser",
    73: "Celotape"
}

def detect_objects(frame):
    results = yolo_model(frame, verbose=False)[0]
    detections = []
    labels_to_show = []
    if results.boxes is not None:
        for result in results.boxes.data.tolist():
            if len(result) == 6:
                x1, y1, x2, y2, score, class_id = result
                if score > 0.4:
                    cid = int(class_id)
                    label = OBJECT_MAPPING.get(cid, None)
                    if label:
                        detections.append({'box': (int(x1), int(y1), int(x2), int(y2)), 'label': label, 'score': score})
                        labels_to_show.append(f"{label} ({score:.1%})")
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = f"{det['label']} ({det['score']:.1%})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame, labels_to_show

def detect_colors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    labels_to_show = []
    color_ranges = {
        "Red": ([0, 150, 50], [10, 255, 255]),
        "Blue": ([100, 150, 50], [130, 255, 255]),
        "Yellow": ([25, 100, 100], [35, 255, 255]),
        "White": ([0, 0, 180], [180, 50, 255])
    }

    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if contours and cv2.contourArea(contours[0]) > 1000:
            x, y, w, h = cv2.boundingRect(contours[0])
            area = cv2.contourArea(contours[0])
            total_box_area = w * h
            accuracy = min((area / total_box_area) * 100, 99.9) if total_box_area > 0 else 0
            
            label = f"{color_name} ({accuracy:.1f}%)"
            labels_to_show.append(label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame, labels_to_show

def detect_shapes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    labels_to_show = []
 
    # For a black background, objects are likely bright.
    # THRESH_BINARY + THRESH_OTSU should make objects white and background black.
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 1500: continue
        
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(cnt)
        
        shape = ""
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            aspect_ratio = float(w) / h
            # Eraser is Square-like
            if 0.85 <= aspect_ratio <= 1.15:
                shape = "Eraser (Square)"
            # Pen is a long Rectangle (high or low aspect ratio)
            elif aspect_ratio > 3.0 or aspect_ratio < 0.33:
                shape = "Pen (Long Rectangle)"
            else:
                shape = "Rectangle"
        elif len(approx) > 6:
            area = cv2.contourArea(cnt)
            circularity = 4 * np.pi * area / (peri * peri)
            # Celotape is Circular
            if circularity > 0.7:
                shape = "Celotape (Circle)"
            
        if shape:
            accuracy = 95.0
            label = f"{shape}"
            labels_to_show.append(label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
    return frame, labels_to_show



class VisionProcessor:
    def __init__(self, mode, shared_state):
        self.mode = mode
        self.shared_state = shared_state

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        labels = []
        
        if self.mode == "Object":
            img, labels = detect_objects(img)
        elif self.mode == "Color":
            img, labels = detect_colors(img)
        elif self.mode == "Shape":
            img, labels = detect_shapes(img)
        
        self.shared_state.update(labels)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")



tab1, tab2, tab3 = st.tabs(["Object Detection", "Color Detection", "Shape Detection"])

def start_webrtc(key, mode):
    shared_state = st.session_state.shared_state
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        webrtc_streamer(
            key=key,
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: VisionProcessor(mode, shared_state),
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Display detections below the camera
        st.write("---")
        st.subheader("Currently Identifying:")
        labels_placeholder = st.empty()
        
        # In a real app, we might use a fragment or a custom component for live updates.
        # For now, we'll show the latest detections from the shared state.
        current_labels = shared_state.get()
        if current_labels:
            for label in current_labels:
                labels_placeholder.markdown(f"- **{label}**")
        else:
            labels_placeholder.info("No objects detected yet. Please ensure good lighting and positioning.")

with tab1:
    st.markdown("### Object Identification: Pen, Eraser, Celotape")
    start_webrtc("obj", "Object")

with tab2:
    st.markdown("### Color Identification: Red, Blue, Yellow, White")
    start_webrtc("col", "Color")

with tab3:
    st.markdown("### Object Identification by Shape: Pen (Long), Eraser (Square), Celotape (Circle), Triangle")
    start_webrtc("shp", "Shape")

st.sidebar.info("""
### Setup Instructions
1. Run `python -m pip install -r requirements.txt`
2. Run `python -m streamlit run app.py`
3. Ensure your camera is not being used by another app.
""")

