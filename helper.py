from ultralytics import YOLO
import streamlit as st
import cv2
import yt_dlp
import settings
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import logging
logging.basicConfig(level=logging.INFO)


TRANSLATIONS = {
    'person': 'persona',
    'bicycle': 'bicicleta',
    'car': 'coche',
    'motorcycle': 'motocicleta',
    'airplane': 'avión',
    'bus': 'autobús',
    'train': 'tren',
    'truck': 'camión',
    'boat': 'barco',
    'traffic light': 'semáforo',
    'fire hydrant': 'hidrante',
    'stop sign': 'señal de stop',
    'parking meter': 'parquímetro',
    'bench': 'banco',
    'bird': 'pájaro',
    'cat': 'gato',
    'dog': 'perro',
    'horse': 'caballo',
    'sheep': 'oveja',
    'cow': 'vaca',
    'elephant': 'elefante',
    'bear': 'oso',
    'zebra': 'cebra',
    'giraffe': 'jirafa',
    # Añade más traducciones según sea necesario
}

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # Translate labels to Spanish
    for r in res:
        for box in r.boxes:
            class_id = int(box.cls[0])
            original_name = model.names[class_id]
            translated_name = TRANSLATIONS.get(original_name.lower(), original_name)
            box.cls[0] = list(TRANSLATIONS.keys()).index(original_name.lower())
            model.names[int(box.cls[0])] = translated_name

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Video Detectado',
                   channels="BGR",
                   use_column_width=True
                   )


def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'no_warnings': True,
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']


def play_youtube_video(conf, model):
    source_youtube = st.sidebar.text_input("YouTube Video url")
    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        if not source_youtube:
            st.sidebar.error("Please enter a YouTube URL")
            return

        try:
            st.sidebar.info("Extracting video stream URL...")
            stream_url = get_youtube_stream_url(source_youtube)

            st.sidebar.info("Opening video stream...")
            vid_cap = cv2.VideoCapture(stream_url)

            if not vid_cap.isOpened():
                st.sidebar.error(
                    "Failed to open video stream. Please try a different video.")
                return

            st.sidebar.success("Video stream opened successfully!")
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(
                        conf,
                        model,
                        st_frame,
                        image,
                        is_display_tracker,
                        tracker
                    )
                else:
                    break

            vid_cap.release()

        except Exception as e:
            st.sidebar.error(f"An error occurred: {str(e)}")


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption(
        'Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)

def play_webcam(conf, model_path):
    class VideoProcessor:
        def __init__(self):
            self.TRANSLATIONS = {
                'person': 'persona',
                'bicycle': 'bicicleta',
                'car': 'coche',
                'motorcycle': 'motocicleta',
                'airplane': 'avión',
                'bus': 'autobús',
                'train': 'tren',
                'truck': 'camión',
                'boat': 'barco',
                'traffic light': 'semáforo',
                'fire hydrant': 'hidrante',
                'stop sign': 'señal de stop',
                'parking meter': 'parquímetro',
                'bench': 'banco',
                'bird': 'pájaro',
                'cat': 'gato',
                'dog': 'perro',
                'horse': 'caballo',
                'sheep': 'oveja',
                'cow': 'vaca',
                'elephant': 'elefante',
                'bear': 'oso',
                'zebra': 'cebra',
                'giraffe': 'jirafa',
                'couch': 'sofá',
                # Añade más traducciones según sea necesario
            }
            self.model = YOLO(model_path)
            logging.info(f"Model loaded from {model_path}")

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            logging.info(f"Processing frame of shape {img.shape}")
            
            # Perform object detection
            results = self.model(img, conf=conf)
            logging.info(f"Detection results: {results}")
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Translate labels to Spanish and draw bounding boxes
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0].tolist()
                        class_id = int(box.cls[0])
                        original_name = self.model.names[class_id].lower()
                        translated_name = self.TRANSLATIONS.get(original_name, original_name)
                        conf_score = float(box.conf[0])
                        logging.info(f"Detected: {original_name} -> {translated_name} ({conf_score:.2f})")
                        
                        img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                        img = cv2.putText(img, f"{translated_name} {conf_score:.2f}", 
                                          (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                logging.info("No objects detected in this frame")

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        st.write("El streaming de la cámara web está activo. Los objetos detectados se mostrarán en tiempo real.")
    else:
        st.error("Error al iniciar el procesador de video. Por favor, recarga la página.")
    
    st.markdown("Nota: Asegúrate de permitir el acceso a la cámara cuando el navegador lo solicite.")


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
