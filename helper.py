from ultralytics import YOLO
import streamlit as st
import cv2
import yt_dlp
import settings
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

# Diccionario de traducción de etiquetas al español
SPANISH_LABELS = {
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
    Carga un modelo de detección de objetos YOLO desde la ruta especificada.
    """
    model = YOLO(model_path)
    return model

def translate_label(label):
    """
    Traduce una etiqueta al español si está en el diccionario, si no, devuelve la etiqueta original.
    """
    return SPANISH_LABELS.get(label.lower(), label)

def display_tracker_options():
    display_tracker = st.radio("Mostrar Rastreador", ('Sí', 'No'))
    is_display_tracker = True if display_tracker == 'Sí' else False
    if is_display_tracker:
        tracker_type = st.radio("Tipo de Rastreador", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Muestra los objetos detectados en un frame de video usando el modelo YOLOv8.
    """
    image = cv2.resize(image, (720, int(720*(9/16))))

    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)

    # Traducir las etiquetas al español
    for r in res:
        for box in r.boxes:
            class_id = int(box.cls[0])
            original_label = model.names[class_id]
            box.cls[0] = model.names[class_id] = translate_label(original_label)

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
    source_youtube = st.sidebar.text_input("URL del video de YouTube")
    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detectar Objetos'):
        if not source_youtube:
            st.sidebar.error("Por favor, ingrese una URL de YouTube")
            return

        try:
            st.sidebar.info("Extrayendo URL del stream de video...")
            stream_url = get_youtube_stream_url(source_youtube)

            st.sidebar.info("Abriendo stream de video...")
            vid_cap = cv2.VideoCapture(stream_url)

            if not vid_cap.isOpened():
                st.sidebar.error(
                    "No se pudo abrir el stream de video. Por favor, intente con un video diferente.")
                return

            st.sidebar.success("¡Stream de video abierto exitosamente!")
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
            st.sidebar.error(f"Ocurrió un error: {str(e)}")

def play_rtsp_stream(conf, model):
    source_rtsp = st.sidebar.text_input("URL del stream RTSP:")
    st.sidebar.caption(
        'Ejemplo de URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detectar Objetos'):
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
            st.sidebar.error("Error al cargar el stream RTSP: " + str(e))

def play_webcam(conf, model):
    """
    Captura y procesa video en vivo desde la webcam usando streamlit-webrtc.
    Detecta objetos en tiempo real usando el modelo de detección de objetos YOLOv8.
    """
    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Realizar detección de objetos
            results = model(img, conf=conf)
            
            # Traducir las etiquetas al español
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    original_label = model.names[class_id]
                    box.cls[0] = model.names[class_id] = translate_label(original_label)
            
            # Dibujar los objetos detectados en la imagen
            annotated_frame = results[0].plot()
            
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        st.write("El streaming de la cámara web está activo. Los objetos detectados se mostrarán en tiempo real.")
    
    st.markdown("Nota: Asegúrate de permitir el acceso a la cámara cuando el navegador lo solicite.")

def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox(
        "Elige un video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detectar Objetos en el Video'):
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
            st.sidebar.error("Error al cargar el video: " + str(e))
