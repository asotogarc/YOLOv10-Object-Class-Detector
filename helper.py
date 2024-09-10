from ultralytics import YOLO
import streamlit as st
import cv2
import yt_dlp
import settings
import numpy as np

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

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )





def play_webcam_streamlit(conf, model):
    """
    Captures image from webcam using Streamlit's camera_input and performs object detection.

    Parameters:
        conf (float): Confidence threshold for object detection.
        model (YOLO): An instance of the YOLO model.

    Returns:
        None
    """
    st.header("Webcam Object Detection")
    
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # Convert image buffer to numpy array
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        
        # Perform object detection
        results = model(cv2_img_rgb, conf=conf)
        
        # Plot the detected objects on the image
        res_plotted = results[0].plot()
        
        # Display the image with detected objects
        st.image(res_plotted, caption="Detected Objects", use_column_width=True)
        
        # Display detection results
        with st.expander("Detection Results"):
            for result in results:
                for box in result.boxes:
                    st.write(f"Class: {model.names[int(box.cls)]}, Confidence: {float(box.conf):.2f}")
