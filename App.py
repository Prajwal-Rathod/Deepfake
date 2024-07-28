import streamlit as st
import cv2
import numpy as np
import tempfile
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU
from keras.optimizers import Adam
from base64 import b64encode
import face_recognition
import imageio

# Define the Meso4 model
class Classifier:
    def __init__(self):
        self.model = None
    
    def predict(self, x):
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)

class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    
    def init_model(self):
        x = Input(shape=(256, 256, 3))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return KerasModel(inputs=x, outputs=y)

# Load the pre-trained Meso4 model
model_path = r'C:\Users\prajw\Documents\Deepfake_demo\Data Sources\Meso_pretrain\Meso4_DF'
classifier = Meso4()
classifier.load(model_path)

def process_and_display_video(video_file, classifier, save_interval=150, margin=0.2):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    video_capture = cv2.VideoCapture(video_path)
    frame_index = 0

    # Get video properties
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            offset = round(margin * h)
            y0 = max(y - offset, 0)
            x1 = min(x + w + offset, frame.shape[1])
            y1 = min(y + h + offset, frame.shape[0])
            x0 = max(x - offset, 0)
            face = frame[y0:y1, x0:x1]

            if frame_index % save_interval == 0:
                inp = cv2.resize(face, (256, 256)) / 255.0
                inp = np.expand_dims(inp, axis=0)
                re_img = classifier.predict(inp)
                deepfake_score = re_img[0][0]
            else:
                deepfake_score = 0.0  # Skip deepfake detection for other frames

            # Draw bounding box and annotate deepfake score
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Deepfake Score: {deepfake_score:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        frame_index += 1

    video_capture.release()
    out.release()

    return output_path

st.title("Deepfake Detection")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
if uploaded_video is not None:
    st.video(uploaded_video)

    try:
        output_path = process_and_display_video(uploaded_video, classifier)
        st.success("Video processed successfully!")

        # Display the processed video in Streamlit
        with open(output_path, 'rb') as file:
            video_bytes = file.read()
            st.video(video_bytes)

        # Provide a download button for the processed video
        with open(output_path, "rb") as f:
            video_data = f.read()
            st.download_button(
                label="Download Video",
                data=video_data,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
    except Exception as e:
        st.error(f"An error occurred: {e}")
