# Deepfake Detection App

This Streamlit application performs deepfake detection on uploaded videos using a pre-trained Meso4 model. The app processes the video, detects faces, and outputs a video with bounding boxes around detected faces along with their deepfake scores.

## Features

- Upload a video file for deepfake detection.
- Process the video to detect faces and deepfakes.
- Display the processed video with bounding boxes and deepfake scores.
- Download the processed video.

the linkes to the dataset and the pre-trained model is geven in the below link 
https://www.kaggle.com/code/srijonichakraborty/fake-detect-basic/input
please go throgh the format to acces the dataset for analysieis the pre trained model is provieded in the input section 

## Requirements

- Python 3.7+
- Streamlit
- OpenCV
- Keras
- TensorFlow
- face_recognition
- numpy
- albumentations
- tqdm

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/deepfake-detection-app.git
    cd deepfake-detection-app
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

3. Download the pre-trained Meso4 model and place it in the `Data Sources/Meso_pretrain` directory.

## Usage

1. Run the Streamlit app:

    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload a video file in mp4, mov, avi, or mkv format.

4. The app will process the video and display the processed video with bounding boxes and deepfake scores.

5. Download the processed video using the provided download button.

## Directory Structure

```plaintext
deepfake-detection-app/
│
├── Data Sources/
│   ├── Meso_pretrain/
│   │   ├── Meso4_DF
│   │   └── MesoInception_DF
│   └── Deepfake Detection Challenge/
│       ├── test_videos/
│       │   └── aassnaulhq.mp4
│       └── ... (other videos)
│
├── app.py
├── requirements.txt
└── README.md

