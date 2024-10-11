# Face Recognition and Emotion Detection System

This project is a **Face Recognition and Emotion Detection System** built using OpenCV, FER (Facial Emotion Recognition), and Streamlit. The system allows users to capture their faces, train a recognition model, and detect both faces and emotions in real time via a webcam.

## Features

- **Face Capture**: Capture multiple images of a user's face for training.
- **Face Training**: Train the face recognition model using the captured images.
- **Real-time Recognition and Emotion Detection**: Detect and recognize faces along with emotions (such as happiness, sadness, etc.) from live webcam feed.

## Technology Stack

- **Python**
- **OpenCV**: For face detection and recognition.
- **FER**: For detecting emotions from faces.
- **Streamlit**: For creating a web-based user interface.

## Setup Instructions

1. Clone this repository:
    ```
 
    git clone https://github.com/BharathChandu3/Face-Recognition-and-Emotion-Recognition.git
    ```
    
3. Navigate to the project directory:
   
    cd face-emotion-detection
    ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

5. Run the application:
    ```bash
    streamlit run app.py
    ```

6. Open your browser and go to `http://localhost:8501` to use the system.

## Usage

1. **Capture Faces**: Enter your name and click the "Capture Face" button to store your facial images.
2. **Train Model**: After capturing your face, click the "Train Faces" button to train the recognition model.
3. **Recognize Faces and Emotions**: Click the "Recognize Faces and Emotions" button to start the real-time face recognition and emotion detection.

## Folder Structure

 ├── faces/ # Folder to store captured face images 
 ├── app.py # Main application file 
 ├── trainer.yml # Saved trained face recognition model 
 ├── README.md # Project documentation 
 └── requirements.txt # Project dependencies
 ## Dependencies

- **OpenCV**: For face detection and recognition.
- **FER**: For emotion detection.
- **Streamlit**: For building the frontend interface.
- **Pillow**: For image processing.

Install all dependencies using the following command:
```bash
pip install -r requirements.txt
