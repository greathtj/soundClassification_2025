# Sound Classification Project

This project is a GUI-based application for real-time sound classification. It allows users to record audio, manage sound classes, train a classification model, and classify sounds using either pre-recorded files or a microphone in real-time.

## Features

- **Audio Recording:** Record 20-second audio clips for training or classification.
- **Class Management:** Dynamically add or remove sound classes.
- **Model Training:** Train a Convolutional Neural Network (CNN) for sound classification using the recorded audio clips.
- **Sound Classification:**
    - Classify audio from a `.wav` file.
    - Classify sound in real-time using the microphone.
- **Audio Playback:** Listen to the recorded audio files.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd sound_classification
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**
    - On Windows:
      ```bash
      .venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source .venv/bin/activate
      ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Application

To start the main application, run the following command:

```bash
python main.py
```

This will open the main window of the Sound Classifier application.

### Main Window Overview

-   **Load Audio File:** Load a `.wav` or `.mp3` file for classification.
-   **Record Audio (20s):** Record a 20-second audio clip. You can choose to record for training or as a general clip.
-   **Play Audio:** Open a dialog to play recorded audio files.
-   **Classify Sound:** Classify the loaded audio file or classify in real-time using the microphone.
-   **Add New Class:** Add a new class for sound classification.
-   **Remove Selected Class:** Remove an existing class and its associated audio files.
-   **Select Class for Recording:** Select the class for which you want to record a training audio clip.

### Training the Model

To train the sound classification model, run the `train.py` script:

```bash
python train.py
```

This script will use the audio files in the `data` directory to train the model and save it as `sound_classifier_model.pth`.

## File Structure

```
.
├── clips/                # Directory for general recorded audio clips
├── data/                 # Directory for training audio data, organized in subdirectories by class
├── .venv/                # Python virtual environment
├── ah.wav                # Sample audio file
├── ka.wav                # Sample audio file
├── class_names.txt       # File containing the names of the classes
├── main.py               # Main application entry point
├── train.py              # Script for training the model
├── sound_classifier.py   # Core sound classification logic and model definition
├── sound_classifier_model.pth # Trained sound classification model
├── requirements.txt      # Python dependencies
└── *.py                  # Other Python files for GUI dialogs
```

## Dependencies

This project uses the following major Python libraries:

-   **PySide6:** For the graphical user interface.
-   **PyTorch:** For building and training the neural network.
-   **sounddevice:** For audio recording and playback.
-   **NumPy & SciPy:** For numerical operations and audio file handling.

