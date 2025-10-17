import sys
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QProgressBar, QMessageBox
from PySide6.QtCore import Qt, QTimer, Signal, QThread
import sounddevice as sd
import numpy as np
import torch
import torchaudio.transforms as T

from sound_classifier import preprocess_audio, classify_audio, SoundClassifierCNN # Import necessary functions and model

class AudioProcessor(QThread):
    # Signal to emit the processed audio chunk for classification
    audio_chunk_ready = Signal(np.ndarray)
    # Signal to emit the current RMS level for UI update
    rms_ready = Signal(float)

    def __init__(self, sample_rate, channels, chunk_size, parent=None):
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size # Number of frames per audio chunk
        self.running = False
        self.stream = None

    def run(self):
        self.running = True
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels,
                                blocksize=self.chunk_size, callback=self._audio_callback) as self.stream:
                print("Real-time audio stream started.")
                while self.running:
                    sd.sleep(100) # Keep the thread alive and responsive
        except Exception as e:
            print(f"Error in audio stream: {e}")
            self.running = False

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        
        print(f"Min: {indata.min():.6f}, Max: {indata.max():.6f}")

        # Emit the audio chunk for processing in the main thread or another worker
        self.audio_chunk_ready.emit(indata.copy())

        # Calculate RMS for sound level indicator
        rms = np.sqrt(np.mean(indata**2))
        self.rms_ready.emit(rms)

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.wait() # Wait for the thread to finish

class RealtimeClassifierDialog(QDialog):
    def __init__(self, model, class_names, sample_rate=44100, channels=1, chunk_size=44100, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Real-time Sound Classification")
        self.setFixedSize(400, 250)

        self.model = model
        self.class_names = class_names
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size # Process 1 second of audio at a time

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device) # Ensure model is on the correct device
        self.model.eval() # Set model to evaluation mode

        self.audio_processor = AudioProcessor(self.sample_rate, self.channels, self.chunk_size, self)
        self.audio_processor.audio_chunk_ready.connect(self._process_audio_chunk)
        self.audio_processor.rms_ready.connect(self._update_level_meter)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.status_label = QLabel("Ready to start real-time classification.", self)
        layout.addWidget(self.status_label)

        self.prediction_label = QLabel("Prediction: N/A", self)
        self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prediction_label)

        self.level_meter = QProgressBar(self)
        self.level_meter.setRange(0, 100) # 0-100% for sound level
        self.level_meter.setValue(0)
        layout.addWidget(self.level_meter)

        self.start_stop_button = QPushButton("Start Real-time Classification", self)
        self.start_stop_button.clicked.connect(self.toggle_realtime_classification)
        layout.addWidget(self.start_stop_button)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)

    def toggle_realtime_classification(self):
        if self.audio_processor.running:
            self.stop_realtime_classification()
        else:
            self.start_realtime_classification()

    def start_realtime_classification(self):
        self.status_label.setText("Starting audio stream...")
        self.start_stop_button.setText("Stop Real-time Classification")
        self.audio_processor.start()
        self.status_label.setText("Listening for sounds...")

    def stop_realtime_classification(self):
        self.status_label.setText("Stopping audio stream...")
        self.start_stop_button.setText("Start Real-time Classification")
        self.audio_processor.stop()
        self.status_label.setText("Real-time classification stopped.")
        self.prediction_label.setText("Prediction: N/A")
        self.level_meter.setValue(0)

    def _process_audio_chunk(self, indata):
        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(indata).float()
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        else:
            audio_tensor = audio_tensor.mean(dim=1, keepdim=True).T

        # Preprocess the audio chunk using the refactored preprocess_audio
        input_tensor = preprocess_audio(audio_tensor, sample_rate=self.sample_rate)

        if input_tensor is None:
            self.prediction_label.setText("Prediction: Error (Preprocessing)")
            return

        # Add batch dimension and move to device
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = self.class_names[predicted_idx.item()]
            self.prediction_label.setText(f"Prediction: {predicted_class}")

    def _update_level_meter(self, rms):
        level_value = int(rms * 200) # Scale RMS to 0-100
        if level_value > 100: level_value = 100
        self.level_meter.setValue(level_value)

    def closeEvent(self, event):
        self.stop_realtime_classification()
        event.accept()
