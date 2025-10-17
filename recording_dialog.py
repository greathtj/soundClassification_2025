import sys
import os
from datetime import datetime
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QMessageBox
from PySide6.QtCore import Qt, QTimer, Signal
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write as write_wav

class RecordingDialog(QDialog):
    recording_finished = Signal(np.ndarray, int, str) # Signal to emit audio data, sample rate, and class name

    def __init__(self, duration_seconds=20, sample_rate=44100, channels=1, class_name="unknown", recording_mode="training", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Recording Audio for {class_name}")
        self.setModal(True) # Make it a modal dialog
        self.setFixedSize(300, 220) # Increased height for new button

        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_buffer = []
        self.stream = None
        self.current_rms = 0.0 # To store the current RMS level
        self.is_recording_active = False # New state variable
        self.class_name = class_name # Store the class name
        self.recording_mode = recording_mode

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.status_label = QLabel("Ready to record...", self)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, self.duration_seconds * 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.level_meter = QProgressBar(self)
        self.level_meter.setRange(0, 100) # 0-100% for sound level
        self.level_meter.setValue(0)
        self.level_meter.setTextVisible(False)
        layout.addWidget(self.level_meter)

        self.start_button = QPushButton("Start Recording", self)
        self.start_button.clicked.connect(self._actual_start_recording)
        layout.addWidget(self.start_button)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.cancel_recording)
        self.cancel_button.setEnabled(False) # Initially disabled
        layout.addWidget(self.cancel_button)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.elapsed_time_ms = 0

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.audio_buffer.append(indata.copy())

        # Calculate RMS for sound level indicator
        rms = np.sqrt(np.mean(indata**2))
        self.current_rms = rms

    def _actual_start_recording(self):
        self.is_recording_active = True
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setEnabled(True)
        self.level_meter.setEnabled(True)
        self.status_label.setText(f"Recording for {self.duration_seconds} seconds...")

        try:
            self.stream = sd.InputStream(samplerate=self.sample_rate, channels=self.channels, callback=self.audio_callback)
            self.stream.start()
            self.timer.start(50) # Update progress and level meter more frequently (e.g., 50ms)
        except Exception as e:
            QMessageBox.critical(self, "Recording Error", f"Failed to start recording: {e}")
            self.cancel_recording() # Clean up and close dialog

    def start_recording(self): # This method now just shows the dialog
        self.exec()

    def update_progress(self):
        self.elapsed_time_ms += self.timer.interval()
        self.progress_bar.setValue(self.elapsed_time_ms)

        # Update sound level meter
        level_value = int(self.current_rms * 200)
        if level_value > 100: level_value = 100
        self.level_meter.setValue(level_value)

        if self.elapsed_time_ms >= self.duration_seconds * 1000:
            self.stop_recording()

    def stop_recording(self):
        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()
        self.timer.stop()
        self.status_label.setText("Processing and saving audio chunks...")
        self.level_meter.setValue(0) # Reset level meter
        self.is_recording_active = False

        if self.audio_buffer:
            recorded_audio = np.concatenate(self.audio_buffer, axis=0)

            if self.recording_mode == "training":
                save_folder = os.path.join(os.getcwd(), "data", self.class_name)
            else: # clips mode
                save_folder = os.path.join(os.getcwd(), "clips")

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                print(f"Created folder: {save_folder}")

            chunk_duration_seconds = 1
            samples_per_chunk = self.sample_rate * chunk_duration_seconds
            num_chunks = len(recorded_audio) // samples_per_chunk

            if num_chunks == 0:
                QMessageBox.warning(self, "Recording Issue", "Recorded audio is too short to create 1-second chunks.")
                self.reject()
                return

            timestamp_base = datetime.now().strftime("%Y%m%d_%H%M%S")
            last_file_path = ""
            last_audio_chunk = None

            for i in range(num_chunks):
                chunk_start = i * samples_per_chunk
                chunk_end = chunk_start + samples_per_chunk
                audio_chunk = recorded_audio[chunk_start:chunk_end]

                if self.recording_mode == "training":
                    filename = f"{self.class_name}_{timestamp_base}_chunk{i+1}.wav"
                else:
                    filename = f"clip_{timestamp_base}_chunk{i+1}.wav"

                file_path = os.path.join(save_folder, filename)
                write_wav(file_path, self.sample_rate, audio_chunk)
                last_file_path = file_path
                last_audio_chunk = audio_chunk

            if last_file_path:
                self.recording_finished.emit(last_audio_chunk, self.sample_rate, last_file_path)
            else:
                QMessageBox.warning(self, "Recording Issue", "No audio data was recorded to create chunks.")
        else:
            QMessageBox.warning(self, "Recording Issue", "No audio data was recorded.")
        self.accept() # Close dialog with accept status

    def cancel_recording(self):
        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()
        self.timer.stop()
        self.audio_buffer = [] # Clear buffer
        self.level_meter.setValue(0) # Reset level meter
        self.is_recording_active = False
        self.reject() # Close dialog with reject status

    def closeEvent(self, event):
        # Ensure stream is stopped if dialog is closed by other means (e.g., X button)
        if self.is_recording_active: # Only cancel if recording was actually started
            self.cancel_recording()
        event.accept()
