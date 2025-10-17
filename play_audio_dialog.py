import sys
import os
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QLabel, QMessageBox
from PySide6.QtCore import Qt, Signal, QTimer
import sounddevice as sd
from scipy.io.wavfile import read as read_wav
import numpy as np
import threading # Import threading

class PlayAudioDialog(QDialog):
    def __init__(self, data_folder, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Play Recorded Audio")
        self.setModal(True)
        self.setFixedSize(400, 300)

        self.data_folder = data_folder
        self.current_audio_data = None
        self.current_sample_rate = None
        self.is_playing = False
        self.playback_thread = None # To hold the playback thread

        self.init_ui()
        self.load_audio_files()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        main_layout.addWidget(QLabel("Select an audio file to play:"))

        self.file_list_widget = QListWidget(self)
        self.file_list_widget.itemSelectionChanged.connect(self.on_file_selected)
        main_layout.addWidget(self.file_list_widget)

        button_layout = QHBoxLayout()
        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.play_selected_audio)
        self.play_button.setEnabled(False)
        button_layout.addWidget(self.play_button)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_playback)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        main_layout.addLayout(button_layout)

        self.status_label = QLabel("Ready.", self)
        main_layout.addWidget(self.status_label)

    def load_audio_files(self):
        self.file_list_widget.clear()
        if not os.path.exists(self.data_folder):
            QMessageBox.warning(self, "Folder Not Found", f"The data folder '{self.data_folder}' does not exist.")
            return

        audio_files = [f for f in os.listdir(self.data_folder) if f.endswith(".wav")]
        if not audio_files:
            self.file_list_widget.addItem("No .wav files found in 'data' folder.")
            self.play_button.setEnabled(False)
            return

        self.file_list_widget.addItems(audio_files)
        self.play_button.setEnabled(False)

    def on_file_selected(self):
        selected_items = self.file_list_widget.selectedItems()
        if selected_items:
            self.play_button.setEnabled(True)
            self.stop_playback() # Stop any current playback and reset buttons
            self.status_label.setText(f"Selected: {selected_items[0].text()}")
        else:
            self.play_button.setEnabled(False)
            self.stop_playback() # Ensure stop is also disabled
            self.status_label.setText("Ready.")

    def _reset_playback_buttons(self):
        self.is_playing = False
        self.stop_button.setEnabled(False)
        if self.file_list_widget.selectedItems():
            self.play_button.setEnabled(True)
            self.status_label.setText(f"Finished playing: {self.file_list_widget.selectedItems()[0].text()}")
        else:
            self.play_button.setEnabled(False)
            self.status_label.setText("Ready.")

    def _playback_worker(self, audio_data, sample_rate):
        try:
            sd.play(audio_data, sample_rate)
            sd.wait() # Wait until playback is finished
        except Exception as e:
            print(f"Error during playback in thread: {e}")
        finally:
            # Use QTimer.singleShot to update GUI from the main thread
            QTimer.singleShot(0, self._reset_playback_buttons)


    def play_selected_audio(self):
        selected_items = self.file_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No File Selected", "Please select an audio file to play.")
            return

        file_name = selected_items[0].text()
        file_path = os.path.join(self.data_folder, file_name)

        if not os.path.exists(file_path):
            QMessageBox.critical(self, "File Not Found", f"The selected file '{file_name}' does not exist.")
            return

        self.stop_playback() # Stop any currently playing audio

        try:
            self.current_sample_rate, self.current_audio_data = read_wav(file_path)
            # Ensure audio data is float32 for sounddevice
            if self.current_audio_data.dtype != np.float32:
                max_val = np.iinfo(self.current_audio_data.dtype).max
                self.current_audio_data = self.current_audio_data.astype(np.float32) / max_val

            self.playback_thread = threading.Thread(target=self._playback_worker, args=(self.current_audio_data, self.current_sample_rate))
            self.playback_thread.start()

            self.is_playing = True
            self.stop_button.setEnabled(True) # Enable stop button during play
            self.play_button.setEnabled(False) # Disable play button while playing
            self.status_label.setText(f"Playing: {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Playback Error", f"Failed to play audio: {e}")
            self.stop_playback()

    def stop_playback(self):
        if self.is_playing:
            sd.stop() # Stop sounddevice playback
            if self.playback_thread and self.playback_thread.is_alive():
                # It's generally not safe to forcefully stop a thread,
                # but sd.stop() should make sd.wait() return.
                # We can add a timeout for join if needed, but for now, assume sd.stop() is effective.
                pass # The _reset_playback_buttons will be called by the thread's finally block or directly below
            self.is_playing = False
            self._reset_playback_buttons() # Ensure buttons are reset immediately
            self.status_label.setText("Playback stopped.")
        else:
            self._reset_playback_buttons() # Just reset buttons if not playing but state needs cleanup

    def closeEvent(self, event):
        self.stop_playback()
        event.accept()

