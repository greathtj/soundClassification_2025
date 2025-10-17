import sys
import torch
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox, QComboBox, QLineEdit, QHBoxLayout
from PySide6.QtCore import Qt
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
import numpy as np
import os
from datetime import datetime
import shutil # Import shutil for directory removal

from recording_dialog import RecordingDialog
from recording_mode_dialog import RecordingModeDialog
from play_audio_dialog import PlayAudioDialog
from sound_classifier import SoundClassifierCNN, preprocess_audio, classify_audio # Import new components
from classification_options_dialog import ClassificationOptionsDialog # Import new dialog
from realtime_classifier_dialog import RealtimeClassifierDialog # Import new dialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sound Classifier")
        self.setGeometry(100, 100, 400, 500) # Increased height for new elements

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.file_path_label = QLabel("No file loaded.")
        self.load_button = QPushButton("Load Audio File")
        self.record_button = QPushButton("Record Audio (20s)")
        self.play_button = QPushButton("Play Audio")
        self.classify_button = QPushButton("Classify Sound")
        self.result_label = QLabel("Classification Result: N/A")

        # --- Class Management ---
        self.class_names = [] # Start with an empty list of class names
        self.load_existing_classes() # Load classes from disk

        self.class_input_label = QLabel("Add New Class:")
        self.class_input_field = QLineEdit()
        self.add_class_button = QPushButton("Add Class")
        self.add_class_button.clicked.connect(self.add_new_class)

        class_input_layout = QHBoxLayout()
        class_input_layout.addWidget(self.class_input_label)
        class_input_layout.addWidget(self.class_input_field)
        class_input_layout.addWidget(self.add_class_button)
        layout.addLayout(class_input_layout)

        self.remove_class_button = QPushButton("Remove Selected Class")
        self.remove_class_button.clicked.connect(self.remove_selected_class)
        layout.addWidget(self.remove_class_button)
        # --- End Class Management ---

        # --- Class Selection ---
        self.class_selection_label = QLabel("Select Class for Recording:")
        self.class_combo_box = QComboBox()
        self.class_combo_box.addItems(self.class_names) # Populate with loaded classes
        layout.addWidget(self.class_selection_label)
        layout.addWidget(self.class_combo_box)
        # --- End Class Selection ---

        layout.addWidget(self.file_path_label)
        layout.addWidget(self.load_button)
        layout.addWidget(self.record_button)
        layout.addWidget(self.play_button)
        layout.addWidget(self.classify_button)
        layout.addWidget(self.result_label)

        self.load_button.clicked.connect(self.load_audio_file)
        self.record_button.clicked.connect(self.record_audio)
        self.play_button.clicked.connect(self.open_play_dialog)
        self.classify_button.clicked.connect(self.classify_sound)

        self.current_audio_file = None
        self.recording_duration = 20
        self.sample_rate = 44100
        self.channels = 1

        # --- PyTorch Model Initialization ---
        # Model will be initialized/re-initialized when classes are added
        self.sound_model = None # Initialize as None
        self.init_classifier() # Call to initialize/update the classifier

    def load_existing_classes(self):
        data_folder = os.path.join(os.getcwd(), "data")
        if os.path.exists(data_folder):
            for item in os.listdir(data_folder):
                item_path = os.path.join(data_folder, item)
                if os.path.isdir(item_path):
                    self.class_names.append(item)
        self.class_names.sort() # Keep classes sorted alphabetically

    def init_classifier(self):
        # Initialize or re-initialize the model based on current class_names
        model_path = os.path.join(os.getcwd(), "sound_classifier_model.pth")
        class_names_path = os.path.join(os.getcwd(), "class_names.txt")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determine device here

        if os.path.exists(model_path) and os.path.exists(class_names_path):
            # Load class names
            with open(class_names_path, "r") as f:
                self.class_names = [line.strip() for line in f.readlines()]
            # Update the combo box with loaded class names
            self.class_combo_box.clear()
            self.class_combo_box.addItems(self.class_names)

            # Instantiate the model with the correct number of classes
            self.sound_model = SoundClassifierCNN(num_classes=len(self.class_names))
            self.sound_model.load_state_dict(torch.load(model_path, map_location=device)) # Load to device
            self.sound_model.to(device) # Move model to device
            self.sound_model.eval() # Set to evaluation mode
            print("Trained model and class names loaded successfully.")
        else:
            print("No trained model or class names found. Initializing dummy model.")
            if self.class_names:
                self.sound_model = SoundClassifierCNN(num_classes=len(self.class_names))
                self.sound_model.to(device) # Move dummy model to device as well
                self.sound_model.eval() # Set to evaluation mode
            else:
                self.sound_model = None # No model if no classes

    def add_new_class(self):
        new_class = self.class_input_field.text().strip()
        if new_class:
            if new_class not in self.class_names:
                # Create directory for the new class
                class_folder_path = os.path.join(os.getcwd(), "data", new_class)
                os.makedirs(class_folder_path, exist_ok=True)

                self.class_names.append(new_class)
                self.class_names.sort() # Keep classes sorted alphabetically
                self.class_combo_box.clear() # Clear and re-add to maintain sort order
                self.class_combo_box.addItems(self.class_names)
                self.class_input_field.clear()
                QMessageBox.information(self, "Class Added", f"Class '{new_class}' added. You can now record audio for it.\n\nRemember to retrain the model to use this new class for classification.")
            else:
                QMessageBox.warning(self, "Duplicate Class", f"Class '{new_class}' already exists.")
        else:
            QMessageBox.warning(self, "Invalid Class Name", "Please enter a class name.")

    def remove_selected_class(self):
        current_index = self.class_combo_box.currentIndex()
        if current_index >= 0:
            class_to_remove = self.class_combo_box.currentText()
            reply = QMessageBox.question(self, "Remove Class",
                                         f"Are you sure you want to remove the class '{class_to_remove}' and DELETE its folder and all audio files within it? This action cannot be undone.",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                # Remove from internal list and combo box
                self.class_names.pop(current_index)
                self.class_combo_box.removeItem(current_index)

                # Delete the corresponding folder from disk
                class_folder_path = os.path.join(os.getcwd(), "data", class_to_remove)
                if os.path.exists(class_folder_path):
                    try:
                        shutil.rmtree(class_folder_path)
                        QMessageBox.information(self, "Class Removed", f"Class '{class_to_remove}' and its folder deleted successfully.")
                    except Exception as e:
                        QMessageBox.critical(self, "Error Deleting Folder", f"Failed to delete folder '{class_folder_path}': {e}")
                else:
                    QMessageBox.warning(self, "Folder Not Found", f"Class folder '{class_folder_path}' not found on disk.")

                self.init_classifier() # Re-initialize classifier with new class count
        else:
            QMessageBox.warning(self, "No Class Selected", "Please select a class to remove.")

    def load_audio_file(self):
        print("Load Audio button clicked")
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Audio Files (*.wav *.mp3)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.current_audio_file = selected_files[0]
                self.file_path_label.setText(f"Loaded: {self.current_audio_file}")
                self.result_label.setText("Classification Result: N/A")

    def record_audio(self):
        print("Record Audio button clicked")

        mode_dialog = RecordingModeDialog(self)
        if mode_dialog.exec():
            mode = mode_dialog.mode
            if not mode:
                return # User closed the dialog

            self.record_button.setEnabled(False)
            self.load_button.setEnabled(False)
            self.play_button.setEnabled(False)
            self.classify_button.setEnabled(False)

            selected_class = self.class_combo_box.currentText()
            if mode == "training" and not selected_class:
                QMessageBox.warning(self, "No Class Selected", "Please add and select a class before recording for training.")
                self.record_button.setEnabled(True)
                self.load_button.setEnabled(True)
                self.play_button.setEnabled(True)
                self.classify_button.setEnabled(True)
                return

            recording_dialog = RecordingDialog(
                duration_seconds=self.recording_duration,
                sample_rate=self.sample_rate,
                channels=self.channels,
                class_name=selected_class if mode == "training" else "clip",
                recording_mode=mode,
                parent=self
            )
            recording_dialog.recording_finished.connect(self._handle_recorded_audio)
            recording_dialog.start_recording()

            self.record_button.setEnabled(True)
            self.load_button.setEnabled(True)
            self.play_button.setEnabled(True)
            self.classify_button.setEnabled(True)

    def _handle_recorded_audio(self, audio_data, sample_rate, file_path):
        if file_path:
            self.current_audio_file = file_path
            self.file_path_label.setText(f"Recorded: {self.current_audio_file}")
            self.result_label.setText("Classification Result: N/A")
            QMessageBox.information(self, "Recording Complete", f"Audio recorded and saved to:\n{self.current_audio_file}")
        else:
            self.file_path_label.setText("Recording cancelled or failed.")
            self.result_label.setText("Classification Result: N/A")

    def open_play_dialog(self):
        print("Play Audio button clicked - opening dialog")
        data_folder = os.path.join(os.getcwd(), "data")
        play_dialog = PlayAudioDialog(data_folder, self)
        play_dialog.exec()

    def classify_sound(self):
        print("Classify Sound button clicked")
        if not self.sound_model or not self.class_names:
            QMessageBox.warning(self, "Classification Error", "No trained model or no classes defined for classification.")
            self.result_label.setText("Classification Result: Error")
            return

        options_dialog = ClassificationOptionsDialog(self)
        options_dialog.classify_by_file_selected.connect(self._classify_by_file)
        options_dialog.classify_by_microphone_selected.connect(self._classify_by_microphone)
        options_dialog.exec() # Show the dialog

    def _classify_by_file(self):
        print("Classify by File selected")
        if self.current_audio_file:
            self.result_label.setText(f"Classifying: {os.path.basename(self.current_audio_file)}...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            classification_result = classify_audio(
                self.current_audio_file,
                self.sound_model,
                self.class_names,
                device=device
            )
            self.result_label.setText(f"Classification Result: {classification_result}")
        else:
            QMessageBox.warning(self, "Classification Error", "No audio file loaded for classification.")
            self.result_label.setText("Classification Result: Error")

    def _classify_by_microphone(self):
        print("Classify by Microphone selected")
        if not self.sound_model or not self.class_names:
            QMessageBox.warning(self, "Classification Error", "No trained model or no classes defined for real-time classification.")
            return

        realtime_dialog = RealtimeClassifierDialog(
            model=self.sound_model,
            class_names=self.class_names,
            sample_rate=self.sample_rate, # Use the main window's sample rate
            parent=self
        )
        realtime_dialog.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
