from PySide6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel

class RecordingModeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Recording Mode")
        self.layout = QVBoxLayout()
        self.label = QLabel("How do you want to save the recording?")
        self.training_button = QPushButton("For Training Dataset")
        self.clips_button = QPushButton("As Audio Clips")

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.training_button)
        self.layout.addWidget(self.clips_button)
        self.setLayout(self.layout)

        self.training_button.clicked.connect(self.accept_training)
        self.clips_button.clicked.connect(self.accept_clips)

        self.mode = None

    def accept_training(self):
        self.mode = "training"
        self.accept()

    def accept_clips(self):
        self.mode = "clips"
        self.accept()
