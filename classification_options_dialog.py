from PySide6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Signal

class ClassificationOptionsDialog(QDialog):
    classify_by_file_selected = Signal()
    classify_by_microphone_selected = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Classification Options")
        self.setFixedSize(250, 150)

        layout = QVBoxLayout(self)

        label = QLabel("How would you like to classify sound?", self)
        layout.addWidget(label)

        btn_by_file = QPushButton("By File", self)
        btn_by_file.clicked.connect(self._emit_by_file_signal)
        layout.addWidget(btn_by_file)

        btn_by_microphone = QPushButton("By Microphone (Real-time)", self)
        btn_by_microphone.clicked.connect(self._emit_by_microphone_signal)
        layout.addWidget(btn_by_microphone)

    def _emit_by_file_signal(self):
        self.classify_by_file_selected.emit()
        self.accept() # Close the dialog

    def _emit_by_microphone_signal(self):
        self.classify_by_microphone_selected.emit()
        self.accept() # Close the dialog
