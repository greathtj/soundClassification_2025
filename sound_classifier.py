import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os

# --- 1. Sound Classifier CNN Model ---
class SoundClassifierCNN(nn.Module):
    def __init__(self, num_classes=10, n_mels=64, max_time_frames=128):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (32, n_mels/2, max_time_frames/2)

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (64, n_mels/4, max_time_frames/4)

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (128, n_mels/8, max_time_frames/8)
        )

        # Calculate the size of the flattened features dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_mels, max_time_frames)
            flattened_size = self.features(dummy_input).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Flatten for the linear layer
        x = self.classifier(x)
        return x

# --- 2. Audio Preprocessing ---
def preprocess_audio(audio_input, sample_rate=44100, n_mels=64, n_fft=1024, hop_length=512, max_time_frames=128):
    """
    Loads an audio file or processes a waveform tensor and converts it to a Mel Spectrogram.
    Pads or truncates the spectrogram to a fixed size.
    `audio_input` can be a file path (str) or a torch.Tensor (waveform).
    """
    waveform = None
    sr = sample_rate

    if isinstance(audio_input, str): # It's a file path
        try:
            waveform, sr = torchaudio.load(audio_input)
        except Exception as e:
            print(f"Error loading audio file {audio_input}: {e}")
            return None
    elif isinstance(audio_input, torch.Tensor): # It's a waveform tensor
        waveform = audio_input
        # Assume sample_rate is already correct for the tensor, or pass it explicitly
    else:
        print("Invalid audio_input type. Must be a file path or a torch.Tensor.")
        return None

    if sr != sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Create Mel Spectrogram transform
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spectrogram = mel_spectrogram_transform(waveform)

    # Convert to log scale
    mel_spectrogram = T.AmplitudeToDB()(mel_spectrogram)

    # Pad/Truncate to a fixed size (e.g., 1 channel, n_mels, max_time_frames)
    # Assuming mel_spectrogram is (1, n_mels, time_frames)
    if mel_spectrogram.shape[2] < max_time_frames:
        padding = max_time_frames - mel_spectrogram.shape[2]
        mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, padding))
    elif mel_spectrogram.shape[2] > max_time_frames:
        mel_spectrogram = mel_spectrogram[:, :, :max_time_frames]

    return mel_spectrogram # Shape (1, n_mels, max_time_frames)

# --- 3. Inference Function ---
def classify_audio(audio_input, model, class_names, device="cpu"):
    """
    Preprocesses audio and performs inference with the given model.
    Returns the predicted class name.
    `audio_input` can be a file path (str) or a torch.Tensor (waveform).
    """
    input_tensor = preprocess_audio(audio_input) # Now accepts path or tensor
    if input_tensor is None:
        return "Error: Could not preprocess audio."

    # Add batch dimension and move to device
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Perform inference
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = class_names[predicted_idx.item()]

    return predicted_class

# Example usage (for testing sound_classifier.py directly)
if __name__ == "__main__":
    # Create a dummy WAV file for testing
    if not os.path.exists("data"):
        os.makedirs("data")
    dummy_wav_path = os.path.join("data", "dummy_audio.wav")
    if not os.path.exists(dummy_wav_path):
        from scipy.io.wavfile import write
        samplerate = 44100
        duration = 1  # seconds
        frequency = 440  # Hz
        t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
        amplitude = np.iinfo(np.int16).max * 0.5
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        write(dummy_wav_path, samplerate, data.astype(np.int16))
        print(f"Created dummy audio file: {dummy_wav_path}")

    # Define dummy class names
    dummy_class_names = ["Dog Bark", "Car Horn", "Siren", "Speech", "Music", "Silence", "Alarm", "Rain", "Footsteps", "Keyboard"]

    # Instantiate the new model
    new_model = SoundClassifierCNN(num_classes=len(dummy_class_names))

    # Classify the dummy audio
    result = classify_audio(dummy_wav_path, new_model, dummy_class_names)
    print(f"Classification Result for '{dummy_wav_path}': {result}")

    # Test with a non-existent file
    result_error = classify_audio("non_existent.wav", new_model, dummy_class_names)
    print(f"Classification Result for 'non_existent.wav': {result_error}")
