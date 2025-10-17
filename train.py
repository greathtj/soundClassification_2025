import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os
import glob
from tqdm import tqdm
from sound_classifier import SoundClassifierCNN, preprocess_audio # Import from our existing file

# --- 1. Custom Dataset Class ---
class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.audio_files = []
        self.labels = []
        self.class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_path = os.path.join(root_dir, class_name)
            for audio_file in glob.glob(os.path.join(class_path, "*.wav")):
                self.audio_files.append(audio_file)
                self.labels.append(self.class_to_idx[class_name])

        print(f"Found {len(self.audio_files)} audio files across {len(self.class_names)} classes.")
        print(f"Class names: {self.class_names}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        input_tensor = self.transform(audio_path)
        if input_tensor is None:
            # Handle cases where preprocessing fails (e.g., corrupted file)
            # For now, we'll return a dummy tensor and a dummy label,
            # but in a real scenario, you might want to skip this sample
            # or log the error more robustly.
            print(f"Warning: Preprocessing failed for {audio_path}. Returning dummy data.")
            return torch.zeros((1, 64, 128)), -1 # Dummy tensor and label

        return input_tensor, torch.tensor(label, dtype=torch.long)

# --- 2. Training Function ---
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train() # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Wrap train_loader with tqdm for a progress bar
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # Zero the parameter gradients

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    # Hyperparameters
    ROOT_DIR = os.path.join(os.getcwd(), "data")
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and dataloader
    # We pass preprocess_audio as the transform function
    dataset = AudioDataset(root_dir=ROOT_DIR, transform=preprocess_audio)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Instantiate model
    num_classes = len(dataset.class_names)
    model = SoundClassifierCNN(num_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS, device)
    print("Training finished.")

    # Save the trained model
    model_save_path = os.path.join(os.getcwd(), "sound_classifier_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Optionally, you can also save the class names for later use in inference
    class_names_path = os.path.join(os.getcwd(), "class_names.txt")
    with open(class_names_path, "w") as f:
        for class_name in dataset.class_names:
            f.write(f"{class_name}\n")
    print(f"Class names saved to {class_names_path}")
