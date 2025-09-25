import whisper
import torch

# Path to your audio
audio_path = "/videos/001_050/001_京上ル下ル_20150701_京都市役所周辺/pyavi/audio.wav"

# Device selection
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the large-v3-turbo model
print("Loading Whisper model 'large-v3-turbo'...")
model = whisper.load_model("large-v3-turbo", device=device)

# Transcribe audio
print(f"Transcribing {audio_path}...")
result = model.transcribe(audio_path)

# Print transcription
print("=== Transcription ===")
print(result.get("text", "<no text>"))
