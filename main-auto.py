from transformers import pipeline
from datasets import load_dataset
import torch
import sounddevice as sd
import numpy as np

# Initialize the text-to-speech pipeline
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

# Load embeddings dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

def speak_text(text):
    """Converts text to speech and plays it."""
    print(f"Converting to speech: {text}")
    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
    audio = speech['audio']
    samplerate = speech['sampling_rate']
    audio_np = np.array(audio).astype(np.float32)
    sd.play(audio_np, samplerate=samplerate)
    sd.wait()  # Wait for the audio to finish playing
    print("Playback finished.")

if __name__ == "__main__":
    while True:
        print("\nText to Speech")
        user_input = input("Enter the text you want to convert to speech (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        speak_text(user_input)

