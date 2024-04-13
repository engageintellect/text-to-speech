from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch  # Add this line to import PyTorch

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)



if __name__ == "__main__":
    print("text to speech")
    user_input = input("Enter the text you want to convert to speech: ")
    speech = synthesiser(user_input, forward_params={"speaker_embeddings": speaker_embedding})
    sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
    print("Speech saved to speech.wav")


