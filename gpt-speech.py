from openai import OpenAI
from transformers import pipeline
from datasets import load_dataset
import torch
import sounddevice as sd
import numpy as np

config_file_path = '/etc/python-gpt.json'
openai_api_key = None

if os.path.exists(config_file_path):
    with open(config_file_path) as config_file:
        openai_api_key = json.load(config_file).get('OPENAI_API_KEY', "")
else:
    raise FileNotFoundError(f"Config file not found at {config_file_path}")

client = OpenAI(api_key=openai_api_key)

# Initialize the text-to-speech pipeline
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

# Load embeddings dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Set up OpenAI GPT


def get_gpt_response(prompt):
    """Queries GPT model and returns the response."""
    response = client.chat.completions.create(model="gpt-3.5-turbo",  # or "text-davinci-003" based on what models you have access to
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ])
    return response.choices[0].message.content.strip()

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
        print("\nText to Speech using ChatGPT")
        user_input = input("Enter your query (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        response_text = get_gpt_response(user_input)
        print(f"ChatGPT response: {response_text}")
        speak_text(response_text)

