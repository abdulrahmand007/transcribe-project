import os
import torch
from transformers import pipeline
from huggingface_hub import login
from pydub import AudioSegment
from dotenv import load_dotenv


load_dotenv(override=True)  


hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(hf_token, add_to_git_credential=True)


if not torch.cuda.is_available():
    print("Warning: CUDA is not available. Running on CPU (will be slow).")
    device = -1 # CPU
    dtype = torch.float32
else:
    device = 0 # GPU
    dtype = torch.float16

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    torch_dtype=dtype,
    device=device,
    return_timestamps=True
)


def transcribe_audio(audio_filepath):
    if audio_filepath is None:
        return "Please upload an audio file."

    try:
        audio = AudioSegment.from_file(audio_filepath)

        # ensure minimum duration (Whisper struggles with ultra-short audio)
        if len(audio) < 1000:
            silence = AudioSegment.silent(duration=1000 - len(audio))
            audio += silence

        temp_file = "temp.wav"
        audio.export(temp_file, format="wav")

        result = pipe(temp_file)

        os.remove(temp_file)

        return result["text"]

    except Exception as e:
        return f"Error during transcription: {str(e)}"