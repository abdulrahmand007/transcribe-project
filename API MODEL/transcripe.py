import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)
groq_api_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)



def transcribe_audio(audio_filepath):
    filename = audio_filepath
    if filename is None:
        return "Please upload an audio file."

    try:
        with open(filename, "rb") as file:
            
            transcription = client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3-turbo",
                temperature=0,
                response_format="verbose_json",
            )
            return transcription.text

    except Exception as e:
        return f"Error during transcription: {str(e)}"
    

    
