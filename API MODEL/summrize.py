import os
from groq import Groq
from dotenv import load_dotenv
from openai import OpenAI




load_dotenv(override=True)
gemma = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("open_router_api_key")
)


Groq_api_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=Groq_api_key)

def summarize(text):
    if not text.strip():
        yield "No text to correct."
        return

    try:
        response = gemma.chat.completions.create(
            model="openai/gpt-oss-120b:free",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert text summarizer.

                    Summarize the given text accurately without changing its meaning.

                    Rules:
                    - Write the summary ONLY as bullet points.
                    - Each point must contain one key idea.
                    - Keep the summary concise and clear.
                    - Do not write paragraphs.
                    - Do not add explanations or extra commentary.
                    - Preserve the original meaning.

                    Output format:
                    • Point 1
                    • Point 2
                    • Point 3
                        """
                },
                {
                    "role": "user",
                    "content": f"Summarize this:\n{text}"
                }
            ],
            stream=True
        )

        full_summary = ""

        for chunk in response:
            content = chunk.choices[0].delta.content

            if content:
                full_summary += content
                yield full_summary

    except Exception as e:
        yield f"Summarization error: {str(e)}"
    
def grammar_correction(text):
    if not text.strip():
        yield "No text to correct."
        return

    try:
        response = gemma.chat.completions.create(
            model="openai/gpt-oss-120b:free",
            messages=[
                {
                    "role": "system",
                    "content": "Correct grammar mistakes only. Keep meaning unchanged. corrction should be like this: \n\nOriginal: I has a apple.\nCorrected: I have an apple."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            stream=True,
            max_tokens=400
        )

        full_response = ""

        for chunk in response:
            content = chunk.choices[0].delta.content if chunk.choices[0].delta else None

            if content:
                full_response += content
                yield full_response

    except Exception as e:
        yield f"Grammar correction error: {e}"