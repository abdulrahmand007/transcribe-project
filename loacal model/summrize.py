import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)  


Groq_api_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=Groq_api_key)

def summarize(text):
    if not text:
        return "No text to summarize."

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
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
    if not text:
        return "No text to correct."

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": """You are a grammar corrector.
Correct grammar and spelling mistakes clearly."""
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            stream=True
        )

        full_response = ""

        for chunk in response:
            content = chunk.choices[0].delta.content

            if content is not None:
                full_response += content
                yield full_response

    except Exception as e:
        yield f"Grammar correction error: {str(e)}"