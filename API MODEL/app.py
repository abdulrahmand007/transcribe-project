
import gradio as gr
from transcripe import transcribe_audio
from summrize import summarize, grammar_correction




custom_theme = gr.themes.Soft(
    primary_hue="blue",    
    secondary_hue="sky",   
    neutral_hue="slate",   
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    
    
)



with gr.Blocks(theme=custom_theme, title="Audio Transcription App") as demo:
    
    gr.Markdown("""
    # 🎙️ Audio Transcription Assistant
    Upload an audio file to transcribe, summarize, and correct grammar professionally.
    """)

    with gr.Row(equal_height=True):

        # Left Panel
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📂 Input Audio")
                
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath", 
                    sources=["upload"]
                    
                )

                with gr.Row():
                    transcribe_button = gr.Button(
                        "📝 Transcribe",
                        variant="primary"
                    )
                    summarize_button = gr.Button(
                        "📄 Summarize"
                    )

                grammar_button = gr.Button(
                    "✍️ Grammar Correction"
                    
                )

        # Right Panel
        with gr.Column(scale=2):

            with gr.Group():
                gr.Markdown("### 📝 Transcription")
                transcription_output = gr.Textbox(
                    label="Transcription Output",
                    lines=12,
                    buttons=["copy"],
                    placeholder="Transcribed text will appear here..."
                )

            with gr.Row():
                with gr.Column():
                    summary_output = gr.Textbox(
                        label="📄 Summary",
                        lines=6,
                        buttons=["copy"],
                        placeholder="Summary will appear here..."
                    )

                with gr.Column():
                    grammar_output = gr.Textbox(
                        label="✍️ Grammar Correction",
                        lines=6,
                        buttons=["copy"],
                        placeholder="Corrected text will appear here..."
                    )

    # Button Actions
    transcribe_button.click(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=transcription_output
    )

    summarize_button.click(
        fn=summarize,
        inputs=transcription_output,
        outputs=summary_output
    )

    grammar_button.click(
        fn=grammar_correction,
        inputs=transcription_output,
        outputs=grammar_output
    )
if __name__ == "__main__":
    # run the Gradio app
    demo.launch()