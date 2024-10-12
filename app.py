import os
import torch
from transformers import MarianTokenizer, MarianMTModel, pipeline
import gradio as gr
import torchaudio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor

asr_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
asr_model.eval()
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
translation_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

translation_pipe = pipeline(
    task="translation_en_to_fr",
    model=translation_model,
    tokenizer=tokenizer,
    device=device.index if torch.cuda.is_available() else -1
)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")


def transcribe_audio(audio_file_path):
    audio_array, _ = torchaudio.load(audio_file_path)
    input_values = processor(audio_array.squeeze(0), return_tensors="pt", padding="longest").input_values.to(device)

    with torch.no_grad():
        logits = asr_model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcribed_text = processor.decode(predicted_ids[0])
    return transcribed_text


def transcribe_and_translate(audio, task, return_timestamps):
    audio_file_path = audio  
    try:
        transcribed_text = transcribe_audio(audio_file_path)
        if task == "translate":
            translation_output = translation_pipe(transcribed_text)
            translated_text = translation_output[0]['translation_text']
            return transcribed_text, translated_text
        else:
            return transcribed_text, ""
    except Exception as e:
        return f"Transcription error: {e}", f"Translation error: {e}"


demo = gr.Blocks()

mic_transcribe = gr.Interface(
    fn=transcribe_and_translate,
    inputs=[
        gr.Audio(type="filepath", label="Microphone Input"),
        gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),
    ],
    outputs=[gr.Textbox(label="Transcription"), gr.Textbox(label="Translation")],
    title="Transcribe Audio",
    description="Record an audio file for transcription and translation.",
    allow_flagging="never",
)

file_transcribe = gr.Interface(
    fn=transcribe_and_translate,
    inputs=[
        gr.Audio(label="Audio file", type="filepath"),
        gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),
    ],
    outputs=[gr.Textbox(label="Transcription"), gr.Textbox(label="Translation")],
    title="Transcribe Audio",
    description="Upload an audio file for transcription and translation.",
    examples=[
        ["example.flac"],
    ],
    cache_examples=True,
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([mic_transcribe, file_transcribe], ["Transcribe Microphone", "Transcribe Audio File"])

demo.launch()
