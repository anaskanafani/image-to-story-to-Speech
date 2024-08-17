import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from huggingface_hub import InferenceClient
from datasets import load_dataset
import torch
import numpy as np
import os

# * Initialize models and set environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_TOKEN = os.getenv("HF_API_TOKEN")

# * To get the image captioning model
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base")

# * To get the story generation model
client = InferenceClient(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    token=HF_TOKEN,
)

# * To get the text-to-speech model
speech_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
speech_model = SpeechT5ForTextToSpeech.from_pretrained(
    "microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

embeddings_dataset = load_dataset(
    "Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(
    embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# * Function to generate caption, story, and speech
def generate_story_and_speech(image):
    # * Image Captioning
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    context = blip_processor.decode(out[0], skip_special_tokens=True)

    # * Story Generation
    story = ""

    for message in client.chat_completion(
        messages=[{"role": "user", "content": f"generate a story that is at most 500 Characters long; the story should be about the image above; the story should rhyme and be in a poetic form; the story should be in English; the story should be unique and creative; the story should be interesting and engaging; Act like you are a story teller and you are telling a story about the this scenario. scenario: {context} Story:"}],
            max_tokens=500,
            stream=True,
    ):
        story += message.choices[0].delta.content

    story = story.replace("\n", " ")
    story = story[:600]

    # * Text-to-Speech Conversion
    inputs = speech_processor(text=story, return_tensors="pt")
    speech = speech_model.generate_speech(
        inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # * Convert speech tensor to NumPy array
    speech_numpy = speech.detach().cpu().numpy()

    # * Define sample rate
    sample_rate = 16000

    return context, story, (sample_rate, speech_numpy)


# * Gradio interface
iface = gr.Interface(
    fn=generate_story_and_speech,
    inputs=gr.Image(type="pil"),
    # * Outputs: Image Caption, Story, Speech
    outputs=[
        gr.Textbox(lines=3, label="Image Caption"),
        gr.Textbox(lines=10, label="Story"),
        gr.Audio(type="numpy", label="Speech"),
    ],
    title="Image-to-Story-to-Speech",
    description="Upload an image to generate a caption, a poetic story, and listen to the story in speech form.",
    allow_flagging="never",
)

iface.launch(share=True)
