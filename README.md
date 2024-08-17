# **Image-to-Story-to-Speech Transformer** 🖼️➡️📝➡️🔊

### **About** ✍️  
This project involves the development of an advanced AI pipeline that leverages **Hugging Face models** and **large language models (LLMs)** to seamlessly transform images into rich, auditory experiences. The pipeline automatically generates descriptive captions from images, crafts engaging and contextually relevant stories based on those captions, and finally converts the stories into lifelike speech. This project showcases the integration of vision, language, and speech synthesis technologies to create a comprehensive storytelling experience.

---

### **Impact** 🌍  
The **Image-to-Story-to-Speech Transformer** opens up new avenues in multimedia content creation, education, and entertainment. It enhances accessibility by providing auditory content derived from visual inputs, making information more accessible to individuals with visual impairments. Furthermore, this project exemplifies the potential of AI in creating personalized and engaging experiences, from bedtime stories to interactive educational tools, fostering creativity and learning.

---

### **Methodology** 🔬  
The project is structured in three key stages:

1. **Image Captioning**: Utilizing the **BLIP model** from Hugging Face, images are processed to generate descriptive captions.
2. **Story Generation**: The captions serve as prompts for story generation, powered by the **Mistral AI** model, creating rhymed, poetic, and contextually engaging stories.
3. **Text-to-Speech Conversion**: The generated stories are then converted into speech using the **SpeechT5 model** along with the **HiFi-GAN vocoder** for high-quality audio output.

---

### **Technologies Used** 💻  
- **Python**: Core programming language for developing the pipeline.
- **Hugging Face Models**: Pre-trained models for image captioning, story generation, and text-to-speech conversion.
- **Gradio**: User interface framework for building and launching the web app.

---

### **Code Snippets** 💾  
Below is a code snippet demonstrating the core functionality of the pipeline:

```python
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
        messages=[{"role": "user", "content": f"generate a story that is at most 500 Characters long; the story should be about the image above; the story should rhyme and be in a poetic form; the story should be in English; the story should be unique and creative; the story should be interesting and engaging; Act like you are a storyteller and you are telling a story about this scenario. scenario: {context} Story:"}],
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
```

---

### **Conclusion** 🎯  
The **Image-to-Story-to-Speech Transformer** is a powerful demonstration of the convergence of vision, language, and speech technologies. It not only showcases the potential of AI in multimedia content creation but also emphasizes the importance of integrating multiple AI models to achieve more complex and engaging outputs. This project lays the groundwork for future innovations in interactive storytelling and accessible content creation.

---

### **How to Run on Your Machine** 🖥️  
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/Image-to-Story-to-Speech.git
   cd Image-to-Story-to-Speech
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   export HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
   export HF_API_TOKEN=your_huggingface_token
   ```

4. **Launch the Gradio interface**:
   ```bash
   python app.py
   ```

5. **Access the web interface**:
   Open your browser and go to `http://localhost:7860` to start using the application.
