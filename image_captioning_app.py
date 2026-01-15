import gradio as gr
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load model and processor once at startup
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def caption_image(input_image: np.ndarray) -> str:
    # Convert numpy array to PIL Image and ensure RGB
    raw_image = Image.fromarray(input_image).convert("RGB")

    # Prepare inputs and generate caption
    inputs = processor(images=raw_image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

    


iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."

)

if __name__ == "__main__":
    # share=True if you want a public link
    iface.launch()
