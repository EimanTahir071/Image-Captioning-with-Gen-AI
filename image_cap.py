#Step 1: Import your required tools from the transformers library
import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image, DON'T FORGET TO WRITE YOUR IMAGE NAME
img_path = r"/home/project/PROJECT_IMAGES/1.jpg"
# convert it into an RGB format 
image = Image.open(img_path).convert('RGB')

# Step 2: Load and Preprocess an Image
# You do not need a question for image captioning
text = "Sunflower"
inputs = processor(images=image, text=text, return_tensors="pt")

# Step 3: Generate caption
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print("Generated Caption:", caption)