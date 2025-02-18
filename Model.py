import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

# Load BLIP model & processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Load an image
image = Image.open("e1.jpg").convert("RGB")

# Prepare input
inputs = processor(image, return_tensors="pt")

# Generate caption
with torch.no_grad():
    output = model.generate(**inputs)

caption = processor.batch_decode(output, skip_special_tokens=True)[0]
print("Generated Caption:", caption)
