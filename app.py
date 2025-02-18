import streamlit as st
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

st.title("Image Captioning with BLIP")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # Generate Caption
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.batch_decode(output, skip_special_tokens=True)[0]

    st.write("**Generated Caption:**", caption)
