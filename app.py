import streamlit as st
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# Load the model and processor
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

# Streamlit app title
st.title("Cryptocurrency Price Prediction")

# User input for cryptocurrency and time frame
crypto = st.text_input("Enter Cryptocurrency (e.g., Bitcoin, Ethereum):")
time_frame = st.selectbox("Select Time Frame:", ["1 Hour", "1 Day", "1 Week", "1 Month"])

# Button to predict price
if st.button("Predict Price"):
    if crypto:
        # Prepare input for the model
        input_text = f"Predict the price of {crypto} for the next {time_frame}."
        inputs = processor(input_text, return_tensors="pt", padding=True).to(model.device)

        # Generate prediction
        with torch.no_grad():
            output = model.generate(**inputs)
        
        # Decode the output
        prediction = processor.batch_decode(output, skip_special_tokens=True)[0]
        
        # Display the prediction
        st.success(f"The predicted price of {crypto} for the next {time_frame} is: {prediction}")
    else:
        st.error("Please enter a cryptocurrency name.")
