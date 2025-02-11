import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Load environment variables
load_dotenv()

# Configure GenerativeAI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Generation configuration
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
    "response_mime_type": "text/plain",
}

# Safety settings
safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# System prompt for medical image analysis
system_prompt = """
As a highly skilled medical practitioner specializing in image analysis, you are tasked with examining medical images for a renowned hospital. Your expertise is crucial in identifying any anomalies, diseases, or health issues that may be present in the images.

Your Responsibilities include:
1. **Detailed Analysis**: Thoroughly analyze each image, focusing on identifying any abnormal findings.
2. **Findings Report**: Document all observed anomalies or signs of disease. Clearly articulate these findings in a structured format.
3. **Recommendations and Next Steps**: Based on your analysis, suggest potential next steps, including further tests or treatments.
4. **Treatment Suggestions**: If appropriate, recommend possible treatment options or interventions.

Important Notes:
1. **Scope of Response**: Only respond if the image pertains to human health issues.
2. **Clarity of Image**: In cases where the image quality impedes clear analysis, note that certain aspects are 'Unable to be determined based on the provided image.'
3. **Disclaimer**: Accompany your analysis with the disclaimer: "Consult with a Doctor before making any medical decisions."
4. **Your Insights are Invaluable** in guiding clinical decisions. Please proceed with the analysis, adhering to the structured approach outlined above.

Please provide an output response with these 4 headings: **Detailed Analysis**, **Findings Report**, **Recommendations and Next Steps**, **Treatment Suggestions**. For each heading, use bold text to highlight the heading.
"""

# Initialize the Generative Model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-lite-preview-02-05",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Streamlit App Configuration
st.set_page_config(page_title="Medical Image Analysis", page_icon=":robot:")

# Custom CSS for Dark Mode
dark_mode_css = """
<style>
    /* General dark mode styles */
    body {
        color: #ffffff;
        background-color: #1e1e1e;
    }
    .stApp {
        background-color: #1e1e1e;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select,
    .stFileUploader > div > div > button {
        background-color: #2d2d2d;
        color: #ffffff;
        border-color: #444444;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    .stMarkdown {
        color: #ffffff;
    }
    .stImage > img {
        border: 2px solid #444444;
        border-radius: 10px;
    }
    .stTitle {
        color: #ffffff;
    }
    .stSubheader {
        color: #ffffff;
    }
    .stWarning {
        background-color: #ffcc00;
        color: #000000;
    }
    .stError {
        background-color: #ff4d4d;
        color: #ffffff;
    }
</style>
"""

# Inject custom CSS
st.markdown(dark_mode_css, unsafe_allow_html=True)

# App Title and Subheader
st.title("👨‍⚕️ Medical Image Analysis")
st.subheader("An AI-powered tool to analyze medical images")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, width=400, caption="Uploaded Image")

# Analyze button
if st.button("Analyze") and uploaded_file:
    try:
        # Process image
        image_data = uploaded_file.getvalue()
        files = [{"mime_type": uploaded_file.type, "data": image_data}]

        # Generate response
        response = model.generate_content([files[0], system_prompt])

        # Display analysis report
        st.title("Analysis Report")
        st.markdown(response.text)  # Use markdown for better formatting
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
else:
    st.warning("Please upload an image to proceed.")
