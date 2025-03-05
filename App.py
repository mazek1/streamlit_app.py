import streamlit as st
import pandas as pd
import zipfile
import os
import tempfile
import openai
import json
from io import BytesIO
from PIL import Image

# Definer en fast sti til cache-filen, så den overlever genstarter
CACHE_FILE = ".streamlit/description_cache.json"

def analyze_image_with_openai(image_path):
    """Bruger OpenAI Vision API til at analysere billedet og generere en professionel, inspirerende, kortfattet og salgsmæssig beskrivelse med tre key points."""
    openai.api_key = os.getenv("OPENAI_API_KEY")  # Henter API-nøgle fra miljøvariabler
    
    with open(image_path, "rb") as image_file:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "You are an assistant that describes fashion products in a professional, inspiring, concise, and sales-oriented way. Each description should include three key points formatted as bullet points."},
                {"role": "user", "content": "Describe this fashion product in English, ensuring a professional tone, engaging language, and highlighting three key selling points."}
            ],
            files={"image": image_file}
        )
    return response["choices"][0]["message"]["content"] if "choices" in response else "No description available."

def load_cache():
    """Indlæser cache-filen hvis den findes."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as file:
            return json.load(file)
    return {}

def save_cache(cache):
    """Gemmer cache-filen."""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)  # Sikrer at mappen eksisterer
    with open(CACHE_FILE, "w") as file:
        json.dump(cache, file)

def process_excel_and_zip(excel_file, zip_file):
    # Indlæs Excel-fil
    xls = pd.ExcelFile(excel_file)
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    
    # Udpak ZIP-filen midlertidigt
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            image_files = {file for file in os.listdir(temp_dir) if file.endswith((".jpg", ".png", "jpeg"))}
        
        # Find style-numre fra billedfiler
        style_numbers = {file[:10] for file in image_files if file.startswith("SR")}
        
        # Indlæs cache
        cache = load_cache()
        
        # Progress bar
        progress_bar = st.progress(0)
        total_images = len(style_numbers)
        processed_images = 0
        
        # Opdater description baseret på AI-analyse af billeder
        for index, row in df.iterrows():
            style_no = row["Style No."]
            if style_no in style_numbers:
                image_path = os.path.join(temp_dir, next(f for f in image_files if f.startswith(style_no)))
                
                if style_no in cache:
                    description = cache[style_no]
                else:
                    description = analyze_image_with_openai(image_path)
                    cache[style_no] = description  # Gem i cache
                
                df.at[index, "Description"] = description  # Opdater Excel-data
                processed_images += 1
                progress_bar.progress(processed_images / total_images)
        
        # Gem cache efter behandling
        save_cache(cache)
    
    # Opdater B2C Tags
    df = update_b2c_tags(df)
    
    # Gem den opdaterede fil som en midlertidig fil
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        df.to_excel(tmp.name, index=False, sheet_name='Updated Data')
        tmp_path = tmp.name
    
    st.success("Processing Completed!")
    return tmp_path

# Streamlit UI
st.title("Product Data Processor")

excel_file = st.file_uploader("Upload Excel File", type=["xlsx"])
zip_file = st.file_uploader("Upload ZIP File with Images", type=["zip"])

if excel_file and zip_file:
    st.success("Files uploaded successfully. Processing...")
    processed_file_path = process_excel_and_zip(excel_file, zip_file)
    with open(processed_file_path, "rb") as file:
        st.download_button("Download Processed Excel File", file, "processed_data.xlsx")
