import streamlit as st
import pandas as pd
import zipfile
import os
import tempfile
import openai
from io import BytesIO
from PIL import Image

def analyze_image_with_openai(image_path):
    """Bruger OpenAI Vision API til at analysere billedet og generere en beskrivelse."""
    import os
openai.api_key = os.getenv("OPENAI_API_KEY")
    
    with open(image_path, "rb") as image_file:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "You are an assistant that describes fashion products based on images."},
                {"role": "user", "content": "Describe this fashion product in a concise, professional way."}
            ],
            files=[("image", image_file)]
        )
    return response["choices"][0]["message"]["content"]

def update_b2c_tags(df):
    tag_translations = {
        "shirt": ["shirt", "shirts", "skjorte", "skjorter", "hemd", "hemden"],
        "blouse": ["blouse", "blouses", "blus", "blusar", "bluse", "blusen"],
        "dress": ["dress", "dresses", "klänning", "klänningar", "kleid", "kleider"],
        "pants": ["pants", "trousers", "byxor", "hose"],
        "skirt": ["skirt", "skirts", "kjol", "kjolar", "rock", "röcke"],
        "jacket": ["jacket", "jackets", "jacka", "jackor", "jacke", "jacken"],
        "blazer": ["blazer", "blazers", "kavaj", "kavajer", "sakko", "sakkos"],
        "ecovero": ["ecovero"],
        "gots": ["gots", "_tag_gots"],
        "_tag_grs": ["_tag_grs"]
    }

    df["B2C Tags"] = df["B2C Tags"].fillna("")

    for key, values in tag_translations.items():
        df.loc[df["Style Name"].str.contains(key, case=False, na=False), "B2C Tags"] += "," + ",".join(values)

    df["B2C Tags"] = df["B2C Tags"].apply(lambda x: ",".join(set(x.split(","))) if x else x)
    return df

def process_excel_and_zip(excel_file, zip_file):
    # Indlæs Excel-fil
    xls = pd.ExcelFile(excel_file)
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    
    # Udpak ZIP-filen midlertidigt
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            image_files = {file for file in os.listdir(temp_dir) if file.endswith((".jpg", ".png", ".jpeg"))}
        
        # Find style-numre fra billedfiler
        style_numbers = {file[:10] for file in image_files if file.startswith("SR")}
        
        # Opdater description baseret på AI-analyse af billeder
        for index, row in df.iterrows():
            style_no = row["Style No."]
            if pd.isna(row["Description"]) and style_no in style_numbers:
                image_path = os.path.join(temp_dir, next(f for f in image_files if f.startswith(style_no)))
                df.at[index, "Description"] = analyze_image_with_openai(image_path)
    
    # Opdater B2C Tags
    df = update_b2c_tags(df)
    
    # Gem den opdaterede fil som en midlertidig fil
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        df.to_excel(tmp.name, index=False, sheet_name='Updated Data')
        tmp_path = tmp.name
    
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
