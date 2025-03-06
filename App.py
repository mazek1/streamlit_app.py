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
    
    try:
        with open(image_path, "rb") as image_file:
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {"role": "system", "content": "You are an assistant that describes fashion products in a professional, inspiring, concise, and sales-oriented way. Each description should include three key points formatted as bullet points."},
                    {"role": "user", "content": "Describe this fashion product in English, ensuring a professional tone, engaging language, and highlighting three key selling points."}
                ],
                files={"image": image_file}
            )
        
        if "choices" in response and response["choices"]:
            return response["choices"][0]["message"]["content"]
        else:
            return "No description available."
    except Exception as e:
        return f"Error generating description: {str(e)}"

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

def update_b2c_tags(df):
    """Opdaterer B2C Tags baseret på Style Name og Quality"""
    tag_translations = {
        "shirt": ["shirt", "shirts", "skjorte", "skjorter", "hemd", "hemden"],
        "blouse": ["blouse", "blouses", "blus", "blusar", "bluse", "blusen"],
        "dress": ["dress", "dresses", "klänning", "klänningar", "kleid", "kleider"],
        "pants": ["pants", "trousers", "byxor", "hose"],
        "skirt": ["skirt", "skirts", "kjol", "kjolar", "rock", "röcke"],
        "jacket": ["jacket", "jackets", "jacka", "jackor", "jacke", "jacken"],
        "blazer": ["blazer", "blazers", "kavaj", "kavajer", "sakko", "sakkos"],
        "knit": ["knit", "knitwear", "strik", "stickat", "gestrickt"],
        "rollneck": ["rollneck", "roll neck", "rullekrave", "polo krage", "rollkragen", "polokragen"],
        "cardigan": ["cardigan", "cardigans", "cardigan", "kofta", "strickjacke", "strickjacken"],
        "o-neck": ["o-neck", "o neck", "rund hals", "rundhals", "runda halsen"],
        "v-neck": ["v-neck", "v neck", "v-hals", "v-halsausschnitt"],
        "ecovero": ["ecovero"],
        "gots": ["gots", "_tag_gots"],
        "_tag_grs": ["_tag_grs"],
        "tencel": ["tencel"],
        "lenzing": ["lenzing", "ecovero"]
    }

    df["B2C Tags"] = df["B2C Tags"].fillna("").astype(str)

    for key, values in tag_translations.items():
        mask = df["Style Name"].str.contains(key, case=False, na=False)
        df.loc[mask, "B2C Tags"] = df.loc[mask, "B2C Tags"].apply(
            lambda x: ",".join(set(x.split(",") + values)).strip(",")
        )
    
    # Tilføj materialekvaliteten som et tag uden procentdelen og fjern TM, () og bindestreger
    df["Quality Tags"] = df["Quality"].str.replace(r"\d+%", "", regex=True).str.replace(r"[™()\-]", "", regex=True).str.strip()
    df["Quality Tags"] = df["Quality Tags"].apply(lambda x: ",".join(set(x.split())))
    df["B2C Tags"] = df.apply(lambda row: ",".join(set([row["B2C Tags"], row["Quality Tags"]])) if row["Quality Tags"] else row["B2C Tags"], axis=1)
    df["B2C Tags"] = df["B2C Tags"].str.strip(",")
    df.drop(columns=["Quality Tags"], inplace=True)
    
    return df

def process_excel_and_zip(excel_file, zip_file):
    # Indlæs Excel-fil
    xls = pd.ExcelFile(excel_file)
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    
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

import re

# Sørg for, at excel_file og zip_file er defineret, før koden køres
if "excel_file" in globals() and "zip_file" in globals():
    if excel_file and zip_file:
        # Genindlæs den allerede behandlede Excel-fil med opdaterede B2C Tags
        df = pd.read_excel(processed_file_path)
    
        # Opret mapping fra style number til billedfilsti via ZIP-filen
        image_mapping = extract_images_from_zip(zip_file)
    
        # Bestem hvilken kolonne der skal bruges til style numbers: "Style Number" hvis tilgængelig, ellers "Style Name"
        style_column = "Style Number" if "Style Number" in df.columns else "Style Name"
    
        # Debug: Udskriv unikke værdier fra den valgte kolonne
        st.write("Unikke værdier i style kolonnen:", df[style_column].unique())
        # Debug: Udskriv nøgler fra billed-mapping for at se, hvilke style numbers der blev fundet i ZIP-filen
        st.write("Image mapping keys:", list(image_mapping.keys()))
    
        # Indlæs cache for beskrivelser
        cache = load_cache()
    
        descriptions = []
        for idx, row in df.iterrows():
            style_text = str(row[style_column])
            # Matcher f.eks. "SR123456", "123456", "SR123-456", "SR 123 456" osv.
            match = re.search(r"(?:SR\s*)?(\d{3})[-\s]?(\d{3})", style_text, re.IGNORECASE)
            if match:
                style_no = f"SR{match.group(1)}-{match.group(2)}"
                if style_no in cache:
                    desc = cache[style_no]
                elif style_no in image_mapping:
                    image_path = image_mapping[style_no]
                    desc = analyze_image_with_openai(image_path)
                    cache[style_no] = desc
                else:
                    desc = f"No matching image found for style {style_no}"
            else:
                desc = "No valid style number found."
            descriptions.append(desc)
    
        # Tilføj beskrivelserne som en ny kolonne i DataFrame
        df["Description"] = descriptions
    
        # Gem den opdaterede fil med beskrivelser til en midlertidig fil
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            df.to_excel(tmp.name, index=False, sheet_name='Updated Data with Descriptions')
            desc_file_path = tmp.name
    
        # Gem cache for at genbruge beskrivelser ved eventuelle fremtidige kørsler
        save_cache(cache)
    
        # Samlet download-knap (brug kun denne!)
        with open(desc_file_path, "rb") as file:
            st.download_button("Download Final Excel File", file, "processed_data_with_descriptions.xlsx")
