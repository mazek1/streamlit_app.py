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

import re
import zipfile
import tempfile
import os

def parse_style_number(raw_str: str) -> str or None:
    """
    Forsøger at udtrække og normalisere et stylenummer i formatet "SRxxx-xxx" fra en given streng.
    Eksempler:
      - "SR425-706"         -> "SR425-706"
      - "SR425706"          -> "SR425-706"
      - "SR425-706_103_1"    -> "SR425-706"
    Metoden:
      1. Hvis der er en underscore, behold kun den del før den.
      2. Fjern alle tegn, der ikke er bogstaver, cifre eller bindestreg.
      3. Forsøg at finde et match af typen "SR\d{3}-\d{3}".
      4. Hvis ikke, se om vi kan finde "SR\d{6}" og indsæt bindestreg.
    """
    if not raw_str:
        return None
    # Konverter til streng, fjern ledende/slående whitespace
    s = str(raw_str).upper().strip()
    # Hvis der er en underscore, behold kun den del før den.
    if "_" in s:
        s = s.split("_", 1)[0]
    # Fjern alle tegn, der ikke er A-Z, 0-9 eller bindestreg.
    s = re.sub(r"[^A-Z0-9-]", "", s)
    # Prøv at finde et match med bindestreg
    m = re.search(r"(SR\d{3}-\d{3})", s)
    if m:
        return m.group(1)
    # Hvis intet match, fjern eventuel bindestreg og prøv igen, hvis der er 6 cifre
    s_no_dash = s.replace("-", "")
    m = re.search(r"(SR\d{6})", s_no_dash)
    if m:
        num = m.group(1)
        return f"SR{num[2:5]}-{num[5:]}"
    return None

def extract_images_from_zip(zip_file):
    """
    Udtrækker billeder fra ZIP-filen og returnerer et dictionary,
    der mapper et stylenummer (i formatet "SRxxx-xxx") til en midlertidig filsti.
    Her splittes billedfilnavnet ved den første underscore.
    """
    image_mapping = {}
    with zipfile.ZipFile(zip_file) as z:
        for file_name in z.namelist():
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                base_name = os.path.basename(file_name)
                # Fjern filtypen og alt efter den første underscore
                base_no_ext = os.path.splitext(base_name)[0]
                if "_" in base_no_ext:
                    base_no_ext = base_no_ext.split("_", 1)[0]
                style_no = parse_style_number(base_no_ext)
                if style_no:
                    data = z.read(file_name)
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(base_name)[1])
                    tmp_file.write(data)
                    tmp_file.close()
                    image_mapping[style_no] = tmp_file.name
    return image_mapping

# Hovedkoden – den skal placeres efter den låste kode.
if excel_file and zip_file:
    # Indlæs den allerede behandlede Excel-fil
    df = pd.read_excel(processed_file_path)
    
    # Opret mapping for billeder fra ZIP-filen
    image_mapping = extract_images_from_zip(zip_file)
    
    # Brug "Style Number" hvis den findes, ellers "Style Name"
    style_column = "Style Number" if "Style Number" in df.columns else "Style Name"
    
    cache = load_cache()
    descriptions = []
    
    for _, row in df.iterrows():
        raw_style = str(row[style_column])
        style_no = parse_style_number(raw_style)
        if style_no is None:
            descriptions.append("No valid style number found.")
            continue
        if style_no in cache:
            descriptions.append(cache[style_no])
        elif style_no in image_mapping:
            image_path = image_mapping[style_no]
            desc = analyze_image_with_openai(image_path)
            cache[style_no] = desc
            descriptions.append(desc)
        else:
            descriptions.append(f"No matching image found for style {style_no}")
    
    df["Description"] = descriptions
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        df.to_excel(tmp.name, index=False, sheet_name='Updated Data with Descriptions')
        final_file_path = tmp.name
    
    save_cache(cache)
    
    with open(final_file_path, "rb") as file:
        st.download_button("Download Final Excel File", file, "processed_data_with_descriptions.xlsx")
