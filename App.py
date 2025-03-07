import streamlit as st
import pandas as pd
import zipfile
import os
import tempfile
import openai
import json
import re
from io import BytesIO
from PIL import Image

# Definer en fast sti til cache-filen, så den overlever genstarter
CACHE_FILE = ".streamlit/description_cache.json"

def analyze_image_with_openai(image_path):
    """Bruger OpenAI Vision API til at analysere billedet og generere en beskrivelse."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
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
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
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
    df["Quality Tags"] = df["Quality"].str.replace(r"\d+%", "", regex=True)\
                                      .str.replace(r"[™()\-]", "", regex=True)\
                                      .str.strip()
    df["Quality Tags"] = df["Quality Tags"].apply(lambda x: ",".join(set(x.split())))
    df["B2C Tags"] = df.apply(lambda row: ",".join(set([row["B2C Tags"], row["Quality Tags"]])) if row["Quality Tags"] else row["B2C Tags"], axis=1)
    df["B2C Tags"] = df["B2C Tags"].str.strip(",")
    df.drop(columns=["Quality Tags"], inplace=True)
    
    return df

def process_excel_and_zips(excel_file, zip_files):
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

def parse_style_number(raw_str: str) -> str or None:
    """
    Udtrækker et stylenummer i formatet "SRxxx-xxx" fra en given streng.
    Eksempler:
      - "SR425-706"         -> "SR425-706"
      - "SR425706"          -> "SR425-706"
      - "SR425-706_103_1"    -> "SR425-706"
    """
    if not raw_str:
        return None
    s = str(raw_str).upper().strip()
    if "_" in s:
        s = s.split("_", 1)[0]
    m = re.search(r"(SR\d{3}-\d{3})", s)
    if m:
        return m.group(1)
    m = re.search(r"(SR\d{6})", s)
    if m:
        candidate = m.group(1)
        return candidate[:5] + "-" + candidate[5:]
    return None

def extract_images_from_zip(uploaded_zip):
    """
    Udtrækker billeder fra den uploadede ZIP-fil (et Streamlit UploadedFile-objekt)
    og returnerer et dictionary, der mapper et stylenummer (formatet "SRxxx-xxx")
    til en midlertidig filsti.
    Indeholder debug-udskrifter for at se, hvad der sker.
    """
    image_mapping = {}
    try:
        zip_bytes = uploaded_zip.read()
        bytes_obj = BytesIO(zip_bytes)
        with zipfile.ZipFile(bytes_obj) as z:
            # Debug: Udskriv alle filnavne i ZIP-filen
            st.write("Filer fundet i ZIP:", z.namelist())
            for file_name in z.namelist():
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    base_name = os.path.basename(file_name)
                    base_no_ext = os.path.splitext(base_name)[0]
                    if "_" in base_no_ext:
                        base_no_ext = base_no_ext.split("_", 1)[0]
                    style_no = parse_style_number(base_no_ext)
                    # Debug: Udskriv filnavn og det parse'ede stylenummer
                    st.write(f"Fil: {file_name} => Parsed style: {style_no}")
                    if style_no:
                        data = z.read(file_name)
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(base_name)[1])
                        tmp_file.write(data)
                        tmp_file.close()
                        image_mapping[style_no] = tmp_file.name
    except Exception as e:
        st.write("Fejl i extract_images_from_zip:", e)
    # Debug: Udskriv den samlede mapping
    st.write("Extracted image mapping:", image_mapping)
    return image_mapping

# Streamlit UI
st.title("Product Data Processor")

# Uploader til Excel-fil og ZIP-filer med billeder
excel_file = st.file_uploader("Upload Excel File", type=["xlsx"])
zip_files = st.file_uploader("Upload ZIP Files with Images", type=["zip"], accept_multiple_files=True, key="zip_files")

if excel_file and zip_files:
    # Læs Excel-data og opdater B2C Tags
    df = pd.read_excel(excel_file)
    df = update_b2c_tags(df)
    
    # Debug: Udskriv de udtrukne stylenumre fra Excel
    style_column = "Style Number" if "Style Number" in df.columns else "Style Name"
    for _, row in df.iterrows():
        raw_style = str(row[style_column]).strip()
        style_no = parse_style_number(raw_style)
        st.write("Excel - Raw style:", raw_style, "=> Parsed style:", style_no)
    
    combined_image_mapping = {}
    for uploaded_zip in zip_files:
        mapping = extract_images_from_zip(uploaded_zip)
        combined_image_mapping.update(mapping)
    
    # Debug: Udskriv de stylenumre, der er fundet i billederne
    st.write("Extracted image style numbers:", list(combined_image_mapping.keys()))
    
    cache = load_cache()
    descriptions = []
    
    for _, row in df.iterrows():
        raw_style = str(row[style_column]).strip()
        style_no = parse_style_number(raw_style)
        if style_no is None:
            descriptions.append("No valid style number found.")
            continue
        if style_no in cache:
            descriptions.append(cache[style_no])
        elif style_no in combined_image_mapping:
            image_path = combined_image_mapping[style_no]
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
