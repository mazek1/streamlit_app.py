import streamlit as st
import pandas as pd
import zipfile
import os
import tempfile
import json
import re
from io import BytesIO
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# ------------------------------------------------------------------------
# 1. Lazy-load ViT-GPT2-modelen (fra nlpconnect/vit-gpt2-image-captioning)
# ------------------------------------------------------------------------
def analyze_image(image_path):
    """
    Anvender ViT-GPT2 til billedbeskrivelse.
    Modellen loades først, når funktionen kaldes første gang.
    """
    if "model" not in st.session_state:
        with st.spinner("Loading image captioning model..."):
            # Hent processor og model fra Hugging Face
            st.session_state["feature_extractor"] = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            st.session_state["tokenizer"] = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            st.session_state["model"] = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    feature_extractor = st.session_state["feature_extractor"]
    tokenizer = st.session_state["tokenizer"]
    model = st.session_state["model"]

    # Hvis GPU ikke er tilgængelig, bruges CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Åbn billedet og konverter til RGB
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Parametre til tekstgenerering
    gen_kwargs = {
        "max_length": 30,
        "num_beams": 4
    }

    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    caption = preds[0].strip()
    return caption

# ------------------------------------------------------------------------
# 2. Eksempel på post-proces for at få en bestemt formatering (valgfrit)
# ------------------------------------------------------------------------
def generate_custom_description(row, raw_caption):
    """
    Eksempel på, hvordan du kan forme din tekst:
    1) Kort overskrift (2-3 ord).
    2) Materiale fra Excel (Quality).
    3) Selve billedbeskrivelsen (evt. med filtrering af ord).
    4) Tre bullet points.
    """
    # Fjern uønskede ord (fx "woman", "man", "wearing", osv.)
    cleaned_caption = re.sub(r"\bwoman\b|\bman\b|\bwearing\b|\bperson\b|\bpeople\b", "", raw_caption, flags=re.IGNORECASE).strip()

    # Eksempel på overskrift: "Chic <Style Name>"
    style_name = str(row.get("Style Name", "")).strip()
    if style_name:
        short_title = f"Chic {style_name}"
    else:
        short_title = "Chic piece"

    # Tilføj materiale fra "Quality"
    material = str(row.get("Quality", "")).strip()
    if material:
        material_text = f"Made from {material}. "
    else:
        material_text = ""

    bullet_points = [
        "Comfortable fit",
        "Timeless design",
        "Versatile styling"
    ]

    # Byg endelig tekst
    description = (
        f"{short_title}\n\n"
        f"{material_text}This style is best described as: {cleaned_caption}.\n\n"
        "Key Features:\n"
        + "\n".join(f"- {bp}" for bp in bullet_points)
    )
    return description

# ------------------------------------------------------------------------
# 3. Cache & Tag-funktioner (som før)
# ------------------------------------------------------------------------
CACHE_FILE = ".streamlit/description_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as file:
            return json.load(file)
    return {}

def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as file:
        json.dump(cache, file)

def update_b2c_tags(df):
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
    df["Quality Tags"] = df["Quality"].str.replace(r"\d+%", "", regex=True)\
                                      .str.replace(r"[™()\-]", "", regex=True)\
                                      .str.strip()
    df["Quality Tags"] = df["Quality Tags"].apply(lambda x: ",".join(set(x.split())))
    df["B2C Tags"] = df.apply(lambda row: ",".join(set([row["B2C Tags"], row["Quality Tags"]])) if row["Quality Tags"] else row["B2C Tags"], axis=1)
    df["B2C Tags"] = df["B2C Tags"].str.strip(",")
    df.drop(columns=["Quality Tags"], inplace=True)
    return df

def parse_style_number(raw_str: str) -> str or None:
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
    image_mapping = {}
    zip_bytes = uploaded_zip.read()
    bytes_obj = BytesIO(zip_bytes)
    with zipfile.ZipFile(bytes_obj) as z:
        for file_name in z.namelist():
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                base_name = os.path.basename(file_name)
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

# ------------------------------------------------------------------------
# 4. Streamlit UI
# ------------------------------------------------------------------------
st.title("Product Data Processor")

excel_file = st.file_uploader("Upload Excel File", type=["xlsx"])
zip_files = st.file_uploader("Upload ZIP Files with Images", type=["zip"], accept_multiple_files=True, key="zip_files")

if excel_file and zip_files:
    df = pd.read_excel(excel_file)
    df = update_b2c_tags(df)
    
    if "Style No." in df.columns:
        style_column = "Style No."
    else:
        style_column = "Style Number"
    
    combined_image_mapping = {}
    for uploaded_zip in zip_files:
        mapping = extract_images_from_zip(uploaded_zip)
        combined_image_mapping.update(mapping)
    
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
            # Kald modellen for at få en rå billedbeskrivelse
            raw_caption = analyze_image(image_path)
            # Tilpas teksten til en salgsmæssig form
            final_desc = generate_custom_description(row, raw_caption)
            cache[style_no] = final_desc
            descriptions.append(final_desc)
        else:
            descriptions.append(f"No matching image found for style {style_no}")
    
    df["Description"] = descriptions
    
    # Gem den opdaterede fil som en midlertidig fil
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        df.to_excel(tmp.name, index=False, sheet_name='Updated Data with Descriptions')
        final_file_path = tmp.name
    
    save_cache(cache)
    
    st.download_button("Download Final Excel File", open(final_file_path, "rb"), "processed_data_with_descriptions.xlsx")
