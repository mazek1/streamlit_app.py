import streamlit as st
import pandas as pd
import zipfile
import os
import tempfile
import json
import re
import random
from io import BytesIO
from PIL import Image
import torch
import openai
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# ------------------------------------------------------------------------
# 1) Lazy-load ViT-GPT2-modelen (nlpconnect/vit-gpt2-image-captioning)
# ------------------------------------------------------------------------
def analyze_image(image_path):
    """
    Anvender ViT-GPT2 til billedbeskrivelse (bruges primært til at fange nøgleattributter som 'long sleeve' osv.).
    """
    if "model" not in st.session_state:
        with st.spinner("Loading image captioning model..."):
            st.session_state["feature_extractor"] = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            st.session_state["tokenizer"] = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            st.session_state["model"] = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    feature_extractor = st.session_state["feature_extractor"]
    tokenizer = st.session_state["tokenizer"]
    model = st.session_state["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
    
    gen_kwargs = {"max_length": 30, "num_beams": 4}
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    caption = preds[0].strip()
    return caption

# ------------------------------------------------------------------------
# 2) Hjælpefunktioner til parsing af data
# ------------------------------------------------------------------------
def get_fashion_type(style_name: str) -> str:
    """
    Finder produktets type (fx 'dress', 'shirt', 'blouse', osv.) ud fra style name.
    """
    style_name_lower = style_name.lower()
    for t in ["dress", "blouse", "shirt", "knit", "pants", "skirt", "jacket", "blazer", "cardigan", "rollneck", "o-neck", "v-neck"]:
        if t in style_name_lower:
            return t
    return "piece"

def parse_main_material(quality_str: str) -> str:
    """
    Finder det materiale med højest procent i Quality-kolonnen.
    Eksempel: "80% Viscose (LENZING™ ECOVERO™) 20% Nylon" -> "Viscose"
    """
    if not quality_str:
        return ""
    pattern = r"(\d+)%\s*([^%]+?)(?=\d+%|$)"
    matches = re.findall(pattern, quality_str)
    if not matches:
        return ""
    best_pct = 0
    best_mat = ""
    for (pct_str, mat_str) in matches:
        pct = int(pct_str)
        if pct > best_pct:
            best_pct = pct
            best_mat = mat_str.strip()
    best_mat = re.sub(r"\(.*?\)", "", best_mat)
    best_mat = re.sub(r"™", "", best_mat).strip()
    return best_mat

# ------------------------------------------------------------------------
# 3) Generer en beskrivelse med GPT-4
# ------------------------------------------------------------------------
def generate_description_with_gpt4(row, raw_caption):
    """
    Kombinerer produktdata og billedcaption til en prompt,
    som sendes til GPT-4 for at generere en salgsorienteret, modefokuseret produktbeskrivelse.
    Beskrivelsen skal:
      - Indholde en kort, catchy titel (kun typen, fx "Chic shirt" eller "Cosy dress"),
      - Indholde en sætning om produktet og hovedmaterialet (kun den med højeste procent),
      - Inkludere tre bullet points med nøglefunktioner.
    """
    style_name = str(row.get("Style Name", "")).strip()
    fashion_type = get_fashion_type(style_name)
    quality = str(row.get("Quality", "")).strip()
    main_material = parse_main_material(quality)
    
    # Opbyg prompt til GPT-4
    prompt = f"""You are a professional fashion copywriter. Using the following product data, generate a compelling, sales-oriented product description for a fashion website. The description must:
- Begin with a catchy title consisting of 2-3 words that only mentions the product type (for example: "Chic shirt", "Cosy dress", etc.).
- Include one sentence that describes the product and its main material (only include the material with the highest percentage, e.g. "Viscose").
- End with three bullet points highlighting key features such as comfort, design, and versatility.
Do not mention irrelevant details such as people, backgrounds, or animals.

Product Data:
- Product Type: {fashion_type}
- Main Material: {main_material if main_material else "Unknown"}
- Image Caption (attributes only): {raw_caption}

Write the final description in English."""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional fashion copywriter."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        generated_text = response["choices"][0]["message"]["content"].strip()
        return generated_text
    except Exception as e:
        return f"Error generating description with GPT-4: {str(e)}"

# ------------------------------------------------------------------------
# 4) Cache & Tag-funktioner (samme som tidligere)
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
        df.loc[mask, "B2C Tags"] = df.loc[mask, "B2C Tags"].apply(lambda x: ",".join(set(x.split(",") + values)).strip(","))
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
# 5) Streamlit UI
# ------------------------------------------------------------------------
st.title("Product Data Processor")

excel_file = st.file_uploader("Upload Excel File", type=["xlsx"])
zip_files = st.file_uploader("Upload ZIP Files with Images", type=["zip"], accept_multiple_files=True, key="zip_files")

if excel_file and zip_files:
    df = pd.read_excel(excel_file)
    df = update_b2c_tags(df)
    
    # Brug kun kolonnerne "Style No." eller "Style Number"
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
            # Få en rå caption fra den lokale model
            raw_caption = analyze_image(image_path)
            # Brug GPT-4 til at generere den endelige beskrivelse
            final_desc = generate_description_with_gpt4(row, raw_caption)
            cache[style_no] = final_desc
            descriptions.append(final_desc)
        else:
            descriptions.append(f"No matching image found for style {style_no}")
    
    df["Description"] = descriptions
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        df.to_excel(tmp.name, index=False, sheet_name='Updated Data with Descriptions')
        final_file_path = tmp.name
    
    save_cache(cache)
    
    st.download_button("Download Final Excel File", open(final_file_path, "rb"), "processed_data_with_descriptions.xlsx")
