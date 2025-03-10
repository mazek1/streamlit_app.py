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
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# ------------------------------------------------------------------------
# 1) Lazy-load ViT-GPT2 (nlpconnect/vit-gpt2-image-captioning)
# ------------------------------------------------------------------------
def analyze_image(image_path):
    """
    Bruger ViT-GPT2 til billedbeskrivelse (mest for at fange 'long sleeve'/'short sleeve').
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
# 2) Hjælpefunktioner til at parse type og materiale
# ------------------------------------------------------------------------
def get_fashion_type(style_name: str) -> str:
    """
    Finder typen i Style Name (fx 'dress', 'shirt', 'blouse', 'knit' osv.).
    Hvis intet match, returneres 'piece'.
    """
    style_name_lower = style_name.lower()
    type_map = [
        "dress", "blouse", "shirt", "knit", "pants", "skirt",
        "jacket", "blazer", "cardigan", "rollneck", "o-neck", "v-neck"
    ]
    for t in type_map:
        if t in style_name_lower:
            return t
    return "piece"

def parse_main_material(quality_str: str) -> str:
    """
    Finder det materiale med højest procent i Quality-kolonnen.
    Eksempel: "80% Viscose (LENZING™ ECOVERO™) 20% Nylon" -> "Viscose"
    Hvis intet match, returneres tom streng.
    """
    if not quality_str:
        return ""
    
    # Regex for at fange "xx% <material>"
    # Eksempel: "80% Viscose (LENZING™ ECOVERO™)" -> groups: ("80", "Viscose (LENZING™ ECOVERO™)")
    pattern = r"(\d+)%\s*([^%]+?)(?=\d+%|$)"
    matches = re.findall(pattern, quality_str)
    # matches er en liste af tuples (pct, materiale)
    if not matches:
        return ""
    
    # Find match med højest procent
    best_pct = 0
    best_mat = ""
    for (pct_str, mat_str) in matches:
        pct = int(pct_str)
        if pct > best_pct:
            best_pct = pct
            best_mat = mat_str.strip()
    
    # Fjern paranteser, varemærker og ekstra mellemrum
    # F.eks. "Viscose (LENZING™ ECOVERO™)" -> "Viscose"
    best_mat = re.sub(r"\(.*?\)", "", best_mat)  # fjern alt i ( )
    best_mat = re.sub(r"™", "", best_mat)       # fjern ™
    best_mat = best_mat.strip()
    return best_mat

# Nogle tilfældige ord til overskrift
HEADWORD_CHOICES = [
    "Chic", "Stylish", "Cosy", "Modern", "Refined",
    "Sophisticated", "Fashionable", "Trendy", "Luxurious", "Elegant"
]

# Nogle tilfældige ord til "crafted from"
MATERIAL_PHRASES = [
    "crafted from", "made of", "expertly woven from",
    "produced in", "beautifully composed of"
]

# ------------------------------------------------------------------------
# 3) Opret en modefokuseret tekst (med random ordvalg)
# ------------------------------------------------------------------------
def generate_custom_description(row, raw_caption):
    """
    1) Vælg en tilfældig overskrifts-stil (fx "Chic", "Cosy") + type (dress, blouse, shirt, etc.).
    2) Find hovedmaterialet (højeste procent).
    3) Fanger 'long sleeve' eller 'short sleeve' fra billedets caption.
    4) Saml en lille tekst.
    """
    style_name = str(row.get("Style Name", "")).strip()
    # a) Find type
    fashion_type = get_fashion_type(style_name)
    
    # b) Random headword
    headword = random.choice(HEADWORD_CHOICES)
    
    # c) Parse main material
    quality = str(row.get("Quality", "")).strip()
    main_material = parse_main_material(quality)
    
    # d) Tjek for "long sleeve" eller "short sleeve"
    raw_lower = raw_caption.lower()
    if "long sleeve" in raw_lower:
        sleeve_info = "Long-sleeve"
    elif "short sleeve" in raw_lower:
        sleeve_info = "Short-sleeve"
    else:
        sleeve_info = ""

    # e) Byg overskrift (fx "Chic shirt")
    heading = f"{headword} {fashion_type}"

    # f) Vælg et random synonym for 'crafted from'
    mat_phrase = random.choice(MATERIAL_PHRASES)
    if main_material:
        mat_sentence = f"{mat_phrase} {main_material}."
    else:
        mat_sentence = ""

    # g) Byg en sætning om style (inkl. sleeve info, hvis fundet)
    if sleeve_info:
        style_sentence = f"This {fashion_type} features a {sleeve_info} design, perfect for any occasion."
    else:
        style_sentence = f"This {fashion_type} is designed for timeless versatility."

    # h) Tre bullet points (kan tilpasses)
    bullet_points = [
        "Modern silhouette",
        "Comfortable for daily wear",
        "Effortless styling options"
    ]

    description = (
        f"{heading}\n\n"
        f"{mat_sentence} {style_sentence}\n\n"
        "Key Features:\n" + "\n".join(f"- {bp}" for bp in bullet_points)
    )
    
    return description.strip()

# ------------------------------------------------------------------------
# 4) Cache & Tag-funktioner
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
            raw_caption = analyze_image(image_path)
            final_desc = generate_custom_description(row, raw_caption)
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
