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
    """
    Bruger OpenAI Vision API til at analysere billedet og generere en beskrivelse.
    Hvis du får en fejl vedr. openai.ChatCompletion, skal du sikre, at du bruger openai==0.28.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        with open(image_path, "rb") as image_file:
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an assistant that describes fashion products in a professional, inspiring, concise, and sales-oriented way. Each description should include three key points formatted as bullet points."
                    },
                    {
                        "role": "user", 
                        "content": "Describe this fashion product in English, ensuring a professional tone, engaging language, and highlighting three key selling points."
                    }
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
    """Indlæser cache-filen, hvis den findes."""
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
    """Opdaterer B2C Tags baseret på Style Name og Quality."""
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
    df["B2C Tags"] = df.apply(
        lambda row: ",".join(set([row["B2C Tags"], row["Quality Tags"]])) if row["Quality Tags"] else row["B2C Tags"], 
        axis=1
    )
    df["B2C Tags"] = df["B2C Tags"].str.strip(",")
    df.drop(columns=["Quality Tags"], inplace=True)
    
    return df

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
    Udtrækker billeder fra den uploadede ZIP-fil (et Streamlit 
