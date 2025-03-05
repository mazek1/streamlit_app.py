import streamlit as st
import pandas as pd
import zipfile
import os
from io import BytesIO

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

    for index, row in df.iterrows():
        tags = set(str(row["B2C Tags"]).split(",")) if pd.notna(row["B2C Tags"]) else set()
        
        # Tilføj materiale-tags
        quality = str(row["Quality"]).lower()
        if "ecovero" in quality:
            tags.add("ecovero")
        if "gots" in quality:
            tags.add("gots")
            tags.add("_tag_gots")
        if "grs" in quality:
            tags.add("_tag_grs")

        # Tilføj Style Name-tag (første ord uden "SR")
        style_name = str(row["Style Name"]).strip()
        first_word = style_name.split()[0].replace("SR", "").strip()
        tags.add(first_word)

        # Tilføj kategori-tags baseret på produktnavnet
        for key, values in tag_translations.items():
            if key in style_name.lower() or key in first_word.lower():
                tags.update(values)

        # Sikrer minimum 6 tags
        while len(tags) < 6:
            tags.add("fashion")  # Generisk tag som backup

        # Opdater kolonnen
        df.at[index, "B2C Tags"] = ",".join(tags)

    return df

def process_excel_and_zip(excel_file, zip_file):
    # Indlæs Excel-fil
    xls = pd.ExcelFile(excel_file)
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    
    # Ekstraher billeder fra ZIP-filen
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        image_files = zip_ref.namelist()
    
    # Find style-numre fra billedfiler
    style_numbers = {file[:10] for file in image_files if file.startswith("SR")}
    
    # Opdater description baseret på billeder
    for index, row in df.iterrows():
        if pd.isna(row["Description"]) and row["Style No."] in style_numbers:
            df.at[index, "Description"] = f"Product description for {row['Style Name']} is generated."
    
    # Opdater B2C Tags
    df = update_b2c_tags(df)
    
    # Gem den opdaterede fil
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Updated Data')
    output.seek(0)
    return output

# Streamlit UI
st.title("Product Data Processor")

excel_file = st.file_uploader("Upload Excel File", type=["xlsx"])
zip_file = st.file_uploader("Upload ZIP File with Images", type=["zip"])

if excel_file and zip_file:
    st.success("Files uploaded successfully. Processing...")
    processed_file = process_excel_and_zip(excel_file, zip_file)
    st.download_button("Download Processed Excel File", processed_file, "processed_data.xlsx")
