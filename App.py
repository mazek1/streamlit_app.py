import streamlit as st
import pandas as pd
import zipfile
import os
from io import BytesIO

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
    
    # Opdater B2C Tags med simple logik
    for index, row in df.iterrows():
        tags = set(str(row["B2C Tags"]).split(",")) if pd.notna(row["B2C Tags"]) else set()
        if "ecovero" in str(row["Quality"]).lower():
            tags.add("ecovero")
        if "gots" in str(row["Quality"]).lower():
            tags.add("gots")
            tags.add("_tag_gots")
        df.at[index, "B2C Tags"] = ",".join(tags)
    
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
