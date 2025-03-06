import re

def extract_images_from_zip(zip_file):
    """
    Udtrækker billeder fra den uploadede ZIP-fil og returnerer et dictionary,
    der mapper et style number (f.eks. "SR123-456") til stien for en midlertidigt gemt billedfil.
    """
    image_mapping = {}
    with zipfile.ZipFile(zip_file) as z:
        for file_name in z.namelist():
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Matcher f.eks. "SR123456", "123456", "SR123-456", "SR 123 456" osv.
                match = re.search(r"(?:SR\s*)?(\d{3})[-\s]?(\d{3})", file_name, re.IGNORECASE)
                if match:
                    style_no = f"SR{match.group(1)}-{match.group(2)}"
                    data = z.read(file_name)
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1])
                    tmp_file.write(data)
                    tmp_file.close()
                    image_mapping[style_no] = tmp_file.name
    return image_mapping

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
