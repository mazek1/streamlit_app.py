def generate_description_with_gpt4(row, raw_caption):
    style_name = str(row.get("Style Name", "")).strip()
    fashion_type = get_fashion_type(style_name)
    quality = str(row.get("Quality", "")).strip()
    main_material = parse_main_material(quality)
    
    prompt = f"""You are a professional fashion copywriter specialized in high-end apparel. Using the following product data, generate a unique, compelling, and detailed product description tailored for a fashion website. The description must:
- Begin with a catchy, creative title consisting of 2-3 words that only mention the product type (for example: "Chic Shirt", "Cozy Dress", "Modern Blouse").
- Follow with one sentence describing the product's design, focusing on its unique style, cut, or pattern.
- Include one sentence mentioning the main material in a captivating way, for example "Expertly woven from Viscose".
- Conclude with three bullet points that highlight key features (e.g., comfort, design innovation, attention to detail).
Do not include generic phrases like 'timeless versatility' or unrelated details.

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
            temperature=0.8,
            max_tokens=200
        )
        generated_text = response["choices"][0]["message"]["content"].strip()
        return generated_text
    except Exception as e:
        return f"Error generating description with GPT-4: {str(e)}"
