import re
import fitz  # This is from PyMuPdf
import pandas as pd 
import country_converter as coco

doc = fitz.open('data/raw/geographic_area_definitions.pdf')

area_headers = {
        "WORLD", "EUROPE13", "EUROPEAN UNION", "EURO AREA",
    "LATIN AMERICA AND OTHER WESTERN HEMISPHERE",
    "SOUTH AND CENTRAL AMERICA", "SOUTH AMERICA",
    "CENTRAL AMERICA", "OTHER WESTERN HEMISPHERE",
    "ASIA AND PACIFIC", "MIDDLE EAST", "AFRICA",
    "INTERNATIONAL ORGANIZATIONS AND UNALLOCATED",
    "ADDITIONAL AREAS", "CAFTA-DR"
}

# Extract the text 
data = []
current_area = None
for page in doc:
    lines = page.get_text().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.upper() in area_headers:
            current_area = line.title()
        elif current_area:
            # Match capitalized country/region names possibly followed by a footnote number
            if re.match(r"^[A-Z][\w .&’',()-]+(?:\s\d+)?$", line):
                data.append({"area": current_area, "country_or_entity": line})

doc.close()
# Remove numbers from headers and country/entity names
data = [{"area": re.sub(r"\d+", "", entry["area"]).strip(), 
         "country_or_entity": re.sub(r"\d+", "", entry["country_or_entity"]).strip()} 
        for entry in data]

df = pd.DataFrame(data)
unwanted_values = [
    "Page  of", 
    "States may be the country in which the ultimate beneficial owner (UBO) of a U.S. affiliate is incorporated or"
]
df = df[~df['country_or_entity'].isin(unwanted_values)]
df['country_clean'] = coco.convert(df['country_or_entity'], to='name_short', not_found=None)
df['iso3'] = coco.convert(df['country_or_entity'], to='iso3', not_found=None)
df['iso2'] = coco.convert(df['country_or_entity'], to='iso2', not_found=None)
df.to_csv('data/working/BEA_region_definitions.csv', index=False)
# Drop rows with specific unwanted values in 'country_or_entity'


# BEA TiVA Specific Regions are Canada, China, Europe, Japan, Mexico, Rest of Asia and Pacific, and Rest of World
europe = df[(df['area'] == 'Europe') & (df['iso3'].str.len() == 3)]
asia_and_pacific = df[(df['area'] == 'Asia And Pacific') & (df['iso3'].str.len() == 3) & (~df['country_clean'].isin(['China', 'Japan']))]

europe = europe.drop(columns=['area', 'country_or_entity','iso2'])
asia_and_pacific = asia_and_pacific.drop(columns=['area','country_clean', 'country_or_entity', 'iso2'])

europe.to_csv('data/final/BEA_TiVA_Europe.csv', index=False)
asia_and_pacific.to_csv('data/final/BEA_TiVA_Asia_and_Pacific.csv', index=False)