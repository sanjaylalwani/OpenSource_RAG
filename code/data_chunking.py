import os
import re
import json
import logging
from dotenv import load_dotenv
from pypdf import PdfReader


folder_path = os.path.join("..", "data", "source")

file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for fn in file_names:
    file_path = os.path.join(folder_path, fn)

    reader = PdfReader(file_path)
    number_of_pages = len(reader.pages)

    text = ""
    for i in range(4, number_of_pages):
        page = reader.pages[i]
        text += page.extract_text()

    # Convert the string into a JSON format
    match = re.search(r"(\d{4})", fn)
    year = match.group(1) if match else None
    is_interim = "_interim" in fn  # Check if '_interim' is in the filename
    file_type = "Interim" if is_interim else "Normal"

    json_data = {"content": text, "metadata":{"year":year, "type": file_type}}

    # Save to a JSON file
    # folder_path_target = os.path.join("..", "data", "target")
    file_path_target = os.path.join("..", "data", "target", fn).replace(".pdf",".json")
    # file_path_target = file_path_target.replace(".pdf",".json")
    with open(file_path_target, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4)

    print("String saved to ", file_path_target)