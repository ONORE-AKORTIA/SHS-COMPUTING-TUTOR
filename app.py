import os
import json
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def convert_all_pdfs():
    base_dir = "raw_pdfs"
    output_dir = "textbooks_json"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' not found. Please create it and organize your PDFs.")
        return

    curriculum_data = {}

    for year in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year)
        if not os.path.isdir(year_path):
            continue
        curriculum_data[year] = {}

        for subject in os.listdir(year_path):
            subject_path = os.path.join(year_path, subject)
            if not os.path.isdir(subject_path):
                continue
            curriculum_data[year][subject] = []

            for file in os.listdir(subject_path):
                if file.endswith(".pdf"):
                    pdf_file_path = os.path.join(subject_path, file)
                    print(f"Processing Year: {year} | Subject: {subject} | File: {file}...")
                    content = extract_text_from_pdf(pdf_file_path)
                    
                    curriculum_data[year][subject].append({
                        "filename": file,
                        "content": content
                    })

    output_file = os.path.join(output_dir, "curriculum_database.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(curriculum_data, f, ensure_ascii=False, indent=4)
    print(f"Successfully generated master curriculum database at {output_file}!")

if __name__ == "__main__":
    convert_all_pdfs()
