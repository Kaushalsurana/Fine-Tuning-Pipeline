import os
import markdown2
import pdfplumber
import PyPDF2

def extract_text_from_md(md_file):
    with open(md_file, 'r', encoding='utf-8') as f:
        return markdown2.markdown(f.read())

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def preprocess_text(text):
    text = text.lower().strip()
    text = ' '.join(text.split())  # Remove excessive spaces
    return text

def load_documents(data_dir):
    docs = []
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if file.endswith(".md"):
            text = extract_text_from_md(file_path)
        elif file.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            continue
        docs.append(preprocess_text(text))
    return docs

if __name__ == "__main__":
    docs = load_documents("../Data/")
    print(f"Loaded {len(docs)} documents.")
