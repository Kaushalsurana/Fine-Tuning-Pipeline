import os
import pdfplumber
import markdown2
import json

def extract_text_from_md(md_file):
    """Extract text from a Markdown file"""
    with open(md_file, 'r', encoding='utf-8') as f:
        return markdown2.markdown(f.read())

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def preprocess_text(text):
    """Preprocess extracted text"""
    text = text.lower().strip()
    text = ' '.join(text.split())  # Remove excessive spaces
    return text

def load_documents(path):
    """Load and preprocess documents from a file or directory"""
    docs = []
    
    if os.path.isdir(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if file.endswith(".md"):
                text = extract_text_from_md(file_path)
            elif file.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            else:
                continue
            docs.append(preprocess_text(text))
    elif os.path.isfile(path):
        if path.endswith(".md"):
            text = extract_text_from_md(path)
        elif path.endswith(".pdf"):
            text = extract_text_from_pdf(path)
        else:
            raise ValueError("Unsupported file format. Only .md and .pdf are supported.")
        
        docs.append(preprocess_text(text))
    else:
        raise FileNotFoundError("Provided path does not exist.")
    
    return docs

if __name__ == "__main__":
    path = "/workspaces/Fine-Tuning-Pipeline/Data/documentation.pdf"  # Change this to your file path
    docs = load_documents(path)
    
    print(f"Loaded {len(docs)} document(s).")
    print(docs[0][:500])  # Print first 500 characters of processed text
    
    # ✅ Save preprocessed text
    os.makedirs("./data", exist_ok=True)  # Ensure directory exists
    with open("./data/preprocessed_text.json", "w", encoding="utf-8") as f:
        json.dump(docs, f)
    
    print("Preprocessed data saved successfully!")
