import fitz
import re
import regex
import os
import json

# Use PyMuPDF to extract text from PDF
def get_pdf_text(pdf_path):
    with fitz.open(pdf_path) as pdf:
        texts = []

        # The only desired metadata is the PDF title
        if pdf.metadata.get('title'):
            texts.append(f"{'title'}: {pdf.metadata.get('title')}")

        for page in range(len(pdf)):
            text = pdf[page].get_text(sort=True)
            # Use regular expression to split text into paragraphs
            # Delimiter: newline(s) followed by an upper case character
            paragraphs = regex.split(r'\n+(?=\p{Lu})', text, flags=re.UNICODE)

            for paragraph in paragraphs:
                paragraph = " ".join(paragraph.strip().split())

                # print(paragraph)
                texts.append(paragraph)
        pdf_text = '\n'.join(texts)
    return pdf_text

# Get the text for all pdfs in the given folder
def get_all_pdfs(folder):
    documents = []
    for item in os.listdir(folder):
        path = os.path.join(folder, item)
        if os.path.isfile(path):
            fn, ext = os.path.splitext(path)
            # Only try to read from .pdfs
            if ext == '.pdf':
                print(path)
                text = get_pdf_text(path)
                documents.append({'page_content':text, 'metadata':{"title":fn}})
    print()
    return documents
    
# This folder contains PDFs gathered from TN Ag Extension resources
docs = get_all_pdfs('/gpfs/wolf2/olcf/trn040/proj-shared/knelms3_jyoder5/rag_test/pdfs/')
# Write data into a JSON file for future usage in RAG application
with open('/gpfs/wolf2/olcf/trn040/proj-shared/knelms3_jyoder5/rag_test/pdfs_as_docs.json','w') as f:
    json.dump(docs, f, indent=4)
        