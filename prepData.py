from langchain_core.documents import Document
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open('./pdfs_as_docs.json','r') as f:
    doc_dict = json.load(f)

docs = [Document(**doc) for doc in doc_dict]

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
chunked_docs = splitter.split_documents(docs)