import os
import json
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

METADATA_FILE = "update_metadata.json"
DATA_DIR = 'C:/Users/LENOVO/Desktop/unlimited-context-ai-models/AI_Data/'

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)

metadata = load_metadata()
updated_metadata = {}
files_to_process = []

for root, _, files in os.walk(DATA_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        if "faiss_index" in file_path:
            continue
        
        last_modified = os.path.getmtime(file_path)
        
        if file_path not in metadata or metadata[file_path] < last_modified:
            files_to_process.append(file_path)
        updated_metadata[file_path] = last_modified

if files_to_process:
    print(f"Processing {len(files_to_process)} new or modified files...")
    # Create a temporary DirectoryLoader for only the files that need processing
    # This part needs careful handling as DirectoryLoader expects a directory, not a list of files.
    # A more robust solution would be to load each file individually or use a custom loader.
    # For simplicity, let's assume we can pass a list of files to a custom loader or process them one by one.
    # For now, I'll adapt the existing DirectoryLoader to load from the main directory and filter later.
    # This is not ideal, but a quick fix. A better approach would be to create a custom document loader
    # that accepts a list of file paths.
    
    # Re-initializing DirectoryLoader to load all documents, then filter based on files_to_process
    # This is inefficient if many files are unchanged. A custom loader is preferred.
    loader = DirectoryLoader(DATA_DIR, glob="**/*.*", exclude=["**/faiss_index/**"])
    all_docs = loader.load()
    
    new_docs = []
    for doc in all_docs:
        if doc.metadata['source'] in files_to_process:
            new_docs.append(doc)

    if new_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        new_split_docs = text_splitter.split_documents(new_docs)

        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
        vectorstore = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(new_split_docs)
        vectorstore.save_local("faiss_index")
        print("Vector store updated.")
    else:
        print("No new or modified documents to process.")
else:
    print("No new or modified files found. Vector store is up to date.")

save_metadata(updated_metadata)
