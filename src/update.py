import os
import json
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from enhanced_local_image_analyzer import EnhancedLocalImageAnalyzer

METADATA_FILE = "update_metadata.json"
DATA_DIR = 'C:/Users/LENOVO/Desktop/unlimited-context-ai-models/AI_Data/'
IMAGE_DIR = 'C:/Users/LENOVO/Desktop/unlimited-context-ai-models/Images/'

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)

def process_text_files(metadata):
    files_to_process = []
    updated_metadata = {}
    
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            if "faiss_index" in file_path:
                continue
            
            last_modified = os.path.getmtime(file_path)
            
            if file_path not in metadata or metadata[file_path] < last_modified:
                files_to_process.append(file_path)
            updated_metadata[file_path] = last_modified

    docs = []
    if files_to_process:
        print(f"Processing {len(files_to_process)} new or modified text files...")
        loader = DirectoryLoader(DATA_DIR, glob="**/*.*", exclude=["**/faiss_index/**"])
        all_docs = loader.load()
        
        for doc in all_docs:
            if doc.metadata['source'] in files_to_process:
                docs.append(doc)
    
    return docs, updated_metadata

def process_image_files(metadata, image_analyzer):
    files_to_process = []
    updated_metadata = {}

    for file in os.listdir(IMAGE_DIR):
        file_path = os.path.join(IMAGE_DIR, file)
        if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            continue

        last_modified = os.path.getmtime(file_path)

        if file_path not in metadata or metadata[file_path] < last_modified:
            files_to_process.append(file_path)
        updated_metadata[file_path] = last_modified

    docs = []
    if files_to_process:
        print(f"Processing {len(files_to_process)} new or modified image files...")
        for image_path in files_to_process:
            print(f"Analyzing {image_path}...")
            analysis_result = image_analyzer.analyze_image_advanced(image_path)
            if analysis_result['status'] == 'success':
                doc = Document(page_content=analysis_result['analysis'], metadata={"source": image_path})
                docs.append(doc)

    return docs, updated_metadata

def main():
    metadata = load_metadata()
    
    # Process text files
    text_docs, text_metadata = process_text_files(metadata)
    
    # Process image files
    image_analyzer = EnhancedLocalImageAnalyzer()
    image_docs, image_metadata = process_image_files(metadata, image_analyzer)
    
    new_docs = text_docs + image_docs
    
    if new_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        new_split_docs = text_splitter.split_documents(new_docs)

        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
        
        if os.path.exists("faiss_index"):
            vectorstore = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents(new_split_docs)
        else:
            vectorstore = FAISS.from_documents(new_split_docs, embeddings)

        vectorstore.save_local("faiss_index")
        print("Vector store updated.")
    else:
        print("No new or modified files found. Vector store is up to date.")

    # Combine and save metadata
    combined_metadata = {**metadata, **text_metadata, **image_metadata}
    save_metadata(combined_metadata)

if __name__ == "__main__":
    main()