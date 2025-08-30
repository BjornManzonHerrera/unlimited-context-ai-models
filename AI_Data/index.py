from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

# Ensure Ollama is running (run `ollama serve` in another PowerShell if needed)
loader = DirectoryLoader('C:/AI_Data/')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")