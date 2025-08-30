from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

loader = DirectoryLoader('C:/Users/LENOVO/Desktop/unlimited-context-ai-models/AI_Data/', glob="**/*.*", exclude=["**/faiss_index/**"])
new_docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
new_split_docs = text_splitter.split_documents(new_docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
vectorstore = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
vectorstore.add_documents(new_split_docs)
vectorstore.save_local("faiss_index")