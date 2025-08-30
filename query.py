from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

llm = OllamaLLM(model="llama3.1:8b-q4_0", base_url="http://localhost:11434")
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
vectorstore = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

response = qa_chain.run("Ask about your data here")
print(response)