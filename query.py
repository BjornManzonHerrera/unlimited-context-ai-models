from integrated_multimodal_system import IntegratedMultimodalSystem
import os
import time
from prompt_saver import save_output
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

def your_existing_query_function(question):
    print("Loading LLM model...")
    start_time = time.time()
    llm = OllamaLLM(model="llama3.1:8b-instruct-q4_0", base_url="http://localhost:11434")
    print(f"LLM model loaded in {time.time() - start_time:.2f} seconds.")

    print("Loading embeddings model...")
    start_time = time.time()
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    print(f"Embeddings model loaded in {time.time() - start_time:.2f} seconds.")

    print("Loading vector store...")
    start_time = time.time()
    vectorstore = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    print(f"Vector store loaded in {time.time() - start_time:.2f} seconds.")

    print("Creating QA chain...")
    start_time = time.time()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
    print(f"QA chain created in {time.time() - start_time:.2f} seconds.")

    print(f"Asking question: {question}")
    start_time = time.time()
    result = qa_chain.run(question)
    print(f"Question answered in {time.time() - start_time:.2f} seconds.")
    return result

def enhanced_query(question, use_multimodal=True):
    """Enhanced version of your existing query function."""
    
    if use_multimodal and os.path.exists("Images"):
        print("🔄 Using multimodal analysis...")
        system = IntegratedMultimodalSystem()
        return system.comprehensive_query(question)
    else:
        print("Using text-only analysis...")
        return your_existing_query_function(question)

if __name__ == "__main__":
    question = """Analyze the reciept in the Images Folder"""
    
    try:
        answer = enhanced_query(question, use_multimodal=True)
    except Exception as e:
        print(f"Multimodal failed, using text-only: {e}")
        answer = enhanced_query(question, use_multimodal=False)
    
    save_output(answer, question)
    print(answer)