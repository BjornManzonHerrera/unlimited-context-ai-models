# Add to the top of your existing query.py
from integrated_multimodal_system import IntegratedMultimodalSystem
import os

# Modify your existing query function:
def enhanced_query(question, use_multimodal=True):
    """Enhanced version of your existing query function."""
    
    if use_multimodal and os.path.exists("Images"):
        print("Using multimodal analysis...")
        system = IntegratedMultimodalSystem()
        return system.comprehensive_query(question)
    else:
        print("Using text-only analysis...")
        # Your existing query logic here
        from langchain_ollama import OllamaLLM
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain.chains import RetrievalQA

        llm = OllamaLLM(model="llama3.1:8b-instruct-q4_0", base_url="http://localhost:11434")
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
        vectorstore = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

        return qa_chain.run(question)

# Example usage at the bottom:
if __name__ == "__main__":
    question = "What is a reasonable weekly allowance based on economic factors?"
    
    # Try multimodal first, fallback to text-only
    try:
        answer = enhanced_query(question, use_multimodal=True)
    except Exception as e:
        print(f"Multimodal failed, using text-only: {e}")
        answer = enhanced_query(question, use_multimodal=False)
    
    print(answer)
