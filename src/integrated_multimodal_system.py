# integrated_multimodal_system.py
import faiss
import pickle
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import os
import json
import requests

class IntegratedMultimodalSystem:
    def __init__(self, 
                 faiss_index_path="faiss_index",
                 text_model="llama3.1:8b-instruct-q4_0"):
        """Complete local multimodal system optimized for your hardware."""
        
        self.text_model = text_model
        
        # Load existing text index
        try:
            self.embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
            self.text_index = FAISS.load_local(faiss_index_path, self.embedding_model, allow_dangerous_deserialization=True)
            print("Text index loaded successfully")
        except Exception as e:
            print(f"Error loading text index: {e}")
            self.text_index = None

    def search_content(self, query: str, k: int = 10) -> list[str]:
        """Search existing text and image documents."""
        if not self.text_index:
            return []
        
        results = self.text_index.similarity_search_with_score(query, k=k)
        
        relevant_chunks = []
        for doc, score in results:
            source = doc.metadata.get("source", "unknown")
            chunk_with_score = f"[Relevance: {1-score:.3f}, Source: {source}] {doc.page_content}"
            relevant_chunks.append(chunk_with_score)
        
        return relevant_chunks
    
    def comprehensive_query(self, query: str) -> str:
        """
        Perform comprehensive query using the unified vector store.
        """
        
        print(f"\nProcessing comprehensive query: '{query}'")
        print("="*60)
        
        # Step 1: Search documents
        print("Searching documents...")
        results = self.search_content(query, k=10)
        print(f"Found {len(results)} relevant chunks")
        
        # Step 2: Synthesize comprehensive answer
        print("Synthesizing comprehensive answer...")
        return self.synthesize_comprehensive_answer(query, results)
    
    def synthesize_comprehensive_answer(self, query: str, results: list[str]) -> str:
        """Create final answer using local text model."""
        
        # Prepare comprehensive context
        context = ""
        if results:
            context = "\n\n".join([f"SOURCE {i+1}:\n{res}" for i, res in enumerate(results)])
        
        # Create synthesis prompt optimized for your local LLM
        synthesis_prompt = f"""You are a comprehensive document analysis assistant. Answer this question using ALL available sources:

QUESTION: {query}

AVAILABLE INFORMATION:

{context}

INSTRUCTIONS:
1. Provide a direct, comprehensive answer to the question
2. Synthesize information from the provided sources.
3. When citing information, use the source file path provided in the context.
4. If sources provide conflicting information, mention both perspectives
5. If information is incomplete, clearly state what's missing
6. Focus on being helpful and actionable

COMPREHENSIVE ANSWER:"""

        try:
            response = requests.post(
                f"http://localhost:11434/api/generate",
                json={
                    "model": self.text_model,
                    "prompt": synthesis_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_ctx": 4096  # Larger context for comprehensive analysis
                    }
                },
                timeout=180
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'No response generated')
            else:
                return f"Error generating response: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Synthesis failed: {e}"
