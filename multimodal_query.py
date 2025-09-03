import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import json
import os
from image_analyzer import GeminiImageAnalyzer
import google.generativeai as genai
from prompt_saver import save_prompt

class MultimodalQuerySystem:
    def __init__(self, faiss_index_path="faiss_index", gemini_api_key=None):
        """Initialize multimodal query system."""
        # Load existing text-based system
        self.load_faiss_index(faiss_index_path)
        
        # Initialize image analyzer
        self.image_analyzer = GeminiImageAnalyzer(gemini_api_key)
        
        # Initialize Gemini for final answer synthesis
        genai.configure(api_key=gemini_api_key or os.getenv('GEMINI_API_KEY'))
        self.synthesis_model = genai.GenerativeModel('gemini-1.5-flash')
    
    def load_faiss_index(self, index_path):
        """Load the existing FAISS index and related data."""
        try:
            self.index = faiss.read_index(os.path.join(index_path, "index.faiss"))
            
            with open(os.path.join(index_path, "chunks.pkl"), "rb") as f:
                self.chunks = pickle.load(f)
            
            with open(os.path.join(index_path, "metadata.pkl"), "rb") as f:
                self.metadata = pickle.load(f)
            
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("FAISS index loaded successfully")
            
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            self.index = None
    
    def search_text_documents(self, query: str, k: int = 5) -> List[str]:
        """Search text documents using existing FAISS index."""
        if not self.index:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                relevant_chunks.append(self.chunks[idx])
        
        return relevant_chunks
    
    def analyze_relevant_images(self, query: str, image_directory: str = "Images") -> List[Dict]:
        """Analyze images that might be relevant to the query."""
        if not os.path.exists(image_directory):
            return []
        
        # Create a focused prompt based on the user's query
        focused_prompt = f"""Analyze this image in the context of this question: "{query}"\n\nFocus on:\n1. Any information relevant to the query\n2. Text, charts, graphs, or data visible in the image\n3. Visual elements that might help answer the question\n4. How this image relates to the query topic\n\nIf the image is not relevant to the query, briefly state that."""
        
        save_prompt(focused_prompt, "multimodal_query_focused_prompt")
        
        results = self.image_analyzer.batch_analyze_images(image_directory, focused_prompt)
        
        # Filter for relevant results
        relevant_results = []
        for result in results:
            if result['status'] == 'success' and result['analysis']:
                # Simple relevance check - you could make this more sophisticated
                analysis_lower = result['analysis'].lower()
                query_words = query.lower().split()
                
                relevance_score = sum(1 for word in query_words if word in analysis_lower)
                if relevance_score > 0 or len(query_words) <= 2:  # Include if any relevance or short query
                    relevant_results.append(result)
        
        return relevant_results
    
    def multimodal_query(self, query: str, image_directory: str = "Images") -> str:
        """
        Perform a multimodal query combining text and image analysis.
        
        Args:
            query: The question to answer
            image_directory: Directory containing images to analyze
        
        Returns:
            Comprehensive answer combining text and image sources
        """
        print(f"Processing multimodal query: {query}")
        
        # 1. Search text documents
        print("Searching text documents...")
        text_results = self.search_text_documents(query, k=5)
        
        # 2. Analyze relevant images
        print("Analyzing images...")
        image_results = self.analyze_relevant_images(query, image_directory)
        
        # 3. Synthesize final answer
        return self.synthesize_answer(query, text_results, image_results)
    
    def synthesize_answer(self, query: str, text_results: List[str], image_results: List[Dict]) -> str:
        """Synthesize final answer from text and image sources."""
        
        # Prepare context for synthesis
        text_context = "\n\n".join([f"Text Source {i+1}:\n{text}" for i, text in enumerate(text_results)])
        
        image_context = ""
        for i, img_result in enumerate(image_results):
            if img_result['status'] == 'success':
                image_context += f"\nImage Analysis {i+1} ({os.path.basename(img_result['image_path'])}):\n{img_result['analysis']}\n"
        
        synthesis_prompt = f"""You are answering the following question: "{query}"\n\nBased on the information from text documents and image analysis below, provide a comprehensive answer.\n\nTEXT SOURCES:\n{text_context}\n\nIMAGE ANALYSIS:\n{image_context}\n\nInstructions:\n1. Synthesize information from both text and image sources\n2. Clearly indicate when information comes from images vs text\n3. If sources conflict, mention the discrepancy\n4. If neither source type contains relevant information, state that clearly\n5. Provide a direct, helpful answer to the original question

Answer:"""

        save_prompt(synthesis_prompt, "multimodal_query_synthesis_prompt")


        try:
            response = self.synthesis_model.generate_content(synthesis_prompt)
            return response.text
        except Exception as e:
            return f"Error synthesizing answer: {e}"

# CLI usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python multimodal_query.py '<your_question>' [image_directory]")
        print("Example: python multimodal_query.py 'What are the quarterly sales figures?' Images")
        sys.exit(1)
    
    query = sys.argv[1]
    image_dir = sys.argv[2] if len(sys.argv) > 2 else "Images"
    
    system = MultimodalQuerySystem()
    answer = system.multimodal_query(query, image_dir)
    
    print("\n" + "="*50)
    print("MULTIMODAL QUERY RESULT")
    print("="*50)
    print(answer)
