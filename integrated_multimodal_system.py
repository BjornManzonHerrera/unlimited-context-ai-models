# integrated_multimodal_system.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
from enhanced_local_image_analyzer import EnhancedLocalImageAnalyzer
import requests
from typing import List, Dict

class IntegratedMultimodalSystem:
    def __init__(self, 
                 faiss_index_path="faiss_index",
                 text_model="llama3.1:8b-instruct-q4_0",
                 vision_model="llava:13b"):
        """Complete local multimodal system optimized for your hardware."""
        
        self.text_model = text_model
        self.vision_model = vision_model
        
        # Load existing text index
        self.load_text_index(faiss_index_path)
        
        # Initialize image analyzer
        self.image_analyzer = EnhancedLocalImageAnalyzer(vision_model)
        
        # Image analysis cache to avoid re-processing
        self.image_cache_file = "image_analysis_cache.json"
        self.load_image_cache()
    
    def load_text_index(self, index_path):
        """Load your existing FAISS text index."""
        try:
            self.text_index = faiss.read_index(os.path.join(index_path, "index.faiss"))
            
            with open(os.path.join(index_path, "chunks.pkl"), "rb") as f:
                self.text_chunks = pickle.load(f)
            
            with open(os.path.join(index_path, "metadata.pkl"), "rb") as f:
                self.text_metadata = pickle.load(f)
            
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Text index loaded successfully")
            
        except Exception as e:
            print(f"Error loading text index: {e}")
            self.text_index = None
    
    def load_image_cache(self):
        """Load cached image analyses to avoid reprocessing."""
        try:
            if os.path.exists(self.image_cache_file):
                with open(self.image_cache_file, 'r') as f:
                    self.image_cache = json.load(f)
                print(f"Loaded {len(self.image_cache)} cached image analyses")
            else:
                self.image_cache = {}
        except Exception as e:
            print(f"Warning: Could not load image cache: {e}")
            self.image_cache = {}
    
    def save_image_cache(self):
        """Save image analysis cache."""
        try:
            with open(self.image_cache_file, 'w') as f:
                json.dump(self.image_cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save image cache: {e}")
    
    def search_text_content(self, query: str, k: int = 5) -> List[str]:
        """Search existing text documents."""
        if not self.text_index:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.text_index.search(query_embedding.astype('float32'), k)
        
        relevant_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.text_chunks):
                # Include relevance score
                chunk_with_score = f"[Relevance: {1/(1+distances[0][i]):.3f}] {self.text_chunks[idx]}"
                relevant_chunks.append(chunk_with_score)
        
        return relevant_chunks
    
    def process_images_for_query(self, query: str, image_directory: str = "Images") -> List[Dict]:
        """Process images relevant to the query with caching."""
        if not os.path.exists(image_directory):
            return []
        
        results = []
        
        for filename in os.listdir(image_directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                image_path = os.path.join(image_directory, filename)
                
                # Check cache first
                cache_key = f"{image_path}_{hash(query)}"
                if cache_key in self.image_cache:
                    print(f"Using cached analysis for {filename}")
                    results.append(self.image_cache[cache_key])
                    continue
                
                # Analyze image
                print(f"Analyzing {filename}...")
                result = self.image_analyzer.analyze_image_advanced(image_path, query)
                
                if result['status'] == 'success':
                    # Cache successful analysis
                    self.image_cache[cache_key] = result
                    results.append(result)
        
        # Save cache after processing
        self.save_image_cache()
        return results
    
    def comprehensive_query(self, query: str, image_directory: str = "Images") -> str:
        """
        Perform comprehensive query using both text and image sources.
        Optimized for your hardware specifications.
        """
        
        print(f"\nProcessing comprehensive query: '{query}'")
        print("="*60)
        
        # Step 1: Search text documents
        print("Searching text documents...")
        text_results = self.search_text_content(query, k=5)
        print(f"Found {len(text_results)} relevant text chunks")
        
        # Step 2: Process images
        print("Processing images...")
        image_results = self.process_images_for_query(query, image_directory)
        successful_images = [r for r in image_results if r['status'] == 'success']
        print(f"Successfully analyzed {len(successful_images)} images")
        
        # Step 3: Synthesize comprehensive answer
        print("Synthesizing comprehensive answer...")
        return self.synthesize_comprehensive_answer(query, text_results, successful_images)
    
    def synthesize_comprehensive_answer(self, query: str, text_results: List[str], image_results: List[Dict]) -> str:
        """Create final answer using local text model."""
        
        # Prepare comprehensive context
        text_context = ""
        if text_results:
            text_context = "\n\n".join([f"TEXT SOURCE {i+1}:\n{text}" for i, text in enumerate(text_results)])
        
        image_context = ""
        if image_results:
            for i, img_result in enumerate(image_results):
                filename = img_result.get('filename', 'unknown')
                analysis = img_result.get('analysis', 'No analysis available')
                image_context += f"\nIMAGE SOURCE {i+1} ({filename}):\n{analysis}\n"
        
        # Create synthesis prompt optimized for your local LLM
        synthesis_prompt = f"""You are a comprehensive document analysis assistant. Answer this question using ALL available sources:

QUESTION: {query}

AVAILABLE INFORMATION:

{text_context}

{image_context}

INSTRUCTIONS:
1. Provide a direct, comprehensive answer to the question
2. Synthesize information from both text documents and image analysis
3. When citing information, specify if it comes from "text documents" or "image analysis"
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

# Performance monitoring
def monitor_system_resources():
    """Optional: Monitor system resource usage during processing."""
    try:
        import psutil
        import GPUtil
        
        print("\nSYSTEM STATUS:")
        print(f"CPU Usage: {psutil.cpu_percent()}%")
        print(f"RAM Usage: {psutil.virtual_memory().percent}%")
        
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"GPU Usage: {gpu.load*100:.1f}%")
            print(f"VRAM Usage: {gpu.memoryUtil*100:.1f}%")
            
    except ImportError:
        print("Install psutil and gputil for resource monitoring: pip install psutil gputil")
    except Exception as e:
        print(f"Resource monitoring error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python integrated_multimodal_system.py '<question>' [image_directory]")
        print("Example: python integrated_multimodal_system.py 'What are the quarterly sales figures?' Images")
        sys.exit(1)
    
    query = sys.argv[1]
    image_dir = sys.argv[2] if len(sys.argv) > 2 else "Images"
    
    # Monitor resources
    monitor_system_resources()
    
    # Run comprehensive analysis
    system = IntegratedMultimodalSystem()
    answer = system.comprehensive_query(query, image_dir)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS RESULT")
    print("="*60)
    print(answer)
    print("\n" + "="*60)
    
    # Final resource check
    monitor_system_resources()
