# GEMINI.md - Complete Local Implementation Guide

## Hardware Assessment

**Your System Specifications:**
- **CPU**: 10-core Intel i5-13450HX ✅ Excellent
- **GPU**: NVIDIA RTX 3050 (6GB VRAM) ✅ Good for medium models
- **RAM**: 25GB ✅ Sufficient for large models
- **Storage**: 512GB SSD ✅ Fast I/O

**Verdict**: Your hardware is well-suited for running local vision-language models with good performance.

## Recommended Implementation Strategy

### Phase 1: LLaVA Integration (Week 1)
**Target**: Get basic image analysis working locally

**Model Recommendation for Your Hardware:**
- **Primary**: `llava:13b` (uses ~8GB RAM + 4GB VRAM) - Best balance
- **Fallback**: `llava:7b` (uses ~5GB RAM + 3GB VRAM) - If 13b is too heavy
- **Future**: `llava:34b` (quantized) - When you want maximum quality

### Phase 2: Performance Optimization (Week 2)
**Target**: Optimize for speed and efficiency

### Phase 3: Advanced Features (Week 3+)
**Target**: Multi-image analysis, batch processing, advanced queries

---

## Implementation Roadmap

### Step 1: Environment Setup

```bash
# 1. Ensure Ollama is installed and running
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull recommended models
ollama pull llava:13b                    # Primary vision model
ollama pull llama3.1:8b-instruct-q4_0   # Your existing text model (keep)

# 3. Install additional Python dependencies
pip install requests pillow opencv-python pytesseract numpy faiss-cpu sentence-transformers

# 4. Optional: Install Tesseract as backup OCR
sudo apt-get install tesseract-ocr  # Linux
# brew install tesseract            # macOS
```

### Step 2: Core Implementation Files

#### A. Enhanced Local Image Analyzer

```python
# enhanced_local_image_analyzer.py
import requests
import json
import base64
import os
import time
from PIL import Image
from typing import Dict, List, Optional
import concurrent.futures
import threading

class EnhancedLocalImageAnalyzer:
    def __init__(self, vision_model="llava:13b", ollama_url="http://localhost:11434"):
        """Optimized for your hardware specs."""
        self.vision_model = vision_model
        self.ollama_url = ollama_url
        self.request_lock = threading.Lock()  # Prevent VRAM conflicts
        
        # Verify model availability
        self.verify_model_availability()
    
    def verify_model_availability(self):
        """Check if required models are available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            available_models = [m['name'] for m in response.json().get('models', [])]
            
            if self.vision_model not in available_models:
                print(f"Warning: {self.vision_model} not found. Available models: {available_models}")
                print(f"Run: ollama pull {self.vision_model}")
            else:
                print(f"✅ {self.vision_model} is ready")
                
        except Exception as e:
            print(f"Could not verify models: {e}")
    
    def optimize_image(self, image_path: str, max_size=(1024, 1024)) -> str:
        """Optimize image size for faster processing while maintaining quality."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Save optimized version temporarily
                optimized_path = f"/tmp/optimized_{os.path.basename(image_path)}"
                img.save(optimized_path, "JPEG", quality=85)
                return optimized_path
                
        except Exception as e:
            print(f"Image optimization failed: {e}")
            return image_path
    
    def analyze_image_advanced(self, image_path: str, query_context: str = "") -> Dict:
        """Advanced image analysis optimized for your system."""
        
        # Optimize image first
        optimized_path = self.optimize_image(image_path)
        
        try:
            # Prepare focused prompt
            focused_prompt = f"""Analyze this image comprehensively for a document analysis system.

Query Context: {query_context}

Provide detailed analysis covering:
1. **Text Content**: Extract and transcribe ALL visible text, numbers, labels
2. **Visual Data**: Describe charts, graphs, tables, diagrams in detail
3. **Document Type**: Identify if this is a report, chart, invoice, etc.
4. **Key Information**: Highlight important data points, trends, or insights
5. **Searchable Keywords**: List key terms that would help find this image later

Be thorough and precise - this analysis will be used for document search and retrieval."""

            with open(optimized_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Use thread lock to prevent VRAM conflicts
            with self.request_lock:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.vision_model,
                        "prompt": focused_prompt,
                        "images": [image_base64],
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # More focused responses
                            "top_p": 0.9
                        }
                    },
                    timeout=180  # Longer timeout for complex images
                )
            
            # Cleanup optimized image if it was created
            if optimized_path != image_path and os.path.exists(optimized_path):
                os.remove(optimized_path)
            
            if response.status_code == 200:
                analysis_text = response.json().get('response', '')
                
                return {
                    'image_path': image_path,
                    'filename': os.path.basename(image_path),
                    'analysis': analysis_text,
                    'model_used': self.vision_model,
                    'status': 'success',
                    'processing_time': time.time()
                }
            else:
                return {
                    'image_path': image_path,
                    'status': 'error',
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            # Cleanup on error
            if optimized_path != image_path and os.path.exists(optimized_path):
                os.remove(optimized_path)
            
            return {
                'image_path': image_path,
                'status': 'error',
                'error': str(e)
            }
    
    def batch_process_optimized(self, image_directory: str, query_context: str = "", max_workers: int = 2) -> List[Dict]:
        """
        Optimized batch processing for your hardware.
        Uses threading but limits concurrent requests to prevent VRAM issues.
        """
        results = []
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
        
        image_files = [
            f for f in os.listdir(image_directory) 
            if os.path.splitext(f)[1].lower() in supported_formats
        ]
        
        print(f"Found {len(image_files)} images to process")
        
        # Process with limited concurrency to manage VRAM
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    self.analyze_image_advanced, 
                    os.path.join(image_directory, filename), 
                    query_context
                ): filename for filename in image_files
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✅ Completed: {filename}")
                except Exception as e:
                    print(f"❌ Failed: {filename} - {e}")
                    results.append({
                        'image_path': os.path.join(image_directory, filename),
                        'status': 'error',
                        'error': str(e)
                    })
        
        return results
```

#### B. Integrated Multimodal System

```python
# integrated_multimodal_system.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
from enhanced_local_image_analyzer import EnhancedLocalImageAnalyzer
import requests

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
            print("✅ Text index loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading text index: {e}")
            self.text_index = None
    
    def load_image_cache(self):
        """Load cached image analyses to avoid reprocessing."""
        try:
            if os.path.exists(self.image_cache_file):
                with open(self.image_cache_file, 'r') as f:
                    self.image_cache = json.load(f)
                print(f"✅ Loaded {len(self.image_cache)} cached image analyses")
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
                    print(f"📋 Using cached analysis for {filename}")
                    results.append(self.image_cache[cache_key])
                    continue
                
                # Analyze image
                print(f"🔍 Analyzing {filename}...")
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
        
        print(f"\n🚀 Processing comprehensive query: '{query}'")
        print("="*60)
        
        # Step 1: Search text documents
        print("📚 Searching text documents...")
        text_results = self.search_text_content(query, k=5)
        print(f"Found {len(text_results)} relevant text chunks")
        
        # Step 2: Process images
        print("🖼️  Processing images...")
        image_results = self.process_images_for_query(query, image_directory)
        successful_images = [r for r in image_results if r['status'] == 'success']
        print(f"Successfully analyzed {len(successful_images)} images")
        
        # Step 3: Synthesize comprehensive answer
        print("🧠 Synthesizing comprehensive answer...")
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
                return f"❌ Error generating response: HTTP {response.status_code}"
                
        except Exception as e:
            return f"❌ Synthesis failed: {e}"

# Performance monitoring
def monitor_system_resources():
    """Optional: Monitor system resource usage during processing."""
    try:
        import psutil
        import GPUtil
        
        print(f"\n📊 SYSTEM STATUS:")
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
        print("Usage: python enhanced_local_image_analyzer.py '<question>' [image_directory]")
        print("Example: python enhanced_local_image_analyzer.py 'What are the quarterly sales figures?' Images")
        sys.exit(1)
    
    query = sys.argv[1]
    image_dir = sys.argv[2] if len(sys.argv) > 2 else "Images"
    
    # Monitor resources
    monitor_system_resources()
    
    # Run comprehensive analysis
    system = IntegratedMultimodalSystem()
    answer = system.comprehensive_query(query, image_dir)
    
    print("\n" + "="*60)
    print("🎯 COMPREHENSIVE ANALYSIS RESULT")
    print("="*60)
    print(answer)
    print("\n" + "="*60)
    
    # Final resource check
    monitor_system_resources()
```

### Step 3: Integration with Existing System

#### Update your existing `query.py`:

```python
# Add to the top of your existing query.py
from integrated_multimodal_system import IntegratedMultimodalSystem
import os

# Modify your existing query function:
def enhanced_query(question, use_multimodal=True):
    """Enhanced version of your existing query function."""
    
    if use_multimodal and os.path.exists("Images"):
        print("🔄 Using multimodal analysis...")
        system = IntegratedMultimodalSystem()
        return system.comprehensive_query(question)
    else:
        print("📝 Using text-only analysis...")
        # Your existing query logic here
        return your_existing_query_function(question)

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
```

### Step 4: Performance Testing Script

```python
# performance_test.py
import time
import os
from integrated_multimodal_system import IntegratedMultimodalSystem

def test_system_performance():
    """Test and benchmark your local multimodal system."""
    
    system = IntegratedMultimodalSystem()
    
    test_queries = [
        "What are the main financial trends?",
        "Summarize the key data points",
        "What charts or graphs are available?",
        "Extract all numerical data",
        "What is the quarterly performance?"
    ]
    
    print("🧪 PERFORMANCE TESTING")
    print("="*50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}/5: {query}")
        start_time = time.time()
        
        try:
            result = system.comprehensive_query(query)
            duration = time.time() - start_time
            
            print(f"✅ Completed in {duration:.2f} seconds")
            print(f"Response length: {len(result)} characters")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print(f"\n📊 Cache status: {len(system.image_analyzer.image_cache)} cached analyses")

if __name__ == "__main__":
    test_system_performance()
```

---

## Hardware-Specific Optimizations

### For Your RTX 3050 (6GB VRAM):
- **Model Choice**: `llava:13b` is optimal (uses ~4GB VRAM)
- **Concurrent Processing**: Max 2 simultaneous image analyses
- **Image Optimization**: Resize to 1024x1024 to reduce VRAM usage
- **Memory Management**: Use threading locks to prevent conflicts

### For Your 25GB RAM:
- **Large Context**: Use 4096+ token context for comprehensive analysis
- **Caching**: Aggressive image analysis caching
- **Batch Processing**: Process multiple images efficiently

### For Your 10-core CPU:
- **Parallel Processing**: Use 2-4 worker threads for batch operations
- **Background Tasks**: OCR and preprocessing in parallel

---

## Usage Examples

### Basic Setup:
```bash
# Create project structure
mkdir Images
cp your_charts_and_documents/* Images/

# Test single image
python enhanced_local_image_analyzer.py "Analyze this financial chart" Images/quarterly_report.png

# Test full system
python enhanced_local_image_analyzer.py "What are the revenue trends?" Images/
```

### Advanced Queries:
```bash
# Complex financial analysis
python enhanced_local_image_analyzer.py "Compare the budget allocation shown in charts with the written financial reports"

# Data extraction
python enhanced_local_image_analyzer.py "Extract all numerical data and create a summary of key metrics"

# Trend analysis
python enhanced_local_image_analyzer.py "What trends can you identify across all documents and images?"
```

---

## Troubleshooting & Optimization

### Common Issues:
1. **VRAM Errors**: Reduce concurrent workers or use `llava:7b`
2. **Slow Processing**: Enable image optimization and caching
3. **Model Not Found**: Run `ollama pull llava:13b`

### Performance Tips:
1. **Pre-process Images**: Resize large images beforehand
2. **Use Caching**: Let the system cache analyses for repeated queries
3. **Monitor Resources**: Use the performance monitoring script
4. **Batch Processing**: Process multiple images at once for efficiency

### Expected Performance on Your Hardware:
- **Single Image Analysis**: 10-30 seconds
- **Batch Processing (10 images)**: 2-5 minutes
- **Full Query with Text + Images**: 30-90 seconds

---

## Next Steps

1. **Week 1**: Implement basic LLaVA integration
2. **Week 2**: Add performance optimizations and caching
3. **Week 3**: Create specialized prompts for your specific document types
4. **Week 4**: Build advanced query interfaces and batch processing tools

Your hardware setup is excellent for this implementation - you should get great performance with the 13b model while keeping everything completely local and private!