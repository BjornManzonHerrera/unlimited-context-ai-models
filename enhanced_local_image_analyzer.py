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