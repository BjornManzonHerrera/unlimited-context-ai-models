# Adding Gemini CLI Image Analysis to Your Local AI Project

## Prerequisites

1. **Install Gemini CLI**:
   ```bash
   npm install -g @google/generative-ai-cli
   ```

2. **Set up API Key**:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   # Or add to your .bashrc/.zshrc for persistence
   echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
   ```

3. **Install Required Python Packages**:
   ```bash
   pip install google-generativeai pillow requests
   ```

## Project Structure Updates

Your updated project structure should look like:
```
your-project/
├── AI_Data/
├── Images/                 # New: Store images for analysis
├── faiss_index/
├── update.py
├── query.py
├── image_analyzer.py       # New: Image analysis module
├── multimodal_query.py     # New: Combined text + image queries
└── requirements.txt        # Updated with new dependencies
```

## Implementation Files

### 1. Create `image_analyzer.py`

```python
import google.generativeai as genai
import os
from PIL import Image
import json
from typing import Dict, List, Optional

class GeminiImageAnalyzer:
    def __init__(self, api_key: str = None):
        """Initialize Gemini image analyzer."""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def analyze_image(self, image_path: str, prompt: str = None) -> Dict:
        """
        Analyze an image with optional custom prompt.
        
        Args:
            image_path: Path to the image file
            prompt: Custom prompt for analysis (optional)
        
        Returns:
            Dictionary with analysis results
        """
        try:
            # Load and validate image
            image = Image.open(image_path)
            
            # Default prompt if none provided
            if not prompt:
                prompt = """Analyze this image in detail. Describe:
                1. What you see in the image
                2. Key objects, people, or elements
                3. Context or setting
                4. Any text visible in the image
                5. Relevant details for document/data analysis
                
                Provide a structured analysis that could be useful for search and retrieval."""
            
            # Generate content
            response = self.model.generate_content([prompt, image])
            
            return {
                'image_path': image_path,
                'analysis': response.text,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'image_path': image_path,
                'analysis': None,
                'status': 'error',
                'error': str(e)
            }
    
    def batch_analyze_images(self, image_directory: str, custom_prompt: str = None) -> List[Dict]:
        """
        Analyze all images in a directory.
        
        Args:
            image_directory: Path to directory containing images
            custom_prompt: Optional custom prompt for all images
        
        Returns:
            List of analysis results
        """
        results = []
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        if not os.path.exists(image_directory):
            print(f"Directory {image_directory} does not exist")
            return results
        
        for filename in os.listdir(image_directory):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in supported_formats:
                image_path = os.path.join(image_directory, filename)
                print(f"Analyzing: {filename}")
                
                result = self.analyze_image(image_path, custom_prompt)
                results.append(result)
        
        return results
    
    def save_analysis_results(self, results: List[Dict], output_file: str):
        """Save analysis results to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")

# CLI usage example
if __name__ == "__main__":
    import sys
    
    analyzer = GeminiImageAnalyzer()
    
    if len(sys.argv) < 2:
        print("Usage: python image_analyzer.py <image_path_or_directory> [custom_prompt]")
        sys.exit(1)
    
    path = sys.argv[1]
    custom_prompt = sys.argv[2] if len(sys.argv) > 2 else None
    
    if os.path.isfile(path):
        # Single image analysis
        result = analyzer.analyze_image(path, custom_prompt)
        print(json.dumps(result, indent=2))
    elif os.path.isdir(path):
        # Batch analysis
        results = analyzer.batch_analyze_images(path, custom_prompt)
        output_file = "image_analysis_results.json"
        analyzer.save_analysis_results(results, output_file)
    else:
        print(f"Path {path} is neither a file nor directory")
```

### 2. Create `multimodal_query.py`

```python
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import json
import os
from image_analyzer import GeminiImageAnalyzer
import google.generativeai as genai

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
        focused_prompt = f"""Analyze this image in the context of this question: "{query}"
        
        Focus on:
        1. Any information relevant to the query
        2. Text, charts, graphs, or data visible in the image
        3. Visual elements that might help answer the question
        4. How this image relates to the query topic
        
        If the image is not relevant to the query, briefly state that."""
        
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
        
        synthesis_prompt = f"""You are answering the following question: "{query}"

Based on the information from text documents and image analysis below, provide a comprehensive answer.

TEXT SOURCES:
{text_context}

IMAGE ANALYSIS:
{image_context}

Instructions:
1. Synthesize information from both text and image sources
2. Clearly indicate when information comes from images vs text
3. If sources conflict, mention the discrepancy
4. If neither source type contains relevant information, state that clearly
5. Provide a direct, helpful answer to the original question

Answer:"""

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
```

### 3. Update `requirements.txt`

Add these dependencies to your requirements.txt:
```
google-generativeai
pillow
requests
sentence-transformers
faiss-cpu
numpy
```

### 4. Enhanced `update.py` (Optional - to include image metadata)

You can modify your existing `update.py` to also index image metadata:

```python
# Add this to your existing update.py after your current implementation

def update_with_images():
    """Enhanced update function that includes image analysis in the index."""
    from image_analyzer import GeminiImageAnalyzer
    
    # Your existing update logic here...
    
    # Add image analysis
    if os.path.exists("Images"):
        print("Analyzing images for indexing...")
        analyzer = GeminiImageAnalyzer()
        
        image_results = analyzer.batch_analyze_images("Images")
        
        # Add image analyses to your chunks for indexing
        for result in image_results:
            if result['status'] == 'success':
                image_chunk = f"IMAGE ANALYSIS ({os.path.basename(result['image_path'])}): {result['analysis']}"
                chunks.append(image_chunk)
                metadata.append({
                    'source': result['image_path'],
                    'type': 'image_analysis',
                    'filename': os.path.basename(result['image_path'])
                })
    
    # Continue with your existing FAISS index creation...
```

## Usage Instructions

### 1. Setup Your Environment

```bash
# Create Images directory
mkdir Images

# Copy your images to analyze into the Images directory
cp /path/to/your/images/* Images/

# Set your Gemini API key
export GEMINI_API_KEY="your-gemini-api-key"
```

### 2. Analyze Individual Images

```bash
# Analyze a single image
python image_analyzer.py Images/chart.png

# Analyze with custom prompt
python image_analyzer.py Images/financial_report.png "Extract all numerical data and financial metrics from this image"

# Analyze all images in directory
python image_analyzer.py Images/
```

### 3. Perform Multimodal Queries

```bash
# Query combining text documents and images
python multimodal_query.py "What were the Q3 sales figures?"

# Specify custom image directory
python multimodal_query.py "Analyze the budget allocation" Images/budget_charts/

# Complex query example
python multimodal_query.py "How do the visual charts compare to the written financial reports?"
```

### 4. Integration with Existing System

You can also integrate this into your existing query.py:

```python
# Add to your existing query.py
from multimodal_query import MultimodalQuerySystem

def enhanced_query(question, use_images=True):
    if use_images and os.path.exists("Images"):
        # Use multimodal system
        system = MultimodalQuerySystem()
        return system.multimodal_query(question)
    else:
        # Use your existing text-only system
        # ... your existing query logic
        pass
```

## Command Examples

### Basic Image Analysis
```bash
python image_analyzer.py Images/quarterly_report.png "Extract financial data from this chart"
```

### Multimodal Financial Query
```bash
python multimodal_query.py "What is the trend in revenue growth based on both documents and charts?"
```

### Batch Processing
```bash
python image_analyzer.py Images/ "Identify any charts, graphs, or financial data in this image"
```

## Tips for Best Results

1. **Image Quality**: Ensure images are clear and readable (300+ DPI for text)
2. **Specific Prompts**: Use targeted prompts for better analysis
3. **Organize Images**: Keep related images in subdirectories
4. **File Naming**: Use descriptive filenames for better organization
5. **API Limits**: Be mindful of Gemini API rate limits for batch processing

## Troubleshooting

### Common Issues:
- **API Key Error**: Ensure GEMINI_API_KEY is set correctly
- **Image Format**: Supported formats: JPG, PNG, GIF, BMP, WebP
- **File Size**: Keep images under 20MB for best performance
- **Rate Limits**: Add delays between requests if hitting rate limits

### Performance Optimization:
- Resize large images before analysis
- Use specific prompts to reduce processing time
- Cache results to avoid re-analyzing the same images

## Security Notes

- Keep your API key secure and never commit it to version control
- Consider using environment files (.env) for local development
- Monitor your API usage to avoid unexpected charges