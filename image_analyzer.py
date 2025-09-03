import google.generativeai as genai
import os
from PIL import Image
import json
from typing import Dict, List, Optional
from prompt_saver import save_prompt

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
            
            save_prompt(prompt, "image_analyzer_prompt")
            
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
