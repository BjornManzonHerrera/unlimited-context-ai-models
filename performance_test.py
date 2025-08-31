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
