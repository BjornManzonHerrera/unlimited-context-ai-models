from integrated_multimodal_system import IntegratedMultimodalSystem
import os
from prompt_saver import save_output
import sys

def query(question: str):
    """
    This function exclusively uses the offline IntegratedMultimodalSystem to answer a question.
    """
    print("Initializing offline multimodal system...")
    system = IntegratedMultimodalSystem()
    
    print(f"Processing query: \"{question}\"")
    answer = system.comprehensive_query(question)
    
    return answer

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query.py \"<your_question>\"")
        sys.exit(1)
        
    question = sys.argv[1]
    
    try:
        answer = query(question)
        save_output(answer, question)
        print("\n" + "="*50)
        print("QUERY RESULT")
        print("="*50)
        print(answer)
    except Exception as e:
        print(f"An error occurred: {e}")
