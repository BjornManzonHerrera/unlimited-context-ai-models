import os
import datetime

def save_prompt(prompt_text, prompt_name):
    """Saves the given prompt text to a file in the 'prompts' directory."""
    if not os.path.exists("prompts"):
        os.makedirs("prompts")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prompts/{timestamp}_{prompt_name}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_text)

def save_output(output_text, query):
    """Saves the given output text to a file in the 'llm_outputs' directory."""
    if not os.path.exists("llm_outputs"):
        os.makedirs("llm_outputs")
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize query for filename
    sanitized_query = "".join(c for c in query if c.isalnum() or c in (' ', '_')).rstrip()
    sanitized_query = sanitized_query.replace(' ', '_')
    if len(sanitized_query) > 50:
        sanitized_query = sanitized_query[:50]
        
    filename = f"llm_outputs/{timestamp}_{sanitized_query}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(output_text)