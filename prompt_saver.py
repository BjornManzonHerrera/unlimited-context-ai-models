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
