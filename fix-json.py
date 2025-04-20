import json
import os
import datetime

def fix_jupyter_notebook_aggressive(notebook_path):
    # Create backup with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{notebook_path}.backup_{timestamp}"
    
    try:
        # Create backup
        with open(notebook_path, 'r', encoding='utf-8') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print(f"Backup created at: {backup_path}")
        
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            try:
                notebook = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Error at line {e.lineno}, column {e.colno}")
                return False
        
        # Aggressive fix: Clean metadata completely
        if 'metadata' in notebook:
            # Preserve only essential metadata, remove problematic parts
            clean_metadata = {}
            if 'kernelspec' in notebook['metadata']:
                clean_metadata['kernelspec'] = notebook['metadata']['kernelspec']
            
            # Set clean metadata
            notebook['metadata'] = clean_metadata
            print("Cleaned metadata structure")
        
        # Ensure nbformat is correct
        notebook['nbformat'] = 4
        notebook['nbformat_minor'] = 5
        
        # Save the fixed notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        
        print(f"Notebook fixed and saved to: {notebook_path}")
        return True
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Usage
notebook_path = "./Multimodal_Analysis_SSLV_Florence2.ipynb"  # Replace with actual path
fix_jupyter_notebook_aggressive(notebook_path)
