import sys
from unittest.mock import patch
import heretic.main

def auto_prompt_select(message, choices=None, **kwargs):
    if "Which trial do you want to use" in message:
        # We always want the numerically best trial which is the first one presented
        return choices[0].value
    elif "What do you want to do" in message:
        # If we haven't saved yet, save. Otherwise return.
        if not hasattr(auto_prompt_select, "saved"):
            auto_prompt_select.saved = True
            return "Save the model to a local folder"
        return "Return to the trial selection menu"
    elif "How do you want to proceed" in message:
        return "merge"
    elif "Select a trial" in message:
        # If it asks this instead, it means the Pareto front choices are presented
        return choices[0].value if choices else ""
    return None

def auto_prompt_path(message, **kwargs):
    if "Path to the folder" in message:
        return "abliterated_model"
    return ""

def auto_prompt_text(message, **kwargs):
    return ""

def auto_prompt_password(message, **kwargs):
    return ""

print("==== Starting Automated Heretic Pipeline ====")
print("Mocking interactive CLI inputs for fully automatic execution on RunPod...")

# Apply all patches simultaneously
with patch('heretic.main.prompt_select', side_effect=auto_prompt_select), \
     patch('heretic.main.prompt_path', side_effect=auto_prompt_path), \
     patch('heretic.main.prompt_text', side_effect=auto_prompt_text), \
     patch('heretic.main.prompt_password', side_effect=auto_prompt_password):
    
    # We remove 'run_heretic.py' from sys.argv so Heretic parses args correctly
    if sys.argv[0].endswith('run_heretic.py'):
        sys.argv.pop(0)
    
    # Manually insert 'heretic' as arg0 to simulate the CLI
    sys.argv.insert(0, 'heretic')
    
    heretic.main.run()

print("==== Automated Heretic Pipeline Finished ====")
