import sys
from unittest.mock import patch
import heretic.main

def _normalize_message(message):
    return (message or "").strip().lower()


def _first_choice_value(choices):
    if not choices:
        raise RuntimeError("Heretic hat keine auswaehlbaren Trials geliefert.")
    return choices[0].value


def auto_prompt_select(message, choices=None, **kwargs):
    normalized = _normalize_message(message)
    print(f"[AUTO-PROMPT] Select: {message!r}")
    
    if "trial" in normalized and "use" in normalized:
        return _first_choice_value(choices)
    elif "what do you want to do with the decensored model" in normalized:
        # State machine to save once then return
        if not hasattr(auto_prompt_select, "saved_flag"):
            auto_prompt_select.saved_flag = True
            print("[AUTO-PROMPT] Triggering Save...")
            return "Save the model to a local folder"
        print("[AUTO-PROMPT] Already saved, returning to menu...")
        return "Return to the trial selection menu"
    elif "how would you like to proceed" in normalized or "how do you want to proceed" in normalized:
        # This can be the resume prompt OR the merge strategy prompt
        if choices and any("merge" in c.title.lower() for c in choices):
            print("[AUTO-PROMPT] Selecting 'Merge' strategy...")
            for c in choices:
                if "merge" in c.title.lower(): return c.value
        
        # Resume prompt handling
        if choices and len(choices) > 1 and "scratch" in choices[1].title.lower():
            print("[AUTO-PROMPT] Starting from scratch as requested...")
            return choices[1].value
        return _first_choice_value(choices)
    elif "select a trial" in normalized or "pareto" in normalized:
        return _first_choice_value(choices)
    elif "what do you want to do" in normalized:
        # Main menu after trial selection
        if not hasattr(auto_prompt_select, "done_it"):
            auto_prompt_select.done_it = True
            return _first_choice_value(choices) # Usually 'Pick a trial' or similar
        return "Quit"

    if choices:
        return _first_choice_value(choices)
    raise RuntimeError(f"Unbekannter Heretic-Auswahldialog: {message!r}")

def auto_prompt_path(message, **kwargs):
    print(f"[AUTO-PROMPT] Path: {message!r}")
    normalized = _normalize_message(message)
    if "path" in normalized and "folder" in normalized:
        return "abliterated_model"
    return "abliterated_model"

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
