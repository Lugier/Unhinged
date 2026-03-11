import os
import subprocess
import sys
import argparse
from importlib.util import find_spec

def check_axolotl():
    """Prüft, ob Axolotl installiert ist."""
    try:
        subprocess.run(["axolotl", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    parser = argparse.ArgumentParser(description="Finetuning (ORPO Healing) Wrapper")
    parser.add_argument("--config", type=str, default="./finetuning/config.yaml", help="Pfad zur Axolotl yaml Config")
    parser.add_argument("--merge", action="store_true", default=True, help="LoRA nach dem Training mergen")
    
    args = parser.parse_args()

    if not check_axolotl():
        print("FEHLER: 'axolotl' ist nicht installiert oder nicht im PATH.")
        print("Bitte installiere es via: pip install 'axolotl[flash-attn,deepspeed] @ git+https://github.com/OpenAccess-AI-Collective/axolotl'")
        sys.exit(1)

    print(f"Starte Axolotl Training mit Config: {args.config}")
    
    try:
        # Axolotl Training ausführen
        subprocess.run(["axolotl", "train", args.config], check=True)
        print("Training erfolgreich beendet!")
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Training: {e}")
        sys.exit(1)

    if args.merge:
        print("Starte Merging von LoRA Adapter in das Basismodell...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            base_model_path = "./abliteration/abliterated_model"
            lora_path = "./orpo_output"
            output_dir = "./final_model"

            print(f"Lade Basismodell von: {base_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

            print(f"Lade LoRA Adapter von: {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)

            print("Merge und Unload Adapter...")
            model = model.merge_and_unload()

            print(f"Speichere finales Modell nach {output_dir}...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print("Erfolgreich gespeichert!")

        except ImportError:
            print("Fehler: 'peft', 'transformers' oder 'torch' fehlen für den Merge-Prozess.")
            sys.exit(1)
        except Exception as e:
            print(f"Fehler beim Mergen: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
