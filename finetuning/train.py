import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def check_axolotl():
    """Prüft, ob Axolotl installiert ist."""
    try:
        subprocess.run(["axolotl", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config


def resolve_path(base_dir, configured_path, fallback):
    path_value = configured_path or fallback
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def ensure_directory(path, label):
    if not path.exists():
        print(f"FEHLER: {label} existiert nicht: {path}")
        sys.exit(1)
    if not path.is_dir():
        print(f"FEHLER: {label} ist kein Verzeichnis: {path}")
        sys.exit(1)


def pick_torch_dtype(torch_module):
    if not torch_module.cuda.is_available():
        return torch_module.float32
    if torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16
    return torch_module.float16


def merge_lora(config_path, config, output_name="final_model"):
    """Merged LoRA-Adapter ins Basismodell."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        project_root = config_path.parent.parent
        base_model_path = resolve_path(project_root, config.get("base_model"), "./abliteration/abliterated_model")
        lora_path = resolve_path(project_root, config.get("output_dir"), "./sft_output")
        output_dir = (project_root / output_name).resolve()
        ensure_directory(base_model_path, "Basismodell-Pfad")
        ensure_directory(lora_path, "LoRA-Output-Pfad")
        torch_dtype = pick_torch_dtype(torch)

        print(f"Lade Basismodell von: {base_model_path}")
        print(f"Verwende Torch-Datentyp: {torch_dtype}")
        tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            torch_dtype=torch_dtype,
            device_map="auto"
        )

        print(f"Lade LoRA Adapter von: {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path))

        print("Merge und Unload Adapter...")
        model = model.merge_and_unload()

        print(f"Speichere Modell nach {output_dir}...")
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print("Erfolgreich gespeichert!")
        return output_dir

    except ImportError:
        print("Fehler: 'peft', 'transformers', 'torch' oder 'pyyaml' fehlen für den Merge-Prozess.")
        sys.exit(1)
    except Exception as e:
        print(f"Fehler beim Mergen: {e}")
        sys.exit(1)


def run_axolotl_train(config_path):
    """Führt Axolotl-Training aus."""
    print(f"Starte Axolotl Training mit Config: {config_path}")
    try:
        subprocess.run(["axolotl", "train", str(config_path)], check=True)
        print("Training erfolgreich beendet!")
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Training: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Finetuning Wrapper (SFT + ORPO)")
    parser.add_argument("--config", type=str, required=True, help="Pfad zur Axolotl yaml Config")
    parser.add_argument("--merge", dest="merge", action="store_true", help="LoRA nach dem Training mergen")
    parser.add_argument("--no-merge", dest="merge", action="store_false", help="LoRA nicht mergen")
    parser.add_argument("--merge-output", type=str, default="final_model", help="Verzeichnisname für das gemergte Modell")
    parser.set_defaults(merge=True)

    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    if not check_axolotl():
        print("FEHLER: 'axolotl' ist nicht installiert oder nicht im PATH.")
        print("Bitte installiere es via: python3 -m pip install 'axolotl[flash-attn,deepspeed]'")
        sys.exit(1)

    run_axolotl_train(config_path)

    if args.merge:
        print("Starte Merging von LoRA Adapter in das Basismodell...")
        merge_lora(config_path, config, args.merge_output)


if __name__ == "__main__":
    main()
