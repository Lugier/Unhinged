#!/bin/bash
set -e
export PYTORCH_ALLOC_CONF=expandable_segments:True

PYTHON_BIN=""
PIP_CMD=()
AXOLOTL_VERSION="0.8.1"

require_python_3_10() {
    PYTHON_BIN="$(command -v python3)"
    if [ -z "$PYTHON_BIN" ]; then
        echo "FEHLER: python3 wurde nicht gefunden."
        exit 1
    fi

    "$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 10):
    version = ".".join(str(part) for part in sys.version_info[:3])
    print(
        f"FEHLER: Python 3.10+ wird benoetigt, gefunden wurde Python {version}. "
        "Bitte aktiviere ein passendes Environment."
    )
    raise SystemExit(1)
PY

    PIP_CMD=("$PYTHON_BIN" -m pip)
}

write_model_id_to_heretic_config() {
    local model_id="$1"
    MODEL_ID="$model_id" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

config_path = Path("abliteration/heretic_config.toml")
lines = config_path.read_text(encoding="utf-8").splitlines()
updated = False

for index, line in enumerate(lines):
    if line.startswith("model = "):
        lines[index] = f'model = "{os.environ["MODEL_ID"]}"'
        updated = True
        break

if not updated:
    raise SystemExit("FEHLER: 'model = ...' wurde in abliteration/heretic_config.toml nicht gefunden.")

config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

update_base_model_in_config() {
    local config_file="$1"
    local new_base_model="$2"
    CONFIG_FILE="$config_file" NEW_BASE="$new_base_model" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

config_path = Path(os.environ["CONFIG_FILE"])
content = config_path.read_text(encoding="utf-8")
import re
content = re.sub(
    r'^base_model:.*$',
    f'base_model: {os.environ["NEW_BASE"]}',
    content,
    flags=re.MULTILINE
)
config_path.write_text(content, encoding="utf-8")
print(f"base_model in {config_path} aktualisiert auf: {os.environ['NEW_BASE']}")
PY
}

# Farben für die Ausgabe
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
RED='\033[1;31m'
CYAN='\033[1;36m'
NC='\033[0m' # No Color

echo "=========================================="
echo "      LLM Uncensor Pipeline gestartet     "
echo "=========================================="

require_python_3_10

# 1. CUDA Prüfung
if ! command -v nvidia-smi &> /dev/null
then
    echo -e "${RED}FEHLER: nvidia-smi nicht gefunden. Bitte stelle sicher, dass du eine NVIDIA GPU mit passenden Treibern hast.${NC}"
    exit 1
fi

echo -e "${GREEN}CUDA Check OK. NVIDIA GPU erkannt.${NC}"
nvidia-smi

# 2. User Input für Model ID
MODEL_ID="Qwen/Qwen3.5-4B"
echo "Starte Pipeline für Model: $MODEL_ID"
write_model_id_to_heretic_config "$MODEL_ID"

# ==========================================
#   Phase 1: Abliteration mit Heretic
# ==========================================
echo -e "${YELLOW}=========================================="
echo "  Phase 1/3: Abliteration mit Heretic"
echo -e "==========================================${NC}"

# Installiere Abhängigkeiten nur wenn nötig
if ! "$PYTHON_BIN" -c "import heretic" &>/dev/null; then
    echo "Installiere Abhängigkeiten für Heretic..."
    "${PIP_CMD[@]}" install --break-system-packages optuna bitsandbytes accelerate peft datasets pydantic pydantic-settings questionary psutil hf-transfer kernels
    "${PIP_CMD[@]}" install --break-system-packages --no-deps heretic-llm
    "${PIP_CMD[@]}" install --break-system-packages --force-reinstall "transformers>=5.3.0" "huggingface-hub>=1.0.0"
else
    echo "Heretic Abhängigkeiten sind bereits installiert."
fi

cd abliteration

if [ -d "abliterated_model" ]; then
    echo -e "${GREEN}Abliteriertes Modell gefunden. Überspringe Phase 1...${NC}"
else
    echo "Wende Bugfixes für Heretic-LLM an (für Qwen3.5 & Continuous runs) ..."

"$PYTHON_BIN" - <<'PYEOF'
import sys
import os
import site

# Finde den korrekten Pfad zu heretic dynamisch
site_packages = site.getsitepackages()[0]
path = os.path.join(site_packages, 'heretic/model.py')
print(f"Versuche heretic zu patchen unter: {path}")

try:
    with open(path, 'r') as f: content = f.read()

    # Part A: Fix target_modules initialization (Ensure both o_proj and out_proj get LoRA adapters)
    old_tm = 'target_modules = [\n            comp.split(".")[-1] for comp in self.get_abliterable_components()\n        ]'
    new_tm = 'target_modules = list(set([comp.split(".")[-1] for comp in self.get_abliterable_components()] + ["o_proj", "out_proj"]))'
    if old_tm in content:
        content = content.replace(old_tm, new_tm)

    # Part B: Fix get_layer_modules (Unify keys to prevent KeyError)
    unified_logic = """# Qwen 3.5 Hybrid Attention Fix (Unified Keys)
        attn_out = None
        if hasattr(layer, 'self_attn'):
            attn_out = getattr(layer.self_attn, 'out_proj', None) or getattr(layer.self_attn, 'o_proj', None)
        elif hasattr(layer, 'linear_attn'):
            attn_out = getattr(layer.linear_attn, 'out_proj', None)
        
        if attn_out:
            try_add("attn.out_proj", attn_out)"""
    
    # Robust replacement using anchors
    start_anchor = "# Exceptions aren't suppressed here, because there is currently"
    end_anchor = "# Most dense models."
    
    if start_anchor in content and end_anchor in content:
        start_idx = content.find(start_anchor) + len(start_anchor)
        end_idx = content.find(end_anchor)
        # Keep the comment but replace the logic between anchors
        content = content[:start_idx] + "\n        " + unified_logic + "\n\n        " + content[end_idx:]
        with open(path, 'w') as f: f.write(content)
        print("model.py erfolgreich gepatcht (Hybrid & KeyError Fix).")
    else:
        print("Warnung: Anker-Kommentare in model.py nicht gefunden.")
except Exception as e:
    print(f"Warnung: model.py konnte nicht gepatcht werden: {e}")
PYEOF

    "$PYTHON_BIN" run_heretic.py
fi
cd ..

if [ ! -d "abliteration/abliterated_model" ]; then
    echo -e "${RED}Fehler: Abliteriertes Modell wurde nicht gefunden. Heretic fehlgeschlagen!${NC}"
    exit 1
fi
echo -e "${GREEN}Phase 1 abgeschlossen: Abliteration erfolgreich!${NC}"

# Installiere Finetuning-Abhängigkeiten nur wenn nötig
if ! "$PYTHON_BIN" -c "import transformers; import axolotl; import torchvision; assert transformers.__version__.startswith('5.')" &>/dev/null; then
    echo "Installiere Abhängigkeiten für Finetuning (dies kann dauern)..."
    "${PIP_CMD[@]}" install --break-system-packages "huggingface-hub>=1.3.0"
    # Qwen 3.5 support requires transformers from source
    "${PIP_CMD[@]}" install --break-system-packages --force-reinstall git+https://github.com/huggingface/transformers.git
    "${PIP_CMD[@]}" install --break-system-packages -r finetuning/requirements.txt
    "${PIP_CMD[@]}" install --break-system-packages torchvision torchaudio
    "${PIP_CMD[@]}" install --break-system-packages git+https://github.com/axolotl-ai-cloud/axolotl.git
else
    echo "Finetuning Abhängigkeiten (Transformers 5.x & Axolotl) sind bereits installiert."
fi

if [ ! -d "sft_merged_model" ]; then
    # Vorbereitung: OpenOrca für Axolotl mappen (KeyError Fix)
    echo "Bereite Dataset vor..."
    "$PYTHON_BIN" - <<'PY'
from datasets import load_dataset
import os
if not os.path.exists("finetuning/openorca_mapped.jsonl"):
    print("Downloade und mappe OpenOrca...")
    ds = load_dataset("Open-Orca/OpenOrca", split="train[:10000]")
    ds = ds.rename_columns({"question": "instruction", "response": "output", "system_prompt": "system"})
    ds.to_json("finetuning/openorca_mapped.jsonl")
    print("Dataset unter finetuning/openorca_mapped.jsonl gespeichert.")
else:
    print("Gemapptes Dataset bereits vorhanden.")
PY

    # Phase 2 Training
    "$PYTHON_BIN" -m axolotl.cli.train "./finetuning/sft_config.yaml"

    # Phase 2 Merge
    echo "Merge Phase 2 Modell..."
    "$PYTHON_BIN" -m axolotl.cli.merge_lora "./finetuning/sft_config.yaml"
    
    # Axolotl speichert standardmäßig in output_dir/merged
    if [ -d "./sft_output/merged" ]; then
        mv "./sft_output/merged" "./sft_merged_model"
        echo "Modell nach ./sft_merged_model verschoben."
    fi

    if [ ! -d "sft_merged_model" ]; then
        echo -e "${RED}Fehler: SFT-Merged Modell nicht gefunden!${NC}"
        exit 1
    fi
    echo -e "${GREEN}Phase 2 abgeschlossen: Intelligence Boost erfolgreich!${NC}"
else
    echo -e "${GREEN}Phase 2 (SFT) bereits abgeschlossen. Überspringe...${NC}"
fi

# Update ORPO config: base_model zeigt jetzt auf das SFT-gemergte Modell
update_base_model_in_config "finetuning/config.yaml" "./sft_merged_model"

if [ ! -d "final_model" ]; then
    # Phase 3 Training
    "$PYTHON_BIN" -m axolotl.cli.train "./finetuning/config.yaml"

    # Phase 3 Merge
    echo "Merge finales Modell..."
    "$PYTHON_BIN" -m axolotl.cli.merge_lora "./finetuning/config.yaml"
    
    if [ -d "./orpo_output/merged" ]; then
        mv "./orpo_output/merged" "./final_model"
        echo "Finales Modell nach ./final_model verschoben."
    fi
else
    echo -e "${GREEN}Phase 3 (ORPO) bereits abgeschlossen. Das finale Modell ist bereit!${NC}"
fi

echo -e "${GREEN}=========================================="
echo "      Pipeline erfolgreich beendet!       "
echo "==========================================${NC}"
echo ""
echo "Ergebnisse:"
echo "  Phase 1 (Abliteration):  ./abliteration/abliterated_model/"
echo "  Phase 2 (SFT Boost):    ./sft_merged_model/"
echo "  Phase 3 (ORPO Healing):  ./final_model/"
echo ""
echo -e "${CYAN}Dein finales, unzensiertes und intelligentes Modell liegt in: ./final_model/${NC}"
