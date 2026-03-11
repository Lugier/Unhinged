#!/bin/bash
set -e

echo "=========================================="
echo "      LLM Uncensor Pipeline gestartet     "
echo "=========================================="

# 1. CUDA Prüfung
if ! command -v nvidia-smi &> /dev/null
then
    echo "FEHLER: nvidia-smi nicht gefunden. Bitte stelle sicher, dass du eine NVIDIA GPU mit passenden Treibern hast."
    exit 1
fi

echo "CUDA Check OK. NVIDIA GPU erkannt."
nvidia-smi

# 3. User Input für Model ID
read -p "Bitte gib die Model-ID ein (Standard: Qwen/Qwen3.5-4B): " MODEL_ID
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-4B}
echo "Starte Pipeline für Model: $MODEL_ID"

# 2. Environments und Requirements
# Installiere Abhängigkeiten für Abliteration (Heretic)
echo "Installiere Abhängigkeiten für Heretic..."
pip install -r abliteration/requirements.txt

# Farben für die Ausgabe
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
RED='\033[1;31m'
NC='\033[0m' # No Color

# 3. Abliteration mit Heretic (Optuna TPE Optimization)
echo -e "${YELLOW}Phase 1: Abliteration mit Heretic (TPE Optuna)${NC}"
cd abliteration
echo -e "${GREEN}Führe Heretic aus (Suche nach perfekten Layern)... Dies dauert ca. 20-30 Minuten auf einer RTX 3090.${NC}"
# Da heretic eine CLI ist und User-Eingaben für das Speichern benötigt,
# nutzen wir einen Python-Wrapper, der die Eingaben "mockt" (automatisiert).
python run_heretic.py --config heretic_config.toml
cd ..

if [ ! -d "abliteration/abliterated_model" ]; then
    echo -e "${RED}Fehler: Abliteriertes Modell wurde nicht gefunden. Heretic fehlgeschlagen!${NC}"
    exit 1
fi
echo -e "${GREEN}Abliteration erfolgreich abgeschlossen!${NC}"

echo "Installiere Abhängigkeiten für Finetuning (ORPO)..."
pip install -r finetuning/requirements.txt
# Axolotl muss vom github repo installiert werden für die neuesten Features
pip install 'axolotl[flash-attn,deepspeed] @ git+https://github.com/OpenAccess-AI-Collective/axolotl'

# 5. ORPO Training ausführen
echo "=========================================="
echo "      Schritt 2: ORPO Healing (LoRA)      "
echo "=========================================="
# Update base_model in config falls nötig (wird aus script gesteuert)
python finetuning/train.py --config "./finetuning/config.yaml" --merge

echo "=========================================="
echo "      Pipeline erfolgreich beendet!       "
echo "=========================================="
echo "Dein finales, unzensiertes und geheiltes Modell liegt in: ./final_model/"
