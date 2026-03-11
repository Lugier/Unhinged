# Uncensor Pipeline: Abliteration + ORPO Healing

Dieses Projekt dient dem Entfernen von Zensur und Restriktionen aus LLMs (Fokus auf kleine Modelle wie Qwen2.5-4B oder Gemma-3-4B) durch eine zweistufige Pipeline:

## Phase 1: Die "Smarte" Abliteration (Heretic + Optuna)
Anstatt die Zensur statisch bei allen Layern gleich zu berechnen, nutzen wir den **Heretic-Algorithmus** inkl. **Optuna TPE-Optimization**.
- **Was passiert:** Das Heretic-Framework sucht in ca. 100 Trial-Läufen (~20 Min auf RTX 3090) die **mathematisch perfekten Layer**, in denen Zensur entfernt werden muss, *ohne* die restliche Intelligenz (KL Divergenz) des Modells zu zerstören.
- **SOTA Features:** Wir nutzen Gram-Schmidt ("Projected Direction") und Row-Norm-Preserving, um die ursprünglichen Vektorlängen in den Gewichten zu erhalten. Resultat: Das Modell vergisst die Moral, bleibt aber genauso schlau wie das Basis-Modell!
- **Dauer:** ca. 20-30 Minuten.

1. **Abliteration (Orthogonalization)**: Entfernt mathematisch die "Refusal"-Richtung aus den Gewichten des Modells, ohne dass ein vollständiges Finetuning nötig ist.
2. **ORPO Healing (LoRA)**: Trainiert das abliterierte Modell mit ORPO (Odds Ratio Preference Optimization), um mögliche Qualitätsverluste aus der Abliteration zu beheben ("Heilung"), während das Modell unzensiert bleibt. Im Gegensatz zu DPO benötigt ORPO kein Referenzmodell, was massiv VRAM spart.

## Voraussetzungen

- **OS**: Linux (Ubuntu empfohlen, nutzbar z.B. auf RunPod).
- **GPU**: NVIDIA GPU, VRAM-Bedarf für 4B-Modelle: mind. 12-16 GB (dank ORPO + QLoRA), empfohlen 24 GB für größere Batches. Das Script funktioniert optimal auf einer RTX A40, A100 oder RTX 3090/4090.
- **Python**: Python 3.10 oder höher.

## Intelligence Booster (Reasoning/CoT)

Wir haben das Dataset `crownelius/Opus-4.6-Reasoning-3300x` integriert. 
- **Was es bewirkt:** Es verleiht dem Modell "Thinking"-Fähigkeiten (Chain-of-Thought).
- **Vorteil:** Das Modell antwortet nicht nur unzensiert, sondern geht dabei logischer und schrittweiser vor. Es macht das unzensierte 4B-Modell deutlich "klüger" (DeepSeek-R1-Stil).
- **Konfiguration:** In `finetuning/config.yaml` wird dieses Dataset automatisch mit dem ORPO-Alignment gemischt.

## Quick Start

1. Repository klonen oder dieses Verzeichnis betreten:
   ```bash
   cd uncensor-pipeline
   ```

2. Ausführbarkeitsrechte für das Master-Skript vergeben:
   ```bash
   chmod +x run_pipeline.sh
   ```

3. Pipeline starten:
   ```bash
   ./run_pipeline.sh
   ```

Das Skript wird dich nach der HuggingFace Model ID fragen (Standard ist `Qwen/Qwen2.5-4B-Instruct`).

## Projektstruktur und Parameter

- `abliteration/abliterate.py`: Führt den chirurgischen Eingriff durch. Du kannst Parameter wie `--layer_fraction` anpassen (Standard 0.6 = nur die hinteren 60% der Layer werden "abliteriert", da sich hier oft die Safety-Features befinden).
- `finetuning/config.yaml`: Die Axolotl-Konfiguration. Sie nutzt QLoRA (4-bit loading) für Effizienz und trainiert mit ORPO und Flash Attention.
- `finetuning/train.py`: Wrapper, um Axolotl auszuführen und am Ende den finalen LoRA-Adapter in das fertige Modell (`./final_model`) zu mergen.

## Ergebnisse

Nach dem erfolgreichen Durchlauf befinden sich die Endergebnisse in:
- `./abliterated_model/`: Das modifizierte Modell aus Stufe 1.
- `./orpo_output/`: Die LoRA-Checkpoints aus Stufe 2.
- `./final_model/`: Das finale, "geheilte" und unzensierte Modell, bereit zur Nutzung (z.B. mit vLLM, Ollama via llama.cpp, etc.).
