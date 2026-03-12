# 🚀 Unhinged: The Ultimate Uncensored Pipeline
### Abliteration + SFT Intelligence Boost + ORPO Healing

**Unhinged** ist eine hochgradig automatisierte "Ultra Turbo" Pipeline zur Entfernung von Zensur und Restriktionen aus Large Language Models (optimiert für Qwen 2.5/3.5 & Gemma 2/3). 

Dieses Projekt kombiniert modernste mathematische Abliteration mit spezialisierten Fine-Tuning-Verfahren, um Modelle zu erschaffen, die sowohl **absolut unzensiert** als auch **intelligent und kohärent** bleiben.

---

## 💎 Key Features
- **🤖 Phase 1: Auto-Heretic Interface**: Vollautomatisierte Abliteration dank neuem Auto-Prompting-Handler. Kein manuelles Klicken mehr in der CLI.
- **⚡ Axolotl Integration**: Phase 2 (SFT) und Phase 3 (ORPO) nutzen jetzt die volle Power von Axolotl für effizientes 4-bit QLoRA Training.
- **🎯 Optuna TPE-Optimization**: Automatische Suche nach den perfekten Layern, um die KL-Divergenz minimal zu halten.
- **💊 ORPO Healing**: "Heilung" des Modells nach der Abliteration ohne Bedarf an einem Referenzmodell – spart massiv VRAM.
- **🏎️ Ultra Turbo Mode**: Optimierte Skripte für maximale Geschwindigkeit auf RTX 3090/4090 oder RunPod-Instanzen.

---

## 🛠️ Die 3 Phasen der Erleuchtung

### 🟦 Phase 1: Die "Smarte" Abliteration (Heretic)
Anstatt die Zensur statisch bei allen Layern gleich zu berechnen, nutzen wir den **Heretic-Algorithmus**.
- **Was passiert:** Das Framework sucht die mathematisch perfekten Layer, in denen Zensur entfernt werden muss.
- **SOTA Features:** Gram-Schmidt ("Projected Direction") und Row-Norm-Preserving zur Gewichts-Erhaltung.
- **Dauer:** ca. 20-30 Minuten.

### 🟨 Phase 2: SFT Intelligence Booster (LoRA)
Trainiert das Modell mit **Supervised Fine-Tuning** auf hochwertigen Instruction-Datasets.
- **Vorteil:** Das Modell gewinnt an Reasoning-Fähigkeit zurück, die durch die Abliteration leicht beeinträchtigt sein könnte.
- **Umsetzung:** Axolotl-driven QLoRA (Rank 32+).

### 🟩 Phase 3: ORPO Healing (LoRA)
Finaler Schliff mit **ORPO (Odds Ratio Preference Optimization)**.
- **Vorteil:** Richtet das Modell auf "gute" (hilfreiche, unzensierte) vs. "schlechte" Antworten aus.
- **Dauer:** ca. 1-2 Stunden.

---

## 🚀 Quick Start

1. **Voraussetzungen**: 
   - Linux (Ubuntu empfohlen/RunPod)
   - NVIDIA GPU (mind. 16GB VRAM, 24GB+ empfohlen)
   - Python 3.10+

2. **Installation & Start**:
   ```bash
   git clone https://github.com/your-username/Unhinged.git
   cd Unhinged
   chmod +x run_pipeline.sh
   ./run_pipeline.sh
   ```

Das Skript führt dich durch alle Installationen und fragt nach der HuggingFace Model ID (Standard: `Qwen/Qwen3.5-4B`).

---

## 📂 Projektstruktur

```
Unhinged/
├── run_pipeline.sh                 # Master-Skript (Vollautomatisch)
├── start.sh                        # Hilfsskript für Environment-Setup
├── abliteration/
│   ├── run_heretic.py              # Auto-Prompting Wrapper für Heretic
│   ├── heretic_config.toml         # Heretic-Konfiguration
│   └── ...
└── finetuning/
    ├── sft_config.yaml             # Axolotl SFT Config
    ├── config.yaml                 # Axolotl ORPO Config
    ├── train.py                    # Axolotl Training & Merge Wrapper
    └── ...
```

---

## 🏆 Ergebnisse

Nach Abschluss findest du deine Modelle hier:
1. `abliteration/abliterated_model/` -> Rein mathematisch unzensiert.
2. `./sft_merged_model/` -> Unzensiert + Intelligent.
3. `./final_model/` -> **Das Endprodukt: Unzensiert, intelligent & perfekt ausgerichtet.**

*Ready for Deployment via vLLM, Ollama oder LM Studio.*

