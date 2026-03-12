#!/bin/bash
# start.sh — Startet die Pipeline in einer tmux-Session mit vollem Logging.
# Verhindert, dass die Pipeline abbricht wenn die SSH-Verbindung zu RunPod getrennt wird.
#
# NUTZUNG:
#   chmod +x start.sh
#   ./start.sh                     # Pipeline starten (oder wieder verbinden)
#
# LOGS VERFOLGEN (zweites Terminal):
#   tail -f pipeline.log           # Live Log-Stream
#   less +F pipeline.log           # Interaktiv (Shift+F = live mode, q = exit)
#
# SESSION VERWALTEN:
#   tmux attach -t pipeline        # Zurück zur laufenden Pipeline
#   tmux ls                        # Alle Sessions anzeigen

SESSION="pipeline"
LOGFILE="pipeline.log"

# Wenn Session schon läuft → einfach reattachen
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Pipeline-Session läuft bereits. Verbinde..."
    echo "(Drücke Ctrl+B, dann D um die Session im Hintergrund zu lassen)"
    sleep 2
    tmux attach -t "$SESSION"
    exit 0
fi

# Sicherstellen dass tmux installiert ist
if ! command -v tmux &>/dev/null; then
    echo "Installiere tmux..."
    apt-get install -y tmux 2>/dev/null || yum install -y tmux 2>/dev/null
fi

echo "=========================================="
echo "  Starte Pipeline in tmux-Session: $SESSION"
echo "  Logs werden geschrieben nach: $LOGFILE"
echo "=========================================="
echo ""
echo "Nützliche Befehle:"
echo "  tmux attach -t $SESSION    (wieder verbinden)"
echo "  tail -f $LOGFILE           (Logs verfolgen)"
echo ""
echo "Drücke Ctrl+B, dann D um die Session im Hintergrund zu lassen."
echo ""
sleep 3

# Starte Pipeline in tmux, output wird gleichzeitig in Logfile geschrieben
# 'script -q /dev/null' erzwingt unbuffered output (damit tail -f sofort sieht)
tmux new-session -d -s "$SESSION" \
    "bash run_pipeline.sh 2>&1 | tee -a $LOGFILE; echo ''; echo '=== Pipeline beendet ==='; bash"

tmux attach -t "$SESSION"
