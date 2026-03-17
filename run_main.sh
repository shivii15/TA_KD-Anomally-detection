#!/bin/bash
# run_and_track.sh - Run for student distillation

# 1. Setup
TS=$(date +%Y%m%d_%H%M)
LOG_FILE="logs/student_train_${TS}.log"
TEACHER_MODEL="./models/teacher_best.pth"
source venv_tgkd/bin/activate

# 2. Safety Check: Does the teacher exist?
if [ ! -f "$TEACHER_MODEL" ]; then
    echo "❌ ERROR: Teacher model not found at $TEACHER_MODEL"
    echo "👉 Please run 'bash train_teacher.sh' first."
    exit 1
fi

echo "🚀 Teacher found. Starting Student Distillation... Logs: $LOG_FILE"

# Read the path that download_data.py just saved
DATA_DIR=$(cat .data_path)


# 3. Run Distillation
nohup python3 -u main.py \
    --data_path "$DATA_DIR" \
    --teacher_path "$TEACHER_MODEL" \
    --epochs 50 \
    --alpha 0.5 \
    --beta 0.1 \
    --student_type "small" > "$LOG_FILE" 2>&1 &

PID=$!

# 4. Background Git Sync
(
    wait $PID
    git add "$LOG_FILE"
    git commit -m "📊 Student Distillation Log: $TS"
    git push origin model_change1_15march
    echo "✅ Logs uploaded to GitHub."
) &