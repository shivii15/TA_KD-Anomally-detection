#!/bin/bash
# train_teacher.sh - Train the Teacher model first

source venv_tgkd/bin/activate

TS=$(date +%Y%m%d_%H%M)
LOG_FILE="logs/teacher_train_${TS}.log"

echo "👩‍🏫 Starting Teacher Training... Logging to $LOG_FILE"

# Read the path that download_data.py just saved
DATA_DIR=$(cat .data_path)

# Assuming you have a script named train_teacher_only.py or similar
# If you use main.py for this, ensure it has a --mode flag
nohup python3 -u train_teacher.py \
    --data_path "$DATA_DIR" \
    --save_path "./models/teacher_best.pth" \
    --epochs 30 \
    --batch_size 512 > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ Teacher training PID: $PID. You can monitor with: tail -f $LOG_FILE"

# Auto-push the teacher log when done
(
    wait $PID
    git add "$LOG_FILE"
    git commit -m "📈 Teacher Training Complete: $TS"
    git push origin model_change1_15march
) &