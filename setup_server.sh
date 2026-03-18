#!/bin/bash

# --- CONFIGURATION ---
PROJECT_NAME="TA_KD-Anomally-detection"
VENV_NAME="venv_tgkd"

echo "🌐 Setting up TGKD-IoT Framework on Server..."

# 1. Create Folders
mkdir -p data models outputs logs checkpoints

# 2. Setup Virtual Environment
if [ ! -d "$VENV_NAME" ]; then
    echo "📦 Creating Virtual Environment..."
    python3 -m venv $VENV_NAME
fi

source $VENV_NAME/bin/activate

# 3. Install Requirements
echo "📥 Installing Dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy scikit-learn tqdm matplotlib seaborn

# 4. GPU Verification
echo "🔍 Verifying GPU availability..."
python3 -c "import torch; print('✅ CUDA Available: ', torch.cuda.is_available()); print('🎮 Device Name: ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Configure Git Identity (Important for the auto-push to work)
git config --local user.email "shivangi.nigam@bennett.edu.in"
git config --local user.name "Shivangi Nigam"

echo "✨ Setup Complete!"
echo "👉 Place your dataset in: ./data/"
echo "👉 Place your teacher model in: ./models/"