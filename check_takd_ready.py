import os
import torch
import joblib
import argparse

def verify_setup(args):
    print("🔍 Starting TGKD Pre-Flight Check...\n")
    all_pass = True

    # 1. Check Data Path
    if os.path.exists(args.data_path):
        files = [f for f in os.listdir(args.data_path) if f.endswith('.csv')]
        print(f"✅ Data Path Found: {args.data_path}")
        print(f"   📊 Found {len(files)} CSV parts.")
    else:
        print(f"❌ Data Path NOT Found: {args.data_path}")
        all_pass = False

    # 2. Check Teacher Model
    if os.path.exists(args.teacher_path):
        try:
            ckpt = torch.load(args.teacher_path, map_location='cpu')
            print(f"✅ Teacher Weights Found: {args.teacher_path}")
            if 'model_state_dict' in ckpt:
                print("   🏷️ Format: Metadata-wrapped (Correct)")
            else:
                print("   ⚠️ Format: Raw State Dict (Will still work)")
        except Exception as e:
            print(f"❌ Teacher Weights Corrupt: {e}")
            all_pass = False
    else:
        print(f"❌ Teacher Weights NOT Found: {args.teacher_path}")
        all_pass = False

    # 3. Check Anomaly Module (Isolation Forest)
    if os.path.exists(args.iso_path):
        try:
            iso = joblib.load(args.iso_path)
            print(f"✅ Anomaly Module Found: {args.iso_path}")
            print(f"   🌲 Estimators: {iso.n_estimators}")
        except Exception as e:
            print(f"❌ Anomaly Module Corrupt: {e}")
            all_pass = False
    else:
        print(f"❌ Anomaly Module NOT Found: {args.iso_path}")
        print("   💡 Run 'python3 train_anomaly_detector.py' first.")
        all_pass = False

    # 4. Final Verdict
    print("\n" + "="*30)
    if all_pass:
        print("🚀 ALL SYSTEMS GO! You are ready to run main.py")
    else:
        print("🛑 STOP! Fix the errors above before running distillation.")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--iso_path', type=str, default="models/iso_forest_iot.pkl")
    args = parser.parse_args()
    verify_setup(args)