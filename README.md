
# **TGKD-IoT: Trust-Gated Knowledge Distillation for Robust IoT Anomaly Detection**

## **📌 Overview**

Standard Knowledge Distillation (KD) assumes the "Teacher" model is an infallible oracle. However, in adversarial IoT environments, a Teacher can be compromised by noise or malicious traffic, passing "toxic knowledge" to the Student.

**TGKD-IoT** introduces a **Trust-Gating mechanism** $T(x) = C(x) \times A(x)$ that dynamically filters the distillation process. This ensures the edge-optimized Student only learns from the Teacher when the data is high-confidence ($C$) and non-anomalous ($A$).

## **🚀 Key Contributions**

* **Trust-Gating:** A novel $T(x)$ function that combines Bayesian confidence and density-based anomaly detection.
* **Adversarial Resilience:** Robustness against FGSM and PGD attacks on the Teacher.
* **Edge Optimized:** Distills a complex DNN into a 0.08MB Student suitable for constrained IoT devices.
* **Benchmark:** Evaluated on the modern **CIC-IoT-2023** dataset (34 classes, 47 features).

---

## **📂 Repository Structure**

```text
├── core/
│   ├── distiller.py        # TGKD Loss implementation
│   └── trust_module.py     # T(x) calculation logic
├── models/
│   ├── teacher.py          # 5-layer DNN Teacher
│   └── student_mlp.py      # Lightweight 3-layer Student (64-32-34)
├── scripts/
│   ├── train_baseline.py   # Standard Student training
│   ├── train_distill.py    # Proposed TGKD training loop
│   └── eval_adversarial.py # FGSM/Adversarial stress tests
├── data/
│   └── preprocess.py       # CIC-IoT-2023 feature selection (47 features)
└── README.md

```

---

## **📊 Performance Summary**

| Method | Accuracy | Size (MB) | Robustness (at $\epsilon=0.1$) |
| --- | --- | --- | --- |
| Teacher DNN |  | | |
| Baseline Student |  |  |  |
| **TGKD (Proposed)** |  |  | |

---

## **⚙️ Getting Started**

### **1. Installation**

```bash
git clone https://github.com/your-username/TGKD-IoT.git
cd TGKD-IoT
pip install -r requirements.txt

```

### **2. Preprocessing**

Place your CIC-IoT-2023 `.csv` files in the `data/raw/` directory and run:

```bash
python data/preprocess.py

```

### **3. Training**

To train the student using the Trust-Gated approach:

```bash
python scripts/train_distill.py --alpha 0.5 --temperature 2.0

```

---

## **📖 Citation**

If you use this code in your research, please cite:

```bibtex
@article{yourname2026tgkd,
  title={Trust-Gated Knowledge Distillation for Robust IoT Anomaly Detection},
  author={Your Name, Arpita Ma'am},
  journal={TBD (Under Review)},
  year={2026}
}

```

