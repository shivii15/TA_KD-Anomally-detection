# config/default_config.py

DEFAULT_CONFIG = {

    # ----------------------------
    # Training
    # ----------------------------
    "epochs": 100,
    "batch_size": 1024,
    "lr": 1e-3,

    # ----------------------------
    # TGKD
    # ----------------------------
    "temp_base": 3.0,
    "feat_weight": 0.5,

    # ----------------------------
    # Trust Module
    # ----------------------------
    "confidence_weight": 0.40,
    "anomaly_weight": 0.30,
    "entropy_weight": 0.20,
    "disagreement_weight": 0.10,

    # ----------------------------
    # Isolation Forest
    # ----------------------------
    "n_estimators": 100,
    "contamination": 0.01,

    # ----------------------------
    # Dataset
    # ----------------------------
    "num_parts": -1
}
