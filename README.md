# gps-spoofing-detection
AV-GPS-Dataset을 활용하여 TS-LIF로 GPS spoofing 공격을 탐지합니다.

# AV-GPS-Dataset
https://github.com/mehrab-abrar/AV-GPS-Dataset/

# Directory Structure
```
├── data/ # (gitignore)
│   ├── raw/
│   │   ├── AV-GPS-Dataset-1.csv
│   │   ├── AV-GPS-Dataset-2.csv
│   │   └── AV-GPS-Dataset-3.csv
│   └── processed/
│       ├── X_train.npy
│       ├── y_train.npy
│       ├── X_val.npy
│       ├── y_val.npy
│       ├── X_test.npy
│       ├── y_test.npy
│       └── scalers.pkl
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── preprocessing.py
    │   └── loader.py
    └── models/ # (TODO)
```

# Quick Start
```bash
pip install -r requirements.txt
python src/data/preprocessing.py
python src/data/loader.py
```