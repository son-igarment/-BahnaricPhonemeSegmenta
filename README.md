# Bahnaric Phoneme Segmentation

## Introduction
The "Bahnaric Phoneme Segmentation" project was developed by Pham Le Ngoc Son to support phoneme segmentation for the Bahnaric language. The main objective of the project is to empower Bahnaric language speakers, fostering communication within their community and with other ethnic groups through language processing technology.

## Project Structure
```
bahnaric-phoneme/
├── .devcontainer/        # Docker container configuration for development
├── .git/                 # Git directory
├── docs/                 # Project documentation
├── img/                  # Illustrative images
├── src/                  # Source code
│   ├── __init__.py
│   ├── acoustic_features.py  # Acoustic feature extraction
│   ├── constants.py          # Project constants
│   ├── dataset.py            # Data processing
│   ├── evaluate.py           # Model evaluation
│   ├── find_phoneme2.py      # Phoneme detection
│   ├── train_lgbm.py         # LightGBM model training
│   ├── train_xgboost.py      # XGBoost model training
│   ├── xgb_model.bin         # Trained XGBoost model
│   ├── op.txt                # Output data
│   └── ov.txt                # Output data
├── .gitignore           # Git ignore list
├── Dockerfile           # Docker configuration for deployment
├── Dockerfile.base      # Base Docker configuration
├── LICENSE              # License
├── README.md            # Documentation guide
├── poetry.lock          # Poetry dependency lock
├── pyproject.toml       # Project configuration
└── requirements.txt     # Dependency list
```

## Data
The data used in this project originates from a broader research initiative. Please ensure to obtain Pham Le Ngoc Son's consent prior to using the data for any other purposes.

## Feature Engineering
1. The project utilizes widely-used acoustic features, including MFCC (Mel-frequency cepstral coefficients), F0 (fundamental frequency), and energy.
2. In order to address the challenge of varying speech signal lengths, the project employs a sliding window approach to divide the speech signals into frames. Subsequently, it computes the average of each feature within each frame.

![Features](img/features.png)

## Machine Learning Models
The project utilizes two main machine learning models:
1. **XGBoost**: A popular Gradient Boosting model with high performance
2. **LightGBM**: A lightweight and efficient Gradient Boosting model

## Usage Guide
### Installing Libraries
```bash
pip install -r requirements.txt
```
or using Poetry:
```bash
poetry install
```

### Feature Generation
```bash
python src/dataset.py
```

### Model Training
```bash
# Using XGBoost
python src/train_xgboost.py

# Or using LightGBM
python src/train_lgbm.py
```

### Model Evaluation
```bash
python src/evaluate.py
```

### For Windows Users
1. Install Docker Desktop for Windows. [Guide](https://docs.docker.com/desktop/install/windows-install/)
2. Choose `Rebuild and Reopen in Container` in the popup window.

## Results
The project has achieved good results in phoneme segmentation for the Bahnaric language, contributing to the preservation and development of this language.

## Contact
For inquiries and contributions, please contact Pham Le Ngoc Son.

## License
The project is distributed under the license specified in the LICENSE file.
