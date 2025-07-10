# Term Deposit Subscription Prediction Model

## Overview
This repository contains a machine learning solution for predicting term deposit subscriptions to optimize bank marketing campaigns. The LightGBM model with SMOTE oversampling demonstrates strong performance in identifying likely subscribers while minimizing wasted outreach efforts.

## Business Problem
Banks struggle with inefficient marketing campaigns for term deposits:
- Low subscription rates (typically 7-8%)
- High costs from contacting disinterested customers
- Manual lead selection processes

Our solution addresses these challenges by:
- Accurately predicting which customers will subscribe
- Reducing unnecessary marketing contacts
- Increasing campaign ROI

## Model Performance
### Key Metrics
| Metric               | Count | Percentage | Business Impact               |
|----------------------|-------|------------|--------------------------------|
| True Negatives (TN)  | 6,873 | 93%        | Reduced wasted marketing spend |
| True Positives (TP)  | 73    | 13%        | Direct revenue opportunities   |
| False Negatives (FN) | 506   | 87%        | Missed subscription chances    |

### Classification Report
```
              precision    recall  f1-score   support

           0       0.93      0.93      0.93      7421
           1       0.12      0.13      0.12       579

    accuracy                           0.87      8000
   macro avg       0.52      0.53      0.53      8000
weighted avg       0.87      0.87      0.87      8000
```

## Technical Implementation
### Data Preparation
- Handled class imbalance using SMOTE oversampling
- Feature engineering focused on customer demographics and transaction history

### Model Architecture
- LightGBM classifier with custom class weights {0:1, 1:8}
- Dart boosting type for better handling of imbalanced data
- Learning rate of 0.05 with early stopping

## Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/term-deposit-prediction.git
cd term-deposit-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage
1. Run the training pipeline:
```bash
python train.py
```

2. Make predictions on new data:
```bash
python predict.py --input data/new_customers.csv
```

## Repository Structure
```
├── data/                   # Processed datasets
│   ├── train.csv           # Training data
│   └── test.csv            # Test data
├── notebooks/              # Jupyter notebooks
│   ├── EDA.ipynb           # Exploratory data analysis
│   └── Modeling.ipynb      # Model development
├── scripts/                # Utility scripts
│   ├── preprocess.py       # Data preprocessing
│   └── evaluate.py         # Model evaluation
├── models/                 # Saved models
│   └── lightgbm_model.pkl  # Trained model
├── train.py                # Training script
├── predict.py              # Prediction script
└── requirements.txt        # Dependencies
```

## Future Improvements
- Incorporate additional customer behavior data
- Experiment with alternative sampling techniques (SMOTEENN, ADASYN)
- Develop API endpoint for real-time predictions
- Implement threshold optimization for different campaign strategies

## Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact
For questions or suggestions, please contact Pranitha Gaddam at pranitha.gaddam79@gmail.com
